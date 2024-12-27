from gfn.gflownet import TBGFlowNet, FMGFlowNet
from gfn.modules import DiscretePolicyEstimator
from gfn.utils.modules import NeuralNet, Tabular
import torch
from torch.nn.parameter import Parameter
from gfn.containers import Trajectories
from gfn.utils.common import get_terminating_state_dist_pmf


class TrajectoryBalance:
    def __init__(self, config, env):
        pb_module = None
        if config.tabular:
            pf_module = Tabular(n_states=env.n_states, output_dim=env.n_actions)
            if not config.uniform_pb:
                pb_module = Tabular(n_states=env.n_states, output_dim=env.n_actions - 1)
        else:
            pf_module = NeuralNet(
                input_dim=env.preprocessor.output_dim,
                output_dim=env.n_actions,
                hidden_dim=config.hidden_dim,
                n_hidden_layers=config.n_hidden,
            )
            if not config.uniform_pb:
                pb_module = NeuralNet(
                    input_dim=env.preprocessor.output_dim,
                    output_dim=env.n_actions - 1,
                    hidden_dim=config.hidden_dim,
                    n_hidden_layers=config.n_hidden,
                    torso=pf_module.torso if config.tied else None,
                )

        self.pf_estimator = DiscretePolicyEstimator(
            module=pf_module,
            n_actions=env.n_actions,
            preprocessor=env.preprocessor,
        )
        self.pb_estimator = DiscretePolicyEstimator(
            module=pb_module,
            n_actions=env.n_actions,
            is_backward=True,
            preprocessor=env.preprocessor,
        )

        self.gflownet = TBGFlowNet(
            pf=self.pf_estimator,
            pb=self.pb_estimator,
            on_policy=True if config.replay_buffer_size == 0 else False,
        )
        self.average_flow_loss = config.average_flow_loss
        self.lr = config.lr
        self.lrZ = config.lrZ

    def load(
        self,
    ):
        pass

    def save(
        self,
    ):
        pass

    def validate(
        self,
        env,
        n_samples,
        visited_terminating_states=None,
    ):
        true_logZ = env.log_partition
        logZ = self.gflownet.logZ.item()
        true_dist_pmf = env.true_dist_pmf
        if isinstance(true_dist_pmf, torch.Tensor):
            true_dist_pmf = true_dist_pmf.cpu()
        else:
            # The environment does not implement a true_dist_pmf property, nor a log_partition property
            # We cannot validate the gflownet
            return {}
        if visited_terminating_states is None:
            terminating_states = self.gflownet.sample_terminating_states(env, n_samples)
        else:
            terminating_states = visited_terminating_states[-n_samples:]
        true_rewards = env.reward(terminating_states)
        avg_true_reward = true_rewards.mean().item()
        avg_log_reward = torch.log(true_rewards).mean().item()

        final_states_dist_pmf = get_terminating_state_dist_pmf(env, terminating_states)
        l1_dist = (final_states_dist_pmf - true_dist_pmf).abs().mean().item()
        validation_info = {
            "l1_dist": l1_dist,
            "logZ_diff": abs(logZ - true_logZ),
            "avg_true_reward": avg_true_reward,
            "avg_log_reward": avg_log_reward,
        }
        return validation_info, avg_true_reward, avg_log_reward

    def sample_trajectories(self, env, n_samples):
        return self.gflownet.sample_trajectories(env, n_samples)

    def to_training_samples(self, trajectories):
        return self.gflownet.to_training_samples(trajectories)

    def parameters(
        self,
    ) -> list[dict[str, Parameter | float]]:
        params = [
            {
                "params": [
                    v
                    for k, v in dict(self.gflownet.named_parameters()).items()
                    if k != "logZ"
                ],
                "lr": self.lr,
            }
        ]
        params.append(
            {
                "params": [dict(self.gflownet.named_parameters())["logZ"]],
                "lr": self.lrZ,
            }
        )

        return params

    def loss(self, env, training_object: Trajectories) -> torch.Tensor:
        del env
        total_log_pf_trajectories, _, scores = self.gflownet.get_trajectories_scores(
            training_object
        )
        loss = (scores + self.gflownet.logZ).pow(2).mean()
        if self.average_flow_loss:
            log_rewards = training_object.log_rewards.clamp_min(
                self.gflownet.log_reward_clip_min
            )
            addition = (
                (log_rewards + total_log_pf_trajectories + self.gflownet.logZ)
                .pow(2)
                .mean()
            )
            loss = loss + addition

        if torch.isnan(loss):
            raise ValueError("loss is nan")

        return loss
