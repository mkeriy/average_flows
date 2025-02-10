from gfn.gflownet import GFlowNet, TBGFlowNet
from torch.optim import Optimizer
from gfn.env import Env
from gfn.containers import ReplayBuffer
from tqdm import tqdm, trange
from gfn.containers import Trajectories

import torch
from torchtune.training.metric_logging import MetricLoggerInterface
from collections import Counter
from typing import Dict, Optional

from src.models.trajectory_balance import AvgTBGFlowNet
from torchtyping import TensorType as TT
from gfn.states import States


def get_terminating_state_dist_pmf(env: Env, states: States) -> TT["n_states", float]:
    states_indices = env.get_terminating_states_indices(states).cpu().numpy().tolist()
    counter = Counter(states_indices)
    counter_list = [
        counter[state_idx] if state_idx in counter else 0
        for state_idx in range(env.n_terminating_states)
    ]

    return torch.tensor(counter_list, dtype=torch.float) / len(states_indices)


def validate(
    env: Env,
    gflownet: GFlowNet,
    n_validation_samples: int = 1000,
    visited_terminating_states: Optional[States] = None,
) -> Dict[str, float]:
    """Evaluates the current gflownet on the given environment.

    This is for environments with known target reward. The validation is done by
    computing the l1 distance between the learned empirical and the target
    distributions.

    Args:
        env: The environment to evaluate the gflownet on.
        gflownet: The gflownet to evaluate.
        n_validation_samples: The number of samples to use to evaluate the pmf.
        visited_terminating_states: The terminating states visited during training. If given, the pmf is obtained from
            these last n_validation_samples states. Otherwise, n_validation_samples are resampled for evaluation.

    Returns: A dictionary containing the l1 validation metric. If the gflownet
        is a TBGFlowNet, i.e. contains LogZ, then the (absolute) difference
        between the learned and the target LogZ is also returned in the dictionary.
    """

    true_logZ = env.log_partition
    true_dist_pmf = env.true_dist_pmf
    if isinstance(true_dist_pmf, torch.Tensor):
        true_dist_pmf = true_dist_pmf.cpu()
    else:
        # The environment does not implement a true_dist_pmf property, nor a log_partition property
        # We cannot validate the gflownet
        return {}

    logZ = None
    if isinstance(gflownet, TBGFlowNet) or isinstance(gflownet, AvgTBGFlowNet):
        logZ = gflownet.logZ.item()
    if visited_terminating_states is None:
        terminating_states = gflownet.sample_terminating_states(n_validation_samples)
    else:
        terminating_states = visited_terminating_states[-n_validation_samples:]

    final_states_dist_pmf = get_terminating_state_dist_pmf(env, terminating_states)
    l1_dist = (final_states_dist_pmf - true_dist_pmf).abs().mean().item()
    validation_info = {"l1_dist": l1_dist}
    if logZ is not None:
        validation_info["logZ_diff"] = abs(logZ - true_logZ)
    return validation_info


class Trainer:
    def __init__(
        self,
        model: GFlowNet,
        optimizer: Optimizer,
        env: Env,
        buffer: ReplayBuffer | None,
        n_trajectories: int,
        batch_size: int,
        logger: MetricLoggerInterface,
        validation_interval: int,
        validation_samples: int,
        save_dir: str,
    ):
        self.model = model
        self.optimizer = optimizer
        self.env = env
        self.buffer = buffer
        self.batch_size = batch_size
        self.n_trajectories = n_trajectories
        self.n_iterations = n_trajectories // batch_size
        self.logger = logger
        self.validation_interval = validation_interval
        self.validation_samples = validation_samples
        self.save_dir = save_dir

    def train(
        self,
    ) -> None:
        visited_terminating_states = self.env.states_from_batch_shape((0,))
        states_visited = 0
        for iteration in trange(self.n_iterations):

            loss, trajectories = self.train_step()

            visited_terminating_states.extend(trajectories.last_states)
            states_visited += len(trajectories)

            self.logger.log("loss", loss, step=iteration)

            if iteration % self.validation_interval == 0:
                validation_info = validate(
                    self.env,
                    self.model,
                    self.validation_samples,
                    visited_terminating_states,
                )
                self.logger.log_dict(validation_info, step=iteration)
                tqdm.write(f"{iteration}: loss - {loss}, {validation_info}")

        torch.save(self.model.state_dict(), self.save_dir + "/model.pt")

    def train_step(
        self,
    ) -> tuple[float, Trajectories]:
        trajectories = self.model.sample_trajectories(
            self.env,
            n_samples=self.batch_size,
            save_logprobs=self.buffer.capacity == 0,
            save_estimator_outputs=False,
        )
        training_samples = self.model.to_training_samples(trajectories)
        if self.buffer is not None:
            with torch.no_grad():
                self.buffer.add(training_samples)
                training_objects = self.buffer.sample(n_trajectories=self.batch_size)
        else:
            training_objects = training_samples

        self.optimizer.zero_grad()
        loss = self.model.loss(self.env, training_objects)
        loss.backward()
        self.optimizer.step()
        return loss.item(), trajectories


# def validate(

#     env,
#     n_samples,
#     visited_terminating_states=None,
# ):
#     true_logZ = env.log_partition
#     logZ = self.gflownet.logZ.item()
#     true_dist_pmf = env.true_dist_pmf
#     if isinstance(true_dist_pmf, torch.Tensor):
#         true_dist_pmf = true_dist_pmf.cpu()
#     else:
#         # The environment does not implement a true_dist_pmf property, nor a log_partition property
#         # We cannot validate the gflownet
#         return {}
#     if visited_terminating_states is None:
#         terminating_states = self.gflownet.sample_terminating_states(env, n_samples)
#     else:
#         terminating_states = visited_terminating_states[-n_samples:]
#     true_rewards = env.reward(terminating_states)
#     avg_true_reward = true_rewards.mean().item()
#     avg_log_reward = torch.log(true_rewards).mean().item()

#     final_states_dist_pmf = get_terminating_state_dist_pmf(env, terminating_states)
#     l1_dist = (final_states_dist_pmf - true_dist_pmf).abs().mean().item()
#     validation_info = {
#         "l1_dist": l1_dist,
#         "logZ_diff": abs(logZ - true_logZ),
#         "avg_true_reward": avg_true_reward,
#         "avg_log_reward": avg_log_reward,
#     }
#     return validation_info, avg_true_reward, avg_log_reward
