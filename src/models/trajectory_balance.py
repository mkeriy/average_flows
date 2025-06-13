from gfn.gflownet.base import TrajectoryBasedGFlowNet
import torch
from torchtyping import TensorType as TT
from torch.nn.parameter import Parameter
from gfn.containers import Trajectories
from gfn.modules import GFNModule, ScalarEstimator


class AvgTBGFlowNet(TrajectoryBasedGFlowNet):
    def __init__(
        self,
        pf: GFNModule,
        pb: GFNModule,
        logZ: float | ScalarEstimator = 0.0,
        log_reward_clip_min: float = -float("inf"),
    ):
        super().__init__(pf, pb)

        if isinstance(logZ, float):
            self.logZ = Parameter(torch.tensor(logZ))
        else:
            assert isinstance(
                logZ, ScalarEstimator
            ), "logZ must be either float or a ScalarEstimator"
            self.logZ = logZ

        self.log_reward_clip_min = log_reward_clip_min

    def loss(
        self,
        env,
        training_objects: Trajectories,
        recalculate_all_logprobs: bool = True,
    ) -> TT[0, float]:
        del env
        total_log_pf_trajectories, _, scores = self.get_trajectories_scores(
            training_objects, recalculate_all_logprobs=recalculate_all_logprobs
        )
        loss = (scores + self.logZ).pow(2).mean()
        log_rewards = training_objects.log_rewards.clamp_min(self.log_reward_clip_min)
        addition = (log_rewards + total_log_pf_trajectories + self.logZ).mean()
        loss = loss - addition
        
        if torch.isnan(loss):
            raise ValueError("loss is nan")

        return loss