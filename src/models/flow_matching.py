from gfn.gflownet import FMGFlowNet
from gfn.modules import DiscretePolicyEstimator
from gfn.states import DiscreteStates
from gfn.env import Env
import torch
from torchtyping import TensorType as TT

class AvgFMGFlowNet(FMGFlowNet):
    
    def reward_matching_loss(
        self, env: Env, terminating_states: DiscreteStates
    ) -> TT[0, float]:
        """Calculates the reward matching loss from the terminating states."""
        del env  # Unused
        assert terminating_states.log_rewards is not None
        log_edge_flows = self.logF(terminating_states)

        # Handle the boundary condition (for all x, F(X->S_f) = R(x)).
        terminating_log_edge_flows = log_edge_flows[:, -1]
        log_rewards = terminating_states.log_rewards
        addition = (log_rewards + terminating_log_edge_flows).mean()
        return (terminating_log_edge_flows - log_rewards).pow(2).mean() + addition