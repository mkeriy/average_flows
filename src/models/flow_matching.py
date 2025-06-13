from gfn.gflownet import FMGFlowNet
from gfn.modules import DiscretePolicyEstimator
from gfn.states import DiscreteStates
from gfn.env import Env
import torch
from torchtyping import TensorType as TT
from typing import Optional, Tuple


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
        addition = (log_rewards - terminating_log_edge_flows)
        return addition, log_rewards
    
    def flow_matching_loss(
        self,
        env: Env,
        states: DiscreteStates,
    ) -> TT["n_trajectories", torch.float]:
        """Computes the FM for the provided states.

        The Flow Matching loss is defined as the log-sum incoming flows minus log-sum
        outgoing flows. The states should not include $s_0$. The batch shape should be
        `(n_states,)`. As of now, only discrete environments are handled.

        Raises:
            AssertionError: If the batch shape is not linear.
            AssertionError: If any state is at $s_0$.
        """

        assert len(states.batch_shape) == 1
        assert not torch.any(states.is_initial_state)

        incoming_log_flows = torch.full_like(
            states.backward_masks, -float("inf"), dtype=torch.float
        )
        outgoing_log_flows = torch.full_like(
            states.forward_masks, -float("inf"), dtype=torch.float
        )

        for action_idx in range(env.n_actions - 1):
            valid_backward_mask = states.backward_masks[:, action_idx]
            valid_forward_mask = states.forward_masks[:, action_idx]
            valid_backward_states = states[valid_backward_mask]
            valid_forward_states = states[valid_forward_mask]

            backward_actions = torch.full_like(
                valid_backward_states.backward_masks[:, 0], action_idx, dtype=torch.long
            ).unsqueeze(-1)
            backward_actions = env.actions_from_tensor(backward_actions)

            valid_backward_states_parents = env._backward_step(
                valid_backward_states, backward_actions
            )

            incoming_log_flows[valid_backward_mask, action_idx] = self.logF(
                valid_backward_states_parents
            )[:, action_idx]

            outgoing_log_flows[valid_forward_mask, action_idx] = self.logF(
                valid_forward_states
            )[:, action_idx]

        # Now the exit action
        
        valid_forward_mask = states.forward_masks[:, -1]
        
        outgoing_log_flows[valid_forward_mask, -1] = self.logF(
            states[valid_forward_mask]
        )[:, -1]
        log_incoming_flows = torch.logsumexp(incoming_log_flows, dim=-1)
        log_outgoing_flows = torch.logsumexp(outgoing_log_flows, dim=-1)
        
        return (log_incoming_flows - log_outgoing_flows), log_incoming_flows

    def loss(
        self, env: Env, states_tuple: Tuple[DiscreteStates, DiscreteStates]
    ) -> TT[0, float]:
        """Given a batch of non-terminal and terminal states, compute a loss.

        Unlike the GFlowNets Foundations paper, we allow more flexibility by passing a
        tuple of states, the first one being the internal states of the trajectories
        (i.e. non-terminal states), and the second one being the terminal states of the
        trajectories."""
        
        intermediary_states, terminating_states = states_tuple
        fm_loss, in_f = self.flow_matching_loss(env, intermediary_states)
        rm_loss, log_rewards = self.reward_matching_loss(env, terminating_states)
        return 0.01 * fm_loss.pow(2).mean() + rm_loss.pow(2).mean()+ (log_rewards + in_f).abs().mean()
