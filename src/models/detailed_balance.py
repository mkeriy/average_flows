import math
from typing import Tuple

import torch
from torchtyping import TensorType as TT

from gfn.containers import Trajectories, Transitions
from gfn.env import Env
from gfn.gflownet.base import PFBasedGFlowNet
from gfn.modules import GFNModule, ScalarEstimator
from gfn.utils.common import has_log_probs



class AvgDBGFlowNet(PFBasedGFlowNet[Transitions]):
    r"""
    """

    def __init__(
        self,
        pf: GFNModule,
        pb: GFNModule,
        logF: ScalarEstimator,
        forward_looking: bool = False,
        log_reward_clip_min: float = -float("inf"),
    ):
        super().__init__(pf, pb)
        assert isinstance(logF, ScalarEstimator), "logF must be a ScalarEstimator"
        self.logF = logF
        self.forward_looking = forward_looking
        self.log_reward_clip_min = log_reward_clip_min

    def logF_named_parameters(self):
        try:
            return {k: v for k, v in self.named_parameters() if "logF" in k}
        except KeyError as e:
            print(
                "logF not found in self.named_parameters. Are the weights tied with PF? {}".format(
                    e
                )
            )

    def logF_parameters(self):
        try:
            return [v for k, v in self.named_parameters() if "logF" in k]
        except KeyError as e:
            print(
                "logF not found in self.named_parameters. Are the weights tied with PF? {}".format(
                    e
                )
            )

    def get_scores(
        self, env: Env, transitions: Transitions, recalculate_all_logprobs: bool = False
    ) -> Tuple[
        TT["n_transitions", float],
        TT["n_transitions", float],
        TT["n_transitions", float],
    ]:
        """Given a batch of transitions, calculate the scores.

        Args:
            transitions: a batch of transitions.

        Unless recalculate_all_logprobs=True, in which case we re-evaluate the logprobs of the transitions with
        the current self.pf. The following applies:
            - If transitions have log_probs attribute, use them - this is usually for on-policy learning
            - Else, re-evaluate the log_probs using the current self.pf - this is usually for
              off-policy learning with replay buffer

        Raises:
            ValueError: when supplied with backward transitions.
            AssertionError: when log rewards of transitions are None.
        """
        if transitions.is_backward:
            raise ValueError("Backward transitions are not supported")
        states = transitions.states
        actions = transitions.actions

        # uncomment next line for debugging
        # assert transitions.states.is_sink_state.equal(transitions.actions.is_dummy)

        if states.batch_shape != tuple(actions.batch_shape):
            raise ValueError("Something wrong happening with log_pf evaluations")

        if has_log_probs(transitions) and not recalculate_all_logprobs:
            valid_log_pf_actions = transitions.log_probs
        else:
            # Evaluate the log PF of the actions
            module_output = self.pf(
                states
            )  # TODO: Inefficient duplication in case of tempered policy
            # The Transitions container should then have some
            # estimator_outputs attribute as well, to avoid duplication here ?
            # See (#156).
            valid_log_pf_actions = self.pf.to_probability_distribution(
                states, module_output
            ).log_prob(actions.tensor)

        valid_log_F_s = self.logF(states).squeeze(-1)
        
        log_rewards = env.log_reward(states)  # TODO: RM unsqueeze(-1) ?
        if math.isfinite(self.log_reward_clip_min):
            log_rewards = log_rewards.clamp_min(self.log_reward_clip_min)
        addition = valid_log_F_s + log_rewards

        preds = valid_log_pf_actions + valid_log_F_s
        targets = torch.zeros_like(preds)

        # uncomment next line for debugging
        # assert transitions.next_states.is_sink_state.equal(transitions.is_done)

        # automatically removes invalid transitions (i.e. s_f -> s_f)
        valid_next_states = transitions.next_states[~transitions.is_done]
        non_exit_actions = actions[~actions.is_exit]

        module_output = self.pb(valid_next_states)
        valid_log_pb_actions = self.pb.to_probability_distribution(
            valid_next_states, module_output
        ).log_prob(non_exit_actions.tensor)

        valid_transitions_is_done = transitions.is_done[
            ~transitions.states.is_sink_state
        ]

        valid_log_F_s_next = self.logF(valid_next_states).squeeze(-1)
        targets[~valid_transitions_is_done] = valid_log_pb_actions
        log_pb_actions = targets.clone()
        targets[~valid_transitions_is_done] += valid_log_F_s_next
        assert transitions.log_rewards is not None
        valid_transitions_log_rewards = transitions.log_rewards[
            ~transitions.states.is_sink_state
        ]
        targets[valid_transitions_is_done] = valid_transitions_log_rewards[
            valid_transitions_is_done
        ]

        scores = preds - targets
        return (valid_log_pf_actions, log_pb_actions, scores, valid_log_F_s, addition)

    def loss(self, env: Env, transitions: Transitions) -> TT[0, float]:
        """Detailed balance loss.

        The detailed balance loss is described in section
        3.2 of [GFlowNet Foundations](https://arxiv.org/abs/2111.09266)."""
        _, _, scores, valid_log_F_s, addition = self.get_scores(env, transitions)
        loss = torch.mean(torch.log1p(scores.abs()**1) * ( 1 + 0.06*addition))
        # loss = torch.mean(torch.log1p(scores**2) * (1 + 0.001 * torch.mean(addition)))
 
        if torch.isnan(loss):
            raise ValueError("loss is nan")

        return loss

    def to_training_samples(self, trajectories: Trajectories) -> Transitions:
        return trajectories.to_transitions()