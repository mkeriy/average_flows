from gfn.containers import ReplayBuffer, PrioritizedReplayBuffer

from gfn.env import Env


def get_buffer(env: Env, loss: str, buffer_size: int = 0, buffer: str = ""):

    if buffer_size > 0:
        if loss in ("TB", "SubTB", "ZVar"):
            objects_type = "trajectories"
        elif loss in ("DB", "ModifiedDB"):
            objects_type = "transitions"
        elif loss == "FM":
            objects_type = "states"
        else:
            raise NotImplementedError(f"Unknown loss: {loss}")

        if buffer == "prioritized":
            return PrioritizedReplayBuffer(
                env,
                objects_type=objects_type,
                capacity=buffer_size,
                p_norm_distance=1,  # Use L1-norm for diversity estimation.
                cutoff_distance=0,  # -1 turns off diversity-based filtering.
            )
        elif buffer == "base":
            return ReplayBuffer(env, objects_type=objects_type, capacity=buffer_size)
    return None
