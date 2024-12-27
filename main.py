from argparse import ArgumentParser
import json
from omegaconf import DictConfig
import torch
import wandb
from tqdm import trange

from gfn.containers import ReplayBuffer

from gfn.gym import HyperGrid

from src.models import TrajectoryBalance
from src.environments import HyperGridNT


def main(config: DictConfig, use_wandb: bool):  # noqa: C901
    seed = (
        config.seed if config.seed != 0 else torch.randint(int(10e10), (1,))[0].item()
    )
    torch.manual_seed(seed)

    device_str = "cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu"
# 1. Create the environment
    env = HyperGridNT(
        config.ndim,
        config.height,
        config.R0,
        config.R1,
        config.R2,
        device_str=device_str,
    )

    # 2. Create the gflownets.
    #    For this we need modules and estimators.
    #    Depending on the loss, we may need several estimators:
    #       one (forward only) for FM loss,
    #       two (forward and backward) or other losses
    #       three (same, + logZ) estimators for TB.
    models = {"TB": TrajectoryBalance, "FM": None, "DB": None}

    gflownet = models[config.loss](config, env)
    # Initialize the replay buffer ?

    replay_buffer = None
    if config.replay_buffer_size > 0:
        if config.loss in ("TB", "SubTB", "ZVar"):
            objects_type = "trajectories"
        elif config.loss in ("DB", "ModifiedDB"):
            objects_type = "transitions"
        elif config.loss == "FM":
            objects_type = "states"
        else:
            raise NotImplementedError(f"Unknown loss: {config.loss}")
        replay_buffer = ReplayBuffer(
            env, objects_type=objects_type, capacity=config.replay_buffer_size
        )

    # 3. Create the optimizer

    # Policy parameters have their own LR.

    # optimizer = torch.optim.Adam
    optimizer = torch.optim.Adam(gflownet.parameters())
    # trainer = Trainer(env, model, optimizer, replay_buffer, config.batch_size)
    visited_terminating_states = env.States.from_batch_shape((0,))

    states_visited = 0
    n_iterations = config.n_trajectories // config.batch_size
    validation_info = {"l1_dist": float("inf")}
    for iteration in trange(n_iterations):
        trajectories = gflownet.sample_trajectories(env, n_samples=config.batch_size)
        training_samples = gflownet.to_training_samples(trajectories)
        if replay_buffer is not None:
            with torch.no_grad():
                replay_buffer.add(training_samples)
                training_objects = replay_buffer.sample(
                    n_trajectories=config.batch_size
                )
        else:
            training_objects = training_samples

        optimizer.zero_grad()
        loss = gflownet.loss(env, training_objects)
        loss.backward()
        optimizer.step()

        visited_terminating_states.extend(trajectories.last_states)

        states_visited += len(trajectories)

        to_log = {"loss": loss.item(), "states_visited": states_visited}
        if use_wandb:
            wandb.log(to_log, step=iteration)
        if iteration % config.validation_interval == 0:
            validation_info, _, _ = gflownet.validate(
                env,
                config.validation_samples,
                visited_terminating_states,
            )

            if use_wandb:
                wandb.log(validation_info, step=iteration)
            to_log.update(validation_info)
            # tqdm.write(f"{iteration}: {to_log}")
    cum_reward = 0
    cum_log_reward = 0
    for i in trange(config.n_inf_validation):
        _, avg_reward, avg_log_reward = gflownet.validate(env, config.n_val_samples)
        cum_reward += avg_reward
        cum_log_reward += avg_log_reward
        info = {
            "validation_avg_reward": avg_reward,
            "validation_avg_log_reward": avg_log_reward,
            "validation_cummulative_reward": cum_reward,
            "validation_cummulative_log_reward": cum_log_reward,
        }
        wandb.log(info)

    return validation_info["l1_dist"]


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--config", default="", help="Config file")

    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as file:
        conf = json.load(file)
    config = DictConfig(conf)
    use_wandb = len(config.wandb_project) > 0
    if use_wandb:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.wandb_run_name,
            config=conf,
        )
    print(main(config, use_wandb))
