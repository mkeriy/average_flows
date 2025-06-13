from argparse import ArgumentParser
import json
from omegaconf import DictConfig
import torch

from datetime import datetime


from torchtune.training.metric_logging import WandBLogger, TensorBoardLogger
from gfn.gym import HyperGrid
from gfn.utils.common import set_seed


from src.models import get_model
from src.buffer import get_buffer
from src.trainer import Trainer
import os

DEFAULT_SEED = 4444


def main(args: DictConfig):  # noqa: C901
    seed = args.seed if args.seed != 0 else DEFAULT_SEED
    set_seed(seed)
    device_str = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    exp = f"{args.wandb_run_name}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
    args.checkpointer.checkpoint_dir = f"./ckpts/{exp}"
    os.mkdir(args.checkpointer.checkpoint_dir)
    logger = TensorBoardLogger(log_dir=f"./logs/{args.wandb_run_name}")
    # logger = WandBLogger(
    #     project=args.wandb_project,
    #     entity=args.wandb_entity,
    #     # group=...
    #     name=exp,
    # )
    logger.log_config(args)

    # 1. Create the environment
    env = HyperGrid(
        args.ndim, args.height, args.R0, args.R1, args.R2, device_str=device_str
    )

    # 2. Create the gflownets.
    #    For this we need modules and estimators.
    #    Depending on the loss, we may need several estimators:
    #       one (forward only) for FM loss,
    #       two (forward and backward) or other losses
    #       three (same, + logZ) estimators for TB.
    gflownet = get_model(env, args)

    assert gflownet is not None, f"No gflownet for loss {args.loss}"

    # Initialize the replay buffer ?
    replay_buffer = get_buffer(
        env, args.loss, args.replay_buffer_size, args.buffer_name
    )

    # 3. Create the optimizer
    # Policy parameters have their own LR.
    params = [
        {
            "params": [
                v for k, v in dict(gflownet.named_parameters()).items() if k != "logZ"
            ],
            "lr": args.lr,
        }
    ]

    # Log Z gets dedicated learning rate (typically higher).
    if "logZ" in dict(gflownet.named_parameters()):

        params.append(
            {
                "params": [dict(gflownet.named_parameters())["logZ"]],
                "lr": args.lr_Z,
            }
        )

    optimizer = torch.optim.Adam(params)

    trainer = Trainer(
        model=gflownet,
        optimizer=optimizer,
        env=env,
        buffer=replay_buffer,
        n_trajectories=args.n_trajectories,
        batch_size=args.batch_size,
        logger=logger,
        validation_interval=args.validation_interval,
        validation_samples=args.validation_samples,
        save_dir=args.checkpointer.checkpoint_dir,
    )

    trainer.train()
    
    logger.close()

    return "--La Finale--"


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--config", default="", help="Config file")
    parser.add_argument("--name", default="experiment", help="Name of experiment")

    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as file:
        conf = json.load(file)
    config = DictConfig(conf)
    config.wandb_run_name = args.name

    print(main(config))
