from gfn.utils.modules import DiscreteUniform, NeuralNet, Tabular
from gfn.modules import DiscretePolicyEstimator, ScalarEstimator
from gfn.gflownet.base import GFlowNet
from gfn.env import Env
from omegaconf import DictConfig
from torch import nn

from gfn.gflownet import (
    DBGFlowNet,
    FMGFlowNet,
    LogPartitionVarianceGFlowNet,
    ModifiedDBGFlowNet,
    SubTBGFlowNet,
    TBGFlowNet,
)

from .trajectory_balance import AvgTBGFlowNet


def create_estimators(
    env: Env,
    pb_uniform: bool = False,
    tabular: bool = False,
    hidden_dim: int = None,
    n_layers: int = None,
    tied: bool = None,
) -> tuple[DiscretePolicyEstimator, DiscretePolicyEstimator, nn.Sequential]:
    if tabular:
        pf_module = Tabular(n_states=env.n_states, output_dim=env.n_actions)
        pb_module = Tabular(n_states=env.n_states, output_dim=env.n_actions - 1)
    else:
        pf_module = NeuralNet(
            input_dim=env.preprocessor.output_dim,
            output_dim=env.n_actions,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_layers,
        )
        pb_module = NeuralNet(
            input_dim=env.preprocessor.output_dim,
            output_dim=env.n_actions - 1,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_layers,
            torso=pf_module.torso if tied else None,
        )
    if pb_uniform:
        pb_module = DiscreteUniform(env.n_actions - 1)

    assert pf_module is not None, f"pf_module is None."
    assert pb_module is not None, f"pb_module is None."

    pf_estimator = DiscretePolicyEstimator(
        module=pf_module,
        n_actions=env.n_actions,
        preprocessor=env.preprocessor,
    )
    pb_estimator = DiscretePolicyEstimator(
        module=pb_module,
        n_actions=env.n_actions,
        is_backward=True,
        preprocessor=env.preprocessor,
    )

    return pf_estimator, pb_estimator, pf_module.torso


def get_model(env: Env, args: DictConfig) -> GFlowNet:

    pf_estimator, pb_estimator, pf_torso = create_estimators(
        env, args.pb_uniform, args.tabular, args.hidden_dim, args.n_hidden, args.tied
    )

    if args.loss == "FM":
        return FMGFlowNet(pf_estimator)
    elif args.loss == "TB":
        return TBGFlowNet(pf_estimator, pb_estimator)
    elif args.loss == "AvgTB":
        return AvgTBGFlowNet(pf_estimator, pb_estimator)

    ##Not implemented
    # elif args.loss == "ZVar":
    #     return LogPartitionVarianceGFlowNet(
    #         pf=pf_estimator,
    #         pb=pb_estimator,
    #     )
    # elif args.loss == "ModifiedDB":
    #     return ModifiedDBGFlowNet(
    #         pf_estimator,
    #         pb_estimator,
    #     )

    # if args.tabular:
    #     module = Tabular(n_states=env.n_states, output_dim=1)
    # else:
    #     module = NeuralNet(
    #         input_dim=env.preprocessor.output_dim,
    #         output_dim=1,
    #         hidden_dim=args.hidden_dim,
    #         n_hidden_layers=args.n_hidden,
    #         torso=pf_torso if args.tied else None,
    #     )

    # logF_estimator = ScalarEstimator(module=module, preprocessor=env.preprocessor)

    # if args.loss == "DB":
    #     return DBGFlowNet(
    #         pf=pf_estimator,
    #         pb=pb_estimator,
    #         logF=logF_estimator,
    #     )
    # elif args.loss == "SubTB":
    #     return SubTBGFlowNet(
    #         pf=pf_estimator,
    #         pb=pb_estimator,
    #         logF=logF_estimator,
    #         weighting=args.subTB_weighting,
    #         lamda=args.subTB_lambda,
    #     )
