from shimmer.dataset import RepeatedDataset
from shimmer.modules.contrastive_loss import (
    ContrastiveLoss,
    ContrastiveLossType,
    ContrastiveLossWithUncertainty,
    VarContrastiveLossType,
    contrastive_loss,
    contrastive_loss_with_uncertainty,
)
from shimmer.modules.domain import DomainModule, LossOutput
from shimmer.modules.global_workspace import (
    GlobalWorkspace,
    GlobalWorkspaceBase,
    GlobalWorkspaceWithUncertainty,
    GWPredictions,
    SchedulerArgs,
    pretrained_global_workspace,
)
from shimmer.modules.gw_module import (
    GWDecoder,
    GWEncoder,
    GWEncoderLinear,
    GWEncoderWithUncertainty,
    GWModule,
    GWModuleBase,
    GWModuleWithUncertainty,
)
from shimmer.modules.losses import (
    GWLosses,
    GWLossesBase,
    GWLossesWithUncertainty,
    LossCoefs,
)
from shimmer.modules.vae import (
    VAE,
    VAEDecoder,
    VAEEncoder,
    gaussian_nll,
    kl_divergence_loss,
    reparameterize,
)

__all__ = [
    "SchedulerArgs",
    "GWPredictions",
    "GlobalWorkspaceBase",
    "GlobalWorkspace",
    "GlobalWorkspaceWithUncertainty",
    "pretrained_global_workspace",
    "LossOutput",
    "DomainModule",
    "GWDecoder",
    "GWEncoder",
    "GWEncoderLinear",
    "GWEncoderWithUncertainty",
    "GWModuleBase",
    "GWModule",
    "GWModuleWithUncertainty",
    "ContrastiveLossType",
    "VarContrastiveLossType",
    "contrastive_loss",
    "ContrastiveLoss",
    "contrastive_loss_with_uncertainty",
    "ContrastiveLossWithUncertainty",
    "LossCoefs",
    "GWLossesBase",
    "GWLosses",
    "GWLossesWithUncertainty",
    "RepeatedDataset",
    "reparameterize",
    "kl_divergence_loss",
    "gaussian_nll",
    "VAEEncoder",
    "VAEDecoder",
    "VAE",
]
