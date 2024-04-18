from shimmer.dataset import RepeatedDataset
from shimmer.modules.contrastive_loss import (
    ContrastiveLoss,
    ContrastiveLossType,
    ContrastiveLossWithConfidence,
    ContrastiveLossWithConfidenceType,
    contrastive_loss,
    contrastive_loss_with_confidence,
)
from shimmer.modules.domain import DomainModule, LossOutput
from shimmer.modules.global_workspace import (
    GlobalWorkspace,
    GlobalWorkspaceBase,
    GlobalWorkspaceWithConfidence,
    GWPredictions,
    SchedulerArgs,
    pretrained_global_workspace,
)
from shimmer.modules.gw_module import (
    GWDecoder,
    GWEncoder,
    GWEncoderLinear,
    GWModule,
    GWModuleBase,
    GWModuleWithConfidence,
)
from shimmer.modules.losses import (
    BroadcastLossCoefs,
    GWLosses,
    GWLossesBase,
    GWLossesWithConfidence,
    LossCoefs,
)
from shimmer.modules.selection import (
    RandomSelection,
    SelectionBase,
    SingleDomainSelection,
)
from shimmer.modules.utils import (
    batch_cycles,
    batch_demi_cycles,
    batch_translations,
    cycle,
    translation,
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
    "GlobalWorkspaceWithConfidence",
    "pretrained_global_workspace",
    "LossOutput",
    "DomainModule",
    "GWDecoder",
    "GWEncoder",
    "GWEncoderLinear",
    "GWModuleBase",
    "GWModule",
    "GWModuleWithConfidence",
    "ContrastiveLossType",
    "ContrastiveLossWithConfidenceType",
    "contrastive_loss",
    "ContrastiveLoss",
    "contrastive_loss_with_confidence",
    "ContrastiveLossWithConfidence",
    "LossCoefs",
    "BroadcastLossCoefs",
    "GWLossesBase",
    "GWLosses",
    "GWLossesWithConfidence",
    "RepeatedDataset",
    "reparameterize",
    "kl_divergence_loss",
    "gaussian_nll",
    "VAEEncoder",
    "VAEDecoder",
    "VAE",
    "batch_cycles",
    "batch_demi_cycles",
    "batch_translations",
    "cycle",
    "translation",
    "RandomSelection",
    "SelectionBase",
    "SingleDomainSelection",
]
