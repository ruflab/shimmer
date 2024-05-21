from shimmer.dataset import RepeatedDataset
from shimmer.modules.contrastive_loss import (
    ContrastiveLoss,
    ContrastiveLossBayesianType,
    ContrastiveLossType,
    contrastive_loss,
)
from shimmer.modules.domain import DomainModule, LossOutput
from shimmer.modules.global_workspace import (
    GlobalWorkspace2Domains,
    GlobalWorkspaceBase,
    GlobalWorkspaceBayesian,
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
    GWModuleBayesian,
)
from shimmer.modules.losses import (
    BroadcastLossCoefs,
    GWLosses2Domains,
    GWLossesBase,
    GWLossesBayesian,
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
    "GlobalWorkspace2Domains",
    "GlobalWorkspaceBayesian",
    "pretrained_global_workspace",
    "LossOutput",
    "DomainModule",
    "GWDecoder",
    "GWEncoder",
    "GWEncoderLinear",
    "GWModuleBase",
    "GWModule",
    "GWModuleBayesian",
    "ContrastiveLossType",
    "ContrastiveLossBayesianType",
    "contrastive_loss",
    "ContrastiveLoss",
    "LossCoefs",
    "BroadcastLossCoefs",
    "GWLossesBase",
    "GWLosses2Domains",
    "GWLossesBayesian",
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
