from shimmer.dataset import RepeatedDataset
from shimmer.modules.contrastive_loss import (
    ContrastiveLoss,
    ContrastiveLossType,
    ContrastiveLossWithUncertainty,
    ContrastiveLossWithUncertaintyType,
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
from shimmer.types import (
    LatentsDomainGroupDT,
    LatentsDomainGroupsDT,
    LatentsDomainGroupsT,
    LatentsDomainGroupT,
    ModelModeT,
    RawDomainGroupDT,
    RawDomainGroupsDT,
    RawDomainGroupsT,
    RawDomainGroupT,
)
from shimmer.version import __version__

__all__ = [
    "__version__",
    "LatentsDomainGroupDT",
    "LatentsDomainGroupsDT",
    "LatentsDomainGroupsT",
    "LatentsDomainGroupT",
    "RawDomainGroupDT",
    "RawDomainGroupsDT",
    "RawDomainGroupsT",
    "RawDomainGroupT",
    "ModelModeT",
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
    "ContrastiveLossWithUncertaintyType",
    "contrastive_loss",
    "ContrastiveLoss",
    "contrastive_loss_with_uncertainty",
    "ContrastiveLossWithUncertainty",
    "LossCoefs",
    "GWLossesBase",
    "GWLosses",
    "GWLossesWithUncertainty",
    "RepeatedDataset",
]
