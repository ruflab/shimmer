from collections.abc import Callable, Iterable, Mapping
from enum import Enum, auto
from pathlib import Path
from typing import Any, Generic, TypedDict, TypeVar, cast

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch.nn import Module, ModuleDict
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LRScheduler, OneCycleLR
from torch.optim.optimizer import Optimizer

from shimmer.modules.contrastive_loss import ContrastiveLoss, ContrastiveLossType
from shimmer.modules.domain import DomainModule, LossOutput
from shimmer.modules.gw_module import (
    GWModule,
    GWModuleBase,
    GWModulePrediction,
    broadcast_cycles,
    cycle,
    translation,
)
from shimmer.modules.losses import (
    BroadcastLossCoefs,
    GWLosses,
    GWLosses2Domains,
    GWLossesBase,
    LossCoefs,
)
from shimmer.modules.selection import (
    RandomSelection,
    SelectionBase,
    SingleDomainSelection,
)
from shimmer.types import (
    LatentsDomainGroupsDT,
    LatentsDomainGroupsT,
    ModelModeT,
    RawDomainGroupsDT,
    RawDomainGroupsT,
    RawDomainGroupT,
)
from shimmer.utils import groups_batch_size


class SchedulerArgs(TypedDict, total=False):
    """TypedDict of arguments passed to the OneCycle scheduler"""

    max_lr: float
    """Maximum learning rate"""

    total_steps: int
    """Total number of steps"""


class GWPredictionsBase(TypedDict):
    """TypedDict of the output given when calling `GlobalWorkspaceBase.predict`"""

    states: dict[frozenset[str], torch.Tensor]
    """
    GW state representation from domain groups with only one domain.
    The key represent the domain's name.
    """

    broadcasts: dict[frozenset[str], dict[str, torch.Tensor]]
    """
    broadcasts predictions of the model for each domain. It contains demi-cycles,
    translations, and fused.
    """

    cycles: dict[frozenset[str], dict[str, torch.Tensor]]
    """
    Cycle predictions of the model from one domain through another one.
    """


_T_gw_mod = TypeVar("_T_gw_mod", bound=GWModuleBase)
_T_selection_mod = TypeVar("_T_selection_mod", bound=SelectionBase)
_T_loss_mod = TypeVar("_T_loss_mod", bound=GWLossesBase)


def batch_demi_cycles(
    gw_mod: GWModuleBase,
    selection_mod: SelectionBase,
    latent_domains: LatentsDomainGroupsT,
) -> dict[str, torch.Tensor]:
    """
    Computes demi-cycles of a batch of groups of domains.

    Args:
        gw_mod (`GWModuleBase`): the GWModuleBase
        selection_mod (`SelectionBase`): selection module
        latent_domains (`LatentsT`): the batch of groups of domains

    Returns:
        `dict[str, torch.Tensor]`: demi-cycles predictions for each domain.
    """
    predictions: dict[str, torch.Tensor] = {}
    for domains, latents in latent_domains.items():
        if len(domains) > 1:
            continue
        domain_name = list(domains)[0]
        z = translation(gw_mod, selection_mod, latents, to=domain_name)
        predictions[domain_name] = z
    return predictions


def batch_cycles(
    gw_mod: GWModuleBase,
    selection_mod: SelectionBase,
    latent_domains: LatentsDomainGroupsT,
    through_domains: Iterable[str],
) -> dict[tuple[str, str], torch.Tensor]:
    """
    Computes cycles of a batch of groups of domains.

    Args:
        gw_mod (`GWModuleBase`): GWModule to use for the cycle
        selection_mod (`SelectionBase`): selection module
        latent_domains (`LatentsT`): the batch of groups of domains
        out_domains (`Iterable[str]`): iterable of domain names to do the cycle through.
            Each domain will be done separetely.

    Returns:
        `dict[tuple[str, str], torch.Tensor]`: cycles predictions for each
            couple of (start domain, intermediary domain).
    """
    predictions: dict[tuple[str, str], torch.Tensor] = {}
    for domains_source, latents_source in latent_domains.items():
        if len(domains_source) > 1:
            continue
        domain_name_source = next(iter(domains_source))
        for domain_name_through in through_domains:
            if domain_name_source == domain_name_through:
                continue
            z = cycle(
                gw_mod, selection_mod, latents_source, through=domain_name_through
            )
            domains = (domain_name_source, domain_name_through)
            predictions[domains] = z[domain_name_source]
    return predictions


def batch_translations(
    gw_mod: GWModuleBase,
    selection_mod: SelectionBase,
    latent_domains: LatentsDomainGroupsT,
) -> dict[tuple[str, str], torch.Tensor]:
    """
    Computes translations of a batch of groups of domains.

    Args:
        gw_mod (`GWModuleBase`): GWModule to do the translation
        selection_mod (`SelectionBase`): selection module
        latent_domains (`LatentsT`): the batch of groups of domains

    Returns:
        `dict[tuple[str, str], torch.Tensor]`: translation predictions for each
            couple of (start domain, target domain).
    """
    predictions: dict[tuple[str, str], torch.Tensor] = {}
    for domains, latents in latent_domains.items():
        if len(domains) < 2:
            continue
        for domain_name_source in domains:
            for domain_name_target in domains:
                if domain_name_source == domain_name_target:
                    continue
                prediction = translation(
                    gw_mod,
                    selection_mod,
                    {domain_name_source: latents[domain_name_source]},
                    to=domain_name_target,
                )
                predictions[(domain_name_source, domain_name_target)] = prediction
    return predictions


def batch_broadcasts(
    gw_mod: GWModuleBase,
    selection_mod: SelectionBase,
    latent_domains: LatentsDomainGroupsT,
) -> tuple[
    dict[frozenset[str], dict[str, torch.Tensor]],
    dict[frozenset[str], dict[str, torch.Tensor]],
]:
    """
    Computes all possible broadcast of a batch for each group of domains.

    Args:
        gw_mod (`GWModuleBase`): the GWModuleBase
        selection_mod (`SelectionBase`): selection module
        latent_domains (`LatentsT`): the batch of groups of domains

    Returns:
        `tuple[dict[frozenset[str], dict[str, torch.Tensor]],
        dict[frozenset[str], dict[str, torch.Tensor]], ]`: broadcast predictions
        for each domain."""
    predictions: dict[frozenset[str], dict[str, torch.Tensor]] = {}
    cycles: dict[frozenset[str], dict[str, torch.Tensor]] = {}
    for domains, latents in latent_domains.items():
        pred_broadcast, pred_cycles = broadcast_cycles(gw_mod, selection_mod, latents)
        predictions[domains] = pred_broadcast
        cycles[domains] = pred_cycles
    return predictions, cycles


class OneCycleSchedulerSentinel(Enum):
    """
    Used for backward-compatibility issues to use One-Cycle Scheduler by default
    """

    DEFAULT = auto()


class GlobalWorkspaceBase(
    Generic[_T_gw_mod, _T_selection_mod, _T_loss_mod], LightningModule
):
    """
    Global Workspace Lightning Module.

    This is the base class to build the Global Workspace.
    """

    def __init__(
        self,
        gw_mod: _T_gw_mod,
        selection_mod: _T_selection_mod,
        loss_mod: _T_loss_mod,
        optim_lr: float = 1e-3,
        optim_weight_decay: float = 0.0,
        scheduler_args: SchedulerArgs | None = None,
        scheduler: Callable[[Optimizer], LRScheduler]
        | None
        | OneCycleSchedulerSentinel = OneCycleSchedulerSentinel.DEFAULT,
    ) -> None:
        """
        Initializes a GW

        Args:
            gw_mod (`GWModuleBase`): the GWModule
            selection_mod (`SelectionBase`): selection module
            loss_mod (`GWLossesBase`): module to compute the GW losses.
            optim_lr (`float`): learning rate
            optim_weight_decay (`float`): weight decay
            scheduler_args (`SchedulerArgs`): `SchedulerArgs` instance to define
                scheduler parameters.
            scheduler (`Callable[[Optimizer], LRScheduler]`): Callback that returns the
                scheduler to use. If None is explicitely given, no scheduler
                will be used. By default, uses OneCycleScheduler
        """
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "gw_mod",
                "selection_mod",
                "domain_mods",
                "loss_mod",
                "domain_descriptions",
                "contrastive_loss",
                "gw_encoders",
                "gw_decoders",
                "scheduler",
            ]
        )

        self.gw_mod = gw_mod
        """ a `GWModuleBase` implementation."""

        self.selection_mod = selection_mod
        """A `SelectionBase` implementation."""

        self.loss_mod = loss_mod
        """The module that computes losses of the GW"""

        self.optim_lr = optim_lr
        self.optim_weight_decay = optim_weight_decay
        self.scheduler = scheduler
        self.scheduler_args = SchedulerArgs(max_lr=optim_lr, total_steps=1)
        if scheduler_args is not None:
            self.scheduler_args.update(scheduler_args)

    @property
    def domain_mods(self) -> Mapping[str, DomainModule]:
        return self.gw_mod.domain_mods

    @property
    def workspace_dim(self) -> int:
        """Dimension of the GW."""
        return self.gw_mod.workspace_dim

    def encode_and_fuse(
        self, x: LatentsDomainGroupsT, selection_module: SelectionBase
    ) -> dict[frozenset[str], torch.Tensor]:
        """
        Encode a group of latent representations into the GW representation.

        Args:
            x (`LatentsDomainGroupsT`): the input domain representations.
            selection_module (`SelectionBase`): selection module to use to obtain
                selection scores.

        Returns:
            `dict[frozenset[str], torch.Tensor]`: the GW representations.
        """
        return {
            domains: self.gw_mod.encode_and_fuse(latents, selection_module)
            for domains, latents in x.items()
        }

    def encode(self, x: LatentsDomainGroupsT) -> LatentsDomainGroupsDT:
        """
        Encode a group of latent representations into the pre-fusion GW representation.

        Args:
            x (`LatentsDomainGroupsT`): the input domain representations.

        Returns:
            `LatensDomainGroupsDT`: the GW representations.
        """
        return {domains: self.gw_mod.encode(latents) for domains, latents in x.items()}

    def fuse(
        self,
        x: LatentsDomainGroupsT,
        selection_scores: Mapping[frozenset[str], Mapping[str, torch.Tensor]],
    ) -> dict[frozenset[str], torch.Tensor]:
        """
        Fuses a group of latent representations into the GW representation.

        Args:
            x (`LatentsDomainGroupsT`): the pre-fusion latent representations
            selection_scores (`Mapping[frozenset[str], Mapping[str, torch.Tensor]]`):
                selection scores for each group

        Returns:
            `dict[frozenset[str], torch.Tensor]`: GW representation of each group
        """
        return {
            domains: self.gw_mod.fuse(latents, selection_scores[domains])
            for domains, latents in x.items()
        }

    def decode(
        self,
        z: Mapping[frozenset[str], torch.Tensor],
        domains: Iterable[str] | None = None,
    ) -> LatentsDomainGroupsDT:
        """
        Decode the group GW representation into given `domains`.

        Args:
            z (`torch.Tensor`): the GW representation.
            domains (`Iterable[str]`): iterable of domains to decode.

        Returns:
            `dict[str, torch.Tensor]`: the decoded unimodal representations.
        """
        return {
            domain_names: self.gw_mod.decode(gw_rep, domains)
            for domain_names, gw_rep in z.items()
        }

    def forward(  # type: ignore
        self,
        latent_domains: LatentsDomainGroupsT,
    ) -> GWPredictionsBase:
        """
        Computes demi-cycles, cycles, and translations.

        Args:
            latent_domains (`LatentsT`): Groups of domains for the computation.

        Returns:
            `GWPredictionsBase`: the predictions on the batch.
        """
        states: dict[frozenset[str], torch.Tensor] = {}
        broadcasts: dict[frozenset[str], dict[str, torch.Tensor]] = {}
        cycles: dict[frozenset[str], dict[str, torch.Tensor]] = {}
        for domain_group, latent_group in latent_domains.items():
            predictions = cast(
                GWModulePrediction, self.gw_mod(latent_group, self.selection_mod)
            )
            states[domain_group] = predictions["states"]
            broadcasts[domain_group] = predictions["broadcasts"]
            cycles[domain_group] = predictions["cycles"]

        return GWPredictionsBase(states=states, broadcasts=broadcasts, cycles=cycles)

    def batch_gw_states(
        self, latent_domains: LatentsDomainGroupsT
    ) -> dict[frozenset[str], torch.Tensor]:
        """
        Comptues GW states of a batch of groups of domains.

        Args:
            latent_domains (`LatentsT`): the batch of groups of domains

        Returns:
            `dict[str, torch.Tensor]`: states for each domain.
        """
        predictions: dict[frozenset[str], torch.Tensor] = {}
        for domains, latents in latent_domains.items():
            z = self.gw_mod.encode_and_fuse(latents, self.selection_mod)
            predictions[domains] = z
        return predictions

    def encode_domain(self, domain: Any, name: str) -> torch.Tensor:
        """
        Encodes a domain from the domain data into the unimodal representation.

        This is a convenient proxy for the `DomainModule.encode` method and is
        equivalent to:
        ```python
        self.domain_mods[name].encode(domain)
        ```

        Args:
            domain (`Any`): the domain data
            name (`str`): domain name to encode

        Returns:
            `torch.Tensor`: the domain's unimodal representation.
        """
        return self.domain_mods[name].encode(domain)

    def encode_domains(self, batch: RawDomainGroupsT) -> LatentsDomainGroupsDT:
        """
        Encode all domains in the batch.

        Args:
            batch (`RawDomainGroupsT`): the batch of
                domain groups with raw unimodal data to encode into groups of latent
                representations.

        Returns:
            `LatentsDomainGroupsDT`: the domains' unimodal representations.
        """
        return {
            domains: {
                name: self.encode_domain(domain, name) for name, domain in data.items()
            }
            for domains, data in batch.items()
        }

    def decode_domain(self, domain: torch.Tensor, name: str) -> Any:
        """
        Decodes a domain from the unimodal representation into the domain data.

        This is a convenient proxy for the `DomainModule.encode` method and is
        equivalent to:
        ```python
        self.domain_mods[name].decode(domain)
        ```

        Args:
            domain (`torch.Tensor`): the domain data
            name (`str`): domain name to encode

        Returns:
            `Any`: the domain's raw data.
        """
        return self.domain_mods[name].decode(domain)

    def decode_domains(self, latents_domain: LatentsDomainGroupsT) -> RawDomainGroupsDT:
        """
        Decodes all domains in the batch.

        Args:
            batch (`LatentsDomainGroupsT`): the batch of
                domain groups with unimodal latent representation to decode into
                groups of raw data.

        Returns:
            `LatentsDomainGroupsDT`: the domains' raw data.
        """
        return {
            domains: {
                name: self.decode_domain(domain, name)
                for name, domain in latents.items()
            }
            for domains, latents in latents_domain.items()
        }

    def unimodal_losses(self, batch: RawDomainGroupsT) -> LossOutput | None:
        metrics: dict[str, torch.Tensor] = {}
        losses: list[torch.Tensor] = []
        for group_domain_names, domain_group in batch.items():
            if len(group_domain_names) > 1:
                continue
            for domain_name, domain in domain_group.items():
                domain_mod = self.domain_mods[domain_name]
                if not domain_mod.is_frozen:
                    loss = domain_mod.compute_domain_loss(domain)
                    if loss is not None:
                        for name, metric in loss.metrics.items():
                            metrics[f"{domain_name}/{name}"] = metric
                        losses.append(loss.loss)
        if not len(losses):
            return None
        return LossOutput(loss=torch.stack(losses, dim=0).sum(), metrics=metrics)

    def generic_step(self, batch: RawDomainGroupsT, mode: ModelModeT) -> STEP_OUTPUT:
        """
        The generic step used in `training_step`, `validation_step` and
        `test_step`.

        Args:
            batch (`RawDomainGroupsT`): the batch of groups of raw unimodal data.
            mode (`ModelModeT`):

        Returns:
            `torch.Tensor`: the loss to train on.
        """
        domain_latents = self.encode_domains(batch)
        batch_size = groups_batch_size(domain_latents)

        loss_output = self.loss_mod.step(batch, domain_latents, mode)

        for name, metric in loss_output.all.items():
            self.log(
                f"{mode}/{name}",
                metric,
                batch_size=batch_size,
                add_dataloader_idx=False,
            )

        total_loss = loss_output.loss

        unimodal_losses = self.unimodal_losses(batch)
        if unimodal_losses is not None:
            for name, metric in unimodal_losses.all.items():
                self.log(
                    f"{mode}/domain_loss/{name}",
                    metric,
                    batch_size=batch_size,
                    add_dataloader_idx=False,
                )

            total_loss += unimodal_losses.loss

        return loss_output.loss

    def validation_step(  # type: ignore
        self, data: RawDomainGroupT, batch_idx: int, dataloader_idx: int = 0
    ) -> STEP_OUTPUT:
        """Validation step used by lightning"""

        batch = {frozenset(data.keys()): data}
        for domain in data:
            batch[frozenset([domain])] = {domain: data[domain]}
        if dataloader_idx == 0:
            return self.generic_step(batch, mode="val")
        return self.generic_step(batch, mode="val/ood")

    def test_step(  # type: ignore
        self, data: Mapping[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> STEP_OUTPUT:
        """Test step used by lightning"""

        batch = {frozenset(data.keys()): data}
        for domain in data:
            batch[frozenset([domain])] = {domain: data[domain]}
        if dataloader_idx == 0:
            return self.generic_step(batch, mode="test")
        return self.generic_step(batch, mode="test/ood")

    def training_step(  # type: ignore
        self, batch: Mapping[frozenset[str], Mapping[str, Any]], batch_idx: int
    ) -> STEP_OUTPUT:
        """Training step used by lightning"""

        return self.generic_step(batch, mode="train")

    def predict_step(  # type: ignore
        self, data: Mapping[str, Any], batch_idx: int
    ) -> GWPredictionsBase:
        """Predict step used by lightning"""

        batch = {frozenset(data.keys()): data}
        for domain in data:
            batch[frozenset([domain])] = {domain: data[domain]}

        domain_latents = self.encode_domains(batch)
        return self.forward(domain_latents)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """
        Configure models optimizers.

        Here we use `AdamW` for the optimizer and `OneCycleLR` for the learning-rate
        scheduler.
        """

        optimizer = AdamW(
            self.parameters(),
            lr=self.optim_lr,
            weight_decay=self.optim_weight_decay,
        )

        if self.scheduler is None:
            return {"optimizer": optimizer}

        lr_scheduler: LRScheduler
        if isinstance(self.scheduler, OneCycleSchedulerSentinel):
            lr_scheduler = OneCycleLR(optimizer, **self.scheduler_args)
        else:
            lr_scheduler = self.scheduler(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }


def freeze_domain_modules(
    domain_mods: Mapping[str, DomainModule],
) -> dict[str, DomainModule]:
    """
    Freezes weights and set to eval mode the domain modules.

    .. note::
        The output is casted as `dict[str, DomainModule]` type for better
        auto-completion, but is actually a torch `ModuleDict`.

    Args:
        domain_mods (`Mapping[str, DomainModule]`): mapping of domain modules to freeze

    Returns:
        `ModuleDict`: frozen modules.
    """

    for mod in domain_mods.values():
        if mod.is_frozen is None:
            mod.freeze()
    # Cast for better auto-completion at the expense of ModuleDict
    return cast(dict[str, DomainModule], ModuleDict(domain_mods))


class GlobalWorkspace2Domains(
    GlobalWorkspaceBase[GWModule, SingleDomainSelection, GWLosses2Domains]
):
    """
    A simple 2-domains max flavor of GlobalWorkspaceBase.

    This is used to simplify a Global Workspace instanciation and only overrides the
    `__init__` method.
    """

    def __init__(
        self,
        domain_mods: Mapping[str, DomainModule],
        gw_encoders: Mapping[str, Module],
        gw_decoders: Mapping[str, Module],
        workspace_dim: int,
        loss_coefs: LossCoefs | Mapping[str, float],
        optim_lr: float = 1e-3,
        optim_weight_decay: float = 0.0,
        scheduler_args: SchedulerArgs | None = None,
        learn_logit_scale: bool = False,
        contrastive_loss: ContrastiveLossType | None = None,
        scheduler: Callable[[Optimizer], LRScheduler]
        | None
        | OneCycleSchedulerSentinel = OneCycleSchedulerSentinel.DEFAULT,
        fusion_activation_fn: Callable[[torch.Tensor], torch.Tensor] = torch.tanh,
    ) -> None:
        """
        Initializes a Global Workspace

        Args:
            domain_mods (`Mapping[str, DomainModule]`): mapping of the domains
                connected to the GW. Keys are domain names, values are the
                `DomainModule`.
            gw_encoders (`Mapping[str, torch.nn.Module]`): mapping for each domain
                name to a `torch.nn.Module` class which role is to encode a
                unimodal latent representations into a GW representation (pre fusion).
            gw_decoders (`Mapping[str, torch.nn.Module]`): mapping for each domain
                name to a `torch.nn.Module` class which role is to decode a
                GW representation into a unimodal latent representations.
            workspace_dim (`int`): dimension of the GW.
            loss_coefs (`LossCoefs | Mapping[str, float]`): loss coefficients
            optim_lr (`float`): learning rate
            optim_weight_decay (`float`): weight decay
            scheduler_args (`SchedulerArgs | None`): optimization scheduler's arguments
            learn_logit_scale (`bool`): whether to learn the contrastive learning
                contrastive loss when using the default contrastive loss.
            contrastive_loss (`ContrastiveLossType | None`): a contrastive loss
                function used for alignment. `learn_logit_scale` will not affect custom
                contrastive losses.
            scheduler: The scheduler to use for traning. If None is explicitely given,
                no scheduler will be used. Defaults to use OneCycleScheduler
            fusion_activation_fn (`Callable[[torch.Tensor], torch.Tensor]`): activation
                function to fuse the domains.
        """
        domain_mods = freeze_domain_modules(domain_mods)

        gw_mod = GWModule(
            domain_mods, workspace_dim, gw_encoders, gw_decoders, fusion_activation_fn
        )
        if contrastive_loss is None:
            contrastive_loss = ContrastiveLoss(
                torch.tensor([1 / 0.07]).log(), "mean", learn_logit_scale
            )
        selection_mod = SingleDomainSelection()
        loss_mod = GWLosses2Domains(
            gw_mod, selection_mod, domain_mods, loss_coefs, contrastive_loss
        )

        super().__init__(
            gw_mod,
            selection_mod,
            loss_mod,
            optim_lr,
            optim_weight_decay,
            scheduler_args,
            scheduler,
        )


class GlobalWorkspaceFusion(GlobalWorkspaceBase[GWModule, RandomSelection, GWLosses]):
    """The fusion (with broadcast loss) flavor of GlobalWorkspaceBase.

    This is used to simplify a Global Workspace instanciation and only overrides the
    `__init__` method.
    """

    def __init__(
        self,
        domain_mods: Mapping[str, DomainModule],
        gw_encoders: Mapping[str, Module],
        gw_decoders: Mapping[str, Module],
        workspace_dim: int,
        loss_coefs: BroadcastLossCoefs | Mapping[str, float],
        selection_temperature: float = 0.2,
        optim_lr: float = 1e-3,
        optim_weight_decay: float = 0.0,
        scheduler_args: SchedulerArgs | None = None,
        learn_logit_scale: bool = False,
        contrastive_loss: ContrastiveLossType | None = None,
        scheduler: Callable[[Optimizer], LRScheduler]
        | None
        | OneCycleSchedulerSentinel = OneCycleSchedulerSentinel.DEFAULT,
        fusion_activation_fn: Callable[[torch.Tensor], torch.Tensor] = torch.tanh,
    ) -> None:
        """
        Initializes a Global Workspace

        Args:
            domain_mods (`Mapping[str, DomainModule]`): mapping of the domains
                connected to the GW. Keys are domain names, values are the
                `DomainModule`.
            gw_encoders (`Mapping[str, torch.nn.Module]`): mapping for each domain
                name to a `torch.nn.Module` class which role is to encode a
                unimodal latent representations into a GW representation (pre fusion).
            gw_decoders (`Mapping[str, torch.nn.Module]`): mapping for each domain
                name to a `torch.nn.Module` class which role is to decode a
                GW representation into a unimodal latent representations.
            workspace_dim (`int`): dimension of the GW.
            loss_coefs (`BroadcastLossCoefs | Mapping[str, float]`): loss coefs for the
                losses.
            selection_temperature (`float`): temperature value for the RandomSelection
                module.
            optim_lr (`float`): learning rate
            optim_weight_decay (`float`): weight decay
            scheduler_args (`SchedulerArgs | None`): optimization scheduler's arguments
            learn_logit_scale (`bool`): whether to learn the contrastive learning
                contrastive loss when using the default contrastive loss.
            contrastive_loss (`ContrastiveLossType | None`): a contrastive loss
                function used for alignment. `learn_logit_scale` will not affect custom
                contrastive losses.
            scheduler: The scheduler to use for traning. If None is explicitely given,
                no scheduler will be used. Defaults to use OneCycleScheduler
            fusion_activation_fn (`Callable[[torch.Tensor], torch.Tensor]`): activation
                function to fuse the domains.
        """
        domain_mods = freeze_domain_modules(domain_mods)
        gw_mod = GWModule(
            domain_mods, workspace_dim, gw_encoders, gw_decoders, fusion_activation_fn
        )

        if contrastive_loss is None:
            contrastive_loss = ContrastiveLoss(
                torch.tensor([1 / 0.07]).log(), "mean", learn_logit_scale
            )

        selection_mod = RandomSelection(selection_temperature)
        loss_mod = GWLosses(
            gw_mod, selection_mod, domain_mods, loss_coefs, contrastive_loss
        )

        super().__init__(
            gw_mod,
            selection_mod,
            loss_mod,
            optim_lr,
            optim_weight_decay,
            scheduler_args,
            scheduler,
        )


def pretrained_global_workspace(
    checkpoint_path: str | Path,
    domain_mods: Mapping[str, DomainModule],
    gw_encoders: Mapping[str, Module],
    gw_decoders: Mapping[str, Module],
    workspace_dim: int,
    loss_coefs: LossCoefs | Mapping[str, float],
    contrastive_fn: ContrastiveLossType,
    scheduler: LRScheduler
    | None
    | OneCycleSchedulerSentinel = OneCycleSchedulerSentinel.DEFAULT,
    fusion_activation_fn: Callable[[torch.Tensor], torch.Tensor] = torch.tanh,
    **kwargs,
) -> GlobalWorkspace2Domains:
    """
    Load a `GlobalWorkspace` flavor of `GlobalWorkspaceBase` from a checkpoint.

    Args:
        checkpoint_path (`str | Path`): path to checkpoint
        domain_mods (`Mapping[str, DomainModule]`): mapping of the domains
            connected to the GW. Keys are domain names, values are the
            `DomainModule`.
        gw_encoders (`Mapping[str, torch.nn.Module]`): mapping for each domain
            name to a `torch.nn.Module` class which role is to encode a
            unimodal latent representations into a GW representation (pre fusion).
        gw_decoders (`Mapping[str, torch.nn.Module]`): mapping for each domain
            name to a `torch.nn.Module` class which role is to decode a
            GW representation into a unimodal latent representations.
        workspace_dim (`int`): dimension of the GW.
        loss_coefs (`LossCoefs | Mapping[str, float]`): loss coefficients
        contrastive_loss (`ContrastiveLossType`): a contrastive loss
            function used for alignment. `learn_logit_scale` will not affect custom
            contrastive losses.
        scheduler: The scheduler to use for traning. If None is explicitely given,
            no scheduler will be used. Defaults to use OneCycleScheduler
        fusion_activation_fn (`Callable[[torch.Tensor], torch.Tensor]`): activation
            function to fuse the domains.
        **kwargs: additional arguments to pass to
            `GlobalWorkspace2Domains.load_from_checkpoint`.

    Returns:
        `GlobalWorkspace2Domains`: the pretrained `GlobalWorkspace2Domains`.

    Raises:
        `TypeError`: if loaded type is not `GlobalWorkspace`.
    """
    domain_mods = freeze_domain_modules(domain_mods)
    gw_mod = GWModule(
        domain_mods, workspace_dim, gw_encoders, gw_decoders, fusion_activation_fn
    )
    selection_mod = SingleDomainSelection()
    loss_mod = GWLosses2Domains(
        gw_mod, selection_mod, domain_mods, loss_coefs, contrastive_fn
    )

    gw = GlobalWorkspace2Domains.load_from_checkpoint(
        checkpoint_path,
        gw_mod=gw_mod,
        selection_mid=selection_mod,
        loss_coefs=loss_coefs,
        loss_mod=loss_mod,
        scheduler=scheduler,
        **kwargs,
    )
    if not isinstance(gw, GlobalWorkspace2Domains):
        raise TypeError("model should be of type GlobalWorkspace")
    return gw
