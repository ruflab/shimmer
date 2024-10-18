from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TypedDict

import torch

from shimmer.modules.contrastive_loss import ContrastiveLossType
from shimmer.modules.domain import DomainModule, LossOutput
from shimmer.modules.gw_module import GWModule, GWModuleBase
from shimmer.modules.selection import SelectionBase
from shimmer.types import LatentsDomainGroupsT, ModelModeT, RawDomainGroupsT


class GWLossesBase(torch.nn.Module, ABC):
    """
    Base Abstract Class for Global Workspace (GW) losses. This module is used
    to compute the different losses of the GW (typically translation, cycle,
    demi-cycle, contrastive losses).
    """

    @abstractmethod
    def step(
        self,
        raw_data: RawDomainGroupsT,
        domain_latents: LatentsDomainGroupsT,
        mode: ModelModeT,
    ) -> LossOutput:
        """
        Computes the losses.

        Args:
            raw_data (`RawDomainGroupsT`): raw input data
            domain_latents (`LatentsDomainGroupsT`): All latent groups
            mode (`Literal["train", "val", "test", "val/ood", "test/ood"]`): model mode
        Returns:
            `LossOutput`: the losses
        """
        ...


def demi_cycle_loss(
    gw_mod: GWModuleBase,
    selection_mod: SelectionBase,
    domain_mods: Mapping[str, DomainModule],
    latent_domains: LatentsDomainGroupsT,
    raw_data: RawDomainGroupsT,
) -> dict[str, torch.Tensor]:
    """
    Computes the demi-cycle loss.

    This return multiple metrics:
        * `demi_cycle_{domain_name}` with the demi-cycle of a particular domain;
        * `demi_cycle_{domain_name}_{metric}` with additional metrics provided by
            the domain_mod's `compute_dcy_loss` output;
        * `demi_cycles` with the average value of all `demi_cycle_{domain_name}` values.

    Args:
        gw_mod (`shimmer.modules.gw_module.GWModuleBase`): The GWModule to use
        selection_mod (`shimmer.modules.selection.SelectionBase`): Selection mod to use
        domain_mods (`Mapping[str, DomainModule]`): the domain modules
        latent_domains (`shimmer.types.LatentsDomainGroupsT`): the latent unimodal
            groups
        raw_data (`RawDomainGroupsT`): raw input data

    Returns:
        `dict[str, torch.Tensor]`: a dict of metrics.
    """
    losses: dict[str, torch.Tensor] = {}
    metrics: dict[str, torch.Tensor] = {}
    for domains, latents in latent_domains.items():
        if len(domains) > 1:
            continue
        domain_name = next(iter(domains))

        domain_mod = domain_mods[domain_name]
        x_recons = gw_mod.decode(
            gw_mod.encode_and_fuse(latents, selection_mod), domains={domain_name}
        )[domain_name]
        loss_output = domain_mod.compute_dcy_loss(
            x_recons, latents[domain_name], raw_data[domains][domain_name]
        )
        if loss_output is None:
            continue
        losses[f"demi_cycle_{domain_name}"] = loss_output.loss
        metrics.update(
            {f"demi_cycle_{domain_name}_{k}": v for k, v in loss_output.metrics.items()}
        )
    losses["demi_cycles"] = torch.stack(list(losses.values()), dim=0).mean()
    losses.update(metrics)
    return losses


def cycle_loss(
    gw_mod: GWModuleBase,
    selection_mod: SelectionBase,
    domain_mods: Mapping[str, DomainModule],
    latent_domains: LatentsDomainGroupsT,
    raw_data: RawDomainGroupsT,
) -> dict[str, torch.Tensor]:
    """
    Computes the cycle loss.

    This return multiple metrics:
        * `cycle_{domain_source}_through_{domain_target}` with the cycle of
            a particular domain;
        * `cycle_{domain_source}_through_{domain_target}_{metric}` with additional
            metrics provided by the domain_mod's `compute_cy_loss` output;
        * `cycles` with the average value of all
            `cycle_{domain_source}_through_{domain_target}` values.

    Args:
        gw_mod (`GWModuleBase`): The GWModule to use
        selection_mod (`shimmer.modules.selection.SelectionBase`): Selection mod to use
        domain_mods (`Mapping[str, DomainModule]`): the domain modules
        latent_domains (`LatentsDomainGroupsT`): the latent unimodal groups
        raw_data (`RawDomainGroupsT`): raw input data

    Returns:
        `dict[str, torch.Tensor]`: a dict of metrics.
    """
    losses: dict[str, torch.Tensor] = {}
    metrics: dict[str, torch.Tensor] = {}
    for domains_source, latents_source in latent_domains.items():
        if len(domains_source) > 1:
            continue
        domain_name_source = list(domains_source)[0]

        domain_mod = domain_mods[domain_name_source]
        z = gw_mod.encode_and_fuse(latents_source, selection_mod)
        for domain_name_target in domain_mods:
            if domain_name_target == domain_name_source:
                continue

            x_pred = gw_mod.decode(z, domains={domain_name_target})

            x_recons = gw_mod.decode(
                gw_mod.encode_and_fuse(x_pred, selection_mod),
                domains={domain_name_source},
            )

            loss_name = f"{domain_name_source}_through_{domain_name_target}"
            loss_output = domain_mod.compute_cy_loss(
                x_recons[domain_name_source],
                latents_source[domain_name_source],
                raw_data[domains_source][domain_name_source],
            )
            if loss_output is None:
                continue

            metrics.update(
                {f"cycle_{loss_name}_{k}": v for k, v in loss_output.metrics.items()}
            )
            losses[f"cycle_{loss_name}"] = loss_output.loss
    losses["cycles"] = torch.stack(list(losses.values()), dim=0).mean()
    losses.update(metrics)
    return losses


def translation_loss(
    gw_mod: GWModuleBase,
    selection_mod: SelectionBase,
    domain_mods: Mapping[str, DomainModule],
    latent_domains: LatentsDomainGroupsT,
    raw_data: RawDomainGroupsT,
) -> dict[str, torch.Tensor]:
    """
    Computes the translation loss.

    This return multiple metrics:
        * `translation_{domain_source}_to_{domain_target}` with the translation
            from a domain source to a domain target;
        * `translation_{domain_source}_to_{domain_target}_{metric}` with
            additional metrics provided by the domain_mod's
            `compute_tr_loss` output;
        * `translations` with the average value of all
            `translation_{domain_source}_to_{domain_target}` values.

    Args:
        gw_mod (`GWModuleBase`): The GWModule to use
        domain_mods (`Mapping[str, DomainModule]`): the domain modules
        latent_domains (`LatentsDomainGroupsT`): the latent unimodal groups
        raw_data (`RawDomainGroupsT`): raw input data

    Returns:
        `dict[str, torch.Tensor]`: a dict of metrics.
    """
    losses: dict[str, torch.Tensor] = {}
    metrics: dict[str, torch.Tensor] = {}
    for domains, latents in latent_domains.items():
        if len(domains) < 2:
            continue
        for domain_name_target in domains:
            domain_sources = {
                domain: latents[domain]
                for domain in domains
                if domain != domain_name_target
            }

            z = gw_mod.encode_and_fuse(domain_sources, selection_mod)
            mod = domain_mods[domain_name_target]

            domain_source_names = "/".join(domain_sources.keys())
            loss_name = f"{domain_source_names}_to_{domain_name_target}"
            if loss_name in losses:
                raise ValueError(f"{loss_name} is already computed.")

            prediction = gw_mod.decode(z, domains={domain_name_target})[
                domain_name_target
            ]
            loss_output = mod.compute_tr_loss(
                prediction,
                latents[domain_name_target],
                raw_data[domains][domain_name_target],
            )
            if loss_output is None:
                continue

            losses[f"translation_{loss_name}"] = loss_output.loss
            metrics.update(
                {
                    f"translation_{loss_name}_{k}": v
                    for k, v in loss_output.metrics.items()
                }
            )
    losses["translations"] = torch.stack(list(losses.values()), dim=0).mean()
    losses.update(metrics)
    return losses


def contrastive_loss(
    gw_mod: GWModuleBase,
    latent_domains: LatentsDomainGroupsT,
    contrastive_fn: ContrastiveLossType,
) -> dict[str, torch.Tensor]:
    """
    Computes the contrastive loss.

    This return multiple metrics:
        * `contrastive_{domain_1}_and_{domain_2}` with the contrastive
            between 2 domains;
        * `contrastive_{domain_1}_and_{domain_2}_{metric}` with
            additional metrics provided by the domain_mod's
            `compute_cont_loss` output;
        * `contrastives` with the average value of all
            `contrastive_{domain_1}_and_{domain_2}` values.

    Args:
        gw_mod (`GWModuleBase`): The GWModule to use
        latent_domains (`LatentsDomainGroupsT`): the latent unimodal groups
        contrastive_fn (`ContrastiveLossType`): the contrastive function to apply

    Returns:
        `dict[str, torch.Tensor]`: a dict of metrics.
    """
    losses: dict[str, torch.Tensor] = {}
    metrics: dict[str, torch.Tensor] = {}
    keys: list[set[str]] = []

    for latents in latent_domains.values():
        if len(latents) != 2:
            continue

        cont_latents = gw_mod.encode(latents)
        for domain1, z1 in cont_latents.items():
            for domain2, z2 in cont_latents.items():
                selected_domains = {domain1, domain2}
                if domain1 == domain2 or selected_domains in keys:
                    continue

                keys.append(selected_domains)

                loss_name = f"contrastive_{domain1}_and_{domain2}"
                loss_output = contrastive_fn(z1, z2)
                losses[loss_name] = loss_output.loss
                metrics.update(
                    {f"{loss_name}_{k}": v for k, v in loss_output.metrics.items()}
                )

    losses["contrastives"] = torch.stack(list(losses.values()), dim=0).mean()
    losses.update(metrics)
    return losses


class LossCoefs(TypedDict, total=False):
    """
    Dict of loss coefficients used in the GWLosses.

    If one is not provided, the coefficient is assumed to be 0 and will not be logged.
    If the loss is excplicitely set to 0, it will be logged, but not take part in
    the total loss.
    """

    demi_cycles: float
    """Demi-cycle loss coefficient."""

    cycles: float
    """Cycle loss coefficient."""

    translations: float
    """Translation loss coefficient."""

    contrastives: float
    """Contrastive loss coefficient."""


def combine_loss(
    metrics: dict[str, torch.Tensor],
    coefs: Mapping[str, float] | LossCoefs,
) -> torch.Tensor:
    """
    Combines the metrics according to the ones selected in coefs

    Args:
        metrics (`dict[str, torch.Tensor]`): all metrics to combine
        coefs (`Mapping[str, float] | LossCoefs | BroadcastLossCoefs`): coefs for
            selected metrics. Note, every metric does not need to be included here.
            If not specified, the metric will not count in the final loss.
            Also not that some metrics are redundant (e.g. `translations` contains
            all of the `translation_{domain_1}_to_{domain_2}`). You can look at the
            docs of the different losses for available values.

    Returns:
        `torch.Tensor`: the combined loss.
    """
    loss = torch.stack(
        [
            metrics[name] * coef
            for name, coef in coefs.items()
            if name in metrics and isinstance(coef, float) and coef > 0
        ],
        dim=0,
    ).mean()
    return loss


class GWLosses2Domains(GWLossesBase):
    """
    Implementation of `GWLossesBase` used for `GWModule`.
    """

    def __init__(
        self,
        gw_mod: GWModule,
        selection_mod: SelectionBase,
        domain_mods: dict[str, DomainModule],
        loss_coefs: LossCoefs | Mapping[str, float],
        contrastive_fn: ContrastiveLossType,
    ):
        """
        Main loss module to use with the GlobalWorkspace

        Args:
            gw_mod (`GWModule`): the GWModule
            selection_mod (`SelectionBase`): selection module
            domain_mods (`dict[str, DomainModule]`): a dict where the key is the
                domain name and value is the DomainModule
            loss_coefs (`LossCoefs`): loss coefficients. LossCoefs object, or a
                mapping to float with correct keys.
            contrastive_fn (`ContrastiveLossType`): the contrastive function to use
                in contrastive loss
        """

        super().__init__()
        self.gw_mod = gw_mod
        self.selection_mod = selection_mod
        self.domain_mods = domain_mods
        self.loss_coefs = loss_coefs
        self.contrastive_fn = contrastive_fn

    def demi_cycle_loss(
        self, latent_domains: LatentsDomainGroupsT, raw_data: RawDomainGroupsT
    ) -> dict[str, torch.Tensor]:
        """
        Computes the demi-cycle loss.

        See `shimmer.modules.losses.demi_cycle_loss`.

        Args:
            latent_domains (`LatentsDomainGroupsT`): the latent unimodal groups
            raw_data (`RawDomainGroupsT`): raw input data

        Returns:
            `dict[str, torch.Tensor]`: a dict of metrics.
        """
        return demi_cycle_loss(
            self.gw_mod, self.selection_mod, self.domain_mods, latent_domains, raw_data
        )

    def cycle_loss(
        self, latent_domains: LatentsDomainGroupsT, raw_data: RawDomainGroupsT
    ) -> dict[str, torch.Tensor]:
        """
        Computes the cycle loss.

        See `shimmer.modules.losses.cycle_loss`.

        Args:
            latent_domains (`LatentsDomainGroupsT`): the latent unimodal groups
            raw_data (`RawDomainGroupsT`): raw input data

        Returns:
            `dict[str, torch.Tensor]`: a dict of metrics.
        """
        return cycle_loss(
            self.gw_mod, self.selection_mod, self.domain_mods, latent_domains, raw_data
        )

    def translation_loss(
        self, latent_domains: LatentsDomainGroupsT, raw_data: RawDomainGroupsT
    ) -> dict[str, torch.Tensor]:
        """
        Computes the translation loss.

        See `shimmer.modules.losses.translation_loss`.

        Args:
            latent_domains (`LatentsDomainGroupsT`): the latent unimodal groups
            raw_data (`RawDomainGroupsT`): raw input data

        Returns:
            `dict[str, torch.Tensor]`: a dict of metrics.
        """
        return translation_loss(
            self.gw_mod, self.selection_mod, self.domain_mods, latent_domains, raw_data
        )

    def contrastive_loss(
        self, latent_domains: LatentsDomainGroupsT
    ) -> dict[str, torch.Tensor]:
        """
        Computes the contrastive loss.

        See `shimmer.modules.losses.contrastive_loss`.

        Args:
            latent_domains (`LatentsDomainGroupsT`): the latent unimodal groups

        Returns:
            `dict[str, torch.Tensor]`: a dict of metrics.
        """
        return contrastive_loss(self.gw_mod, latent_domains, self.contrastive_fn)

    def step(
        self,
        raw_data: RawDomainGroupsT,
        domain_latents: LatentsDomainGroupsT,
        mode: ModelModeT,
    ) -> LossOutput:
        """
        Computes and returns the losses

        Contains:
            - Demi-cycle metrics (see `GWLosses.demi_cycle_loss`)
            - Cycle metrics (see `GWLosses.cycle_loss`)
            - Translation metrics (see `GWLosses.translation_loss`)
            - Contrastive metrics (see `GWLosses.contrastive_loss`)

        Args:
            raw_data (`RawDomainGroupsT`): raw input data
            domain_latents (`LatentsDomainGroupsT`): All latent groups
            mode (`ModelModeT`): model mode
        Returns:
            `LossOutput`: the losses
        """
        metrics: dict[str, torch.Tensor] = {}

        metrics.update(self.demi_cycle_loss(domain_latents, raw_data))
        metrics.update(self.cycle_loss(domain_latents, raw_data))
        metrics.update(self.translation_loss(domain_latents, raw_data))
        metrics.update(self.contrastive_loss(domain_latents))

        return LossOutput(combine_loss(metrics, self.loss_coefs), metrics)
