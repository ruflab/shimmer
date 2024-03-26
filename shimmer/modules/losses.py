from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TypedDict

import torch
import torch.nn.functional as F

from shimmer.modules.contrastive_loss import (
    ContrastiveLossType,
    ContrastiveLossWithUncertaintyType,
)
from shimmer.modules.domain import DomainModule, LossOutput
from shimmer.modules.gw_module import (
    GWModule,
    GWModuleBase,
    GWModuleFusion,
    GWModuleWithUncertainty,
)
from shimmer.types import LatentsDomainGroupsT, ModelModeT


class GWLossesBase(torch.nn.Module, ABC):
    """
    Base Abstract Class for Global Workspace (GW) losses. This module is used
    to compute the different losses of the GW (typically translation, cycle,
    demi-cycle, contrastive losses).
    """

    @abstractmethod
    def step(
        self,
        domain_latents: LatentsDomainGroupsT,
        mode: ModelModeT,
    ) -> LossOutput:
        """
        Computes the losses.

        Args:
            domain_latents (`LatentsDomainGroupsT`): All latent groups
            mode (`Literal["train", "val", "test", "val/ood", "test/ood"]`): model mode
        Returns:
            `LossOutput`: the losses
        """
        ...


def demi_cycle_loss(
    gw_mod: GWModuleBase,
    domain_mods: Mapping[str, DomainModule],
    latent_domains: LatentsDomainGroupsT,
) -> dict[str, torch.Tensor]:
    """Computes the demi-cycle loss.

    This return multiple metrics:
        * `demi_cycle_{domain_name}` with the demi-cycle of a particular domain;
        * `demi_cycle_{domain_name}_{metric}` with additional metrics provided by
            the domain_mod's `compute_dcy_loss` output;
        * `demi_cycles` with the average value of all `demi_cycle_{domain_name}` values.

    Args:
        gw_mod (`GWModuleBase`): The GWModule to use
        domain_mods (`Mapping[str, DomainModule]`): the domain modules
        latent_domains (`LatentsDomainGroupsT`): the latent unimodal groups

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
            gw_mod.encode_and_fuse(gw_mod.on_before_gw_encode_dcy(latents)),
            domains={domain_name},
        )[domain_name]
        loss_output = domain_mod.compute_dcy_loss(x_recons, latents[domain_name])
        losses[f"demi_cycle_{domain_name}"] = loss_output.loss
        metrics.update(
            {f"demi_cycle_{domain_name}_{k}": v for k, v in loss_output.metrics.items()}
        )
    losses["demi_cycles"] = torch.stack(list(losses.values()), dim=0).mean()
    losses.update(metrics)
    return losses


def cycle_loss(
    gw_mod: GWModuleBase,
    domain_mods: Mapping[str, DomainModule],
    latent_domains: LatentsDomainGroupsT,
) -> dict[str, torch.Tensor]:
    """Computes the cycle loss.

    This return multiple metrics:
        * `cycle_{domain_source}_through_{domain_target}` with the cycle of
            a particular domain;
        * `cycle_{domain_source}_through_{domain_target}_{metric}` with additional
            metrics provided by the domain_mod's `compute_cy_loss` output;
        * `cycles` with the average value of all
            `cycle_{domain_source}_through_{domain_target}` values.

    Args:
        gw_mod (`GWModuleBase`): The GWModule to use
        domain_mods (`Mapping[str, DomainModule]`): the domain modules
        latent_domains (`LatentsDomainGroupsT`): the latent unimodal groups

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
        z = gw_mod.encode_and_fuse(gw_mod.on_before_gw_encode_cy(latents_source))
        for domain_name_target in domain_mods.keys():
            if domain_name_target == domain_name_source:
                continue

            x_pred = gw_mod.decode(z, domains={domain_name_target})
            x_recons = gw_mod.decode(
                gw_mod.encode_and_fuse(x_pred), domains={domain_name_source}
            )

            loss_name = f"{domain_name_source}_through_{domain_name_target}"
            loss_output = domain_mod.compute_cy_loss(
                x_recons[domain_name_source],
                latents_source[domain_name_source],
            )
            metrics.update(
                {f"cycle_{loss_name}_{k}": v for k, v in loss_output.metrics.items()}
            )
            losses[f"cycle_{loss_name}"] = loss_output.loss
    losses["cycles"] = torch.stack(list(losses.values()), dim=0).mean()
    losses.update(metrics)
    return losses


def translation_loss(
    gw_mod: GWModuleBase,
    domain_mods: Mapping[str, DomainModule],
    latent_domains: LatentsDomainGroupsT,
) -> dict[str, torch.Tensor]:
    """Computes the translation loss.

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

            z = gw_mod.encode_and_fuse(gw_mod.on_before_gw_encode_tr(domain_sources))
            mod = domain_mods[domain_name_target]

            domain_source_names = "/".join(domain_sources.keys())
            loss_name = f"{domain_source_names}_to_{domain_name_target}"
            if loss_name in losses.keys():
                raise ValueError(f"{loss_name} is already computed.")

            prediction = gw_mod.decode(z, domains={domain_name_target})[
                domain_name_target
            ]
            loss_output = mod.compute_tr_loss(
                prediction,
                latents[domain_name_target],
            )
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
    """Computes the contrastive loss.

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

        cont_latents = gw_mod.encode(gw_mod.on_before_gw_encode_cont(latents))
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


def contrastive_loss_with_uncertainty(
    gw_mod: GWModuleWithUncertainty,
    latent_domains: LatentsDomainGroupsT,
    contrastive_fn: ContrastiveLossWithUncertaintyType,
) -> dict[str, torch.Tensor]:
    """Computes the contrastive loss with uncertainty.

    This return multiple metrics:
        * `contrastive_{domain_1}_and_{domain_2}` with the contrastive
            between 2 domains;
        * `contrastive_{domain_1}_and_{domain_2}_{metric}` with
            additional metrics provided by the domain_mod's
            `compute_cont_loss` output;
        * `contrastives` with the average value of all
            `contrastive_{domain_1}_and_{domain_2}` values.

    Args:
        gw_mod (`GWModuleWithUncertainty`): The GWModule to use
        latent_domains (`LatentsDomainGroupsT`): the latent unimodal groups
        contrastive_fn (`ContrastiveLossWithUncertaintyType`): the contrastive function
            to apply

    Returns:
        `dict[str, torch.Tensor]`: a dict of metrics.
    """
    losses: dict[str, torch.Tensor] = {}
    metrics: dict[str, torch.Tensor] = {}
    keys: list[set[str]] = []

    for latents in latent_domains.values():
        if len(latents) < 2:
            continue
        for domain1_name, domain1 in latents.items():
            z1_mean, z1_log_uncertainty = gw_mod.encoded_distribution(
                gw_mod.on_before_gw_encode_cont({domain1_name: domain1})
            )
            for domain2_name, domain2 in latents.items():
                selected_domains = {domain1_name, domain2_name}
                if domain1_name == domain2_name or selected_domains in keys:
                    continue

                keys.append(selected_domains)

                loss_name = f"contrastive_{domain1_name}_and_{domain2_name}"
                z2_mean, z2_log_uncertainty = gw_mod.encoded_distribution(
                    gw_mod.on_before_gw_encode_cont({domain2_name: domain2})
                )
                loss_output = contrastive_fn(
                    z1_mean[domain1_name],
                    z1_log_uncertainty[domain1_name],
                    z2_mean[domain2_name],
                    z2_log_uncertainty[domain2_name],
                )
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


class GWLosses(GWLossesBase):
    """
    Implementation of `GWLossesBase` used for `GWModule`.
    """

    def __init__(
        self,
        gw_mod: GWModule,
        domain_mods: dict[str, DomainModule],
        loss_coefs: LossCoefs,
        contrastive_fn: ContrastiveLossType,
    ):
        """
        Main loss module to use with the GlobalWorkspace

        Args:
            gw_mod (`GWModule`): the GWModule
            domain_mods (`dict[str, DomainModule]`): a dict where the key is the
                domain name and value is the DomainModule
            loss_coefs (`LossCoefs`): loss coefficients. LossCoefs object, or a
                mapping to float with correct keys.
            contrastive_fn (`ContrastiveLossType`): the contrastive function to use
                in contrastive loss
        """

        super().__init__()
        self.gw_mod = gw_mod
        self.domain_mods = domain_mods
        self.loss_coefs = loss_coefs
        self.contrastive_fn = contrastive_fn

    def demi_cycle_loss(
        self, latent_domains: LatentsDomainGroupsT
    ) -> dict[str, torch.Tensor]:
        """Computes the demi-cycle loss.

        See `shimmer.modules.losses.demi_cycle_loss`.

        Args:
            latent_domains (`LatentsDomainGroupsT`): the latent unimodal groups

        Returns:
            `dict[str, torch.Tensor]`: a dict of metrics.
        """
        return demi_cycle_loss(self.gw_mod, self.domain_mods, latent_domains)

    def cycle_loss(
        self, latent_domains: LatentsDomainGroupsT
    ) -> dict[str, torch.Tensor]:
        """Computes the cycle loss.

        See `shimmer.modules.losses.cycle_loss`.

        Args:
            latent_domains (`LatentsDomainGroupsT`): the latent unimodal groups

        Returns:
            `dict[str, torch.Tensor]`: a dict of metrics.
        """
        return cycle_loss(self.gw_mod, self.domain_mods, latent_domains)

    def translation_loss(
        self, latent_domains: LatentsDomainGroupsT
    ) -> dict[str, torch.Tensor]:
        """Computes the translation loss.

        See `shimmer.modules.losses.translation_loss`.

        Args:
            latent_domains (`LatentsDomainGroupsT`): the latent unimodal groups

        Returns:
            `dict[str, torch.Tensor]`: a dict of metrics.
        """
        return translation_loss(self.gw_mod, self.domain_mods, latent_domains)

    def contrastive_loss(
        self, latent_domains: LatentsDomainGroupsT
    ) -> dict[str, torch.Tensor]:
        """Computes the contrastive loss.

        See `shimmer.modules.losses.contrastive_loss`.

        Args:
            latent_domains (`LatentsDomainGroupsT`): the latent unimodal groups

        Returns:
            `dict[str, torch.Tensor]`: a dict of metrics.
        """
        return contrastive_loss(self.gw_mod, latent_domains, self.contrastive_fn)

    def step(
        self, domain_latents: LatentsDomainGroupsT, mode: ModelModeT
    ) -> LossOutput:
        """
        Computes and returns the losses

        Contains:
            - Demi-cycle metrics (see `GWLosses.demi_cycle_loss`)
            - Cycle metrics (see `GWLosses.cycle_loss`)
            - Translation metrics (see `GWLosses.translation_loss`)
            - Contrastive metrics (see `GWLosses.contrastive_loss`)

        Args:
            domain_latents (`LatentsDomainGroupsT`): All latent groups
            mode (`ModelModeT`): model mode
        Returns:
            `LossOutput`: the losses
        """
        metrics: dict[str, torch.Tensor] = {}

        metrics.update(self.demi_cycle_loss(domain_latents))
        metrics.update(self.cycle_loss(domain_latents))
        metrics.update(self.translation_loss(domain_latents))
        metrics.update(self.contrastive_loss(domain_latents))

        loss = torch.stack(
            [
                metrics[name] * coef
                for name, coef in self.loss_coefs.items()
                if isinstance(coef, float) and coef > 0
            ],
            dim=0,
        ).mean()

        return LossOutput(loss, metrics)


class GWLossesWithUncertainty(GWLossesBase):
    """
    Implementation of `GWLossesBase` used for `GWModuleWithUncertainty`.
    """

    def __init__(
        self,
        gw_mod: GWModuleWithUncertainty,
        domain_mods: dict[str, DomainModule],
        loss_coefs: LossCoefs,
        contrastive_fn: ContrastiveLossType | None = None,
        cont_fn_with_uncertainty: ContrastiveLossWithUncertaintyType | None = None,
    ):
        """
        Loss module with uncertainty to use with the GlobalWorkspaceWithUncertainty

        Args:
            gw_mod (`GWModuleWithUncertainty`): the GWModule
            domain_mods (`dict[str, DomainModule]`): a dict where the key is the
                domain name and value is the DomainModule
            loss_coefs (`LossCoefsWithUncertainty`): loss coefficients
            contrastive_fn (`ContrastiveLossType | None`): the contrastive function
                to use in contrastive loss
            cont_fn_with_uncertainty (`ContrastiveLossWithUncertaintyType | None`): a contrastive
                function that uses uncertainty
        """

        super().__init__()

        self.gw_mod = gw_mod
        """The GWModule."""

        self.domain_mods = domain_mods
        """Domain modules linked to the GW."""

        self.loss_coefs = loss_coefs
        """The loss coefficients."""

        assert (contrastive_fn is not None) != (
            cont_fn_with_uncertainty is not None
        ), "Should either have contrastive_fn or cont_fn_with_uncertainty"

        self.contrastive_fn = contrastive_fn
        """Contrastive loss to use without the use of uncertainty. This is only
        used in `GWLossesWithUncertainty.step` if 
        `GWLossesWithUncertainty.cont_fn_with_uncertainty` is not set.
        """

        self.cont_fn_with_uncertainty = cont_fn_with_uncertainty
        """Contrastive loss to use with the use of uncertainty."""

    def demi_cycle_loss(
        self, latent_domains: LatentsDomainGroupsT
    ) -> dict[str, torch.Tensor]:
        """Demi-cycle loss. See `GWLosses.demi_cycle_loss`.

        Args:
            latent_domains (`LatentsDomainGroupsT`): the latent unimodal groups

        Returns:
            `dict[str, torch.Tensor]`: a dict of metrics.
        """
        return demi_cycle_loss(self.gw_mod, self.domain_mods, latent_domains)

    def cycle_loss(
        self, latent_domains: LatentsDomainGroupsT
    ) -> dict[str, torch.Tensor]:
        """Cycle loss. See `GWLosses.cycle_loss`.

        Args:
            latent_domains (`LatentsDomainGroupsT`): the latent unimodal groups

        Returns:
            `dict[str, torch.Tensor]`: a dict of metrics.
        """
        return cycle_loss(self.gw_mod, self.domain_mods, latent_domains)

    def translation_loss(
        self, latent_domains: LatentsDomainGroupsT
    ) -> dict[str, torch.Tensor]:
        """Translation loss. See `GWLosses.translation_loss`.

        Args:
            latent_domains (`LatentsDomainGroupsT`): the latent unimodal groups

        Returns:
            `dict[str, torch.Tensor]`: a dict of metrics.
        """
        return translation_loss(self.gw_mod, self.domain_mods, latent_domains)

    def contrastive_loss(
        self, latent_domains: LatentsDomainGroupsT
    ) -> dict[str, torch.Tensor]:
        """Contrastive loss.

        If `GWLossesWithUncertainty.cont_fn_with_uncertainty` is set, will use the
        contrastive loss with uncertainty. Otherwise, use the traditional
        contrastive loss (see `GWLosses.contrastive_loss`).

        Args:
            latent_domains (`LatentsDomainGroupsT`): the latent unimodal groups

        Returns:
            `dict[str, torch.Tensor]`: a dict of metrics.
        """
        if self.cont_fn_with_uncertainty is not None:
            return contrastive_loss_with_uncertainty(
                self.gw_mod, latent_domains, self.cont_fn_with_uncertainty
            )

        assert self.contrastive_fn is not None
        return contrastive_loss(self.gw_mod, latent_domains, self.contrastive_fn)

    def step(
        self, domain_latents: LatentsDomainGroupsT, mode: ModelModeT
    ) -> LossOutput:
        """
        Computes and returns the losses

        Contains:
            - Demi-cycle metrics (see `GWLossesWithUncertainty.demi_cycle_loss`)
            - Cycle metrics (see `GWLossesWithUncertainty.cycle_loss`)
            - Translation metrics (see `GWLossesWithUncertainty.translation_loss`)
            - Contrastive metrics (see `GWLossesWithUncertainty.contrastive_loss`)

        Args:
            domain_latents (`LatentsDomainGroupsT`): All latent groups
            mode (`ModelModeT`): model mode
        Returns:
            `LossOutput`: the losses
        """
        metrics: dict[str, torch.Tensor] = {}

        dcy_losses = self.demi_cycle_loss(domain_latents)
        metrics.update(dcy_losses)
        cy_losses = self.cycle_loss(domain_latents)
        metrics.update(cy_losses)
        tr_losses = self.translation_loss(domain_latents)
        metrics.update(tr_losses)
        cont_losses = self.contrastive_loss(domain_latents)
        metrics.update(cont_losses)

        loss = torch.stack(
            [
                metrics[name] * coef
                for name, coef in self.loss_coefs.items()
                if isinstance(coef, float) and coef > 0
            ],
            dim=0,
        ).mean()

        return LossOutput(loss, metrics)


def sample_scaling_factors(
    binary_scaling_prob: float,
    batch_size: int,
    temperature: float,
    device: torch.device,
):
    """
    Args:
        binary_scaling_prob (`float`): Should be between 0 and 1.
        batch_size (`int`):
        temperature (`float`): Should be greater than 0.
        device (`torch.device`):
    """
    assert 0 <= binary_scaling_prob <= 1

    # TODO: make selection deterministic
    binary_mask = torch.rand(batch_size) < binary_scaling_prob

    binary_factors = torch.randint(0, 2, (batch_size,)).float()
    binary_softmax = torch.stack([binary_factors, 1 - binary_factors], dim=1)

    uniform_samples = torch.rand(batch_size)
    uniform_for_softmax = torch.stack([uniform_samples, 1 - uniform_samples], dim=1)

    uniform_softmax = F.softmax(uniform_for_softmax * temperature, dim=1)

    scaling_factors = torch.where(
        binary_mask.unsqueeze(-1), binary_softmax, uniform_softmax
    ).to(device)

    binary_indices = torch.where(binary_mask)[0]
    softmax_indices = torch.where(~binary_mask)[0]

    binary_scaling_factors = scaling_factors[binary_indices]
    softmax_scaling_factors = scaling_factors[softmax_indices]

    return {
        "binary": (
            binary_scaling_factors[:, 0],
            binary_scaling_factors[:, 1],
            binary_indices,
        ),
        "softmax": (
            softmax_scaling_factors[:, 0],
            softmax_scaling_factors[:, 1],
            softmax_indices,
        ),
    }


class GWLossesFusion(GWLossesBase):
    def __init__(
        self,
        gw_mod: GWModuleFusion,
        domain_mods: dict[str, DomainModule],
        contrastive_fn: ContrastiveLossType,
    ):
        super().__init__()
        self.gw_mod = gw_mod
        self.domain_mods = domain_mods
        self.contrastive_fn = contrastive_fn

    def demi_cycle_loss(
        self, latent_domains: LatentsDomainGroupsT
    ) -> dict[str, torch.Tensor]:
        return demi_cycle_loss(self.gw_mod, self.domain_mods, latent_domains)

    def cycle_loss(
        self, latent_domains: LatentsDomainGroupsT
    ) -> dict[str, torch.Tensor]:
        return cycle_loss(self.gw_mod, self.domain_mods, latent_domains)

    def translation_loss(
        self, latent_domains: LatentsDomainGroupsT
    ) -> dict[str, torch.Tensor]:
        return translation_loss(self.gw_mod, self.domain_mods, latent_domains)

    def contrastive_loss(
        self, latent_domains: LatentsDomainGroupsT
    ) -> dict[str, torch.Tensor]:
        return contrastive_loss(self.gw_mod, latent_domains, self.contrastive_fn)

    def broadcast_loss(
        self, latent_domains: LatentsDomainGroupsT, mode: ModelModeT
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        metrics: dict[str, torch.Tensor] = {}

        for latents in latent_domains.values():
            if len(latents) < 2:
                continue
            batch_size = latents[next(iter(latents))].size(0)
            device = latents[next(iter(latents))].device

            # TODO: don't hardcode the proportions (first param of the sample_scaling_factors function)

            if mode == "val":
                scaling_factors = sample_scaling_factors(0.5, batch_size, 5.0, device)
            else:
                scaling_factors = sample_scaling_factors(0.9, batch_size, 5.0, device)

            for scale_type, (
                scaling_factor_1,
                scaling_factor_2,
                indices,
            ) in scaling_factors.items():
                scaled_latents = {}

                for i, (domain_name, latent) in enumerate(latents.items()):
                    scaling_factor = scaling_factor_1 if i == 0 else scaling_factor_2
                    scaled_latents_subset = latent[indices] * scaling_factor.unsqueeze(
                        -1
                    )
                    scaled_latents_subset = scaled_latents_subset.to(latent)

                    scaled_latents[domain_name] = scaled_latents_subset

                encoded_latents_for_subset = self.gw_mod.encode_and_fuse(scaled_latents)
                encoded_latents_for_subset = torch.tanh(encoded_latents_for_subset)
                decoded_latents_for_subset = self.gw_mod.decode(
                    encoded_latents_for_subset
                )

                for domain_name, latent in latents.items():
                    domain_mod = self.domain_mods[domain_name]
                    decoded_latent_for_domain_subset = decoded_latents_for_subset[
                        domain_name
                    ]
                    original_latent_for_domain_subset = latents[domain_name][indices]
                    loss_output = domain_mod.compute_broadcast_loss(
                        decoded_latent_for_domain_subset,
                        original_latent_for_domain_subset,
                    )
                    loss_key = f"{domain_name}_loss_{scale_type}"

                    metrics.update(
                        {
                            f"broadcast_{loss_key}_{k}": v
                            for k, v in loss_output.metrics.items()
                        }
                    )
                    losses[loss_key] = loss_output.loss.mean()

            binary_count = scaling_factors["binary"][2].size(0)
            softmax_count = scaling_factors["softmax"][2].size(0)
            total_count = binary_count + softmax_count

            for domain_name, latent in latents.items():
                full_loss_key = f"{domain_name}_full_loss"

                binary_loss_key = f"{domain_name}_loss_binary"
                softmax_loss_key = f"{domain_name}_loss_softmax"

                binary_loss = losses[binary_loss_key] * (binary_count / total_count)
                softmax_loss = losses[softmax_loss_key] * (softmax_count / total_count)

                losses[full_loss_key] = binary_loss + softmax_loss

        losses["broadcast"] = torch.stack(
            [loss for name, loss in losses.items() if "full_loss" in name], dim=0
        ).mean()
        losses.update(metrics)
        return losses

    def step(
        self,
        domain_latents: LatentsDomainGroupsT,
        mode: ModelModeT,
    ) -> LossOutput:
        metrics: dict[str, torch.Tensor] = {}

        metrics.update(self.demi_cycle_loss(domain_latents))
        metrics.update(self.cycle_loss(domain_latents))
        metrics.update(self.translation_loss(domain_latents))
        metrics.update(self.contrastive_loss(domain_latents))
        metrics.update(self.broadcast_loss(domain_latents, mode))

        loss = metrics["broadcast"] + metrics["contrastives"]

        return LossOutput(loss, metrics)
