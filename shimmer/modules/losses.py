from abc import ABC, abstractmethod
from collections.abc import Generator, Mapping
from itertools import product
from typing import TypedDict

import torch

from shimmer.modules.contrastive_loss import ContrastiveLossType
from shimmer.modules.domain import DomainModule, LossOutput
from shimmer.modules.gw_module import (
    GWModule,
    GWModuleBase,
    GWModuleBayesian,
)
from shimmer.modules.selection import SelectionBase
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
    selection_mod: SelectionBase,
    domain_mods: Mapping[str, DomainModule],
    latent_domains: LatentsDomainGroupsT,
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
    selection_mod: SelectionBase,
    domain_mods: Mapping[str, DomainModule],
    latent_domains: LatentsDomainGroupsT,
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
    selection_mod: SelectionBase,
    domain_mods: Mapping[str, DomainModule],
    latent_domains: LatentsDomainGroupsT,
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


def contrastive_loss_bayesian(
    gw_mod: GWModuleBayesian,
    latent_domains: LatentsDomainGroupsT,
    contrastive_fn: ContrastiveLossType,
) -> dict[str, torch.Tensor]:
    """
    Computes the contrastive loss with a Bayesian based uncertainty prediction.

    This return multiple metrics:
        * `contrastive_{domain_1}_and_{domain_2}` with the contrastive
            between 2 domains;
        * `contrastive_{domain_1}_and_{domain_2}_{metric}` with
            additional metrics provided by the domain_mod's
            `compute_cont_loss` output;
        * `contrastives` with the average value of all
            `contrastive_{domain_1}_and_{domain_2}` values.

    Args:
        gw_mod (`GWModuleBayesian`): The GWModule to use
        latent_domains (`LatentsDomainGroupsT`): the latent unimodal groups
        contrastive_fn (`ContrastiveLossBayesianType`): the contrastive function
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
            z1 = gw_mod.encode({domain1_name: domain1})[domain1_name]
            z1_precision = gw_mod.get_precision(domain1_name, domain1)
            for domain2_name, domain2 in latents.items():
                selected_domains = {domain1_name, domain2_name}
                if domain1_name == domain2_name or selected_domains in keys:
                    continue

                keys.append(selected_domains)

                loss_name = f"contrastive_{domain1_name}_and_{domain2_name}"
                z2 = gw_mod.encode({domain2_name: domain2})[domain2_name]
                z2_precision = gw_mod.get_precision(domain2_name, domain2)
                coef = torch.softmax(
                    gw_mod.precision_softmax_temp
                    * torch.stack([z1_precision, z2_precision]),
                    dim=0,
                )
                norm = torch.sqrt(coef[0] * coef[1])
                loss_output = contrastive_fn(z1 * norm, z2 * norm)
                loss_output_no_norm = contrastive_fn(z1, z2)
                losses[loss_name] = loss_output.loss
                metrics.update(
                    {f"{loss_name}_{k}": v for k, v in loss_output.metrics.items()}
                )
                metrics[f"unnorm_{loss_name}"] = loss_output_no_norm.loss

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


class GWLosses2Domains(GWLossesBase):
    """
    Implementation of `GWLossesBase` used for `GWModule`.
    """

    def __init__(
        self,
        gw_mod: GWModule,
        selection_mod: SelectionBase,
        domain_mods: dict[str, DomainModule],
        loss_coefs: LossCoefs,
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
        self, latent_domains: LatentsDomainGroupsT
    ) -> dict[str, torch.Tensor]:
        """
        Computes the demi-cycle loss.

        See `shimmer.modules.losses.demi_cycle_loss`.

        Args:
            latent_domains (`LatentsDomainGroupsT`): the latent unimodal groups

        Returns:
            `dict[str, torch.Tensor]`: a dict of metrics.
        """
        return demi_cycle_loss(
            self.gw_mod, self.selection_mod, self.domain_mods, latent_domains
        )

    def cycle_loss(
        self, latent_domains: LatentsDomainGroupsT
    ) -> dict[str, torch.Tensor]:
        """
        Computes the cycle loss.

        See `shimmer.modules.losses.cycle_loss`.

        Args:
            latent_domains (`LatentsDomainGroupsT`): the latent unimodal groups

        Returns:
            `dict[str, torch.Tensor]`: a dict of metrics.
        """
        return cycle_loss(
            self.gw_mod, self.selection_mod, self.domain_mods, latent_domains
        )

    def translation_loss(
        self, latent_domains: LatentsDomainGroupsT
    ) -> dict[str, torch.Tensor]:
        """
        Computes the translation loss.

        See `shimmer.modules.losses.translation_loss`.

        Args:
            latent_domains (`LatentsDomainGroupsT`): the latent unimodal groups

        Returns:
            `dict[str, torch.Tensor]`: a dict of metrics.
        """
        return translation_loss(
            self.gw_mod, self.selection_mod, self.domain_mods, latent_domains
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


def generate_partitions(n: int) -> Generator[tuple[int, ...], None, None]:
    """
    Generates all possible partitions of zeros and ones for `n` elements,
    excluding the all-zeros partition.

    Args:
        n (`int`): The number of modalities to generate partitions for.

    Yields:
        `tuple[int, ...]`: A partition of zeros and ones, excluding the
        all-zeros partition.
    """
    for perm in product([0, 1], repeat=n):
        if any(perm):
            yield perm


def broadcast_loss(
    gw_mod: GWModuleBase,
    selection_mod: SelectionBase,
    domain_mods: Mapping[str, DomainModule],
    latent_domains: LatentsDomainGroupsT,
) -> dict[str, torch.Tensor]:
    """
    Computes broadcast loss including demi-cycle, cycle, and translation losses.

    Args:
        gw_mod (`shimmer.modules.gw_module.GWModuleBase`): The GWModule to use
        selection_mod (`shimmer.modules.selection.SelectionBase`): Selection mod to use
        domain_mods (`Mapping[str, DomainModule]`): the domain modules
        latent_domains: The latent domain representations.

    Returns:
        A dictionary with the total loss and additional metrics.
    """
    losses: dict[str, torch.Tensor] = {}
    metrics: dict[str, torch.Tensor] = {}

    demi_cycle_losses: list[str] = []
    cycle_losses: list[str] = []
    translation_losses: list[str] = []
    fused_losses: list[str] = []

    for group_domains, latents in latent_domains.items():
        encoded_latents = gw_mod.encode(latents)
        partitions = generate_partitions(len(group_domains))
        domain_names = list(latents)
        group_name = "-".join(group_domains)

        for partition in partitions:
            selected_latents = {
                domain: latents[domain]
                for domain, present in zip(domain_names, partition, strict=True)
                if present
            }
            selected_encoded_latents = {
                domain: encoded_latents[domain] for domain in selected_latents
            }
            selected_group_label = "{" + ", ".join(sorted(selected_latents)) + "}"

            selection_scores = selection_mod(selected_latents, selected_encoded_latents)
            fused_latents = gw_mod.fuse(selected_encoded_latents, selection_scores)
            decoded_latents = gw_mod.decode(fused_latents)

            num_active_domains = sum(partition)
            num_total_domains = len(decoded_latents)

            for domain, pred in decoded_latents.items():
                if domain not in group_domains:  # if we don't have ground truth
                    continue
                ground_truth = latents[domain]
                loss_output = domain_mods[domain].compute_loss(pred, ground_truth)
                loss_label = f"from_{selected_group_label}_to_{domain}"
                losses[loss_label + "_loss"] = loss_output.loss
                metrics.update(
                    {f"{loss_label}_{k}": v for k, v in loss_output.metrics.items()}
                )

                if num_active_domains == 1 and domain in selected_latents:
                    demi_cycle_losses.append(loss_label + "_loss")
                elif domain not in selected_latents:
                    translation_losses.append(loss_label + "_loss")
                else:  # fused loss
                    fused_losses.append(loss_label + "_loss")

            if num_active_domains < num_total_domains:
                inverse_selected_latents = {
                    domain: decoded_latents[domain]
                    for domain in decoded_latents
                    if domain not in selected_latents
                }

                inverse_selected_group_label = (
                    "{" + ", ".join(sorted(inverse_selected_latents)) + "}"
                )

                re_encoded_latents = gw_mod.encode(inverse_selected_latents)
                re_selection_scores = selection_mod(
                    inverse_selected_latents, re_encoded_latents
                )
                re_fused_latents = gw_mod.fuse(re_encoded_latents, re_selection_scores)
                re_decoded_latents = gw_mod.decode(
                    re_fused_latents, domains=selected_latents.keys()
                )

                for domain in selected_latents:
                    re_ground_truth = latents[domain]
                    re_loss_output = domain_mods[domain].compute_loss(
                        re_decoded_latents[domain], re_ground_truth
                    )
                    loss_label = (
                        f"from_{selected_group_label}_"
                        f"through_{inverse_selected_group_label}_to_{domain}_"
                        f"case_{group_name}"
                    )
                    losses[loss_label + "_loss"] = re_loss_output.loss
                    metrics.update(
                        {
                            f"{loss_label}_{k}": v
                            for k, v in re_loss_output.metrics.items()
                        }
                    )
                    cycle_losses.append(loss_label + "_loss")

    if demi_cycle_losses:
        metrics["demi_cycles"] = torch.mean(
            torch.stack([losses[loss_name] for loss_name in demi_cycle_losses])
        )
    if cycle_losses:
        metrics["cycles"] = torch.mean(
            torch.stack([losses[loss_name] for loss_name in cycle_losses])
        )
    if translation_losses:
        metrics["translations"] = torch.mean(
            torch.stack([losses[loss_name] for loss_name in translation_losses])
        )
    if fused_losses:
        metrics["fused"] = torch.mean(
            torch.stack([losses[loss_name] for loss_name in fused_losses])
        )

    metrics.update(losses)
    return metrics


class BroadcastLossCoefs(TypedDict, total=False):
    """
    Dict of loss coefficients used in the GWLossesFusion.

    If one is not provided, the coefficient is assumed to be 0 and will not be logged.
    If the loss is excplicitely set to 0, it will be logged, but not take part in
    the total loss.
    """

    contrastives: float
    """Contrastive loss coefficient."""

    fused: float
    """fused loss coefficient (encode multiple domains and decode to one of them)."""

    demi_cycles: float
    """demi_cycles loss coefficient. Demi-cycles are always one-to-one"""

    cycles: float
    """cycles loss coefficient. Cycles can be many-to-one"""

    translations: float
    """translation loss coefficient. Translation, like cycles, can be many-to-one."""


class GWLosses(GWLossesBase):
    """
    Implementation of `GWLossesBase` for fusion-based models.
    """

    def __init__(
        self,
        gw_mod: GWModule,
        selection_mod: SelectionBase,
        domain_mods: dict[str, DomainModule],
        loss_coefs: BroadcastLossCoefs,
        contrastive_fn: ContrastiveLossType,
    ):
        """
        Initializes the loss computation module for a Global Workspace Fusion model.

        Args:
            gw_mod: The GWModule for the global workspace.
            selection_mod: The selection mechanism for the model.
            domain_mods: A mapping of domain names to their respective DomainModule.
            loss_coefs (`BroadcastLossCoefs`): coefs for the losses
            contrastive_fn: The function used for computing contrastive loss.
        """
        super().__init__()
        self.gw_mod = gw_mod
        self.selection_mod = selection_mod
        self.domain_mods = domain_mods
        self.loss_coefs = loss_coefs
        self.contrastive_fn = contrastive_fn

    def contrastive_loss(
        self, latent_domains: LatentsDomainGroupsT
    ) -> dict[str, torch.Tensor]:
        """
        Computes the contrastive loss for the given latent domains.

        Args:
            latent_domains: The latent domain representations.

        Returns:
            A dictionary of contrastive loss metrics.
        """

        return contrastive_loss(self.gw_mod, latent_domains, self.contrastive_fn)

    def broadcast_loss(
        self, latent_domains: LatentsDomainGroupsT
    ) -> dict[str, torch.Tensor]:
        return broadcast_loss(
            self.gw_mod, self.selection_mod, self.domain_mods, latent_domains
        )

    def step(
        self, domain_latents: LatentsDomainGroupsT, mode: ModelModeT
    ) -> LossOutput:
        """
        Performs a step of loss computation.

        Args:
            domain_latents: Latent representations for all domains.
            mode: The mode in which the model is currently operating.

        Returns:
            A LossOutput object containing the loss and metrics for this step.
        """

        metrics: dict[str, torch.Tensor] = {}

        metrics.update(self.contrastive_loss(domain_latents))
        metrics.update(self.broadcast_loss(domain_latents))

        loss = torch.stack(
            [
                metrics[name] * coef
                for name, coef in self.loss_coefs.items()
                if isinstance(coef, float) and coef > 0
            ],
            dim=0,
        ).mean()

        metrics["broadcast_loss"] = torch.stack(
            [
                metrics[name]
                for name, coef in self.loss_coefs.items()
                if isinstance(coef, float) and coef > 0 and name != "contrastives"
            ],
            dim=0,
        ).mean()

        return LossOutput(loss, metrics)


class GWLossesBayesian(GWLossesBase):
    """
    Implementation of `GWLossesBase` used for `GWModuleBayesian`.
    """

    def __init__(
        self,
        gw_mod: GWModuleBayesian,
        selection_mod: SelectionBase,
        domain_mods: dict[str, DomainModule],
        loss_coefs: BroadcastLossCoefs,
        contrastive_fn: ContrastiveLossType,
        use_normalized_constrastive: bool = True,
    ):
        """
        Loss module with uncertainty prediction to use with the GlobalWorkspaceBayesian

        Args:
            gw_mod (`GWModuleBayesian`): the GWModule
            selection_mod (`SelectionBase`): selection module
            domain_mods (`dict[str, DomainModule]`): a dict where the key is the
                domain name and value is the DomainModule
            loss_coefs (`BroadcastLossCoefs`): loss coefficients
            contrastive_fn (`ContrastiveLossType`): the contrastive function
                to use in contrastive loss
            use_normalized_constrastive (`bool`): whether to use the normalized cont
                loss by the precision coefs
        """
        super().__init__()

        self.gw_mod = gw_mod
        """The GWModule."""

        self.selection_mod = selection_mod
        """Selection module"""

        self.domain_mods = domain_mods
        """Domain modules linked to the GW."""

        self.loss_coefs = loss_coefs
        """The loss coefficients."""

        self.contrastive_fn = contrastive_fn
        """
        Contrastive loss to use.
        """

        self.use_normalized_constrastive = use_normalized_constrastive

    def contrastive_loss(
        self, latent_domains: LatentsDomainGroupsT
    ) -> dict[str, torch.Tensor]:
        """
        Contrastive loss.

        Args:
            latent_domains (`LatentsDomainGroupsT`): the latent unimodal groups

        Returns:
            `dict[str, torch.Tensor]`: a dict of metrics.
        """
        if self.use_normalized_constrastive:
            return contrastive_loss_bayesian(
                self.gw_mod, latent_domains, self.contrastive_fn
            )
        return contrastive_loss(self.gw_mod, latent_domains, self.contrastive_fn)

    def broadcast_loss(
        self, latent_domains: LatentsDomainGroupsT
    ) -> dict[str, torch.Tensor]:
        return broadcast_loss(
            self.gw_mod, self.selection_mod, self.domain_mods, latent_domains
        )

    def step(
        self, domain_latents: LatentsDomainGroupsT, mode: ModelModeT
    ) -> LossOutput:
        """
        Performs a step of loss computation.

        Args:
            domain_latents: Latent representations for all domains.
            mode: The mode in which the model is currently operating.

        Returns:
            A LossOutput object containing the loss and metrics for this step.
        """

        metrics: dict[str, torch.Tensor] = {}

        metrics.update(self.contrastive_loss(domain_latents))
        metrics.update(self.broadcast_loss(domain_latents))

        loss = torch.stack(
            [
                metrics[name] * coef
                for name, coef in self.loss_coefs.items()
                if isinstance(coef, float) and coef > 0
            ],
            dim=0,
        ).mean()

        metrics["broadcast_loss"] = torch.stack(
            [
                metrics[name]
                for name, coef in self.loss_coefs.items()
                if isinstance(coef, float) and coef > 0 and name != "contrastives"
            ],
            dim=0,
        ).mean()

        return LossOutput(loss, metrics)
