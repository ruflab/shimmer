from abc import ABC, abstractmethod
from collections.abc import Mapping

import torch

from shimmer.modules.contrastive_loss import (ContrastiveLossType,
                                              VarContrastiveLossType)
from shimmer.modules.dict_buffer import DictBuffer
from shimmer.modules.domain import DomainModule
from shimmer.modules.gw_module import (GWModule, GWModuleBase,
                                       VariationalGWModule)
from shimmer.modules.vae import kl_divergence_loss

LatentsDomainGroupT = Mapping[str, torch.Tensor]
LatentsT = Mapping[frozenset[str], LatentsDomainGroupT]


class GWLossesBase(torch.nn.Module, ABC):
    """
    Base Abstract Class for Global Workspace (GW) losses. This module is used
    to compute the different losses of the GW (typically translation, cycle,
    demi-cycle, contrastive losses).
    """

    @abstractmethod
    def step(
        self,
        domain_latents: LatentsT,
    ) -> dict[str, torch.Tensor]:
        """
        Computes the losses
        Args:
            domain_latents: All latent groups
        Returns:
            a dict with loss name as keys and loss value as values.
        """
        ...


def _demi_cycle_loss(
    gw_mod: GWModuleBase,
    domain_mods: dict[str, DomainModule],
    latent_domains: LatentsT,
) -> dict[str, torch.Tensor]:
    losses: dict[str, torch.Tensor] = {}
    metrics: dict[str, torch.Tensor] = {}
    for domains, latents in latent_domains.items():
        if len(domains) > 1:
            continue
        domain_name = next(iter(domains))
        domain_mod = domain_mods[domain_name]
        x_recons = gw_mod.decode(
            gw_mod.encode(gw_mod.on_before_gw_encode_dcy(latents)),
            domains={domain_name},
        )[domain_name]
        loss_output = domain_mod.compute_dcy_loss(
            x_recons, latents[domain_name]
        )
        losses[f"demi_cycle_{domain_name}"] = loss_output.loss
        metrics.update(
            {
                f"demi_cycle_{domain_name}_{k}": v
                for k, v in loss_output.metrics.items()
            }
        )
    losses["demi_cycles"] = torch.stack(list(losses.values()), dim=0).mean()
    losses.update(metrics)
    return losses


def _cycle_loss(
    gw_mod: GWModuleBase,
    domain_mods: dict[str, DomainModule],
    latent_domains: LatentsT,
) -> dict[str, torch.Tensor]:
    losses: dict[str, torch.Tensor] = {}
    metrics: dict[str, torch.Tensor] = {}
    for domains_source, latents_source in latent_domains.items():
        if len(domains_source) > 1:
            continue
        domain_name_source = list(domains_source)[0]

        domain_mod = domain_mods[domain_name_source]
        z = gw_mod.encode(gw_mod.on_before_gw_encode_cy(latents_source))
        for domain_name_target in domain_mods.keys():
            if domain_name_target == domain_name_source:
                continue

            x_pred = gw_mod.decode(z, domains={domain_name_target})
            x_recons = gw_mod.decode(
                gw_mod.encode(x_pred), domains={domain_name_source}
            )

            loss_name = f"{domain_name_source}_through_{domain_name_target}"
            loss_output = domain_mod.compute_cy_loss(
                x_recons[domain_name_source],
                latents_source[domain_name_source],
            )
            metrics.update(
                {
                    f"cycle_{loss_name}_{k}": v
                    for k, v in loss_output.metrics.items()
                }
            )
            losses[f"cycle_{loss_name}"] = loss_output.loss
    losses["cycles"] = torch.stack(list(losses.values()), dim=0).mean()
    losses.update(metrics)
    return losses


def _translation_loss(
    gw_mod: GWModuleBase,
    domain_mods: dict[str, DomainModule],
    latent_domains: LatentsT,
) -> dict[str, torch.Tensor]:
    losses: dict[str, torch.Tensor] = {}
    metrics: dict[str, torch.Tensor] = {}
    for domains, latents in latent_domains.items():
        if len(domains) < 2:
            continue
        for domain_name_source in domains:
            z = gw_mod.encode(
                gw_mod.on_before_gw_encode_tr(
                    {domain_name_source: latents[domain_name_source]}
                )
            )

            for domain_name_target in domains:
                if domain_name_source == domain_name_target:
                    continue

                mod = domain_mods[domain_name_target]

                loss_name = f"{domain_name_source}_to_{domain_name_target}"
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


def _contrastive_loss(
    gw_mod: GWModuleBase,
    latent_domains: LatentsT,
    contrastive_fn: ContrastiveLossType,
) -> dict[str, torch.Tensor]:
    losses: dict[str, torch.Tensor] = {}
    metrics: dict[str, torch.Tensor] = {}
    keys: list[set[str]] = []

    for latents in latent_domains.values():
        if len(latents) < 2:
            continue
        for domain1_name, domain1 in latents.items():
            z1 = gw_mod.encode(
                gw_mod.on_before_gw_encode_cont({domain1_name: domain1})
            )
            for domain2_name, domain2 in latents.items():
                selected_domains = {domain1_name, domain2_name}
                if domain1_name == domain2_name or selected_domains in keys:
                    continue

                keys.append(selected_domains)

                loss_name = f"contrastive_{domain1_name}_and_{domain2_name}"
                z2 = gw_mod.encode(
                    gw_mod.on_before_gw_encode_cont({domain2_name: domain2})
                )
                loss_output = contrastive_fn(z1, z2)
                losses[loss_name] = loss_output.loss
                metrics.update(
                    {
                        f"{loss_name}_{k}": v
                        for k, v in loss_output.metrics.items()
                    }
                )

    losses["contrastives"] = torch.stack(list(losses.values()), dim=0).mean()
    losses.update(metrics)
    return losses


def _contrastive_loss_with_uncertainty(
    gw_mod: VariationalGWModule,
    latent_domains: LatentsT,
    contrastive_fn: VarContrastiveLossType,
) -> dict[str, torch.Tensor]:
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
                    {
                        f"{loss_name}_{k}": v
                        for k, v in loss_output.metrics.items()
                    }
                )

    losses["contrastives"] = torch.stack(list(losses.values()), dim=0).mean()
    losses.update(metrics)
    return losses


class GWLosses(GWLossesBase):
    def __init__(
        self,
        gw_mod: GWModule,
        domain_mods: dict[str, DomainModule],
        coef_buffers: DictBuffer,
        contrastive_fn: ContrastiveLossType,
    ):
        super().__init__()
        self.gw_mod = gw_mod
        self.domain_mods = domain_mods
        self.loss_coefs = coef_buffers
        self.contrastive_fn = contrastive_fn

    def demi_cycle_loss(
        self, latent_domains: LatentsT
    ) -> dict[str, torch.Tensor]:
        return _demi_cycle_loss(self.gw_mod, self.domain_mods, latent_domains)

    def cycle_loss(self, latent_domains: LatentsT) -> dict[str, torch.Tensor]:
        return _cycle_loss(self.gw_mod, self.domain_mods, latent_domains)

    def translation_loss(
        self, latent_domains: LatentsT
    ) -> dict[str, torch.Tensor]:
        return _translation_loss(self.gw_mod, self.domain_mods, latent_domains)

    def contrastive_loss(
        self, latent_domains: LatentsT
    ) -> dict[str, torch.Tensor]:
        return _contrastive_loss(
            self.gw_mod, latent_domains, self.contrastive_fn
        )

    def step(
        self,
        domain_latents: Mapping[frozenset[str], Mapping[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}

        losses.update(self.demi_cycle_loss(domain_latents))
        losses.update(self.cycle_loss(domain_latents))
        losses.update(self.translation_loss(domain_latents))
        losses.update(self.contrastive_loss(domain_latents))

        losses["loss"] = torch.stack(
            [
                losses[name] * coef
                for name, coef in self.loss_coefs.items()
                if coef.item() > 0
            ],
            dim=0,
        ).mean()

        return losses


class VariationalGWLosses(GWLossesBase):
    def __init__(
        self,
        gw_mod: VariationalGWModule,
        domain_mods: dict[str, DomainModule],
        coef_buffers: DictBuffer,
        contrastive_fn: ContrastiveLossType | None = None,
        var_contrastive_fn: VarContrastiveLossType | None = None,
    ):
        super().__init__()

        self.gw_mod = gw_mod
        self.domain_mods = domain_mods
        self.loss_coefs = coef_buffers
        assert (contrastive_fn is not None) != (
            var_contrastive_fn is not None
        ), "Should either have contrastive_fn or var_contrastive_fn"
        self.contrastive_fn = contrastive_fn
        self.var_contrastive_fn = var_contrastive_fn

    def demi_cycle_loss(
        self, latent_domains: LatentsT
    ) -> dict[str, torch.Tensor]:
        return _demi_cycle_loss(self.gw_mod, self.domain_mods, latent_domains)

    def cycle_loss(self, latent_domains: LatentsT) -> dict[str, torch.Tensor]:
        return _cycle_loss(self.gw_mod, self.domain_mods, latent_domains)

    def translation_loss(
        self, latent_domains: LatentsT
    ) -> dict[str, torch.Tensor]:
        return _translation_loss(self.gw_mod, self.domain_mods, latent_domains)

    def contrastive_loss(
        self, latent_domains: LatentsT
    ) -> dict[str, torch.Tensor]:
        if self.var_contrastive_fn is not None:
            return _contrastive_loss_with_uncertainty(
                self.gw_mod, latent_domains, self.var_contrastive_fn
            )

        assert self.contrastive_fn is not None
        return _contrastive_loss(
            self.gw_mod, latent_domains, self.contrastive_fn
        )

    def kl_loss(self, latent_domains: LatentsT) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}

        for domains, latents in latent_domains.items():
            if len(domains) > 1:
                continue
            for domain_name in domains:
                mean, logvar = self.gw_mod.encoded_distribution(
                    {domain_name: latents[domain_name]}
                )
                loss_name = f"kl_{domain_name}"
                norm = mean[domain_name].size(0) + mean[domain_name].size(1)
                losses[loss_name] = (
                    kl_divergence_loss(mean[domain_name], logvar[domain_name])
                    / norm
                )
        losses["kl"] = torch.stack(list(losses.values()), dim=0).mean()
        return losses

    def step(
        self,
        domain_latents: Mapping[frozenset[str], Mapping[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}

        dcy_losses = self.demi_cycle_loss(domain_latents)
        losses.update(dcy_losses)
        cy_losses = self.cycle_loss(domain_latents)
        losses.update(cy_losses)
        tr_losses = self.translation_loss(domain_latents)
        losses.update(tr_losses)
        cont_losses = self.contrastive_loss(domain_latents)
        losses.update(cont_losses)
        kl_losses = self.kl_loss(domain_latents)
        losses.update(kl_losses)

        losses["loss"] = torch.stack(
            [
                losses[name] * coef
                for name, coef in self.loss_coefs.items()
                if (coef > 0).item()
            ],
            dim=0,
        ).mean()

        return losses
