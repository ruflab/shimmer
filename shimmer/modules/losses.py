from collections.abc import Mapping

from info_nce import info_nce, torch
from torch.nn.functional import mse_loss

from shimmer.modules.domain import DomainModule
from shimmer.modules.gw_module import (
    DeterministicGWModule,
    GWModule,
    VariationalGWModule,
)
from shimmer.modules.vae import kl_divergence_loss

LatentsDomainGroupT = Mapping[str, torch.Tensor]
LatentsT = Mapping[frozenset[str], LatentsDomainGroupT]


class GWLosses:
    def step(
        self,
        domain_latents: Mapping[frozenset[str], Mapping[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError


def _demi_cycle_loss(
    gw_mod: GWModule, latent_domains: LatentsT
) -> dict[str, torch.Tensor]:
    losses: dict[str, torch.Tensor] = {}
    for domains, latents in latent_domains.items():
        if len(domains) > 1:
            continue
        domain_name = next(iter(domains))
        x_recons = gw_mod.decode(
            gw_mod.encode(latents), domains={domain_name}
        )[domain_name]
        loss = mse_loss(x_recons, latents[domain_name], reduction="sum")
        losses[f"demi_cycle_{domain_name}"] = loss
    losses["demi_cycles"] = torch.stack(list(losses.values()), dim=0).mean()
    return losses


def _cycle_loss(
    gw_mod: GWModule,
    domain_mods: dict[str, DomainModule],
    latent_domains: LatentsT,
) -> dict[str, torch.Tensor]:
    losses: dict[str, torch.Tensor] = {}
    for domains_source, latents_source in latent_domains.items():
        if len(domains_source) > 1:
            continue
        domain_name_source = list(domains_source)[0]
        z = gw_mod.encode(latents_source)
        for domain_name_target in domain_mods.keys():
            x_pred = gw_mod.decode(z, domains={domain_name_target})
            x_recons = gw_mod.decode(
                gw_mod.encode(x_pred), domains={domain_name_source}
            )

            loss_name = (
                f"cycle_{domain_name_source}_through_{domain_name_target}"
            )
            losses[loss_name] = mse_loss(
                x_recons[domain_name_source],
                latents_source[domain_name_source],
                reduction="sum",
            )

    losses["cycles"] = torch.stack(list(losses.values()), dim=0).mean()
    return losses


def _translation_loss(
    gw_mod: GWModule, latent_domains: LatentsT
) -> dict[str, torch.Tensor]:
    losses: dict[str, torch.Tensor] = {}
    for domains, latents in latent_domains.items():
        if len(domains) < 2:
            continue
        for domain_name_source in domains:
            z = gw_mod.encode(
                {domain_name_source: latents[domain_name_source]}
            )

            for domain_name_target in domains:
                if domain_name_source == domain_name_target:
                    continue

                loss_name = (
                    f"translation_{domain_name_source}"
                    f"_to_{domain_name_target}"
                )
                if loss_name in losses.keys():
                    raise ValueError(f"{loss_name} is already computed.")

                prediction = gw_mod.decode(z, domains={domain_name_target})[
                    domain_name_target
                ]
                losses[loss_name] = mse_loss(
                    prediction,
                    latents[domain_name_target],
                    reduction="sum",
                )
    losses["translations"] = torch.stack(list(losses.values()), dim=0).mean()
    return losses


def _contrastive_loss(
    gw_mod: GWModule, latent_domains: LatentsT
) -> dict[str, torch.Tensor]:
    losses: dict[str, torch.Tensor] = {}
    keys: list[set[str]] = []

    for latents in latent_domains.values():
        if len(latents) < 2:
            continue
        for domain1_name, domain1 in latents.items():
            z1 = gw_mod.encode({domain1_name: domain1})
            for domain2_name, domain2 in latents.items():
                selected_domains = {domain1_name, domain2_name}
                if domain1_name == domain2_name or selected_domains in keys:
                    continue

                keys.append(selected_domains)

                loss_name = f"contrastive_{domain1_name}_and_{domain2_name}"
                z2 = gw_mod.encode({domain2_name: domain2})
                losses[loss_name] = info_nce(z1, z2, reduction="sum")

    losses["contrastives"] = torch.stack(list(losses.values()), dim=0).mean()
    return losses


class DeterministicGWLosses(GWLosses):
    def __init__(
        self,
        gw_mod: DeterministicGWModule,
        domain_mods: dict[str, DomainModule],
        loss_coefs: Mapping[str, float] | None = None,
    ):
        self.gw_mod = gw_mod
        self.domain_mods = domain_mods
        self.loss_coefs = loss_coefs or {}

    def demi_cycle_loss(
        self, latent_domains: LatentsT
    ) -> dict[str, torch.Tensor]:
        return _demi_cycle_loss(self.gw_mod, latent_domains)

    def cycle_loss(self, latent_domains: LatentsT) -> dict[str, torch.Tensor]:
        return _cycle_loss(self.gw_mod, self.domain_mods, latent_domains)

    def translation_loss(
        self, latent_domains: LatentsT
    ) -> dict[str, torch.Tensor]:
        return _translation_loss(self.gw_mod, latent_domains)

    def contrastive_loss(
        self, latent_domains: LatentsT
    ) -> dict[str, torch.Tensor]:
        return _contrastive_loss(self.gw_mod, latent_domains)

    def step(
        self,
        domain_latents: Mapping[frozenset[str], Mapping[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}

        losses.update(self.demi_cycle_loss(domain_latents))
        losses.update(self.cycle_loss(domain_latents))
        losses.update(self.translation_loss(domain_latents))
        losses.update(self.contrastive_loss(domain_latents))

        for name, coef in self.loss_coefs.items():
            losses[name] *= coef

        losses["loss"] = torch.stack(
            [losses[name] for name in self.loss_coefs.keys()], dim=0
        ).mean()

        return losses


class VariationalGWLosses(GWLosses):
    def __init__(
        self,
        gw_mod: VariationalGWModule,
        domain_mods: dict[str, DomainModule],
        loss_coefs: Mapping[str, float] | None = None,
    ):
        self.gw_mod = gw_mod
        self.domain_mods = domain_mods
        self.loss_coefs = loss_coefs or {}

    def demi_cycle_loss(
        self, latent_domains: LatentsT
    ) -> dict[str, torch.Tensor]:
        return _demi_cycle_loss(self.gw_mod, latent_domains)

    def cycle_loss(self, latent_domains: LatentsT) -> dict[str, torch.Tensor]:
        return _cycle_loss(self.gw_mod, self.domain_mods, latent_domains)

    def translation_loss(
        self, latent_domains: LatentsT
    ) -> dict[str, torch.Tensor]:
        return _translation_loss(self.gw_mod, latent_domains)

    def contrastive_loss(
        self, latent_domains: LatentsT
    ) -> dict[str, torch.Tensor]:
        return _contrastive_loss(self.gw_mod, latent_domains)

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
                losses[loss_name] = kl_divergence_loss(
                    mean[domain_name], logvar[domain_name]
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

        kl_scale_coef = self.loss_coefs["demi_cycles"] * (len(dcy_losses) - 1)
        kl_scale_coef += self.loss_coefs["translations"] * (len(tr_losses) - 1)
        kl_scale_coef += self.loss_coefs["cycles"] * (len(cy_losses) - 1)

        kl_losses["kl"] *= kl_scale_coef
        losses.update(kl_losses)

        for name, coef in self.loss_coefs.items():
            losses[name] *= coef

        losses["loss"] = torch.stack(
            [losses[name] for name in self.loss_coefs.keys()],
            dim=0,
        ).mean()

        return losses
