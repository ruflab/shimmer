from collections.abc import Iterable, Mapping
from typing import Any, cast

import torch
import torch.nn as nn
from info_nce import info_nce
from lightning.pytorch import LightningModule
from torch.nn import ModuleDict
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import OneCycleLR

from shimmer.modules.domain import DomainDescription, DomainModule
from shimmer.modules.vae import kl_divergence_loss, reparameterize

LatentsDomainGroupT = Mapping[str, torch.Tensor]
LatentsT = Mapping[frozenset[str], LatentsDomainGroupT]


def get_n_layers(n_layers: int, hidden_dim: int):
    layers = []
    for _ in range(n_layers):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
    return layers


class GWEncoder(nn.Sequential):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int,
    ):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.n_layers = n_layers

        super(GWEncoder, self).__init__(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            *get_n_layers(n_layers, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.out_dim),
        )


class VariationalGWEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int,
    ):
        super(VariationalGWEncoder, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = n_layers

        self.layers = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            *get_n_layers(n_layers, self.hidden_dim),
        )
        self.mean_layer = nn.Linear(self.hidden_dim, self.out_dim)
        self.logvar_layer = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.layers(x)
        return self.mean_layer(z), self.logvar_layer(z)


class GlobalWorkspaceLightningModule(LightningModule):
    def __init__(
        self,
        domain_descriptions: Mapping[str, DomainDescription],
        encoder_type: type[GWEncoder] | type[VariationalGWEncoder],
        latent_dim: int,
        encoder_hidden_dim: Mapping[str, int],
        encoder_n_layers: Mapping[str, int],
        decoder_hidden_dim: Mapping[str, int],
        decoder_n_layers: Mapping[str, int],
        loss_coefficients: Mapping[str, float],
        optim_lr: float,
        optim_weight_decay: float,
        scheduler_args: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        self.domains = set(domain_descriptions.keys())
        self.latent_dim = latent_dim

        self.input_dim = {
            name: domain.latent_dim
            for name, domain in domain_descriptions.items()
        }
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_n_layers = encoder_n_layers
        self.decoder_hidden_dim = decoder_hidden_dim
        self.decoder_n_layers = decoder_n_layers

        self.encoders = nn.ModuleDict(
            {
                domain: encoder_type(
                    self.input_dim[domain],
                    self.encoder_hidden_dim[domain],
                    self.latent_dim,
                    self.encoder_n_layers[domain],
                )
                for domain in self.domains
            }
        )
        self.decoders = nn.ModuleDict(
            {
                domain: GWEncoder(
                    self.latent_dim,
                    self.decoder_hidden_dim[domain],
                    self.input_dim[domain],
                    self.decoder_n_layers[domain],
                )
                for domain in self.domains
            }
        )

        domain_modules = {
            name: domain.module for name, domain in domain_descriptions.items()
        }

        for module in domain_modules.values():
            module.eval().freeze()

        self.domain_modules = cast(
            dict[str, DomainModule], ModuleDict(domain_modules)
        )

        self.loss_coefficients = loss_coefficients
        self.optim_lr = optim_lr
        self.optim_weight_decay = optim_weight_decay
        self.scheduler_args: dict[str, Any] = {
            "max_lr": optim_lr,
            "total_steps": 1,
        }
        self.scheduler_args.update(scheduler_args or {})

    def fusion_mechanism(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.mean(torch.stack(list(x.values())), dim=0)

    def encode(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def decode(
        self, z: torch.Tensor, domains: Iterable[str] | None = None
    ) -> dict[str, torch.Tensor]:
        return {
            domain: self.decoders[domain](z)
            for domain in domains or self.domains
        }

    def translate(
        self, x: Mapping[str, torch.Tensor], to: str
    ) -> torch.Tensor:
        raise NotImplementedError

    def cycle(
        self, x: Mapping[str, torch.Tensor], through: str
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def batch_demi_cycles(
        self, latent_domains: LatentsT
    ) -> dict[str, torch.Tensor]:
        predictions: dict[str, torch.Tensor] = {}
        for domains, latents in latent_domains.items():
            if len(domains) > 1:
                continue
            domain_name = list(domains)[0]
            z = self.translate(latents, to=domain_name)
            predictions[domain_name] = z
        return predictions

    def demi_cycle_loss(
        self, latent_domains: LatentsT
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        for domains, latents in latent_domains.items():
            if len(domains) > 1:
                continue
            domain_name = next(iter(domains))
            x_recons = self.decode(
                self.encode(latents), domains={domain_name}
            )[domain_name]
            loss = mse_loss(x_recons, latents[domain_name], reduction="sum")
            losses[f"demi_cycle_{domain_name}"] = loss
        losses["demi_cycles"] = torch.stack(
            list(losses.values()), dim=0
        ).mean()
        return losses

    def batch_cycles(
        self, latent_domains: LatentsT
    ) -> dict[tuple[str, str], torch.Tensor]:
        predictions: dict[tuple[str, str], torch.Tensor] = {}
        for domains_source, latents_source in latent_domains.items():
            if len(domains_source) > 1:
                continue
            domain_name_source = next(iter(domains_source))
            for domain_name_target in self.domain_modules.keys():
                if domain_name_source == domain_name_target:
                    continue
                z = self.cycle(latents_source, through=domain_name_target)
                domains = (domain_name_source, domain_name_target)
                predictions[domains] = z[domain_name_source]
        return predictions

    def cycle_loss(self, latent_domains: LatentsT) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        for domains_source, latents_source in latent_domains.items():
            if len(domains_source) > 1:
                continue
            domain_name_source = list(domains_source)[0]
            z = self.encode(latents_source)
            for domain_name_target in self.domain_modules.keys():
                x_pred = self.decode(z, domains={domain_name_target})
                x_recons = self.decode(
                    self.encode(x_pred), domains={domain_name_source}
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

    def batch_translations(
        self, latent_domains: LatentsT
    ) -> dict[tuple[str, str], torch.Tensor]:
        predictions: dict[tuple[str, str], torch.Tensor] = {}
        for domains, latents in latent_domains.items():
            if len(domains) < 2:
                continue
            for domain_name_source in domains:
                for domain_name_target in domains:
                    if domain_name_source == domain_name_target:
                        continue
                    prediction = self.translate(
                        {domain_name_source: latents[domain_name_source]},
                        to=domain_name_target,
                    )
                    predictions[
                        (domain_name_source, domain_name_target)
                    ] = prediction
        return predictions

    def translation_loss(
        self, latent_domains: LatentsT
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        for domains, latents in latent_domains.items():
            if len(domains) < 2:
                continue
            for domain_name_source in domains:
                z = self.encode(
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

                    prediction = self.decode(z, domains={domain_name_target})[
                        domain_name_target
                    ]
                    losses[loss_name] = mse_loss(
                        prediction,
                        latents[domain_name_target],
                        reduction="sum",
                    )
        losses["translations"] = torch.stack(
            list(losses.values()), dim=0
        ).mean()
        return losses

    def contrastive_loss(
        self, latent_domains: LatentsT
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        keys: list[set[str]] = []

        for latents in latent_domains.values():
            if len(latents) < 2:
                continue
            for domain1_name, domain1 in latents.items():
                z1 = self.encode({domain1_name: domain1})
                for domain2_name, domain2 in latents.items():
                    selected_domains = {domain1_name, domain2_name}
                    if (
                        domain1_name == domain2_name
                        or selected_domains in keys
                    ):
                        continue

                    keys.append(selected_domains)

                    loss_name = (
                        f"contrastive_{domain1_name}_and_{domain2_name}"
                    )
                    z2 = self.encode({domain2_name: domain2})
                    losses[loss_name] = info_nce(z1, z2, reduction="sum")

        losses["contrastives"] = torch.stack(
            list(losses.values()), dim=0
        ).mean()
        return losses

    def encode_domain(self, domain: Any, name: str) -> torch.Tensor:
        return self.domain_modules[name].encode(domain)

    def encode_domains(
        self,
        batch: Mapping[frozenset[str], Mapping[str, Any]],
    ) -> dict[frozenset[str], dict[str, torch.Tensor]]:
        return {
            domains: {
                name: self.encode_domain(domain, name)
                for name, domain in data.items()
            }
            for domains, data in batch.items()
        }

    def decode_domain(self, domain: torch.Tensor, name: str) -> Any:
        return self.domain_modules[name].decode(domain)

    def decode_domains(
        self,
        latents_domain: LatentsT,
    ) -> dict[frozenset[str], dict[str, Any]]:
        return {
            domains: {
                name: self.decode_domain(domain, name)
                for name, domain in latents.items()
            }
            for domains, latents in latents_domain.items()
        }

    def _get_batch_size(
        self,
        domain_latents: LatentsT,
    ) -> int:
        for data in domain_latents.values():
            for tensor in data.values():
                return tensor.size(0)
        raise ValueError("Empty batch.")

    def generic_step(
        self,
        batch: Mapping[frozenset[str], Mapping[str, Any]],
        mode: str,
    ) -> torch.Tensor:
        raise NotImplementedError

    def validation_step(self, data: Mapping[str, Any], _) -> torch.Tensor:
        batch = {frozenset(data.keys()): data}
        for domain in data.keys():
            batch[frozenset([domain])] = {domain: data[domain]}
        return self.generic_step(batch, mode="val")

    def training_step(
        self, batch: Mapping[frozenset[str], Mapping[str, Any]], _
    ) -> torch.Tensor:
        return self.generic_step(batch, mode="train")

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.optim_lr,
            weight_decay=self.optim_weight_decay,
        )

        lr_scheduler = OneCycleLR(optimizer, **self.scheduler_args)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }


class DeterministicGlobalWorkspaceLightningModule(
    GlobalWorkspaceLightningModule
):
    def __init__(
        self,
        domain_descriptions: Mapping[str, DomainDescription],
        latent_dim: int,
        encoder_hidden_dim: Mapping[str, int],
        encoder_n_layers: Mapping[str, int],
        decoder_hidden_dim: Mapping[str, int],
        decoder_n_layers: Mapping[str, int],
        loss_coefficients: Mapping[str, float],
        optim_lr: float,
        optim_weight_decay: float,
        scheduler_args: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            domain_descriptions,
            GWEncoder,
            latent_dim,
            encoder_hidden_dim,
            encoder_n_layers,
            decoder_hidden_dim,
            decoder_n_layers,
            loss_coefficients,
            optim_lr,
            optim_weight_decay,
            scheduler_args,
        )
        self.save_hyperparameters(ignore=["domain_descriptions"])

    def encode(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.fusion_mechanism(
            {domain: self.encoders[domain](x[domain]) for domain in x.keys()}
        )

    def translate(
        self, x: Mapping[str, torch.Tensor], to: str
    ) -> torch.Tensor:
        return self.decode(self.encode(x), domains={to})[to]

    def cycle(
        self, x: Mapping[str, torch.Tensor], through: str
    ) -> dict[str, torch.Tensor]:
        return {
            domain: self.translate(
                {through: self.translate(x, through)}, domain
            )
            for domain in x.keys()
        }

    def generic_step(
        self,
        batch: Mapping[frozenset[str], Mapping[str, Any]],
        mode: str,
    ) -> torch.Tensor:
        domain_latents = self.encode_domains(batch)
        batch_size = self._get_batch_size(domain_latents)

        losses: dict[str, torch.Tensor] = {}
        losses.update(self.demi_cycle_loss(domain_latents))
        losses.update(self.cycle_loss(domain_latents))
        losses.update(self.translation_loss(domain_latents))
        losses.update(self.contrastive_loss(domain_latents))

        for name, coef in self.loss_coefficients.items():
            losses[name] *= coef

        losses["loss"] = torch.stack(
            [losses[name] for name in self.loss_coefficients.keys()],
            dim=0,
        ).mean()

        for name, loss in losses.items():
            self.log(f"{mode}/{name}", loss, batch_size=batch_size)

        return losses["loss"]


class VariationalGlobalWorkspaceLightningModule(
    GlobalWorkspaceLightningModule
):
    def __init__(
        self,
        domain_descriptions: Mapping[str, DomainDescription],
        latent_dim: int,
        encoder_hidden_dim: Mapping[str, int],
        encoder_n_layers: Mapping[str, int],
        decoder_hidden_dim: Mapping[str, int],
        decoder_n_layers: Mapping[str, int],
        loss_coefficients: Mapping[str, float],
        optim_lr: float,
        optim_weight_decay: float,
        scheduler_args: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            domain_descriptions,
            VariationalGWEncoder,
            latent_dim,
            encoder_hidden_dim,
            encoder_n_layers,
            decoder_hidden_dim,
            decoder_n_layers,
            loss_coefficients,
            optim_lr,
            optim_weight_decay,
            scheduler_args,
        )
        self.save_hyperparameters(ignore=["domain_descriptions"])

    def encode(
        self,
        x: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        latents: dict[str, torch.Tensor] = {}
        for domain in x.keys():
            mean, logvar = self.encoders[domain](x[domain])
            latents[domain] = reparameterize(mean, logvar)
        return self.fusion_mechanism(latents)

    def encoded_distribution(
        self,
        x: Mapping[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        means: dict[str, torch.Tensor] = {}
        logvars: dict[str, torch.Tensor] = {}
        for domain in x.keys():
            mean, logvar = self.encoders[domain](x[domain])
            means[domain] = mean
            logvars[domain] = logvar
        return means, logvars

    def encode_mean(
        self,
        x: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.fusion_mechanism(self.encoded_distribution(x)[0])

    def translate(
        self, x: Mapping[str, torch.Tensor], to: str
    ) -> torch.Tensor:
        return self.decode(self.encode_mean(x), domains={to})[to]

    def cycle(
        self, x: Mapping[str, torch.Tensor], through: str
    ) -> dict[str, torch.Tensor]:
        return {
            domain: self.translate(
                {through: self.translate(x, through)}, domain
            )
            for domain in x.keys()
        }

    def kl_loss(self, latent_domains: LatentsT) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}

        for domains, latents in latent_domains.items():
            if len(domains) > 1:
                continue
            for domain_name in domains:
                mean, logvar = self.encoded_distribution(
                    {domain_name: latents[domain_name]}
                )
                loss_name = f"kl_{domain_name}"
                losses[loss_name] = kl_divergence_loss(
                    mean[domain_name], logvar[domain_name]
                )
        losses["kl"] = torch.stack(list(losses.values()), dim=0).mean()
        return losses

    def generic_step(
        self,
        batch: Mapping[frozenset[str], Mapping[str, Any]],
        mode: str,
    ) -> torch.Tensor:
        domain_latents = self.encode_domains(batch)
        batch_size = self._get_batch_size(domain_latents)

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

        kl_scale_coef = self.loss_coefficients["demi_cycles"] * (
            len(dcy_losses) - 1
        )
        kl_scale_coef += self.loss_coefficients["translations"] * (
            len(tr_losses) - 1
        )
        kl_scale_coef += self.loss_coefficients["cycles"] * (
            len(cy_losses) - 1
        )

        kl_losses["kl"] *= kl_scale_coef
        losses.update(kl_losses)

        for name, coef in self.loss_coefficients.items():
            losses[name] *= coef

        losses["loss"] = torch.stack(
            [losses[name] for name in self.loss_coefficients.keys()],
            dim=0,
        ).mean()

        for name, loss in losses.items():
            self.log(f"{mode}/{name}", loss, batch_size=batch_size)

        return losses["loss"]
