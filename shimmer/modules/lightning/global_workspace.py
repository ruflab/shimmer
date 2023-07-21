from collections.abc import Mapping
from typing import Any, cast

import torch
from info_nce import info_nce
from lightning.pytorch import LightningModule
from torch.nn import ModuleDict
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import OneCycleLR

from shimmer.modules.domain import DomainDescription, DomainModule
from shimmer.modules.global_workspace import (
    DeterministicGlobalWorkspace,
    VariationalGlobalWorkspace,
)
from shimmer.modules.vae import kl_divergence_loss

LatentsDomainGroupT = Mapping[str, torch.Tensor]
LatentsT = Mapping[frozenset[str], LatentsDomainGroupT]


def variational_global_workspace_from_domains(
    domains: Mapping[str, DomainDescription],
    latent_dim: int,
    encoder_hiddent_dim: int,
    encoder_n_layers: int,
    decoder_hidden_dim: int,
    decoder_n_layers: int,
) -> VariationalGlobalWorkspace:
    domain_names = set(domains.keys())
    input_dims = {name: domain.latent_dim for name, domain in domains.items()}
    return VariationalGlobalWorkspace(
        domain_names,
        latent_dim,
        input_dims,
        {name: encoder_hiddent_dim for name in domains.keys()},
        {name: encoder_n_layers for name in domains.keys()},
        {name: decoder_hidden_dim for name in domains.keys()},
        {name: decoder_n_layers for name in domains.keys()},
    )


def deterministic_global_workspace_from_domains(
    domains: Mapping[str, DomainDescription],
    latent_dim: int,
    encoder_hiddent_dim: int,
    encoder_n_layers: int,
    decoder_hidden_dim: int,
    decoder_n_layers: int,
) -> DeterministicGlobalWorkspace:
    domain_names = set(domains.keys())
    input_dims = {name: domain.latent_dim for name, domain in domains.items()}
    return DeterministicGlobalWorkspace(
        domain_names,
        latent_dim,
        input_dims,
        {name: encoder_hiddent_dim for name in domains.keys()},
        {name: encoder_n_layers for name in domains.keys()},
        {name: decoder_hidden_dim for name in domains.keys()},
        {name: decoder_n_layers for name in domains.keys()},
    )


class DeterministicGlobalWorkspaceLightningModule(LightningModule):
    def __init__(
        self,
        domain_descriptions: Mapping[str, DomainDescription],
        latent_dim: int,
        encoders_hidden_dim: int,
        encoders_n_layers: int,
        decoders_hidden_dim: int,
        decoders_n_layers: int,
        demi_cycle_loss_coefficient: float,
        cycle_loss_coefficient: float,
        translation_loss_coefficient: float,
        contrastive_loss_coefficient: float,
        optim_lr: float,
        optim_weight_decay: float,
        scheduler_args: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["domain_descriptions"])

        self.global_workspace = deterministic_global_workspace_from_domains(
            domain_descriptions,
            latent_dim,
            encoders_hidden_dim,
            encoders_n_layers,
            decoders_hidden_dim,
            decoders_n_layers,
        )

        domain_modules = {
            name: domain.module for name, domain in domain_descriptions.items()
        }

        for module in domain_modules.values():
            module.eval().freeze()

        self.domain_modules = cast(
            dict[str, DomainModule], ModuleDict(domain_modules)
        )

        self.loss_coefficients = {
            "demi_cycles": demi_cycle_loss_coefficient,
            "cycles": cycle_loss_coefficient,
            "translations": translation_loss_coefficient,
            "contrastives": contrastive_loss_coefficient,
        }

        self.optim_lr = optim_lr
        self.optim_weight_decay = optim_weight_decay
        self.scheduler_args: dict[str, Any] = {
            "max_lr": optim_lr,
            "total_steps": 1,
        }
        self.scheduler_args.update(scheduler_args or {})

    def demi_cycle(self, latent_domains: LatentsT) -> dict[str, torch.Tensor]:
        predictions: dict[str, torch.Tensor] = {}
        for domains, latents in latent_domains.items():
            if len(domains) > 1:
                continue
            domain_name = list(domains)[0]
            z = self.global_workspace.translate(latents, to=domain_name)
            predictions[domain_name] = z
        return predictions

    def demi_cycle_loss(
        self, latent_domains: LatentsT
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        for domains, latents in latent_domains.items():
            if len(domains) > 1:
                continue
            domain_name = list(domains)[0]
            z = self.global_workspace.translate(latents, to=domain_name)
            losses[f"demi_cycle_{domain_name}"] = mse_loss(
                z,
                latents[domain_name],
                reduction="sum",
            )
        losses["demi_cycles"] = torch.stack(
            list(losses.values()), dim=0
        ).mean()
        return losses

    def cycle(
        self, latent_domains: LatentsT
    ) -> dict[tuple[str, str], torch.Tensor]:
        predictions: dict[tuple[str, str], torch.Tensor] = {}
        for domains_source, latents_source in latent_domains.items():
            if len(domains_source) > 1:
                continue
            domain_name_source = list(domains_source)[0]
            for domain_name_target in self.domain_modules.keys():
                if domain_name_source == domain_name_target:
                    continue
                z = self.global_workspace.cycle(
                    latents_source, through=domain_name_target
                )
                domains = (domain_name_source, domain_name_target)
                predictions[domains] = z[domain_name_source]
        return predictions

    def cycle_loss(self, latent_domains: LatentsT) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        for domains_source, latents_source in latent_domains.items():
            if len(domains_source) > 1:
                continue
            domain_name_source = list(domains_source)[0]
            for domain_name_target in self.domain_modules.keys():
                z = self.global_workspace.cycle(
                    latents_source, through=domain_name_target
                )
                loss_name = (
                    f"cycle_{domain_name_source}_through_{domain_name_target}"
                )
                losses[loss_name] = mse_loss(
                    z[domain_name_source],
                    latents_source[domain_name_source],
                    reduction="sum",
                )
        losses["cycles"] = torch.stack(list(losses.values()), dim=0).mean()
        return losses

    def translation(
        self, latent_domains: LatentsT
    ) -> dict[tuple[str, str], torch.Tensor]:
        predictions: dict[tuple[str, str], torch.Tensor] = {}
        for domains, latents in latent_domains.items():
            if len(domains) < 2:
                continue
            for domain_name_source in domains:
                z = self.global_workspace.encode(
                    {domain_name_source: latents[domain_name_source]}
                )
                for domain_name_target in domains:
                    if domain_name_source == domain_name_target:
                        continue
                    prediction = self.global_workspace.decode(
                        z, domains={domain_name_target}
                    )[domain_name_target]
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
                z = self.global_workspace.encode(
                    {domain_name_source: latents[domain_name_source]}
                )
                for domain_name_target in domains:
                    if domain_name_source == domain_name_target:
                        continue
                    prediction = self.global_workspace.decode(
                        z, domains={domain_name_target}
                    )[domain_name_target]
                    loss_name = (
                        f"translation_{domain_name_source}"
                        f"_to_{domain_name_target}"
                    )
                    if loss_name in losses.keys():
                        raise ValueError(f"{loss_name} is already computed.")
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

        done_domains: list[set[str]] = []
        for domains, latents in latent_domains.items():
            if len(domains) < 2:
                continue
            for domain_name_source in domains:
                z_source = self.global_workspace.encode(
                    {domain_name_source: latents[domain_name_source]}
                )
                for domain_name_target in domains:
                    if domain_name_source == domain_name_target:
                        continue
                    selected_domains = {domain_name_source, domain_name_target}
                    if selected_domains in done_domains:
                        continue
                    done_domains.append(selected_domains)
                    z_target = self.global_workspace.encode(
                        {domain_name_target: latents[domain_name_target]}
                    )
                    loss_name = (
                        f"contrastive_{domain_name_source}"
                        f"_and_{domain_name_target}"
                    )
                    if loss_name in losses.keys():
                        raise ValueError(f"{loss_name} is already computed.")
                    losses[loss_name] = info_nce(
                        z_source,
                        z_target,
                        reduction="sum",
                    )
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
        return 0

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


class VariationalGlobalWorkspaceLightningModule(LightningModule):
    def __init__(
        self,
        domain_descriptions: Mapping[str, DomainDescription],
        latent_dim: int,
        encoders_hidden_dim: int,
        encoders_n_layers: int,
        decoders_hidden_dim: int,
        decoders_n_layers: int,
        demi_cycle_loss_coefficient: float,
        cycle_loss_coefficient: float,
        translation_loss_coefficient: float,
        contrastive_loss_coefficient: float,
        kl_loss_coefficient: float,
        optim_lr: float,
        optim_weight_decay: float,
        scheduler_args: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["domain_descriptions"])

        self.global_workspace = variational_global_workspace_from_domains(
            domain_descriptions,
            latent_dim,
            encoders_hidden_dim,
            encoders_n_layers,
            decoders_hidden_dim,
            decoders_n_layers,
        )

        domain_modules = {
            name: domain.module for name, domain in domain_descriptions.items()
        }

        for module in domain_modules.values():
            module.eval().freeze()

        self.domain_modules = cast(
            dict[str, DomainModule], ModuleDict(domain_modules)
        )

        self.loss_coefficients = {
            "demi_cycles": demi_cycle_loss_coefficient,
            "cycles": cycle_loss_coefficient,
            "translations": translation_loss_coefficient,
            "contrastives": contrastive_loss_coefficient,
            "kl": kl_loss_coefficient,
        }

        self.optim_lr = optim_lr
        self.optim_weight_decay = optim_weight_decay
        self.scheduler_args: dict[str, Any] = {
            "max_lr": optim_lr,
            "total_steps": 1,
        }
        self.scheduler_args.update(scheduler_args or {})

    def demi_cycle(self, latent_domains: LatentsT) -> dict[str, torch.Tensor]:
        predictions: dict[str, torch.Tensor] = {}
        for domains, latents in latent_domains.items():
            if len(domains) > 1:
                continue
            domain_name = list(domains)[0]
            z = self.global_workspace.translate_mean(latents, to=domain_name)
            predictions[domain_name] = z
        return predictions

    def demi_cycle_loss(
        self, latent_domains: LatentsT
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        for domains, latents in latent_domains.items():
            if len(domains) > 1:
                continue
            domain_name = list(domains)[0]
            z = self.global_workspace.translate(latents, to=domain_name)
            losses[f"demi_cycle_{domain_name}"] = mse_loss(
                z, latents[domain_name], reduction="sum"
            )
        losses["demi_cycles"] = torch.stack(
            list(losses.values()), dim=0
        ).mean()
        return losses

    def cycle(
        self, latent_domains: LatentsT
    ) -> dict[tuple[str, str], torch.Tensor]:
        predictions: dict[tuple[str, str], torch.Tensor] = {}
        for domains_source, latents_source in latent_domains.items():
            if len(domains_source) > 1:
                continue
            domain_name_source = list(domains_source)[0]
            for domain_name_target in self.domain_modules.keys():
                if domain_name_source == domain_name_target:
                    continue
                z = self.global_workspace.cycle_mean(
                    latents_source, through=domain_name_target
                )
                domains = (domain_name_source, domain_name_target)
                predictions[domains] = z[domain_name_source]
        return predictions

    def cycle_loss(self, latent_domains: LatentsT) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        for domains_source, latents_source in latent_domains.items():
            if len(domains_source) > 1:
                continue
            domain_name_source = list(domains_source)[0]
            for domain_name_target in self.domain_modules.keys():
                z = self.global_workspace.cycle(
                    latents_source, through=domain_name_target
                )
                loss_name = (
                    f"cycle_{domain_name_source}_through_{domain_name_target}"
                )
                losses[loss_name] = mse_loss(
                    z[domain_name_source],
                    latents_source[domain_name_source],
                    reduction="sum",
                )
        losses["cycles"] = torch.stack(list(losses.values()), dim=0).mean()
        return losses

    def translation(
        self, latent_domains: LatentsT
    ) -> dict[tuple[str, str], torch.Tensor]:
        predictions: dict[tuple[str, str], torch.Tensor] = {}
        for domains, latents in latent_domains.items():
            if len(domains) < 2:
                continue
            for domain_name_source in domains:
                mean, _ = self.global_workspace.encoded_distribution(
                    {domain_name_source: latents[domain_name_source]}
                )
                z = self.global_workspace.fusion_mechanism(mean)
                for domain_name_target in domains:
                    if domain_name_source == domain_name_target:
                        continue
                    prediction = self.global_workspace.decode(
                        z, domains={domain_name_target}
                    )[domain_name_target]
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
                z = self.global_workspace.encode(
                    {domain_name_source: latents[domain_name_source]}
                )[0]
                for domain_name_target in domains:
                    if domain_name_source == domain_name_target:
                        continue
                    prediction = self.global_workspace.decode(
                        z, domains={domain_name_target}
                    )[domain_name_target]
                    loss_name = (
                        f"translation_{domain_name_source}"
                        f"_to_{domain_name_target}"
                    )
                    if loss_name in losses.keys():
                        raise ValueError(f"{loss_name} is already computed.")
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

        done_domains: list[set[str]] = []
        for domains, latents in latent_domains.items():
            if len(domains) < 2:
                continue
            for domain_name_source in domains:
                (
                    mean_source,
                    logvar_source,
                ) = self.global_workspace.encoded_distribution(
                    {domain_name_source: latents[domain_name_source]}
                )
                z_source = mean_source[domain_name_source]
                for domain_name_target in domains:
                    if domain_name_source == domain_name_target:
                        continue
                    selected_domains = {domain_name_source, domain_name_target}
                    if selected_domains in done_domains:
                        continue
                    done_domains.append(selected_domains)
                    (
                        mean_target,
                        logvar_target,
                    ) = self.global_workspace.encoded_distribution(
                        {domain_name_target: latents[domain_name_target]}
                    )
                    z_target = mean_target[domain_name_target]
                    loss_name = (
                        f"contrastive_{domain_name_source}"
                        f"_and_{domain_name_target}"
                    )
                    if loss_name in losses.keys():
                        raise ValueError(f"{loss_name} is already computed.")
                    std_source = logvar_source[domain_name_source].exp()
                    std_target = logvar_target[domain_name_target].exp()
                    normalization = 1 + std_source + std_target
                    normalized_z_source = z_source / normalization
                    normalized_z_target = z_target / normalization

                    losses[loss_name] = info_nce(
                        normalized_z_source,
                        normalized_z_target,
                        reduction="sum",
                    )

        losses["contrastives"] = torch.stack(
            list(losses.values()), dim=0
        ).mean()
        return losses

    def kl_loss(self, latent_domains: LatentsT) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}

        for domains, latents in latent_domains.items():
            if len(domains) > 1:
                continue
            for domain_name in domains:
                mean, logvar = self.global_workspace.encoded_distribution(
                    {domain_name: latents[domain_name]}
                )
                loss_name = f"kl_{domain_name}"
                losses[loss_name] = kl_divergence_loss(
                    mean[domain_name], logvar[domain_name]
                )
        losses["kl"] = torch.stack(list(losses.values()), dim=0).mean()
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
        return 0

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


GlobalWorkspaceLightningModuleType = (
    DeterministicGlobalWorkspaceLightningModule
    | VariationalGlobalWorkspaceLightningModule
)
