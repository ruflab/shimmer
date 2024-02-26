from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, TypedDict, cast

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch.nn import ModuleDict
from torch.optim.lr_scheduler import OneCycleLR

from shimmer.modules.contrastive_loss import (
    ContrastiveLoss,
    ContrastiveLossType,
    ContrastiveLossWithUncertainty,
    VarContrastiveLossType,
)
from shimmer.modules.dict_buffer import DictBuffer
from shimmer.modules.domain import DomainModule
from shimmer.modules.gw_module import (
    GWInterfaceBase,
    GWModule,
    GWModuleBase,
    GWModuleFusion,
    VariationalGWModule,
)
from shimmer.modules.losses import (
    GWLosses,
    GWLossesBase,
    GWLossesFusion,
    LatentsT,
    VariationalGWLosses,
)


class SchedulerArgs(TypedDict, total=False):
    max_lr: float
    total_steps: int


class GWPredictions(TypedDict):
    demi_cycles: dict[str, torch.Tensor]
    cycles: dict[tuple[str, str], torch.Tensor]
    translations: dict[tuple[str, str], torch.Tensor]
    states: dict[str, torch.Tensor]


class GlobalWorkspaceBase(LightningModule):
    def __init__(
        self,
        gw_mod: GWModuleBase,
        domain_mods: Mapping[str, DomainModule],
        coef_buffers: DictBuffer,
        loss_mod: GWLossesBase,
        optim_lr: float = 1e-3,
        optim_weight_decay: float = 0.0,
        scheduler_args: SchedulerArgs | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "gw_mod",
                "domain_mods",
                "loss_mod",
                "domain_descriptions",
                "coef_buffers",
                "contrastive_loss",
            ]
        )

        self.gw_mod = gw_mod
        self.domain_mods = domain_mods
        self.loss_mod = loss_mod
        self.coef_buffers = coef_buffers

        self.optim_lr = optim_lr
        self.optim_weight_decay = optim_weight_decay
        self.scheduler_args = SchedulerArgs(max_lr=optim_lr, total_steps=1)
        if scheduler_args is not None:
            self.scheduler_args.update(scheduler_args)

    @property
    def workspace_dim(self):
        return self.gw_mod.workspace_dim

    def encode(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.gw_mod.encode(x)

    def decode(
        self, z: torch.Tensor, domains: Iterable[str] | None = None
    ) -> dict[str, torch.Tensor]:
        return self.gw_mod.decode(z, domains)

    def forward(self, latent_domains: LatentsT) -> GWPredictions:
        outputs = GWPredictions(
            **{
                "demi_cycles": self.batch_demi_cycles(latent_domains),
                "cycles": self.batch_cycles(latent_domains),
                "translations": self.batch_translations(latent_domains),
                "states": self.batch_gw_states(latent_domains),
            }
        )
        return outputs

    def batch_gw_states(self, latent_domains: LatentsT) -> dict[str, torch.Tensor]:
        predictions: dict[str, torch.Tensor] = {}
        for domains, latents in latent_domains.items():
            if len(domains) > 1:
                continue
            domain_name = list(domains)[0]
            z = self.gw_mod.encode(latents)
            predictions[domain_name] = z
        return predictions

    def batch_demi_cycles(self, latent_domains: LatentsT) -> dict[str, torch.Tensor]:
        predictions: dict[str, torch.Tensor] = {}
        for domains, latents in latent_domains.items():
            if len(domains) > 1:
                continue
            domain_name = list(domains)[0]
            z = self.gw_mod.translate(latents, to=domain_name)
            predictions[domain_name] = z
        return predictions

    def batch_cycles(
        self, latent_domains: LatentsT
    ) -> dict[tuple[str, str], torch.Tensor]:
        predictions: dict[tuple[str, str], torch.Tensor] = {}
        for domains_source, latents_source in latent_domains.items():
            if len(domains_source) > 1:
                continue
            domain_name_source = next(iter(domains_source))
            for domain_name_target in self.domain_mods.keys():
                if domain_name_source == domain_name_target:
                    continue
                z = self.gw_mod.cycle(latents_source, through=domain_name_target)
                domains = (domain_name_source, domain_name_target)
                predictions[domains] = z[domain_name_source]
        return predictions

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
                    prediction = self.gw_mod.translate(
                        {domain_name_source: latents[domain_name_source]},
                        to=domain_name_target,
                    )
                    predictions[(domain_name_source, domain_name_target)] = prediction
        return predictions

    def encode_domain(self, domain: Any, name: str) -> torch.Tensor:
        return self.domain_mods[name].encode(domain)

    def encode_domains(
        self,
        batch: Mapping[frozenset[str], Mapping[str, Any]],
    ) -> dict[frozenset[str], dict[str, torch.Tensor]]:
        return {
            domains: {
                name: self.domain_mods[name].encode(domain)
                for name, domain in data.items()
            }
            for domains, data in batch.items()
        }

    def decode_domain(self, domain: torch.Tensor, name: str) -> Any:
        return self.domain_mods[name].decode(domain)

    def decode_domains(
        self,
        latents_domain: LatentsT,
    ) -> dict[frozenset[str], dict[str, Any]]:
        return {
            domains: {
                name: self.domain_mods[name].decode(domain)
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
        domain_latents = self.encode_domains(batch)
        batch_size = self._get_batch_size(domain_latents)

        loss_output = self.loss_mod.step(domain_latents, mode)

        for name, metric in loss_output.all.items():
            self.log(
                f"{mode}/{name}",
                metric,
                batch_size=batch_size,
                add_dataloader_idx=False,
            )

        return loss_output.loss

    def validation_step(
        self, data: Mapping[str, Any], _, dataloader_idx: int = 0
    ) -> torch.Tensor:
        batch = {frozenset(data.keys()): data}
        for domain in data.keys():
            batch[frozenset([domain])] = {domain: data[domain]}
        if dataloader_idx == 0:
            return self.generic_step(batch, mode="val")
        return self.generic_step(batch, mode="val/ood")

    def test_step(
        self, data: Mapping[str, Any], _, dataloader_idx: int = 0
    ) -> torch.Tensor:
        batch = {frozenset(data.keys()): data}
        for domain in data.keys():
            batch[frozenset([domain])] = {domain: data[domain]}
        if dataloader_idx == 0:
            return self.generic_step(batch, mode="test")
        return self.generic_step(batch, mode="test/ood")

    def training_step(
        self, batch: Mapping[frozenset[str], Mapping[str, Any]], _
    ) -> torch.Tensor:
        return self.generic_step(batch, mode="train")

    def predict_step(self, data: Mapping[str, Any], _) -> GWPredictions:  # type: ignore
        batch = {frozenset(data.keys()): data}
        for domain in data.keys():
            batch[frozenset([domain])] = {domain: data[domain]}

        domain_latents = self.encode_domains(batch)
        return self.forward(domain_latents)

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
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


def freeze_domain_modules(
    domain_mods: Mapping[str, DomainModule],
) -> dict[str, DomainModule]:
    for mod in domain_mods.values():
        mod.freeze()
    # Cast for better auto-completion at the expense of ModuleDict
    return cast(dict[str, DomainModule], ModuleDict(domain_mods))


class GlobalWorkspace(GlobalWorkspaceBase):
    def __init__(
        self,
        domain_mods: Mapping[str, DomainModule],
        gw_interfaces: Mapping[str, GWInterfaceBase],
        workspace_dim: int,
        loss_coefs: Mapping[str, torch.Tensor],
        optim_lr: float = 1e-3,
        optim_weight_decay: float = 0.0,
        scheduler_args: SchedulerArgs | None = None,
        learn_logit_scale: bool = False,
        contrastive_loss: ContrastiveLossType | None = None,
    ) -> None:
        gw_mod = GWModule(gw_interfaces, workspace_dim)
        domain_mods = freeze_domain_modules(domain_mods)
        coef_buffers = DictBuffer(loss_coefs)
        if contrastive_loss is None:
            contrastive_loss = ContrastiveLoss(
                torch.tensor([1 / 0.07]).log(), "mean", learn_logit_scale
            )
        loss_mod = GWLosses(
            gw_mod,
            domain_mods,
            coef_buffers,
            contrastive_loss,
        )

        super().__init__(
            gw_mod,
            domain_mods,
            coef_buffers,
            loss_mod,
            optim_lr,
            optim_weight_decay,
            scheduler_args,
        )


class VariationalGlobalWorkspace(GlobalWorkspaceBase):
    def __init__(
        self,
        domain_mods: Mapping[str, DomainModule],
        gw_interfaces: Mapping[str, GWInterfaceBase],
        workspace_dim: int,
        loss_coefs: Mapping[str, torch.Tensor],
        use_var_contrastive_loss: bool = False,
        optim_lr: float = 1e-3,
        optim_weight_decay: float = 0.0,
        scheduler_args: SchedulerArgs | None = None,
        learn_logit_scale: bool = False,
        contrastive_loss: ContrastiveLossType | None = None,
        var_contrastive_loss: VarContrastiveLossType | None = None,
    ) -> None:
        gw_mod = VariationalGWModule(gw_interfaces, workspace_dim)
        domain_mods = freeze_domain_modules(domain_mods)
        coef_buffers = DictBuffer(loss_coefs)

        if use_var_contrastive_loss:
            if var_contrastive_loss is None:
                var_contrastive_loss = ContrastiveLossWithUncertainty(
                    torch.tensor([1]).log(), "mean", learn_logit_scale
                )
            loss_mod = VariationalGWLosses(
                gw_mod,
                domain_mods,
                coef_buffers,
                var_contrastive_fn=var_contrastive_loss,
            )
        else:
            if contrastive_loss is None:
                contrastive_loss = ContrastiveLoss(
                    torch.tensor([1]).log(), "mean", learn_logit_scale
                )
            loss_mod = VariationalGWLosses(
                gw_mod,
                domain_mods,
                coef_buffers,
                contrastive_fn=contrastive_loss,
            )

        super().__init__(
            gw_mod,
            domain_mods,
            coef_buffers,
            loss_mod,
            optim_lr,
            optim_weight_decay,
            scheduler_args,
        )


class GlobalWorkspaceFusion(GlobalWorkspaceBase):
    def __init__(
        self,
        domain_mods: Mapping[str, DomainModule],
        gw_interfaces: Mapping[str, GWInterfaceBase],
        workspace_dim: int,
        loss_coefs: Mapping[str, torch.Tensor],
        optim_lr: float = 1e-3,
        optim_weight_decay: float = 0.0,
        scheduler_args: SchedulerArgs | None = None,
        learn_logit_scale: bool = False,
        contrastive_loss: ContrastiveLossType | None = None,
    ) -> None:
        gw_mod = GWModuleFusion(gw_interfaces, workspace_dim)
        domain_mods = freeze_domain_modules(domain_mods)
        coef_buffers = DictBuffer(loss_coefs)
        if contrastive_loss is None:
            contrastive_loss = ContrastiveLoss(
                torch.tensor([1 / 0.07]).log(), "mean", learn_logit_scale
            )
        loss_mod = GWLossesFusion(
            gw_mod,
            domain_mods,
            coef_buffers,
            contrastive_loss,
        )

        super().__init__(
            gw_mod,
            domain_mods,
            coef_buffers,
            loss_mod,
            optim_lr,
            optim_weight_decay,
            scheduler_args,
        )


def pretrained_global_workspace(
    checkpoint_path: str | Path,
    domain_mods: Mapping[str, DomainModule],
    gw_interfaces: Mapping[str, GWInterfaceBase],
    workspace_dim: int,
    loss_coefs: Mapping[str, torch.Tensor],
    contrastive_fn: ContrastiveLossType,
    **kwargs,
) -> GlobalWorkspace:
    gw_mod = GWModule(gw_interfaces, workspace_dim)
    domain_mods = freeze_domain_modules(domain_mods)
    coef_buffers = DictBuffer(loss_coefs)
    loss_mod = GWLosses(
        gw_mod,
        domain_mods,
        coef_buffers,
        contrastive_fn,
    )

    gw = GlobalWorkspace.load_from_checkpoint(
        checkpoint_path,
        domain_mods=domain_mods,
        gw_mod=gw_mod,
        coef_buffers=coef_buffers,
        loss_mod=loss_mod,
        **kwargs,
    )
    if not isinstance(gw, GlobalWorkspace):
        raise TypeError("model should be of type GlobalWorkspace")
    return gw
