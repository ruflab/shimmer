import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch import Tensor, nn
from torch.optim.lr_scheduler import OneCycleLR

from shimmer.modules.global_workspace import GlobalWorkspaceBase, SchedulerArgs
from shimmer.modules.gw import GWModuleBase
from shimmer.modules.losses import GWLossesBase
from shimmer.modules.selection import DynamicQueryAttention, SelectionBase
from shimmer.types import (
    LatentsDomainGroupsDT,
    LatentsDomainGroupsT,
    RawDomainGroupsT,
    RawDomainGroupT,
)


class ShapesClassifier(nn.Sequential):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, output_dim)


class DynamicAttention(LightningModule):
    """
    Attention Lightning Module.

    This is a wrapper around the DynamicQueryAttention module.
    It is used to train the Dynamic Query Attention mechanism.
    """

    def __init__(
        self,
        gw: GlobalWorkspaceBase[GWModuleBase, SelectionBase, GWLossesBase],
        domain_dim: int,
        head_size: int,
        domain_names: Sequence[str],
        criterion: Callable[[torch.Tensor, RawDomainGroupT], torch.Tensor],
        optim_lr: float = 1e-3,
        optim_weight_decay: float = 0.0,
        scheduler_args: SchedulerArgs | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "gw",
                "criterion",
            ]
        )

        self.gw = gw
        self.attention = DynamicQueryAttention(head_size, domain_dim, domain_names)
        self.domain_names = domain_names
        self.criterion = criterion
        self.optim_lr = optim_lr
        self.optim_weight_decay = optim_weight_decay
        self.scheduler_args = SchedulerArgs(max_lr=optim_lr, total_steps=1)
        if scheduler_args is not None:
            self.scheduler_args.update(scheduler_args)

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """
        Configure models optimizers.

        Here we use `AdamW` for the optimizer and `OneCycleLR` for the learning-rate
        scheduler.
        """

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

    def forward(
        self,
        single_domain_input: LatentsDomainGroupsT,
        prefusion_encodings: LatentsDomainGroupsT,
    ) -> LatentsDomainGroupsDT:
        return {
            domains: self.attention(latents, prefusion_encodings[domains])
            for domains, latents in single_domain_input.items()
        }

    def apply_corruption(
        self,
        batch: LatentsDomainGroupsT,
        corruption_vector: torch.Tensor | None = None,
        corrupted_domain: str | None = None,
    ) -> LatentsDomainGroupsDT:
        """
        Apply corruption to the batch.

        Args:
            batch: A batch of latent domains.
            corruption_vector: A vector to be added to the corrupted domain.
            corrupted_domain: The domain to be corrupted.

        Returns:
            A batch where one of the latent domains is corrupted.
        """
        if corrupted_domain is None:
            # Specify which domain will be corrupted
            corrupted_domain = random.choice(list(self.domain_names))

        matched_data_dict: LatentsDomainGroupsDT = {}
        for domain_names, domains in batch.items():
            for domain_name, domain in domains.items():
                if domain_names != self.domain_names or domain_name != corrupted_domain:
                    matched_data_dict.setdefault(domain_names, {})[domain_name] = domain
                    continue

                # If corruption vector is not fixed outside the loop
                if corruption_vector is None:
                    corruption_vector = torch.randn_like(domain)

                # Apply element-wise addition to one of the domains
                matched_data_dict.setdefault(domain_names, {})[domain_name] = (
                    domain + corruption_vector
                )

        return matched_data_dict

    def generic_step(self, batch: RawDomainGroupsT, mode: str) -> Tensor:
        latent_domains = self.gw.encode_domains(batch)
        corrupted_batch = self.apply_corruption(latent_domains)
        prefusion_encodings = self.gw.encode(corrupted_batch)
        attention_scores = self.forward(corrupted_batch, prefusion_encodings)
        merged_gw_representation = self.gw.fuse(prefusion_encodings, attention_scores)
        losses = []
        for domain_names, domains in merged_gw_representation.items():
            losses.append(self.criterion(domains, batch[domain_names]))
            domain_names_str = ",".join(domain_names)
            self.log(
                f"{mode}/{domain_names_str}_loss",
                losses[-1],
                batch_size=domains.size(0),
            )
        loss = torch.stack(losses).mean()
        print(f"loss: {loss}")
        self.log(f"{mode}/loss", loss, on_step=True, on_epoch=True)

        return loss

    def training_step(
        self, batch: RawDomainGroupsT, batch_idx: int
    ) -> Tensor | Mapping[str, Any] | None:  # type: ignore
        return self.generic_step(batch, "train")

    def validation_step(  # type: ignore
        self, data: RawDomainGroupT, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Validation step used by lightning"""

        batch = {frozenset(data.keys()): data}
        for domain in data:
            batch[frozenset([domain])] = {domain: data[domain]}
        if dataloader_idx == 0:
            return self.generic_step(batch, mode="val")
        return self.generic_step(batch, mode="val/ood")

    def test_step(  # type: ignore
        self, data: RawDomainGroupT, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Test step used by lightning"""

        batch = {frozenset(data.keys()): data}
        for domain in data:
            batch[frozenset([domain])] = {domain: data[domain]}
        if dataloader_idx == 0:
            return self.generic_step(batch, mode="test")
        return self.generic_step(batch, mode="test/ood")
