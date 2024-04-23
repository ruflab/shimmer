import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from modules.selection import DynamicQueryAttention, SelectionBase
from torch import Tensor, nn
from torch.optim.lr_scheduler import OneCycleLR

from shimmer.modules.global_workspace import GlobalWorkspaceBase, SchedulerArgs
from shimmer.modules.gw_module import GWModuleBase
from shimmer.modules.losses import GWLossesBase
from shimmer.types import (
    LatentsDomainGroupsDT,
    LatentsDomainGroupsT,
    RawDomainGroupsT,
)


class ClassificationHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Increasing the number of layers for more complexity
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        return self.fc5(x)


class DynamicAttention(LightningModule):
    """
    Attention Lightning Module.

    This is a wrapper around the DynamicQueryAttention module
    """

    def __init__(
        self,
        gw_module: GlobalWorkspaceBase[GWModuleBase, SelectionBase, GWLossesBase],
        batch_size: int,
        domain_dim: int,
        head_size: int,
        domain_names: Sequence[str],
        criterion: Callable[[torch.Tensor, RawDomainGroupsT], torch.Tensor],
        optim_lr: float,
        scheduler_args: SchedulerArgs | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["criterion"])

        self.gw_module = gw_module
        self.attention = DynamicQueryAttention(
            batch_size, domain_dim, head_size, domain_names
        )
        self.domain_names = domain_names
        self.criterion = criterion
        self.optim_lr = optim_lr
        self.scheduler_args = SchedulerArgs(max_lr=optim_lr, total_steps=1)

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

    def forward(self, single_domain_input, prefusion_encodings):
        return self.attention(single_domain_input, prefusion_encodings)

    def apply_corruption(
        self,
        batch: LatentsDomainGroupsT,
        corruption_vector: torch.Tensor | None = None,
        corrupted_domain: str | None = None,
    ) -> LatentsDomainGroupsT:
        """
        Apply corruption to the batch.

        Args:
            batch: The batch of data.
            dict_domain_names: The domain names to look for in the batch
            gw_dim: The global workspace dimension.
            corruption_vector: The corruption vector, if defined outside training.
        """
        if corrupted_domain is None:
            # Specify which domain will be corrupted
            corrupted_domain = random.choice(self.domain_names)

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

    def generic_step(
        self, batch: RawDomainGroupsT, batch_idx: int
    ) -> Tensor | Mapping[str, Any] | None:
        latent_domains = self.gw_module.encode_domains(batch)

        corrupted_batch = self.apply_corruption(latent_domains)

        prefusion_encodings = self.gw_module.encode(corrupted_batch)
        attention_scores = self.forward(corrupted_batch, prefusion_encodings)
        merged_gw_representation = self.gw_module.gw_mod.fuse(
            prefusion_encodings, attention_scores
        )

        loss = self.criterion(merged_gw_representation, batch)

        for name, metric in loss.all.items():
            self.log(
                f"{name}",
                metric,
                batch_size=self.batch_size,
                add_dataloader_idx=False,
            )
        return loss

    def training_step(
        self, batch: RawDomainGroupsT, batch_idx: int
    ) -> Tensor | Mapping[str, Any] | None:  # type: ignore
        return self.generic_step(batch, batch_idx)

    def validation_step(
        self, *args: Any, **kwargs: Any
    ) -> Tensor | Mapping[str, Any] | None:
        return super().validation_step(*args, **kwargs)

    def test_step(self, *args: Any, **kwargs: Any) -> Tensor | Mapping[str, Any] | None:
        return super().test_step(*args, **kwargs)
