import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch import Tensor
from torch.optim.lr_scheduler import OneCycleLR

from shimmer.modules.global_workspace import (
    GlobalWorkspaceBase,
    GWModuleBase,
    SchedulerArgs,
)
from shimmer.modules.losses import GWLossesBase
from shimmer.modules.selection import (
    SelectionBase,
)
from shimmer.types import (
    LatentsDomainGroupsDT,
    LatentsDomainGroupsT,
    RawDomainGroupsT,
    RawDomainGroupT,
)
from shimmer.utils import group_device, groups_batch_size


class AttentionBase(LightningModule):
    """
    Attention Lightning Module.

    This is a wrapper around the different attention modules.
    It is used to train an attention/selection mechanism.
    """

    def __init__(
        self,
        gw: GlobalWorkspaceBase[GWModuleBase, SelectionBase, GWLossesBase],
        attention: SelectionBase,
        domain_names: Sequence[str],
        criterion: Callable[
            [torch.Tensor, RawDomainGroupT], tuple[torch.Tensor, torch.Tensor]
        ],
        domain_dim: int,
        fixed_corruption_vector: torch.Tensor | None = None,
        corruption_scaling: list[float] | None = None,
        corrupt_batch: bool = False,
        optim_lr: float = 1e-3,
        optim_weight_decay: float = 0.0,
        scheduler_args: SchedulerArgs | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "gw",
                "attention",
                "criterion",
            ]
        )

        self.gw = gw
        self.attention = attention
        self.list_domain_names = domain_names
        self.domain_names = frozenset(domain_names)
        self.criterion = criterion
        self.domain_dim = domain_dim
        self.fixed_corruption_vector = fixed_corruption_vector
        self.corruption_scaling = corruption_scaling
        self.corrupt_batch = corrupt_batch
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
    ) -> LatentsDomainGroupsDT:
        """
            Apply corruption to each tensor of the matched data
            by use of masking.
        Args:
            batch: A batch of latent domains.
        Returns:
            A batch where either one (of the domains) of each tensor is corrupted.
        """
        matched_data_dict: LatentsDomainGroupsDT = {}
        # Make a copy of the batch
        for domain_names, domains in batch.items():
            for domain_name, domain in domains.items():
                matched_data_dict.setdefault(domain_names, {})[domain_name] = domain
                continue
        device = group_device(domains)
        batch_size = groups_batch_size(batch)
        n_domains = len(self.domain_names)

        selected_domains = torch.randint(0, n_domains, (batch_size,), device=device)
        masked_domains = torch.nn.functional.one_hot(selected_domains, n_domains).to(
            device, torch.bool
        )
        # Inverse (which means this only works for two domains now)
        masked_domains_inversed = ~masked_domains

        if self.fixed_corruption_vector:
            corruption_vector = self.fixed_corruption_vector.expand(
                batch_size, self.domain_dim
            )
        else:
            corruption_vector = torch.randn(
                (batch_size, self.domain_dim), device=device
            )

        # Normalize the corruption vector
        corruption_vector = (
            corruption_vector - corruption_vector.mean(dim=1, keepdim=True)
        ) / corruption_vector.std(dim=1, keepdim=True)

        # Choose randomly from corruption scaling
        amount_corruption = (
            random.choice(self.corruption_scaling) if self.corruption_scaling else 1.0
        )
        # Scale the corruption vector based on the amount of corruption
        scaled_corruption_vector = (corruption_vector * 5) * amount_corruption

        for k, (domain_names, domains) in enumerate(matched_data_dict.items()):
            if domain_names == self.domain_names:
                for domain_name, domain in domains.items():
                    if domain_name == self.list_domain_names[0]:
                        domain[masked_domains[:, k]] += scaled_corruption_vector[
                            masked_domains[:, k]
                        ]
                    if domain_name == self.list_domain_names[1]:
                        domain[masked_domains_inversed[:, k]] += (
                            scaled_corruption_vector[masked_domains_inversed[:, k]]
                        )

        return matched_data_dict

    def old_apply_corruption(
        self,
        batch: LatentsDomainGroupsT,
    ) -> LatentsDomainGroupsDT:
        """
            Apply corruption to one of the latent domains.
        Args:
                batch: A batch of latent domains.
                corruption_vector: A vector to be added to the corrupted domain.
                corrupted_domain: The domain to be corrupted.
            Returns:
                A batch where one of the latent domains is corrupted.
        """
        # Check if batch or instance corruption is applied
        if self.corrupt_batch:
            corrupted_domain = random.choice(list(self.domain_names))
        matched_data_dict: LatentsDomainGroupsDT = {}
        for domain_names, domains in batch.items():
            if not self.corrupt_batch:
                corrupted_domain = random.choice(list(self.domain_names))
            for domain_name, domain in domains.items():
                if domain_names != self.domain_names or domain_name != corrupted_domain:
                    matched_data_dict.setdefault(domain_names, {})[domain_name] = domain
                    continue

                # Create corruption vector if not given
                if self.fixed_corruption_vector is None:
                    corruption_vector = torch.randn_like(domain)
                    corruption_vector = torch.randn_like(domain[:, 0])
                    print(domain.shape)
                    print("dynamic corruption vector")
                    print(corruption_vector)
                elif self.fixed_corruption_vector.shape != domain.shape:
                    corruption_vector = self.fixed_corruption_vector[: domain.shape[0]]
                else:
                    corruption_vector = self.fixed_corruption_vector
                normalized_corruption_vector = (
                    corruption_vector - corruption_vector.mean()
                ) / corruption_vector.std()
                # Random choose corruption from 1 to 10 (1.0 means no scaling)
                amount_corruption = (
                    random.choice(self.corruption_scaling)
                    if self.corruption_scaling
                    else 1.0
                )
                # Scale the corruption vector based on the amount of corruption
                scaled_corruption_vector = (
                    normalized_corruption_vector * 50
                ) * amount_corruption
                # Apply element-wise addition to one of the domains
                matched_data_dict.setdefault(domain_names, {})[domain_name] = (
                    domain + scaled_corruption_vector
                )
        return matched_data_dict

    def generic_step(self, batch: RawDomainGroupsT, mode: str) -> Tensor:
        latent_domains = self.gw.encode_domains(batch)
        corrupted_batch = self.apply_corruption(latent_domains)
        prefusion_encodings = self.gw.encode(corrupted_batch)
        attention_scores = self.forward(corrupted_batch, prefusion_encodings)
        merged_gw_representation = self.gw.fuse(prefusion_encodings, attention_scores)
        losses = []
        accuracies = []

        for domain_names, domains in merged_gw_representation.items():
            loss, accuracy = self.criterion(domains, batch[domain_names])
            losses.append(loss)
            accuracies.append(accuracy)
            domain_names_str = ",".join(domain_names)
            self.log(
                f"{mode}/{domain_names_str}_loss",
                losses[-1],
                batch_size=domains.size(0),
            )
            self.log(
                f"{mode}/{domain_names_str}_accuracy",
                accuracies[-1],
                batch_size=domains.size(0),
            )
        loss = torch.stack(losses).mean()
        self.log(f"{mode}/loss", loss, on_step=True, on_epoch=True)
        self.log(f"{mode}/accuracy", torch.stack(accuracies).mean())

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
