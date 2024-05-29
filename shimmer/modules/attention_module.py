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
        self.domain_names = frozenset({"v_latents", "attr"})
        self.criterion = criterion
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
        Apply corruption to the batch.

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
        print("batch")
        print(batch)
        for domain_names, domains in batch.items():
            if domain_names == self.domain_names:
                for domain_name, domain in domains.items():
                    matched_data_dict.setdefault(domain_names, {})[domain_name] = domain
            # if not self.corrupt_batch:
            #     corrupted_domain = random.choice(list(self.domain_names))
            #     print("corrupted domain")
            #     print(corrupted_domain)
            for domain_name, domain in domains.items():
                if domain_names != self.domain_names or domain_name != corrupted_domain:
                    matched_data_dict.setdefault(domain_names, {})[domain_name] = domain
                    print("non corrupted matched data dict")
                    print(matched_data_dict)
                    continue

                # Create corruption vector if not given
                if self.fixed_corruption_vector is None:
                    corruption_vector = torch.randn(domain.size(1)).to("cuda:0")
                else:
                    corruption_vector = self.fixed_corruption_vector
                print("corruption vector")
                print(corruption_vector)
                normalized_corruption_vector = (
                    corruption_vector - corruption_vector.mean()
                ) / corruption_vector.std()
                print("normalized corruption vector")
                print(normalized_corruption_vector)
                # Random choose corruption from 1 to 10 (1.0 means no scaling)
                amount_corruption = (
                    random.choice(self.corruption_scaling)
                    if self.corruption_scaling
                    else 1.0
                )
                print("amount of corruption")
                print(amount_corruption)
                # Scale the corruption vector based on the amount of corruption
                scaled_corruption_vector = (
                    normalized_corruption_vector * 5
                ) * amount_corruption
                print("scaled corruption vector")
                print(scaled_corruption_vector)
                # Apply element-wise addition to one of the domains
                matched_data_dict.setdefault(domain_names, {})[domain_name] = (
                    domain + scaled_corruption_vector
                )
        print("matched_data_dict")
        print(matched_data_dict)

        return matched_data_dict

    def apply_corruption2(
        self,
        batch: LatentsDomainGroupsT,
    ) -> LatentsDomainGroupsT:
        """
        Apply corruption to the batch.

        Args:
            batch: A batch of latent domains.
            corruption_vector: A vector to be added to the corrupted domain.
            corrupted_domain: The domain to be corrupted.

        Returns:
            A batch where one of the latent domains is corrupted.
        """
        # matched_data_dict: LatentsDomainGroupsDT = {}
        print(" batch before corruption")
        print(batch)
        # Make a copy of the batch
        for domain_names, domains in batch.items():
            # Check if the domain-names are {frozenset({'v_latents', 'attr'})
            if domain_names == self.domain_names:
                # Iterate over each row index
                for row in range(domains[list(self.domain_names)[0]].size(1)):
                    # Randomly choose whether to corrupt 'v_latents' or 'attr'
                    domain_to_corrupt = random.choice(list(self.domain_names))

                    # Create corruption vector if not given
                    if self.fixed_corruption_vector is None:
                        corruption_vector = torch.randn(
                            domains[domain_to_corrupt].size(2)
                        ).to("cuda:0")
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
                    print("amount of corruption")
                    print(amount_corruption)
                    # Scale the corruption vector based on the amount of corruption
                    scaled_corruption_vector = (
                        normalized_corruption_vector * 5
                    ) * amount_corruption

                    domains[domain_to_corrupt][0][row] = (
                        domains[domain_to_corrupt][0][row] + scaled_corruption_vector
                    )
            batch[domain_names] = domains
        print("batch after corruption")
        print(batch)
        return batch

    def generic_step(self, batch: RawDomainGroupsT, mode: str) -> Tensor:
        latent_domains = self.gw.encode_domains(batch)
        corrupted_batch = self.apply_corruption2(latent_domains)
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
