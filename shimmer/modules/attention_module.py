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
        corrupt_single_side: str | None = None,
        corrupt_sides: bool = False,
        test_variable_corruption: dict[str, float] | None = None,
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
        self.corrupt_single_side = corrupt_single_side
        self.corrupt_sides = corrupt_sides
        self.test_variable_corruption = test_variable_corruption
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

    def apply_row_corruption(
        self,
        batch: LatentsDomainGroupsT,
    ) -> LatentsDomainGroupsDT:
        """
            Apply corruption to each tensor of the matched data
            by use of masking. Only for two domains.

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

        if self.corrupt_single_side is not None:
            corrupted_domain_index = self.list_domain_names.index(
                self.corrupt_single_side
            )
            masked_domains = torch.zeros(batch_size, n_domains, dtype=torch.bool)
            masked_domains[:, corrupted_domain_index] = True
        else:
            selected_domains = torch.randint(0, n_domains, (batch_size,), device=device)
            masked_domains = torch.nn.functional.one_hot(
                selected_domains, n_domains
            ).to(device, torch.bool)

        if self.fixed_corruption_vector is not None:
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
            random.choice(self.corruption_scaling) if self.corruption_scaling else 0.0
        )
        # Scale the corruption vector based on the amount of corruption
        scaled_corruption_vector = (corruption_vector * 5) * amount_corruption
        for _, (domain_names, domains) in enumerate(matched_data_dict.items()):
            if domain_names == self.domain_names:
                for domain_name, domain in domains.items():
                    if domain_name == self.list_domain_names[0]:
                        domain[masked_domains[:, 0]] += scaled_corruption_vector[
                            masked_domains[:, 0]
                        ]
                    if domain_name == self.list_domain_names[1]:
                        domain[~masked_domains[:, 0]] += scaled_corruption_vector[
                            ~masked_domains[:, 0]
                        ]
        return matched_data_dict

    def apply_variable_row_corruption(
        self,
        batch: LatentsDomainGroupsT,
    ) -> LatentsDomainGroupsDT:
        """
            Apply corruption to each tensor of the matched data
            by use of masking. Only for two domains.

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

        if self.fixed_corruption_vector is not None:
            corruption_vector = self.fixed_corruption_vector.expand(
                batch_size, self.domain_dim
            )
        else:
            corruption_vector = torch.randn(
                (batch_size, self.domain_dim), device=device
            )

        # Normalize the corruption vector
        normalized_corruption_vector = (
            corruption_vector - corruption_vector.mean(dim=1, keepdim=True)
        ) / corruption_vector.std(dim=1, keepdim=True)

        corruption_vectors = {}
        if self.test_variable_corruption is not None:
            for domain_name, amount_corruption in self.variable_corruption.items():
                scaled_corruption_vector = (
                    normalized_corruption_vector * 5
                ) * amount_corruption
                corruption_vectors[domain_name] = scaled_corruption_vector
        else:
            for domain in self.list_domain_names:
                amount_corruption = (
                    random.choice(self.corruption_scaling)
                    if self.corruption_scaling
                    else 1.0
                )
                scaled_corruption_vector = (
                    normalized_corruption_vector * 5
                ) * amount_corruption
                corruption_vectors[domain] = scaled_corruption_vector

        # print(f"corruption_vectors: {corruption_vectors}")
        # print(f"batch: {batch}")
        for _, (domain_names, domains) in enumerate(matched_data_dict.items()):
            if domain_names == self.domain_names:
                for domain_name, domain in domains.items():
                    domain += corruption_vectors[domain_name]

                    #     domain[masked_domains[:, 0]] += corruption_vectors[domain_name][
                    #         masked_domains[:, 0]
                    #     ]
                    # # Hier zat een ~ voor masked_domains
                    # if domain_name == self.list_domain_names[1]:
                    #     domain[~masked_domains[:, 0]] += corruption_vectors[domain_name][
                    #         ~masked_domains[:, 0]
                    #     ]
        # print(f"matched_data_dict: {matched_data_dict}")
        return matched_data_dict

    def apply_probabilistic_row_corruption(
        self,
        batch: LatentsDomainGroupsT,
    ) -> LatentsDomainGroupsDT:
        """
            Apply corruption to each tensor of the matched data
            by use of masking. Only for two domains.

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

        domain_names = list(self.variable_corruption.keys())
        corrupt_ratio = list(self.variable_corruption.values())

        assert sum(corrupt_ratio) == 1.0

        probabilities = torch.tensor(corrupt_ratio)

        corrupt_choice = torch.multinomial(probabilities, batch_size, replacement=True)

        # Initialize the masked_domains tensor
        masked_domains = torch.zeros(batch_size, n_domains, dtype=torch.bool)

        # Set the chosen domains to be corrupted
        for i in range(batch_size):
            masked_domains[i, corrupt_choice[i]] = True

        if self.fixed_corruption_vector is not None:
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
        for _, (domain_names, domains) in enumerate(matched_data_dict.items()):
            if domain_names == self.domain_names:
                for domain_name, domain in domains.items():
                    if domain_name == self.list_domain_names[0]:
                        domain[masked_domains[:, 0]] += scaled_corruption_vector[
                            masked_domains[:, 0]
                        ]
                    if domain_name == self.list_domain_names[1]:
                        domain[~masked_domains[:, 0]] += scaled_corruption_vector[
                            ~masked_domains[:, 0]
                        ]
        return matched_data_dict

    def calculate_mean_attention(
        self,
        attention_scores: dict[frozenset[str], dict[str, Tensor]],
    ) -> dict:
        """
        Calculate the mean attention scores for each domain.
        """
        # Initialize variables to accumulate mean scores
        mean_attention_dict = {}
        # Iterate through attention_dicts
        for _, scores in attention_scores.items():
            # Check if more than 1 domains are present
            if len(scores) > 1:
                for key, values in scores.items():
                    # Accumulate mean scores for each key
                    mean_score = values.mean().item()
                    mean_attention_dict[key] = mean_score
        return mean_attention_dict

    def generic_step(self, batch: RawDomainGroupsT, mode: str) -> Tensor:
        latent_domains = self.gw.encode_domains(batch)
        if self.corrupt_sides is True:
            corrupted_batch = self.apply_variable_row_corruption(latent_domains)
        else:
            corrupted_batch = self.apply_row_corruption(latent_domains)
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
        mean_attention_scores = self.calculate_mean_attention(attention_scores)
        for domain_name, score in mean_attention_scores.items():
            self.log(f"{mode}/{domain_name}_mean_attention_score", score)

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
