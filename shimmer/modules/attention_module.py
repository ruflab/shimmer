from collections.abc import Callable, Mapping
from typing import Any

import torch
from lightning.pytorch import LightningModule
from modules.selection import DynamicQueryAttention
from torch import Tensor, nn


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
        gw_module: nn.Module,
        batch_size: int,
        domain_dim: int,
        head_size: int,
        domain_names: list[str],
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["criterion"])

        self.gw_module = gw_module
        self.attention = DynamicQueryAttention(
            batch_size, domain_dim, head_size, domain_names
        )
        self.domain_names = domain_names
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(list(self.attention.parameters()), lr=0.0005)

    def forward(self, single_domain_input, prefusion_encodings):
        return self.attention(single_domain_input, prefusion_encodings)

    def apply_corruption(
        batch: dict,
        dict_domain_names: frozenset,
        corruption_vector=None,
        corrupted_domain=None,
    ):
        """
        Apply corruption to the batch.

        Args:
            batch: The batch of data.
            dict_domain_names: The domain names to look for in the batch
            gw_dim: The global workspace dimension.
            corruption_vector: The corruption vector, if defined outside training.
        """

        corrupted_domain = "attr"
        if corrupted_domain is None:
            # Specify which domain will be corrupted
            corrupted_domain = random.choice(list(dict_domain_names))

        # Get the data tensor with only matched data
        matched_data_dict = batch[dict_domain_names]

        # Remove the list around value of 'attr'
        for key, _ in matched_data_dict.items():
            if key == "attr":
                matched_data_dict[key] = matched_data_dict[key][0]

        tensor_size = matched_data_dict[corrupted_domain].shape

        # If corruption vector is not fixed outside the loop
        if corruption_vector is None:
            corruption_vector = torch.randn(tensor_size, device=torch.device("cuda:0"))

        # Some tensors have type long, some type float
        matched_data_dict[corrupted_domain] = matched_data_dict[corrupted_domain].to(
            torch.float32
        )
        corruption_vector = corruption_vector.to(
            matched_data_dict[corrupted_domain].device
        )

        # Apply element-wise addition to one of the domains
        matched_data_dict[corrupted_domain] += corruption_vector

        return matched_data_dict

    def training_step(self, batch, batch_idx) -> Tensor | Mapping[str, Any] | None:
        # Assume its one batch
        filtered_batch = batch[0]
        # How to get the shape info?
        target = shape_info_attr
        corrupted_batch = self.apply_corruption(filtered_batch, self.domain_names)
        attention_scores = self.forward(corrupted_batch, prefusion_encodings)
        merged_gw_representation = module.encode_and_fuse(
            attention_scores, self.attention
        )

        loss = self.criterion(merged_gw_representation, target)
        return loss

    def validation_step(
        self, *args: Any, **kwargs: Any
    ) -> Tensor | Mapping[str, Any] | None:
        return super().validation_step(*args, **kwargs)
