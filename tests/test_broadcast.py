from typing import Any

import torch
from torch import nn

from shimmer.modules.domain import DomainModule, LossOutput
from shimmer.modules.global_workspace import GlobalWorkspaceFusion
from shimmer.modules.losses import BroadcastLossCoefs


class DummyDomainModule(DomainModule):
    def __init__(self, latent_dim: int):
        super().__init__(latent_dim)
        self.encoder = nn.Linear(latent_dim, latent_dim)  # Simplified encoder
        self.decoder = nn.Linear(latent_dim, latent_dim)  # Simplified decoder

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)  # Simple forward pass through encoder

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)  # Simple forward pass through decoder

    def compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor, raw_target: Any
    ) -> LossOutput:
        loss = torch.mean((pred - target) ** 2)  # Simple MSE loss
        return LossOutput(loss=loss)  # Constructing LossOutput with the loss


def test_broadcast():
    domain_mods: dict[str, DomainModule] = {
        "domain1": DummyDomainModule(latent_dim=10),
        "domain2": DummyDomainModule(latent_dim=10),
    }
    gw_encoders = {"domain1": nn.Linear(10, 10), "domain2": nn.Linear(10, 10)}
    gw_decoders = {"domain1": nn.Linear(10, 10), "domain2": nn.Linear(10, 10)}
    workspace_dim = 10
    loss_coefs: BroadcastLossCoefs = {
        "cycles": 1.0,
        "demi_cycles": 1.0,
        "translations": 1.0,
        "contrastives": 0.1,
    }

    gw_fusion = GlobalWorkspaceFusion(
        domain_mods,
        gw_encoders,
        gw_decoders,
        workspace_dim,
        loss_coefs,
        selection_temperature=0.2,
        optim_lr=1e-3,
        optim_weight_decay=0.0,
        scheduler_args=None,  # Simplified for testing
        learn_logit_scale=False,
    )

    # Adjusting the dummy data to fit the expected input structure for broadcast
    # Now using a frozenset for the keys to match LatentsDomainGroupsT
    latent_domains = {
        frozenset(["domain1", "domain2"]): {
            "domain1": torch.rand(5, 10),  # Batch size of 5, feature dimension of 10
            "domain2": torch.rand(5, 10),
        }
    }

    # Test broadcast with the corrected structure
    result = gw_fusion.loss_mod.broadcast(latent_domains, latent_domains)
    assert "metrics" in result and "cycle_cases" in result
    # Cycle metrics are computed in step(), not within broadcast
    metrics = result["metrics"]
    assert all(metric in metrics for metric in ["demi_cycles", "translations"])

    # Broadcast metrics should be logged from step (but not the deprecated aggregate)
    step_output = gw_fusion.loss_mod.step(latent_domains, latent_domains, mode="train")
    assert "broadcast_loss" not in step_output.metrics
    for metric in ["demi_cycles", "translations", "cycles"]:
        assert metric in step_output.metrics

    er_msg = "Losses should be scalar tensors or 1D tensor with size equal to one."
    assert all(
        (loss.dim() == 0 or (loss.dim() == 1 and loss.size(0) == 1))
        for key, loss in metrics.items()
        if key.endswith("_loss")
    ), er_msg
