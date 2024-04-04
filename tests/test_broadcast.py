import torch
from torch import nn
import pytorch_lightning as pl
from shimmer.modules.global_workspace import GlobalWorkspaceFusion
from shimmer.modules.selection import RandomSelection
from shimmer.modules.losses import GWLossesFusion
from shimmer.modules.gw_module import GWModule
from shimmer.modules.contrastive_loss import ContrastiveLoss, ContrastiveLossType
from shimmer.modules.domain import DomainModule, LossOutput
from typing import Dict, Literal
from torch.nn.functional import normalize, cross_entropy
from dataclasses import dataclass, field


def contrastive_loss(x: torch.Tensor, y: torch.Tensor) -> LossOutput:
    """
    Simplified CLIP-like contrastive loss that matches the expected signature.

    Args:
        x (torch.Tensor): Predictions.
        y (torch.Tensor): Targets.

    Returns:
        LossOutput: A dataclass containing the computed loss and optionally additional metrics.
    """
    # Assuming logit_scale is a pre-defined tensor if needed for the calculation
    # For the sake of matching the function signature, we'll remove it from the parameters
    # Similarly, we assume a fixed reduction mode for simplicity
    logit_scale = torch.tensor(1.0)  # Placeholder for an actual logit scale if necessary
    reduction = "mean"  # Fixed reduction mode

    xn = normalize(x, dim=-1)
    yn = normalize(y, dim=-1)
    logits = torch.matmul(xn, yn.t())
    labels = torch.arange(xn.size(0), device=xn.device)
    ce_loss = 0.5 * (cross_entropy(logits * logit_scale.exp(), labels, reduction=reduction) + 
                     cross_entropy(logits.t() * logit_scale.exp(), labels, reduction=reduction))

    return LossOutput(loss=ce_loss)


class DummyDomainModule(DomainModule):
    def __init__(self, latent_dim: int):
        super().__init__(latent_dim)
        self.encoder = nn.Linear(latent_dim, latent_dim)  # Simplified encoder
        self.decoder = nn.Linear(latent_dim, latent_dim)  # Simplified decoder

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)  # Simple forward pass through encoder

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)  # Simple forward pass through decoder

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> LossOutput:
        loss = torch.mean((pred - target) ** 2)  # Simple MSE loss
        return LossOutput(loss=loss)  # Constructing LossOutput with the loss

# Setting up the test environment for GlobalWorkspaceFusion
def setup_global_workspace_fusion():
    domain_mods: Dict[str, DomainModule] = {
        "domain1": DummyDomainModule(latent_dim=10),
        "domain2": DummyDomainModule(latent_dim=10),
    }
    gw_encoders = {"domain1": nn.Linear(10, 10), "domain2": nn.Linear(10, 10)}
    gw_decoders = {"domain1": nn.Linear(10, 10), "domain2": nn.Linear(10, 10)}
    workspace_dim = 10
    selection_mod = RandomSelection(temperature=0.2)
    contrastive_fn: ContrastiveLossType = contrastive_loss

    gw_mod = GWModule(domain_mods, workspace_dim, gw_encoders, gw_decoders)
    loss_mod = GWLossesFusion(gw_mod, selection_mod, domain_mods, contrastive_fn)

    gw_fusion = GlobalWorkspaceFusion(
        domain_mods=domain_mods,
        gw_encoders=gw_encoders,
        gw_decoders=gw_decoders,
        workspace_dim=workspace_dim,
        optim_lr=1e-3,
        optim_weight_decay=0.0,
        scheduler_args=None,  # Simplified for testing
        learn_logit_scale=False,
        contrastive_loss=contrastive_fn,
    )

    return gw_fusion

def test_broadcast_loss():
    gw_fusion = setup_global_workspace_fusion()

    # Adjusting the dummy data to fit the expected input structure for broadcast_loss
    # Now using a frozenset for the keys to match LatentsDomainGroupsT
    latent_domains = {
        frozenset(["domain1", "domain2"]): {
            "domain1": torch.rand(5, 10),  # Batch size of 5, feature dimension of 10
            "domain2": torch.rand(5, 10),
        }
    }

    # Test broadcast_loss with the corrected structure
    output = gw_fusion.loss_mod.broadcast_loss(latent_domains, "train")
    print(output)

# Call the test function to execute the test
test_broadcast_loss()
