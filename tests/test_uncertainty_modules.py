import torch
from utils import DummyDomainModule

from shimmer import GWDecoder, GWEncoder, GWModuleWithUncertainty


def test_uncertainty_fusion():
    domains = {
        "v": DummyDomainModule(latent_dim=2),
        "t": DummyDomainModule(latent_dim=4),
        "a": DummyDomainModule(latent_dim=8),
    }

    workspace_dim = 16

    gw_encoders = {
        domain_name: GWEncoder(
            domain.latent_dim,
            hidden_dim=64,
            out_dim=workspace_dim,
            n_layers=1,
        )
        for domain_name, domain in domains.items()
    }

    gw_decoders = {
        domain_name: GWDecoder(
            workspace_dim,
            hidden_dim=64,
            out_dim=domain.latent_dim,
            n_layers=1,
        )
        for domain_name, domain in domains.items()
    }

    gw_module = GWModuleWithUncertainty(
        domains, workspace_dim, gw_encoders, gw_decoders
    )

    batch_size = 32
    batch = {
        domain_name: torch.randn(batch_size, domain.latent_dim)
        for domain_name, domain in domains.items()
    }

    pre_fusion_reps = gw_module.encode(batch)
    selection_scores = torch.full(
        (
            len(gw_encoders),
            batch_size,
        ),
        1.0 / 3.0,
    )
    _, scores = gw_module._fuse_and_scores(pre_fusion_reps, selection_scores)
    assert torch.allclose(scores.sum(dim=0), torch.ones_like(scores.sum(dim=0)))
