import torch
from utils import DummyDomainModule

from shimmer import GWDecoder, GWEncoder, GWModuleBayesian


def test_bayesian_fusion():
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

    gw_module = GWModuleBayesian(domains, workspace_dim, gw_encoders, gw_decoders)

    batch_size = 32
    batch = {
        domain_name: torch.randn(batch_size, domain.latent_dim)
        for domain_name, domain in domains.items()
    }

    precisions = torch.stack(list(gw_module.get_precision(batch).values()))
    assert torch.allclose(precisions.sum(dim=0), torch.ones_like(precisions.sum(dim=0)))
