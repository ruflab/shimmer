import torch
from utils import DummyDomainModule

from shimmer import GWDecoder, GWEncoder, GWModuleWithConfidence
from shimmer.modules.gw_module import compute_fusion_scores


def test_confidence_fusion():
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

    gw_module = GWModuleWithConfidence(domains, workspace_dim, gw_encoders, gw_decoders)

    batch_size = 32
    batch = {
        domain_name: torch.randn(batch_size, domain.latent_dim)
        for domain_name, domain in domains.items()
    }

    pre_fusion_reps = gw_module.encode(batch)
    selection_scores = {
        domain: torch.full((batch_size,), 1.0 / 3.0) for domain in gw_encoders
    }
    scores: list[torch.Tensor] = []
    precisions: list[torch.Tensor] = []
    domains_: list[torch.Tensor] = []
    for domain, score in selection_scores.items():
        scores.append(score)
        precisions.append(gw_module.get_precision(domain, pre_fusion_reps[domain]))
        domains_.append(pre_fusion_reps[domain])
    combined_scores = compute_fusion_scores(
        torch.stack(scores).unsqueeze(-1),
        torch.softmax(torch.stack(precisions), dim=0),
        1,
        1,
    )
    assert torch.allclose(
        combined_scores.sum(dim=0), torch.ones_like(combined_scores.sum(dim=0))
    )
