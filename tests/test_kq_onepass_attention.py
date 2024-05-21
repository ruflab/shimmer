import torch

from shimmer.modules.selection import KQFixedQSelection


def test_single_domain():
    domain_dim = 12
    head_size = 6
    batch_size = 2056
    domains = ["v_latents"]

    attention = KQFixedQSelection(head_size, domain_dim, domains)
    gw_state = torch.rand(batch_size, domain_dim)
    attention.update_gw_state(gw_state)

    single_domain_input = {"v_latents": torch.rand(batch_size, domain_dim)}
    encodings_pre_fusion = {"v_latents": torch.rand(batch_size, domain_dim)}
    attention_scores = attention(single_domain_input, encodings_pre_fusion)

    expected_scores = torch.ones(batch_size, 1)
    assert torch.allclose(
        attention_scores["v_latents"], expected_scores
    ), "Attention scores for single domain should be all 1s"


def test_multiple_domains_sumis1():
    domain_dim = 12
    head_size = 5
    batch_size = 2056
    domains = ["v_latents", "attr"]
    attention = KQFixedQSelection(head_size, domain_dim, domains)
    gw_state = torch.rand(batch_size, domain_dim)
    attention.update_gw_state(gw_state)

    multiple_domain_input = {
        "v_latents": torch.rand(batch_size, domain_dim),
        "attr": torch.rand(batch_size, domain_dim),
    }
    encodings_pre_fusion = {
        "v_latents": torch.rand(batch_size, domain_dim),
        "attr": torch.rand(batch_size, domain_dim),
    }

    attention_scores = attention(multiple_domain_input, encodings_pre_fusion)

    scores_sum = sum(
        attention_scores[domain].squeeze() for domain in multiple_domain_input
    )
    assert isinstance(scores_sum, torch.Tensor)

    expected_sum = torch.ones(batch_size)

    assert torch.allclose(
        scores_sum, expected_sum
    ), "Sum of attention scores across domains should be 1"
