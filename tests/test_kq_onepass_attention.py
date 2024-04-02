import torch

from shimmer.modules.selection import KQFixedQSelection


def test_single_domain():
    domain_dim = 12
    head_size = 6
    batch_size = 2056

    attention = KQFixedQSelection(domain_dim, head_size)
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
    attention = KQFixedQSelection(domain_dim, head_size)
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


def test_attention_backward():
    domain_dim = 12
    head_size = 6
    batch_size = 2056

    attention = KQFixedQSelection(domain_dim, head_size)
    gw_state = torch.rand(batch_size, domain_dim, requires_grad=True)
    attention.update_gw_state(gw_state)

    domains = {
        "v_latents": torch.rand(batch_size, domain_dim, requires_grad=True),
        "attr": torch.rand(batch_size, domain_dim, requires_grad=True),
    }

    encodings_pre_fusion = {
        "v_latents": torch.rand(batch_size, domain_dim, requires_grad=True),
        "attr": torch.rand(batch_size, domain_dim, requires_grad=True),
    }
    attention_scores = attention(domains, encodings_pre_fusion)
    loss = sum(score.mean() for score in attention_scores.values())
    assert isinstance(loss, torch.Tensor)
    loss.backward()

    assert gw_state.grad is not None, "Gradients should be computed for gw_state"
    for domain, tensor in domains.items():
        assert (
            tensor.grad is not None
        ), f"Gradients should be computed for domain '{domain}' inputs"

    for name, param in attention.named_parameters():
        assert (
            param.grad is not None
        ), f"Gradients should be computed for parameter '{name}'"
