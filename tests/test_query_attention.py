import torch

from shimmer.modules.selection import DynamicQueryAttention


def test_single_domain_simplified():
    domain_dim = 2
    head_size = 1
    batch_size = 2

    # First gw state
    gw_state = torch.tensor([[2.0, 2.0], [2.0, 2.0]])

    # Initialize the attention module
    attention = DynamicQueryAttention(batch_size, domain_dim, head_size)

    # The initial inputs
    single_domain_input = {"v_latents": torch.tensor([[4.0, 2.0], [6.0, 2.0]])}
    prefusion_encodings = {"v_latents": torch.tensor([[4.0, 2.0], [6.0, 2.0]])}

    # This is the forward pass
    attention_scores = attention(single_domain_input, prefusion_encodings)

    print(f"attention scores: {attention_scores}")

    expected_scores = torch.ones(batch_size, 1)
    assert torch.allclose(
        attention_scores["v_latents"], expected_scores
    ), "Attention scores for single domain should be all 1s"


def test_single_domain():
    domain_dim = 12
    head_size = 6
    batch_size = 2056

    attention = DynamicQueryAttention(batch_size, domain_dim, head_size)
    gw_state = torch.rand(batch_size, domain_dim)
    attention.update_gw_state(gw_state)

    single_domain_input = {"v_latents": torch.rand(batch_size, domain_dim)}
    prefusion_encodings = {"v_latents": torch.rand(batch_size, domain_dim)}

    attention_scores = attention(single_domain_input, prefusion_encodings)

    expected_scores = torch.ones(batch_size, 1)
    assert torch.allclose(
        attention_scores["v_latents"], expected_scores
    ), "Attention scores for single domain should be all 1s"


def test_multiple_domains_sumis1():
    domain_dim = 12
    head_size = 5
    batch_size = 2056
    attention = DynamicQueryAttention(batch_size, domain_dim, head_size)
    gw_state = torch.rand(batch_size, domain_dim)
    attention.update_gw_state(gw_state)

    multiple_domain_input = {
        "v_latents": torch.rand(batch_size, domain_dim),
        "attr": torch.rand(batch_size, domain_dim),
    }
    prefusion_encodings = {
        "v_latents": torch.rand(batch_size, domain_dim),
        "attr": torch.rand(batch_size, domain_dim),
    }
    attention_scores = attention(multiple_domain_input, prefusion_encodings)

    scores_sum = sum(
        attention_scores[domain].squeeze() for domain in multiple_domain_input
    )
    expected_sum = torch.ones(batch_size)

    assert torch.allclose(
        scores_sum, expected_sum
    ), "Sum of attention scores across domains should be 1"


def test_attention_backward():
    domain_dim = 12
    head_size = 6
    batch_size = 2056

    attention = DynamicQueryAttention(batch_size, domain_dim, head_size)
    gw_state = torch.rand(batch_size, domain_dim, requires_grad=True)
    attention.update_gw_state(gw_state)

    domains = {
        "v_latents": torch.rand(batch_size, domain_dim, requires_grad=True),
        "attr": torch.rand(batch_size, domain_dim, requires_grad=True),
    }
    prefusion_encodings = {
        "v_latents": torch.rand(batch_size, domain_dim, requires_grad=True),
        "attr": torch.rand(batch_size, domain_dim, requires_grad=True),
    }

    attention_scores = attention(domains, prefusion_encodings)

    loss = sum(score.mean() for score in attention_scores.values())
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
