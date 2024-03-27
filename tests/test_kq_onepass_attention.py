import torch

from shimmer.modules.selection import KQAttentionOnePass


def test_single_domain():
    domain_dim = 12
    head_size = 6
    batch_size = 2056

    attention = KQAttentionOnePass(domain_dim, head_size)
    # Simulate a single domain input
    single_domain_input = {"v_latents": torch.rand(batch_size, domain_dim)}
    gw_state = torch.rand(batch_size, domain_dim)

    expected_scores = torch.ones(batch_size, 1)
    attention_scores = attention(single_domain_input, gw_state)

    assert torch.allclose(
        attention_scores["v_latents"], expected_scores
    ), "Attention scores for single domain should be all 1s"


def test_multiple_domains_sumis1():
    domain_dim = 12
    head_size = 5
    batch_size = 2056
    attention = KQAttentionOnePass(domain_dim, head_size)

    multiple_domain_input = {
        "v_latents": torch.rand(batch_size, domain_dim),
        "attr": torch.rand(batch_size, domain_dim),
    }
    gw_state = torch.rand(batch_size, domain_dim)

    attention_scores = attention(multiple_domain_input, gw_state)

    scores_sum = sum(
        attention_scores[domain] for domain in multiple_domain_input.keys()
    )
    expected_sum = torch.ones(batch_size, 1)

    assert torch.allclose(
        scores_sum, expected_sum
    ), "Sum of attention scores across domains should be 1"


def test_attention_backward():
    torch.manual_seed(42)  # For reproducibility

    domain_dim = 12
    head_size = 6
    batch_size = 2056

    attention = KQAttentionOnePass(domain_dim, head_size)

    # Make sure all parameters and inputs require gradients
    attention.train()  # Ensure the module is in training mode
    for param in attention.parameters():
        param.requires_grad_(True)

    domains = {
        "v_latents": torch.rand(batch_size, domain_dim, requires_grad=True),
        "attr": torch.rand(batch_size, domain_dim, requires_grad=True),
    }
    gw_state = torch.rand(batch_size, domain_dim, requires_grad=True)

    attention_scores = attention(domains, gw_state)

    # Dummy loss computation (e.g., mean of all attention scores for simplicity)
    loss = sum(score.mean() for score in attention_scores.values())
    loss.backward()

    # Test if gradients are computed (not None) for the gw_state and input domains
    assert gw_state.grad is not None, "Gradients should be computed for gw_state"
    for domain, tensor in domains.items():
        assert (
            tensor.grad is not None
        ), f"Gradients should be computed for domain '{domain}' inputs"

    # Optionally, you could also check if gradients are computed for the module's parameters
    for name, param in attention.named_parameters():
        assert (
            param.grad is not None
        ), f"Gradients should be computed for parameter '{name}'"
