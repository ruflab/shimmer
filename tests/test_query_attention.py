import torch

from shimmer.modules.selection import DynamicQueryAttention


def test_single_domain():
    domain_dim = 12
    head_size = 6
    batch_size = 2056
    domains = ["v_latents"]
    attention = DynamicQueryAttention(batch_size, domain_dim, head_size, domains)

    single_domain_input = {"v_latents": torch.rand(batch_size, domain_dim)}
    prefusion_encodings = torch.rand(1, batch_size, domain_dim)

    attention_scores = attention(single_domain_input, prefusion_encodings)

    expected_scores = torch.ones(batch_size)
    assert torch.allclose(
        attention_scores[0], expected_scores
    ), "Attention scores for single domain should be all 1s"


def test_multiple_domains_sumis1():
    domain_dim = 12
    head_size = 5
    batch_size = 2056
    domains = ["v_latents", "attr"]

    attention = DynamicQueryAttention(batch_size, domain_dim, head_size, domains)

    multiple_domain_input = {
        "v_latents": torch.rand(batch_size, domain_dim),
        "attr": torch.rand(batch_size, domain_dim),
    }
    prefusion_encodings = torch.rand(2, batch_size, domain_dim)
    attention_scores = attention(multiple_domain_input, prefusion_encodings)

    scores_sum = attention_scores.sum(dim=0)
    assert isinstance(scores_sum, torch.Tensor)

    expected_sum = torch.ones(batch_size)

    assert torch.allclose(
        scores_sum, expected_sum
    ), "Sum of attention scores across domains should be 1"


def test_attention_backward():
    domain_dim = 12
    head_size = 6
    batch_size = 2056
    domain_names = ["v_latents", "attr"]

    attention = DynamicQueryAttention(batch_size, domain_dim, head_size, domain_names)

    domains = {
        "v_latents": torch.rand(batch_size, domain_dim, requires_grad=True),
        "attr": torch.rand(batch_size, domain_dim, requires_grad=True),
    }
    prefusion_encodings = torch.rand(2, batch_size, domain_dim, requires_grad=True)

    attention_scores = attention(domains, prefusion_encodings)

    loss = attention_scores.mean(dim=1).sum(dim=0)
    assert isinstance(loss, torch.Tensor)
