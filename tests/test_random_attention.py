import torch

from shimmer.modules.selection import RandomSelection


def test_multiple_domains():
    temperature = 1.0
    domain_dim = 12
    batch_size = 2056

    selection = RandomSelection(temperature)
    multiple_domain_input = {
        "v_latents": torch.rand(batch_size, domain_dim),
        "attr": torch.rand(batch_size, domain_dim),
    }

    prefusion_encodings = {
        "v_latents": torch.rand(batch_size, domain_dim),
        "attr": torch.rand(batch_size, domain_dim),
    }

    selection_scores = selection(multiple_domain_input, prefusion_encodings)

    # Ensure the sum of attention scores across domains equals 1
    scores_sum = sum(
        selection_scores[domain].squeeze() for domain in multiple_domain_input
    )
    assert isinstance(scores_sum, torch.Tensor)

    expected_sum = torch.ones(batch_size)

    assert torch.allclose(
        scores_sum, expected_sum
    ), "Sum of selection scores across domains should be 1"


def test_three_domains():
    temperature = 1.0
    domain_dim = 12
    batch_size = 2056

    selection = RandomSelection(temperature)
    three_domain_input = {
        "v_latents": torch.rand(batch_size, domain_dim),
        "attr": torch.rand(batch_size, domain_dim),
        "audio": torch.rand(batch_size, domain_dim),
    }

    prefusion_encodings = {
        "v_latents": torch.rand(batch_size, domain_dim),
        "attr": torch.rand(batch_size, domain_dim),
        "audio": torch.rand(batch_size, domain_dim),
    }

    selection_scores = selection(three_domain_input, prefusion_encodings)

    # Ensure that the shape of the selection scores matches the input domains
    for domain in three_domain_input:
        assert selection_scores[domain].shape == (
            batch_size,
        ), f"Scores shape mismatch for {domain}"

    # Ensure the sum of attention scores across domains equals 1
    scores_sum = sum(
        selection_scores[domain].squeeze() for domain in three_domain_input
    )
    assert isinstance(scores_sum, torch.Tensor)

    expected_sum = torch.ones(batch_size)

    assert torch.allclose(
        scores_sum, expected_sum
    ), "Sum of selection scores across three domains should be 1"
