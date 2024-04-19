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

    prefusion_encodings = torch.rand(2, batch_size, domain_dim)
    selection_scores = selection(multiple_domain_input, prefusion_encodings)

    # Ensure the sum of attention scores across domains equals 1
    scores_sum = selection_scores.sum(dim=0)
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

    prefusion_encodings = torch.rand(3, batch_size, domain_dim)
    selection_scores = selection(three_domain_input, prefusion_encodings)

    # Ensure that the shape of the selection scores matches the input domains
    assert selection_scores.shape == (
        3,
        batch_size,
    ), "Scores shape mismatch"

    # Ensure the sum of attention scores across domains equals 1
    scores_sum = selection_scores.sum(dim=0)
    expected_sum = torch.ones(batch_size)

    assert torch.allclose(
        scores_sum, expected_sum
    ), "Sum of selection scores across three domains should be 1"
