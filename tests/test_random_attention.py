import numpy as np
import torch

from shimmer.modules.selection import RandomSelection


def test_multiple_domains():
    binary_proportion = 0.5
    temperature = 1.0
    domain_dim = 12
    batch_size = 2056

    selection = RandomSelection(binary_proportion, temperature)
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
    binary_proportion = 0.5
    temperature = 1.0
    domain_dim = 12
    batch_size = 2056

    selection = RandomSelection(binary_proportion, temperature)
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
            1,
        ), f"Scores shape mismatch for {domain}"

    # Check if the binary scores are as expected
    # This part might need adjustments based on how binary scores are distributed
    # and combined with uniform scores in your actual implementation

    # Check if the sum of selection scores across domains equals 1
    scores_sum = sum(
        selection_scores[domain].squeeze() for domain in three_domain_input
    )
    assert isinstance(scores_sum, torch.Tensor)

    expected_sum = torch.ones(batch_size)

    assert torch.allclose(
        scores_sum, expected_sum
    ), "Sum of selection scores across three domains should be 1"


def test_binary_scores_xor_check_for_multiple_proportions():
    temperature = 1.0
    domain_dim = 12
    batch_size = 2056
    num_tests = 10  # Number of random proportions to test

    for _ in range(num_tests):
        binary_proportion = np.random.rand()  # Random proportion between 0 and 1

        selection = RandomSelection(binary_proportion, temperature)
        domains_input = {
            "v_latents": torch.rand(batch_size, domain_dim),
            "attr": torch.rand(batch_size, domain_dim),
            "audio": torch.rand(batch_size, domain_dim),
        }

        prefusion_encodings = {
            "v_latents": torch.rand(batch_size, domain_dim),
            "attr": torch.rand(batch_size, domain_dim),
            "audio": torch.rand(batch_size, domain_dim),
        }

        selection_scores = selection(domains_input, prefusion_encodings)

        scores_matrix = torch.cat(
            [selection_scores[domain] for domain in domains_input], dim=1
        )
        binary_scores_mask = scores_matrix == 1
        xor_binary_check = binary_scores_mask.sum(dim=1) == 1
        num_binary_rows = xor_binary_check.sum().item()
        expected_num_binary_rows = int(batch_size * binary_proportion)

        assert num_binary_rows == expected_num_binary_rows, (
            "Incorrect number of binary score rows for proportion"
            f"{binary_proportion:.2f}: expected {expected_num_binary_rows}, "
            "got {num_binary_rows}"
        )
