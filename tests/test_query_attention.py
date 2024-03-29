import torch

from shimmer.modules.selection import KQDynamicQSelection


def test_single_domain_simplified():
    domain_dim = 2  # 12 # amount of columns
    head_size = 1  # 6
    batch_size = 2  # 2056 # amount of rows

    # initial naive (random) gw_state
    # gw_state = torch.rand(batch_size, domain_dim)
    gw_state = torch.tensor([[2.0, 2.0], [2.0, 2.0]])

    # Initialize the attention module
    attention = KQDynamicQSelection(domain_dim, head_size)
    attention.update_gw_state(gw_state)

    single_domain_input = {"v_latents": torch.tensor([[4.0, 2.0], [6.0, 2.0]])}
    # print(f"input: {single_domain_input}")

    # This is the forward pass
    attention_scores = attention(single_domain_input)
    # print(f"attention scores: {attention_scores}")

    # Calculate the next gw_state
    new_gw_state = attention.calculate_gw_state_with_attention(attention_scores)
    # print(f"new state: {new_gw_state}")

    # Update the gw_state
    attention.update_gw_state(new_gw_state)

    expected_scores = torch.ones(batch_size, 1)
    assert torch.allclose(
        attention_scores["v_latents"], expected_scores
    ), "Attention scores for single domain should be all 1s"


# def test_single_domain():
#     domain_dim = 12
#     head_size = 6
#     batch_size = 2056

#     attention = KQDynamicQSelection(domain_dim, head_size)
#     gw_state = torch.rand(batch_size, domain_dim)
#     attention.update_gw_state(gw_state)

#     single_domain_input = {"v_latents": torch.rand(batch_size, domain_dim)}
#     attention_scores = attention(single_domain_input)

#     expected_scores = torch.ones(batch_size, 1)
#     assert torch.allclose(
#         attention_scores["v_latents"], expected_scores
#     ), "Attention scores for single domain should be all 1s"


# def test_multiple_domains_sumis1():
#     domain_dim = 12
#     head_size = 5
#     batch_size = 2056
#     attention = KQFixedQSelection(domain_dim, head_size)
#     gw_state = torch.rand(batch_size, domain_dim)
#     attention.update_gw_state(gw_state)

#     multiple_domain_input = {
#         "v_latents": torch.rand(batch_size, domain_dim),
#         "attr": torch.rand(batch_size, domain_dim),
#     }
#     attention_scores = attention(multiple_domain_input)

#     scores_sum = sum(
#         attention_scores[domain].squeeze() for domain in multiple_domain_input.keys()
#     )
#     expected_sum = torch.ones(batch_size)

#     assert torch.allclose(
#         scores_sum, expected_sum
#     ), "Sum of attention scores across domains should be 1"


# def test_attention_backward():
#     domain_dim = 12
#     head_size = 6
#     batch_size = 2056

#     attention = KQFixedQSelection(domain_dim, head_size)
#     gw_state = torch.rand(batch_size, domain_dim, requires_grad=True)
#     attention.update_gw_state(gw_state)

#     domains = {
#         "v_latents": torch.rand(batch_size, domain_dim, requires_grad=True),
#         "attr": torch.rand(batch_size, domain_dim, requires_grad=True),
#     }

#     attention_scores = attention(domains)
#     loss = sum(score.mean() for score in attention_scores.values())
#     loss.backward()

#     assert gw_state.grad is not None, "Gradients should be computed for gw_state"
#     for domain, tensor in domains.items():
#         assert (
#             tensor.grad is not None
#         ), f"Gradients should be computed for domain '{domain}' inputs"

#     for name, param in attention.named_parameters():
#         assert (
#             param.grad is not None
#         ), f"Gradients should be computed for parameter '{name}'"
