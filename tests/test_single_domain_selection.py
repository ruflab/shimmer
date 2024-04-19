import torch

from shimmer.modules.selection import SingleDomainSelection


def test_selection_1_domain():
    selection_mod = SingleDomainSelection()

    bs = 32
    domains = {"v": torch.randn(bs, 8)}
    prefusion_encodings = torch.randn(1, bs, 8)
    selection = selection_mod(domains, prefusion_encodings)

    assert len(selection) == len(domains)
    target = torch.ones(bs)
    assert torch.allclose(selection[0], target), "Everything should be selected."


def test_selection_2_domains():
    selection_mod = SingleDomainSelection()

    bs = 32
    domains = {"v": torch.randn(bs, 8), "t": torch.randn(bs, 12)}
    prefusion_encodings = torch.randn(2, bs, 8)

    selection = selection_mod(domains, prefusion_encodings)

    assert len(selection) == len(domains)
    target = torch.ones(bs)
    assert torch.allclose(
        selection.sum(dim=0), target
    ), "Everything should be selected once and only once."


def test_selection_3_domains():
    selection_mod = SingleDomainSelection()

    bs = 32
    domains = {
        "v": torch.randn(bs, 8),
        "t": torch.randn(bs, 12),
        "attr": torch.randn(bs, 4),
    }
    prefusion_encodings = torch.randn(3, bs, 8)

    selection = selection_mod(domains, prefusion_encodings)

    assert len(selection) == len(domains)
    target = torch.ones(bs)
    assert torch.allclose(
        selection.sum(dim=0), target
    ), "Everything should be selected once and only once."
