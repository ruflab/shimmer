import torch

from shimmer.modules.selection import SingleDomainSelection


def test_selection_1_domain():
    selection_mod = SingleDomainSelection()

    bs = 32
    domains = {"v": torch.randn(bs, 8)}
    prefusion_encodings = {"v": torch.randn(bs, 8)}

    selection: dict[str, torch.Tensor] = selection_mod(domains, prefusion_encodings)

    assert len(selection) == len(domains)
    assert next(iter(selection.keys())) == "v"
    assert (selection["v"] == 0).sum() == 0, "Everything should be selected."


def test_selection_2_domains():
    selection_mod = SingleDomainSelection()

    bs = 32
    domains = {"v": torch.randn(bs, 8), "t": torch.randn(bs, 12)}
    prefusion_encodings = {"v": torch.randn(bs, 8), "t": torch.randn(bs, 12)}

    selection: dict[str, torch.Tensor] = selection_mod(domains, prefusion_encodings)

    assert len(selection) == len(domains)
    assert (
        (selection["v"] + selection["t"]) == 1
    ).sum() == bs, "Everything should be selected once and only once."


def test_selection_3_domains():
    selection_mod = SingleDomainSelection()

    bs = 32
    domains = {
        "v": torch.randn(bs, 8),
        "t": torch.randn(bs, 12),
        "attr": torch.randn(bs, 4),
    }
    prefusion_encodings = {
        "v": torch.randn(bs, 8),
        "t": torch.randn(bs, 12),
        "attr": torch.randn(bs, 4),
    }

    selection: dict[str, torch.Tensor] = selection_mod(domains, prefusion_encodings)

    assert len(selection) == len(domains)
    assert (
        (selection["v"] + selection["t"] + selection["attr"]) == 1
    ).sum() == bs, "Everything should be selected once and only once."
