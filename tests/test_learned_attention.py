import pytest
import torch
import torch.nn as nn

from shimmer.modules.domain import DomainModule
from shimmer.modules.global_workspace import (
    GlobalWorkspaceFusion,
    freeze_domain_modules,
)
from shimmer.modules.selection import LearnedAttention


def _make_latents(batch_size: int, dim: int) -> dict[str, torch.Tensor]:
    return {
        "a": torch.randn(batch_size, dim, requires_grad=True),
        "b": torch.randn(batch_size, dim, requires_grad=True),
    }


def test_learned_attention_probs_sum_to_one() -> None:
    selector = LearnedAttention(gw_dim=4, domain_names=["a", "b"], head_size=2)
    latents = _make_latents(batch_size=8, dim=4)

    weights = selector(latents, encodings_pre_fusion=latents)

    for domain in ["a", "b"]:
        assert weights[domain].shape == (8,)

    stacked = torch.stack([weights["a"], weights["b"]], dim=1)
    assert torch.allclose(stacked.sum(dim=1), torch.ones(8))


def test_learned_attention_stopgrad_toggle() -> None:
    base_latents = _make_latents(batch_size=4, dim=6)

    frozen_latents = {
        k: v.detach().clone().requires_grad_(True) for k, v in base_latents.items()
    }
    frozen_selector = LearnedAttention(
        gw_dim=6, domain_names=["a", "b"], head_size=3, stopgrad=True
    )
    frozen_weights = frozen_selector(
        frozen_latents, encodings_pre_fusion=frozen_latents
    )
    torch.stack(list(frozen_weights.values())).sum().backward()
    assert frozen_latents["a"].grad is None
    assert frozen_latents["b"].grad is None

    train_latents = {
        k: v.detach().clone().requires_grad_(True) for k, v in base_latents.items()
    }
    trainable_selector = LearnedAttention(
        gw_dim=6, domain_names=["a", "b"], head_size=3, stopgrad=False
    )
    trainable_weights = trainable_selector(
        train_latents, encodings_pre_fusion=train_latents
    )
    torch.stack(list(trainable_weights.values())).sum().backward()
    assert train_latents["a"].grad is not None
    assert train_latents["b"].grad is not None


def test_learned_attention_domain_key_path() -> None:
    domain_dims = {"a": 3, "b": 5}
    selector = LearnedAttention(
        gw_dim=4,
        domain_names=domain_dims.keys(),
        head_size=3,
        per_domain_keys=True,
        stopgrad=False,
        key_on_prefusion=False,
        domain_dims=domain_dims,
    )

    domain_latents = {
        "a": torch.randn(6, 3, requires_grad=True),
        "b": torch.randn(6, 5, requires_grad=True),
    }
    prefusion_latents = {
        "a": torch.randn(6, 4, requires_grad=True),
        "b": torch.randn(6, 4, requires_grad=True),
    }

    weights = selector(domain_latents, encodings_pre_fusion=prefusion_latents)

    stacked = torch.stack([weights["a"], weights["b"]], dim=1)
    assert torch.allclose(stacked.sum(dim=1), torch.ones(6))


def test_learned_attention_domain_key_shared_layer_error() -> None:
    domain_dims = {"a": 3, "b": 5}
    with pytest.raises(ValueError):
        LearnedAttention(
            gw_dim=4,
            domain_names=domain_dims.keys(),
            head_size=3,
            per_domain_keys=False,
            stopgrad=True,
            key_on_prefusion=False,
            domain_dims=domain_dims,
        )


class _DummyDomain(DomainModule):
    def __init__(self, latent_dim: int):
        super().__init__(latent_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple stub
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple stub
        return z


def test_global_workspace_init_learned_attention_domain_dims() -> None:
    domain_mods = freeze_domain_modules({"a": _DummyDomain(3), "b": _DummyDomain(5)})
    gw_encoders = {"a": nn.Identity(), "b": nn.Identity()}
    gw_decoders = {"a": nn.Identity(), "b": nn.Identity()}

    gw = GlobalWorkspaceFusion(
        domain_mods=domain_mods,
        gw_encoders=gw_encoders,
        gw_decoders=gw_decoders,
        workspace_dim=4,
        loss_coefs={"contrastives": 0.0},
    )

    selector = gw.init_learned_attention(
        head_size=2,
        per_domain_keys=True,
        stopgrad=False,
        key_on_prefusion=False,
    )

    assert selector.key_on_prefusion is False
    assert selector.per_key_layers is not None
    assert selector.per_key_layers["a"].weight.shape[1] == 3
    assert selector.per_key_layers["b"].weight.shape[1] == 5


def test_global_workspace_init_learned_attention_shared_error() -> None:
    domain_mods = freeze_domain_modules({"a": _DummyDomain(3), "b": _DummyDomain(5)})
    gw_encoders = {"a": nn.Identity(), "b": nn.Identity()}
    gw_decoders = {"a": nn.Identity(), "b": nn.Identity()}

    gw = GlobalWorkspaceFusion(
        domain_mods=domain_mods,
        gw_encoders=gw_encoders,
        gw_decoders=gw_decoders,
        workspace_dim=4,
        loss_coefs={"contrastives": 0.0},
    )

    with pytest.raises(ValueError):
        gw.init_learned_attention(
            head_size=2,
            per_domain_keys=False,
            stopgrad=True,
            key_on_prefusion=False,
        )
