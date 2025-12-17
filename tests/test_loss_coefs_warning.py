import warnings

import torch

from shimmer.modules.losses import combine_loss


def test_missing_loss_coef_warns_and_defaults_to_zero() -> None:
    metrics = {
        "demi_cycles": torch.tensor(1.0),
        "contrastives": torch.tensor(2.0),
    }
    coefs = {"contrastives": 1.0}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loss = combine_loss(metrics, coefs)

    assert any("demi_cycles" in str(w.message) for w in caught)
    assert torch.isclose(loss, torch.tensor(2.0))
