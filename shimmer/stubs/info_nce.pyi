from typing import Literal

import torch

reductionLiteral = Literal["mean", "sum", "none"]
negativeModeLiteral = Literal["paired", "unpaired"]

def info_nce(
    query: torch.Tensor,
    positive_key: torch.Tensor,
    negative_keys: torch.Tensor | None = None,
    temperature: float = 0.1,
    reduction: reductionLiteral = "mean",
    negative_mode: negativeModeLiteral = "unpaired",
) -> torch.Tensor: ...
