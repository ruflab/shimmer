import os
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import AdamW
import json
import numpy as np

from shimmer import DomainModule, LossOutput



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import AdamW
from dataclasses import dataclass, field
from typing import Dict, Tuple, Any



class BasicImageDomain(DomainModule):
    def __init__(self, latent_dim: int):
        super().__init__(latent_dim)
        # load the model parameters

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        self.eval()

        return z

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        loss = torch.tensor(0.).to(self.device)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        loss = torch.tensor(0.).to(self.device)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=1e-3, weight_decay=1e-6)

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor, raw_target: torch.Tensor) -> LossOutput:
        # Computes an illustrative loss, can be tailored for specific use cases
        return LossOutput(loss=F.mse_loss(pred, target))



import sys
import torch
import torch.nn.functional as F
from torch.optim import AdamW

# Assume DomainModule and LossOutput are available in your project.
# from shimmer.domain_base import DomainModule, LossOutput

class TextDomain(DomainModule):
    def __init__(self, latent_dim: int):
        super().__init__(latent_dim)
        self._loss_debug_printed = False  # print once, then exit

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Just pass through the embedding
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # Text embeddings do not decode back to text, return input as placeholder
        return z

    # ---------- lightning hooks ----------
    def training_step(self, batch: torch.Tensor, batch_idx: int):
        (domain,) = batch
        # This is a self-comparison; expected to be exactly zero.
        loss = F.mse_loss(domain, domain)
        # Print diagnostics on first ever loss call, then exit
        if not self._loss_debug_printed:
            self._print_and_exit(where="training_step (MSE(domain, domain))")
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        (domain,) = batch
        loss = F.mse_loss(domain, domain)
        if not self._loss_debug_printed:
            self._print_and_exit(where="validation_step (MSE(domain, domain))")
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=1e-3, weight_decay=1e-6)

    # ---------- used by GW reconstruction paths ----------
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor, raw_target: torch.Tensor) -> "LossOutput":
        return LossOutput(loss=F.mse_loss(pred, target))



class LabelDomain(DomainModule):
    """
    Multi-label (23D). Decoder outputs logits.
    Loss = BCE-with-logits against multi-hot targets (optionally hardened).
    """
    def __init__(self, num_classes: int, pos_weight: torch.Tensor | None = None, harden_targets: bool = False):
        super().__init__(num_classes)
        self.harden_targets = harden_targets
        self.register_buffer("pos_weight", None, persistent=False)
        if pos_weight is not None:
            if pos_weight.ndim != 1 or pos_weight.shape[0] != num_classes:
                raise ValueError(f"pos_weight must be 1D of length {num_classes}")
            self.pos_weight = pos_weight.to(dtype=torch.float32)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # labels come in as [0,1] multi-hot
        return x.to(torch.float32)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # raw logits; do NOT apply sigmoid here (the loss does it stably)
        return z

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor, raw_target: torch.Tensor):
        # Optionally harden; otherwise use the provided multi-hot directly
        tgt = (target >= 0.5).to(pred.dtype) if self.harden_targets else target.to(pred.dtype).clamp_(0, 1)
        loss = F.binary_cross_entropy_with_logits(pred, tgt, pos_weight=self.pos_weight, reduction="mean")
        return LossOutput(loss=loss)


class CleanTargetDomain(DomainModule):
    """
    Target-only domain for *_clean modalities.
    - encode/decode: identity
    - training/validation steps: return zero loss (logged once)
    - compute_loss: forbidden → raises RuntimeError
    """
    def __init__(self, latent_dim: int, name: str = "clean"):
        super().__init__(latent_dim)
        self.name = name
        self._warned_once = False

    # -------- enc/dec (pass-through) --------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z

    # -------- lightning hooks (no-op with diagnostics) --------
    def _note_once(self, where: str) -> None:
        if not self._warned_once:
            print(f"[CleanTargetDomain:{self.name}] {where}: returning zero loss; this domain is target-only.")
            self._warned_once = True

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        # Expect batch like (domain_tensor,)
        (domain,) = batch
        loss = torch.zeros([], device=domain.device, dtype=domain.dtype)
        self._note_once("training_step")
        try:
            self.log("train_loss", loss)
        except Exception:
            pass
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        (domain,) = batch
        loss = torch.zeros([], device=domain.device, dtype=domain.dtype)
        self._note_once("validation_step")
        try:
            self.log("val_loss", loss)
        except Exception:
            pass
        return loss

    def configure_optimizers(self):
        # No parameters/optimizer needed for a target-only, identity module
        return []

    # -------- used by GW reconstruction paths --------
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor, raw_target: torch.Tensor) -> "LossOutput":
        raise RuntimeError(
            f"CleanTargetDomain({self.name}) should never have compute_loss() called. "
            "It is a target-only modality (used for supervision, not prediction)."
        )