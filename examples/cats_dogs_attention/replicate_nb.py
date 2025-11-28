#!/usr/bin/env python3
# train_cats_dogs_gw_basic.py
# ------------------------------------------------------------
# Minimal GlobalWorkspaceFusion training on cats & dogs BLIP latents.
#
# Pipeline:
#   • Uses dataset_cats_dogs_blip_aug.py (our cats & dogs datamodule).
#   • Two-modality training (image_latents, caption_embeddings) — no labels.
#   • Assumes embeddings were produced by make_cats_dogs_captions_vlm.py
#     and saved under the default CATS_DOGS_BLIP_DIR.
#   • No W&B; local CSV logs + checkpoints only.
#
# After training, best and last checkpoints are copied to:
#   checkpoints_cats_dogs_gw/<RUN_NAME>__best.ckpt
#   checkpoints_cats_dogs_gw/<RUN_NAME>__last.ckpt
# ------------------------------------------------------------

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

# ---- GW bits (your codebase) ----
from shimmer import BroadcastLossCoefs
from shimmer.modules.global_workspace import GlobalWorkspaceFusion
from domains import BasicImageDomain, TextDomain

# ---- Our cats & dogs datamodule ----
from dataset import make_datamodule_cats_dogs_blip


# ============================================================
# Fixed config (no CLI)
# ============================================================

SEED       = 42
VIEW_MODE  = "all"      # "clean", "mean", "all", or "view_k"
BATCH_SIZE = 512
MAX_STEPS  = 5_000
VAL_CHECK  = 0.2
PRECISION  = "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"

# Optim
LR           = 5e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP    = 0.0

# Workspace / stacks (small, safe defaults)
WORKSPACE_DIM = 512
N_LAYERS      = 2
HIDDEN_DIM    = 768
DROPOUT       = 0.1

# Logging / checkpoints (LOCAL ONLY)
BASE_LOG_DIR  = Path("./logs_cats_dogs_gw")
BASE_CKPT_DIR = Path("./checkpoints_cats_dogs_gw")

RUN_NAME = f"cats_dogs_gw_basic__{VIEW_MODE}__s{SEED}"
LOG_DIR  = BASE_LOG_DIR / RUN_NAME
CKPT_DIR = BASE_CKPT_DIR / RUN_NAME


# ============================================================
# Tiny MLP stacks for encoders/decoders
# ============================================================

class ResMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1  = nn.Linear(dim, hidden)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2  = nn.Linear(hidden, dim)

    def forward(self, x):
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        return x + h


def make_encoder(in_dim: int, workspace_dim: int, n_layers: int, hidden: int, dropout: float) -> nn.Sequential:
    layers = [nn.Linear(in_dim, workspace_dim), nn.GELU()]
    for _ in range(n_layers):
        layers.append(ResMLPBlock(workspace_dim, hidden, dropout))
    return nn.Sequential(*layers)


def make_decoder(out_dim: int, workspace_dim: int, n_layers: int, hidden: int, dropout: float) -> nn.Sequential:
    layers = []
    for _ in range(n_layers):
        layers.append(ResMLPBlock(workspace_dim, hidden, dropout))
    layers.append(nn.Linear(workspace_dim, out_dim))
    return nn.Sequential(*layers)


# ============================================================
# Main
# ============================================================

def main():
    # Repro
    seed_everything(SEED, workers=True)

    # Datamodule (cats & dogs, BLIP latents)
    dm = make_datamodule_cats_dogs_blip(
        batch_size=BATCH_SIZE,
        num_workers=9,
        pin_memory=True,
        normalize="none",
        view_mode=VIEW_MODE,
        view_k=0,        # ignored unless VIEW_MODE=="view_k"
    )

    # Shapes (from joint image+caption domain)
    train_pair = dm.train_datasets[frozenset(["image_latents", "caption_embeddings"])]
    img_dim = int(train_pair.domain_data["image_latents"].shape[1])
    txt_dim = int(train_pair.domain_data["caption_embeddings"].shape[1])

    print(f"[dims] image_latents dim      = {img_dim}")
    print(f"[dims] caption_embeddings dim = {txt_dim}")

    # Domain modules (2-mod)
    domain_mods: Dict[str, nn.Module] = {
        "image_latents":      BasicImageDomain(latent_dim=img_dim),
        "caption_embeddings": TextDomain(latent_dim=txt_dim),
    }

    # Encoders / Decoders
    gw_encoders = {
        "image_latents":      make_encoder(img_dim, WORKSPACE_DIM, N_LAYERS, HIDDEN_DIM, DROPOUT),
        "caption_embeddings": make_encoder(txt_dim, WORKSPACE_DIM, N_LAYERS, HIDDEN_DIM, DROPOUT),
    }
    gw_decoders = {
        "image_latents":      make_decoder(img_dim, WORKSPACE_DIM, N_LAYERS, HIDDEN_DIM, DROPOUT),
        "caption_embeddings": make_decoder(txt_dim, WORKSPACE_DIM, N_LAYERS, HIDDEN_DIM, DROPOUT),
    }

    # Loss mix (2-mod friendly)
    loss_coefs = BroadcastLossCoefs(
        translations=1.0,   # X→Y, Y→X
        demi_cycles=0.5,    # X→Y→X, Y→X→Y (one side)
        cycles=0.5,         # full cycles (if defined)
        contrastives=0.05,  # CL-like
    )

    # Scheduler: cosine with short warmup (simple & robust)
    def sched_cosine_warmup(optimizer, total_steps: int, warmup_steps: int = 1_000):
        def lr_lambda(step: int):
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    scheduler_fn = lambda optim: sched_cosine_warmup(
        optim,
        total_steps=MAX_STEPS,
        warmup_steps=min(1_000, MAX_STEPS // 5),
    )

    # Model
    model = GlobalWorkspaceFusion(
        domain_mods=domain_mods,
        gw_encoders=gw_encoders,
        gw_decoders=gw_decoders,
        workspace_dim=WORKSPACE_DIM,
        loss_coefs=loss_coefs,
        optim_lr=LR,
        optim_weight_decay=WEIGHT_DECAY,
        scheduler=scheduler_fn,
        scheduler_args=None,
    )

    # Loggers / checkpoints (LOCAL ONLY, no W&B)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    csv_logger = CSVLogger(
        save_dir=str(LOG_DIR),
        name=RUN_NAME,
        flush_logs_every_n_steps=50,
    )

    lr_cb = LearningRateMonitor(logging_interval="step")
    ckpt_cb = ModelCheckpoint(
        dirpath=str(CKPT_DIR),
        filename=RUN_NAME + "-{epoch:02d}-{step:06d}",
        save_last=True,
        save_top_k=3,
        monitor="val/loss",
        mode="min",
    )

    # Trainer
    trainer = Trainer(
        logger=[csv_logger],
        callbacks=[lr_cb, ckpt_cb],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_steps=MAX_STEPS,
        log_every_n_steps=50,
        val_check_interval=VAL_CHECK,
        num_sanity_val_steps=0,
        detect_anomaly=False,
        gradient_clip_val=GRAD_CLIP,
        precision=PRECISION,
        enable_checkpointing=True,
    )

    print(f"\n[Cats & Dogs] Training GlobalWorkspaceFusion (image ↔ caption) | RUN={RUN_NAME}")
    trainer.fit(model=model, datamodule=dm)

    # Final checkpoint copies (best + last) with stable filenames
    best_src = Path(ckpt_cb.best_model_path) if ckpt_cb.best_model_path else None
    last_src = CKPT_DIR / "last.ckpt"

    best_dst = CKPT_DIR / f"{RUN_NAME}__best.ckpt"
    last_dst = CKPT_DIR / f"{RUN_NAME}__last.ckpt"

    try:
        if best_src is not None and best_src.exists():
            shutil.copy2(best_src, best_dst)
        if last_src.exists():
            shutil.copy2(last_src, last_dst)
    except Exception as e:
        print(f"[WARN] Could not finalize checkpoint copies: {e}")

    print("\n✅ Done.")
    print(f"   Logs dir : {LOG_DIR.resolve()}")
    print(f"   Ckpt dir : {CKPT_DIR.resolve()}")
    if best_dst.exists():
        print(f"   Best ckpt: {best_dst.resolve()}")
    else:
        print("   Best ckpt: (not found — did val run / monitor key match?)")
    if last_dst.exists():
        print(f"   Last ckpt: {last_dst.resolve()}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
