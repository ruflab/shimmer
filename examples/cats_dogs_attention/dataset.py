#!/usr/bin/env python3
# dataset_cats_dogs_blip_aug.py — cats & dogs BLIP latents datamodule (raw+augs, NO labels)
#
# - Uses embeddings produced by: make_cats_dogs_captions_vlm.py
# - Image/text embeddings come from the same BLIP checkpoint (image-caption model).
# - Augmented arrays have shape (N, V, D) where:
#       N = number of original samples (≈1000)
#       V = 1 + K views (view 0 = raw, views 1..K = augmentations)
#
# View handling:
#   • "clean"  → use view 0 only    (matches original samples)
#   • "mean"   → average over V views per original (N×D tensors)
#   • "all"    → expand each original to all V views (N*V×D tensors)
#   • "view_k" → use a specific view index k (0 ≤ k < V)
#
# Expected files (under CATS_DOGS_BLIP_DIR):
#   cats_dogs_blip_image_embeds_augmented.npy  (N x V x D)
#   cats_dogs_blip_text_embeds_augmented.npy   (N x V x D)
#   captions_from_vlm.json                     (metadata, same order as embeddings)

from __future__ import annotations
from typing import Dict, Tuple, Literal, Optional, Iterable
from pathlib import Path
import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

# Root directory for the cats & dogs BLIP embeddings
CATS_DOGS_DIR = Path(os.getenv(
    "CATS_DOGS_BLIP_DIR",
    "/home/rbertin/attention/update_shimmer/shimmer/examples/cats_dogs_attention/data/cats_dogs",
)).resolve()

DEFAULT_IMAGE_LATENTS = CATS_DOGS_DIR / "cats_dogs_blip_image_embeds_augmented.npy"
DEFAULT_TEXT_EMBEDS   = CATS_DOGS_DIR / "cats_dogs_blip_text_embeds_augmented.npy"
JSON_METADATA_PATH    = CATS_DOGS_DIR / "captions_from_vlm.json"

# Allow env overrides (similar style to MM-IMDB code)
IMAGE_LATENTS_PATH      = os.getenv("CATS_DOGS_IMAGE_LATENTS", str(DEFAULT_IMAGE_LATENTS))
CAPTION_EMBEDDINGS_PATH = os.getenv("CATS_DOGS_TEXT_EMBEDS",   str(DEFAULT_TEXT_EMBEDS))

# ──────────────────────────────────────────────────────────────────────────────
# Utils
# ──────────────────────────────────────────────────────────────────────────────

def load_npy_as_float32_tensor(file_path: str | Path) -> torch.Tensor:
    """Load .npy → contiguous float32 torch tensor (supports 2D or 3D)."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {file_path}")
    print(f"[load] Loading array from {file_path} ...")
    arr = np.load(str(file_path))
    print(f"[load] Loaded array with shape {arr.shape}, dtype={arr.dtype}")
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=True)
    else:
        arr = np.array(arr, dtype=np.float32, copy=True)
    return torch.from_numpy(arr)

def _build_random_split(
    N: int,
    train_ratio: float = 0.8,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a deterministic random train/val split over N originals.
    Returns (idx_train, idx_val) as np.int64 arrays.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(N, dtype=np.int64)
    rng.shuffle(idx)
    n_train = int(round(train_ratio * N))
    idx_train = np.sort(idx[:n_train])
    idx_val   = np.sort(idx[n_train:])
    print(f"[split] Random split over N={N}: train={idx_train.size}, val={idx_val.size}")
    return idx_train, idx_val

def _select_views_for_split(
    arr: torch.Tensor,
    idx_tr: torch.LongTensor,
    idx_va: torch.LongTensor,
    view_mode: Literal["clean", "mean", "all", "view_k"] = "clean",
    view_k: int = 0,
    *,
    name: str = "modality",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given arr (N, V, D) or (N, D), original indices idx_tr/idx_va (over N),
    and a view_mode, return (train_tensor, val_tensor) with shape:
      - "clean"   : (N_tr, D), view 0
      - "mean"    : (N_tr, D), mean over V
      - "all"     : (N_tr * V, D), flattened
      - "view_k"  : (N_tr, D), specified view
    """
    if arr.ndim == 2:
        # Treat as N x 1 x D (only raw view)
        arr = arr.unsqueeze(1)  # (N, 1, D)
    if arr.ndim != 3:
        raise ValueError(f"Expected {name} embeddings with ndim=2 or 3, got ndim={arr.ndim} and shape={tuple(arr.shape)}")

    N, V, D = arr.shape
    if V <= 0:
        raise ValueError(f"{name} embeddings have V={V} views, expected >= 1")

    # Ensure indices are Torch Long on CPU
    idx_tr = idx_tr.to(dtype=torch.long, device=arr.device)
    idx_va = idx_va.to(dtype=torch.long, device=arr.device)

    if view_mode == "mean":
        arr_tr = arr.index_select(0, idx_tr).mean(dim=1)  # (N_tr, D)
        arr_va = arr.index_select(0, idx_va).mean(dim=1)
        return arr_tr, arr_va

    elif view_mode == "clean":
        k = 0
        arr_tr = arr.index_select(0, idx_tr)[:, k, :]  # (N_tr, D)
        arr_va = arr.index_select(0, idx_va)[:, k, :]
        return arr_tr, arr_va

    elif view_mode == "view_k":
        if not (0 <= view_k < V):
            raise ValueError(f"{name}: view_k={view_k} out of range [0, {V})")
        k = int(view_k)
        arr_tr = arr.index_select(0, idx_tr)[:, k, :]
        arr_va = arr.index_select(0, idx_va)[:, k, :]
        return arr_tr, arr_va

    elif view_mode == "all":
        arr_tr = arr.index_select(0, idx_tr)     # (N_tr, V, D)
        arr_va = arr.index_select(0, idx_va)     # (N_val, V, D)
        arr_tr = arr_tr.reshape(-1, D)           # (N_tr*V, D)
        arr_va = arr_va.reshape(-1, D)
        return arr_tr, arr_va

    else:
        raise ValueError(f"Unsupported view_mode={view_mode!r} for {name} embeddings")

# ──────────────────────────────────────────────────────────────────────────────
# Dataset wrappers
# ──────────────────────────────────────────────────────────────────────────────

class DomainDataset(Dataset):
    """
    Returns a dict[str, Tensor] per item, e.g.:
      - {'image_latents': x}
      - {'caption_embeddings': y}
      - {'image_latents': x, 'caption_embeddings': y}
    """
    def __init__(self, domain_data: Dict[str, torch.Tensor]) -> None:
        self.domain_data = domain_data
        lens = [t.shape[0] for t in domain_data.values()]
        if len(set(lens)) != 1:
            raise ValueError(f"Domain tensors have mismatched lengths: {lens}")
        self._len = lens[0]

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {name: data[index] for name, data in self.domain_data.items()}

class GWDataModule(LightningDataModule):
    """
    Simple multimodal datamodule with:
      - train_datasets: mapping from frozenset(domain_names) → DomainDataset
      - val_datasets  : same, for validation
    Uses CombinedLoader(mode="min_size") like your MM-IMDB pipeline.
    """
    def __init__(
        self,
        val_datasets: Dict[frozenset[str], DomainDataset],
        train_datasets: Dict[frozenset[str], DomainDataset],
        batch_size: int,
        num_workers: int = 9,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.val_datasets = val_datasets
        self.train_datasets = train_datasets
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup_dataloaders(self, datasets: Dict[frozenset[str], DomainDataset], *, shuffle: bool):
        dls: Dict[frozenset[str], DataLoader] = {}
        for domain_set, dataset in datasets.items():
            dls[domain_set] = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        return dls

    def train_dataloader(self):
        train_loaders = self.setup_dataloaders(self.train_datasets, shuffle=True)
        return CombinedLoader(train_loaders, mode="min_size")

    def val_dataloader(self):
        val_loaders = self.setup_dataloaders(self.val_datasets, shuffle=False)
        return CombinedLoader(val_loaders, mode="min_size")

    def get_samples(self, split: Literal["train", "val"], amount: int) -> Dict[frozenset, Dict[str, torch.Tensor]]:
        loader = self.train_dataloader() if split == "train" else self.val_dataloader()
        batch = next(iter(loader))
        out: Dict[frozenset, Dict[str, torch.Tensor]] = {}
        for dom_set, tensors in batch.items():
            out[dom_set] = {k: v[:amount] for k, v in tensors.items()}
        return out

# ──────────────────────────────────────────────────────────────────────────────
# Builders (cats & dogs BLIP, NO labels)
# ──────────────────────────────────────────────────────────────────────────────

def make_datasets(
    image_latents_path: str | Path = IMAGE_LATENTS_PATH,
    caption_embeddings_path: str | Path = CAPTION_EMBEDDINGS_PATH,
    *,
    extra_image_latents: Optional[Iterable[str] | str] = None,      # kept for signature compatibility (ignored)
    extra_caption_embeddings: Optional[Iterable[str] | str] = None, # kept for signature compatibility (ignored)
    normalize: Literal["none", "train_split"] = "none",
    view_mode: Literal["clean", "mean", "all", "view_k"] = "clean",
    view_k: int = 0,
    manifest_path: Optional[str | Path] = None,  # unused, kept for signature compatibility
) -> Tuple[Dict[frozenset[str], DomainDataset], Dict[frozenset[str], DomainDataset]]:
    """
    Load cats & dogs BLIP arrays (augmented views, already L2-normalized in extractor),
    build a deterministic random train/val split, and select/aggregate views identically
    across both modalities.

    - Expects arrays with shape (N, V, D) where V = 1 + K (raw + augmentations).
    - view_mode controls how to handle these views:
        'clean'  → view 0 (raw only)
        'mean'   → average over all V views
        'all'    → flatten all views
        'view_k' → arbitrary k in [0, V)
    """

    if extra_image_latents is not None or extra_caption_embeddings is not None:
        raise NotImplementedError("extra_image_latents/extra_caption_embeddings not supported for cats & dogs datamodule.")

    # Required files
    for p in [image_latents_path, caption_embeddings_path]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Required embedding path not found: {p}")
    if not JSON_METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata JSON not found: {JSON_METADATA_PATH}")

    # Load embeddings
    image_all = load_npy_as_float32_tensor(image_latents_path)      # (N, V, D_img)
    text_all  = load_npy_as_float32_tensor(caption_embeddings_path) # (N, V, D_txt)

    if image_all.ndim == 2:
        image_all = image_all.unsqueeze(1)
    if text_all.ndim == 2:
        text_all = text_all.unsqueeze(1)

    if image_all.ndim != 3 or text_all.ndim != 3:
        raise ValueError(
            f"Expected 3D embeddings (N, V, D). "
            f"Got image={tuple(image_all.shape)}, text={tuple(text_all.shape)}"
        )

    N_img, V_img, D_img = image_all.shape
    N_txt, V_txt, D_txt = text_all.shape
    if N_img != N_txt or V_img != V_txt:
        raise ValueError(
            f"Image/text embeddings mismatch: "
            f"image={tuple(image_all.shape)}, text={tuple(text_all.shape)}"
        )
    N_orig, V = N_img, V_img

    print(f"[cats_dogs] Loaded arrays:")
    print(f"  image_all: N={N_orig}, V={V}, D={D_img}")
    print(f"  text_all : N={N_orig}, V={V}, D={D_txt}")

    # Check against JSON metadata length (sanity check)
    meta = json.loads(JSON_METADATA_PATH.read_text(encoding="utf-8"))
    if len(meta) != N_orig:
        raise ValueError(
            f"Metadata length mismatch: len(captions_from_vlm.json)={len(meta)} vs N_orig={N_orig}"
        )

    # Build deterministic random split over originals
    idx_tr_np, idx_va_np = _build_random_split(N_orig, train_ratio=0.8, seed=0)
    idx_tr = torch.from_numpy(idx_tr_np).long()
    idx_va = torch.from_numpy(idx_va_np).long()

    # Select / aggregate views
    img_tr, img_val = _select_views_for_split(
        image_all, idx_tr, idx_va, view_mode=view_mode, view_k=view_k, name="image"
    )
    txt_tr, txt_val = _select_views_for_split(
        text_all, idx_tr, idx_va, view_mode=view_mode, view_k=view_k, name="text"
    )

    print(f"[cats_dogs] Split ({view_mode}):")
    print(f"  train image_latents  : {tuple(img_tr.shape)}")
    print(f"  train caption_embeds : {tuple(txt_tr.shape)}")
    print(f"  val   image_latents  : {tuple(img_val.shape)}")
    print(f"  val   caption_embeds : {tuple(txt_val.shape)}")

    # Optional re-normalization (default: none)
    if normalize == "train_split":
        img_mean, img_std = img_tr.mean(dim=0, keepdim=True), img_tr.std(dim=0, keepdim=True)
        txt_mean, txt_std = txt_tr.mean(dim=0, keepdim=True), txt_tr.std(dim=0, keepdim=True)
        img_std[img_std == 0] = 1e-6
        txt_std[txt_std == 0] = 1e-6
        img_tr = (img_tr - img_mean) / img_std
        img_val = (img_val - img_mean) / img_std
        txt_tr = (txt_tr - txt_mean) / txt_std
        txt_val = (txt_val - txt_mean) / txt_std
        print("Applied train-split normalization to both modalities.")
    else:
        print("No extra normalization (using extractor’s L2-normalized features).")

    # Domain datasets (same pattern as MM-IMDB nolabels)
    train_datasets = {
        frozenset(["image_latents"]): DomainDataset({"image_latents": img_tr}),
        frozenset(["caption_embeddings"]): DomainDataset({"caption_embeddings": txt_tr}),
        frozenset(["image_latents", "caption_embeddings"]): DomainDataset({
            "image_latents": img_tr,
            "caption_embeddings": txt_tr,
        }),
    }
    val_datasets = {
        frozenset(["image_latents", "caption_embeddings"]): DomainDataset({
            "image_latents": img_val,
            "caption_embeddings": txt_val,
        }),
    }
    return train_datasets, val_datasets

def make_datamodule(
    batch_size: int = 2048,
    image_latents_path: str | Path = IMAGE_LATENTS_PATH,
    caption_embeddings_path: str | Path = CAPTION_EMBEDDINGS_PATH,
    *,
    extra_image_latents: Optional[Iterable[str] | str] = None,
    extra_caption_embeddings: Optional[Iterable[str] | str] = None,
    num_workers: int = 9,
    pin_memory: bool = True,
    normalize: Literal["none", "train_split"] = "none",
    view_mode: Literal["clean", "mean", "all", "view_k"] = "clean",
    view_k: int = 0,
    manifest_path: Optional[str | Path] = None,
) -> GWDataModule:
    """
    Build a cats & dogs BLIP datamodule with the same interface
    as the MM-IMDB aug70 nolabels datamodule.
    """
    train_datasets, val_datasets = make_datasets(
        image_latents_path=image_latents_path,
        caption_embeddings_path=caption_embeddings_path,
        extra_image_latents=extra_image_latents,
        extra_caption_embeddings=extra_caption_embeddings,
        normalize=normalize,
        view_mode=view_mode,
        view_k=view_k,
        manifest_path=manifest_path,
    )
    dm = GWDataModule(
        train_datasets=train_datasets,
        val_datasets=val_datasets,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return dm

# ──────────────────────────────────────────────────────────────────────────────
# Convenience wrapper
# ──────────────────────────────────────────────────────────────────────────────

def make_datamodule_cats_dogs_blip(**kwargs) -> GWDataModule:
    """
    Convenience entry point: BLIP image+caption pair on cats & dogs embeddings.
    """
    return make_datamodule(
        image_latents_path=str(DEFAULT_IMAGE_LATENTS),
        caption_embeddings_path=str(DEFAULT_TEXT_EMBEDS),
        **kwargs,
    )
