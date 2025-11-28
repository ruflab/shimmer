#!/usr/bin/env python3
"""
Build/reuse a tiny cats & dogs dataset, caption 1000 images with BLIP,
then embed images + captions using the same BLIP backbone, apply
augmentations/corruptions, and save everything to disk.

Outputs under:
  examples/cats_dogs_attention/data/cats_dogs/ :

  - captions_from_vlm.json
      (per-sample metadata: image_path, caption, times, augmentation types, raw index)
  - captions_from_vlm.csv
      (image_path, caption, caption_time_sec, augmentation summaries)

  Raw embeddings (no augmentation):
  - cats_dogs_blip_image_embeds_raw.npy     (N x D, float32)
  - cats_dogs_blip_text_embeds_raw.npy      (N x D, float32)

  Raw + augmented embeddings:
  - cats_dogs_blip_image_embeds_augmented.npy  (N x (1+K) x D, float32, K=5)
  - cats_dogs_blip_text_embeds_augmented.npy   (N x (1+K) x D, float32, K=5)
    where index 0 along axis=1 is ALWAYS the raw, non-augmented embedding.

Run from repo root:
    python examples/cats_dogs_attention/make_cats_dogs_captions_vlm.py
"""

from __future__ import annotations

import csv
import json
import random
import statistics
import time
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
from PIL import Image

import torchvision.transforms as T
from torchvision.datasets import CIFAR10

from tqdm.auto import tqdm

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipModel,
)

# ---------------------------------------------------------------------------
# 0. Global config
# ---------------------------------------------------------------------------

MODEL_NAME = "Salesforce/blip-image-captioning-base"
AUGMENTATIONS_PER_SAMPLE = 5  # K = 5
RAW_INDEX = 0                 # index of non-augmented embedding in second dim
IMG_EMB_NOISE_SIGMA = 0.1
TXT_EMB_NOISE_SIGMA = 0.1


# ---------------------------------------------------------------------------
# 1. Dataset prep: CIFAR-10 → cats/dogs dataset (if empty)
# ---------------------------------------------------------------------------

def prepare_cats_dogs_dataset(root: Path, num_per_class: int = 500) -> None:
    """
    Ensure a small cats/dogs dataset exists under `root`.

    - If images are already there, do nothing.
    - Otherwise, pull from CIFAR-10: label 3 (cat) and 5 (dog).
    """
    existing = list(root.rglob("*.png")) + list(root.rglob("*.jpg")) + list(
        root.rglob("*.jpeg")
    )
    if existing:
        print(
            f"[dataset] Found {len(existing)} existing images under {root}, "
            f"skipping CIFAR-10 extraction."
        )
        return

    print(f"[dataset] No images found under {root}, creating from CIFAR-10...")
    root.mkdir(parents=True, exist_ok=True)
    cats_dir = root / "cats"
    dogs_dir = root / "dogs"
    cats_dir.mkdir(parents=True, exist_ok=True)
    dogs_dir.mkdir(parents=True, exist_ok=True)

    cifar_root = root.parent / "cifar10_raw"
    transform = T.ToTensor()
    train_set = CIFAR10(root=str(cifar_root), train=True, download=True, transform=transform)

    CAT_LABEL = 3
    DOG_LABEL = 5
    counters = {CAT_LABEL: 0, DOG_LABEL: 0}

    indices = list(range(len(train_set)))
    random.shuffle(indices)

    for idx in indices:
        img, label = train_set[idx]
        if label not in (CAT_LABEL, DOG_LABEL):
            continue

        if all(counters[l] >= num_per_class for l in (CAT_LABEL, DOG_LABEL)):
            break
        if counters[label] >= num_per_class:
            continue

        pil_img = T.ToPILImage()(img)
        if label == CAT_LABEL:
            out_dir, prefix = cats_dir, "cat"
        else:
            out_dir, prefix = dogs_dir, "dog"

        fname = f"{prefix}_{counters[label]}.png"
        out_path = out_dir / fname
        pil_img.save(out_path)
        counters[label] += 1

    print(
        f"[dataset] Created {counters[CAT_LABEL]} cat and {counters[DOG_LABEL]} dog "
        f"images under {root}."
    )


# ---------------------------------------------------------------------------
# 2. Load BLIP VLM for captioning + embeddings
# ---------------------------------------------------------------------------

def load_caption_vlm(device: torch.device):
    """
    Load a BLIP image captioning model (for generating captions).
    """
    print(f"[vlm] Loading captioning model: {MODEL_NAME} ...")
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(device).eval()
    print("[vlm] Captioning model loaded.")
    return processor, model


def load_blip_embedder(device: torch.device) -> BlipModel:
    """
    Load the BLIP backbone for getting image/text embeddings.
    Same checkpoint as the captioning model.
    """
    print(f"[embed] Loading BlipModel (embeddings) from: {MODEL_NAME} ...")
    model = BlipModel.from_pretrained(MODEL_NAME).to(device).eval()
    print("[embed] BlipModel loaded.")
    return model


def caption_image(
    image: Image.Image,
    processor: BlipProcessor,
    model: BlipForConditionalGeneration,
    device: torch.device,
) -> str:
    """
    Get a caption from the BLIP captioning model for a single image.
    """
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=30,
            num_beams=5,
            do_sample=False,
        )
    caption = processor.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
    return caption


# ---------------------------------------------------------------------------
# 3. Augmentations / corruptions
# ---------------------------------------------------------------------------

# Image augmentations (classical)
IMG_AUG_TRANSFORMS = {
    "identity": lambda img: img,
    "hflip": T.functional.hflip,
    "color_jitter": T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    "gaussian_blur": T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    "random_affine": T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
}

IMG_AUG_NAMES = list(IMG_AUG_TRANSFORMS.keys())


def apply_image_augmentation(img: Image.Image, aug_name: str) -> Image.Image:
    t = IMG_AUG_TRANSFORMS[aug_name]
    # Some transforms are classes (callable) and some are functions
    return t(img)


# Text augmentations (simple masking/shuffling)
TXT_AUG_NAMES = [
    "identity",
    "mask_some_words",
    "drop_random_word",
    "shuffle_middle_words",
]


def mask_some_words(text: str, mask_token: str = "[MASK]", p: float = 0.3) -> str:
    words = text.split()
    if not words:
        return text
    new_words = []
    for w in words:
        if random.random() < p:
            new_words.append(mask_token)
        else:
            new_words.append(w)
    # Avoid all-masked caption
    if all(w == mask_token for w in new_words):
        new_words[random.randrange(len(new_words))] = words[0]
    return " ".join(new_words)


def drop_random_word(text: str) -> str:
    words = text.split()
    if len(words) <= 1:
        return text
    idx = random.randrange(len(words))
    return " ".join(words[:idx] + words[idx + 1 :])


def shuffle_middle_words(text: str) -> str:
    words = text.split()
    if len(words) <= 3:
        return text
    middle = words[1:-1]
    random.shuffle(middle)
    return " ".join([words[0]] + middle + [words[-1]])


def apply_text_augmentation(text: str, aug_name: str) -> str:
    if aug_name == "identity":
        return text
    elif aug_name == "mask_some_words":
        return mask_some_words(text)
    elif aug_name == "drop_random_word":
        return drop_random_word(text)
    elif aug_name == "shuffle_middle_words":
        return shuffle_middle_words(text)
    else:
        raise ValueError(f"Unknown text augmentation: {aug_name}")


def add_gaussian_noise_and_renorm(
    feats: np.ndarray, sigma: float
) -> np.ndarray:
    """
    Add Gaussian noise and renormalize to unit L2 norm along the last dim.
    feats: (N, D)
    """
    noise = np.random.normal(0.0, sigma, size=feats.shape).astype(np.float32)
    noisy = feats + noise
    norms = np.linalg.norm(noisy, axis=-1, keepdims=True) + 1e-8
    noisy = noisy / norms
    return noisy.astype(np.float32)


# ---------------------------------------------------------------------------
# 4. Main loop
# ---------------------------------------------------------------------------

def main():
    # Deterministic-ish behavior
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # Base paths (script lives in examples/cats_dogs_attention/)
    script_dir = Path(__file__).resolve().parent
    data_root = script_dir / "data" / "cats_dogs"

    # 1) Ensure dataset exists (up to 500 cats + 500 dogs)
    prepare_cats_dogs_dataset(data_root, num_per_class=500)

    # 2) Collect images and select exactly 1000 (500 per class if possible)
    cat_paths = sorted(
        p
        for p in (data_root / "cats").glob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    dog_paths = sorted(
        p
        for p in (data_root / "dogs").glob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    MAX_PER_CLASS = 500
    cat_paths = cat_paths[:MAX_PER_CLASS]
    dog_paths = dog_paths[:MAX_PER_CLASS]

    image_paths = cat_paths + dog_paths
    random.shuffle(image_paths)

    MAX_SAMPLES = 1000
    image_paths = image_paths[:MAX_SAMPLES]

    if not image_paths:
        print(f"[main] No images found under {data_root}, nothing to caption.")
        return

    print(
        f"[main] Using {len(image_paths)} images under {data_root} "
        f"(cats={len(cat_paths)}, dogs={len(dog_paths)})."
    )

    # 3) Device & models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] Using device: {device}")

    processor, caption_model = load_caption_vlm(device)
    blip_embedder = load_blip_embedder(device)

    # 4) Caption images with tqdm
    results: List[Dict] = []
    caption_times: List[float] = []
    captions: List[str] = []
    pil_images: List[Image.Image] = []

    print("[main] Captioning images...")
    for img_path in tqdm(image_paths, desc="Captioning images"):
        img = Image.open(img_path).convert("RGB")
        pil_images.append(img)

        t0 = time.perf_counter()
        caption = caption_image(img, processor, caption_model, device=device)
        t1 = time.perf_counter()

        dt = t1 - t0
        caption_times.append(dt)
        captions.append(caption)

        record = {
            "image_path": str(img_path),
            "caption": caption,
            "caption_time_sec": dt,
        }
        results.append(record)

    # 5) Save captions JSON + CSV (metadata will be extended later too)
    json_path = data_root / "captions_from_vlm.json"
    csv_path = data_root / "captions_from_vlm.csv"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[save] JSON annotations saved to {json_path}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "caption", "caption_time_sec"])
        for r in results:
            writer.writerow(
                [
                    r["image_path"],
                    r["caption"],
                    f"{r['caption_time_sec']:.6f}",
                ]
            )
    print(f"[save] CSV annotations saved to {csv_path}")

    # Timing summary for captioning
    print(
        f"[timing] caption: mean={statistics.mean(caption_times):.3f}s "
        f"(min={min(caption_times):.3f}, max={max(caption_times):.3f})"
    )

    N = len(image_paths)

    # ----------------------------------------------------------------------
    # 6) Compute RAW (non-augmented) embeddings for all N samples
    # ----------------------------------------------------------------------
    print("[embed] Computing RAW image and text embeddings...")
    BATCH = 32

    # Image embeddings (raw)
    img_emb_list: List[np.ndarray] = []
    for i in tqdm(range(0, N, BATCH), desc="Embedding RAW images"):
        batch_imgs = pil_images[i : i + BATCH]
        inputs = processor(images=batch_imgs, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feats = blip_embedder.get_image_features(pixel_values=inputs["pixel_values"])
            feats = torch.nn.functional.normalize(feats, dim=-1).cpu().numpy().astype(np.float32)
        img_emb_list.append(feats)
    img_emb_raw = np.concatenate(img_emb_list, axis=0)
    assert img_emb_raw.shape[0] == N

    # Text embeddings (raw)
    txt_emb_list: List[np.ndarray] = []
    for i in tqdm(range(0, N, BATCH), desc="Embedding RAW captions"):
        batch_caps = captions[i : i + BATCH]
        text_inputs = processor.tokenizer(
            batch_caps,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        with torch.no_grad():
            feats = blip_embedder.get_text_features(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
            )
            feats = torch.nn.functional.normalize(feats, dim=-1).cpu().numpy().astype(
                np.float32
            )
        txt_emb_list.append(feats)
    txt_emb_raw = np.concatenate(txt_emb_list, axis=0)
    assert txt_emb_raw.shape[0] == N

    # Save RAW embeddings
    img_emb_raw_path = data_root / "cats_dogs_blip_image_embeds_raw.npy"
    txt_emb_raw_path = data_root / "cats_dogs_blip_text_embeds_raw.npy"
    np.save(img_emb_raw_path, img_emb_raw)
    np.save(txt_emb_raw_path, txt_emb_raw)
    print(f"[save] RAW image embeds → {img_emb_raw_path} (shape={img_emb_raw.shape})")
    print(f"[save] RAW text embeds  → {txt_emb_raw_path} (shape={txt_emb_raw.shape})")

    # ----------------------------------------------------------------------
    # 7) Augmented embeddings: 1 raw + K augs per sample
    # ----------------------------------------------------------------------
    print(
        f"[augment] Generating {AUGMENTATIONS_PER_SAMPLE} augmentations per sample "
        f"(total per modality: 1 raw + {AUGMENTATIONS_PER_SAMPLE} aug)."
    )

    D = img_emb_raw.shape[1]
    all_img_embeds = np.zeros((N, 1 + AUGMENTATIONS_PER_SAMPLE, D), dtype=np.float32)
    all_txt_embeds = np.zeros((N, 1 + AUGMENTATIONS_PER_SAMPLE, D), dtype=np.float32)

    # Set raw embeddings at index 0
    all_img_embeds[:, RAW_INDEX, :] = img_emb_raw
    all_txt_embeds[:, RAW_INDEX, :] = txt_emb_raw

    # For metadata: record augmentation types
    img_aug_types_per_sample: List[List[str]] = [[] for _ in range(N)]
    txt_aug_types_per_sample: List[List[str]] = [[] for _ in range(N)]

    # For each augmentation index j = 1..K, apply random augment per sample
    for j in range(1, 1 + AUGMENTATIONS_PER_SAMPLE):
        print(f"[augment] Pass {j}/{AUGMENTATIONS_PER_SAMPLE}: building augmented batch...")
        # Choose random image + text augmentations per sample
        chosen_img_augs: List[str] = [
            random.choice(IMG_AUG_NAMES) for _ in range(N)
        ]
        chosen_txt_augs: List[str] = [
            random.choice(TXT_AUG_NAMES) for _ in range(N)
        ]

        # Store names (in same order as samples)
        for i in range(N):
            img_aug_types_per_sample[i].append(chosen_img_augs[i])
            txt_aug_types_per_sample[i].append(chosen_txt_augs[i])

        # Build augmented images/captions
        aug_images: List[Image.Image] = []
        aug_captions: List[str] = []
        for i in range(N):
            aug_img = apply_image_augmentation(pil_images[i], chosen_img_augs[i])
            aug_text = apply_text_augmentation(captions[i], chosen_txt_augs[i])
            aug_images.append(aug_img)
            aug_captions.append(aug_text)

        # Embed images in batches
        aug_img_emb_list: List[np.ndarray] = []
        for i in tqdm(
            range(0, N, BATCH), desc=f"Embedding AUG images (pass {j})"
        ):
            batch_imgs = aug_images[i : i + BATCH]
            inputs = processor(images=batch_imgs, return_tensors="pt", padding=True).to(
                device
            )
            with torch.no_grad():
                feats = blip_embedder.get_image_features(
                    pixel_values=inputs["pixel_values"]
                )
                feats = torch.nn.functional.normalize(feats, dim=-1).cpu().numpy().astype(
                    np.float32
                )
            aug_img_emb_list.append(feats)
        aug_img_emb = np.concatenate(aug_img_emb_list, axis=0)
        assert aug_img_emb.shape[0] == N

        # Embed captions in batches
        aug_txt_emb_list: List[np.ndarray] = []
        for i in tqdm(
            range(0, N, BATCH), desc=f"Embedding AUG captions (pass {j})"
        ):
            batch_caps = aug_captions[i : i + BATCH]
            text_inputs = processor.tokenizer(
                batch_caps,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)
            with torch.no_grad():
                feats = blip_embedder.get_text_features(
                    input_ids=text_inputs["input_ids"],
                    attention_mask=text_inputs["attention_mask"],
                )
                feats = torch.nn.functional.normalize(feats, dim=-1).cpu().numpy().astype(
                    np.float32
                )
            aug_txt_emb_list.append(feats)
        aug_txt_emb = np.concatenate(aug_txt_emb_list, axis=0)
        assert aug_txt_emb.shape[0] == N

        # Add Gaussian noise at embedding level + renormalize
        aug_img_emb_noisy = add_gaussian_noise_and_renorm(
            aug_img_emb, sigma=IMG_EMB_NOISE_SIGMA
        )
        aug_txt_emb_noisy = add_gaussian_noise_and_renorm(
            aug_txt_emb, sigma=TXT_EMB_NOISE_SIGMA
        )

        # Store in slot j along axis 1
        all_img_embeds[:, j, :] = aug_img_emb_noisy
        all_txt_embeds[:, j, :] = aug_txt_emb_noisy

    # 8) Save augmented embeddings
    img_emb_aug_path = data_root / "cats_dogs_blip_image_embeds_augmented.npy"
    txt_emb_aug_path = data_root / "cats_dogs_blip_text_embeds_augmented.npy"
    np.save(img_emb_aug_path, all_img_embeds)
    np.save(txt_emb_aug_path, all_txt_embeds)

    print(
        f"[save] RAW+aug image embeds → {img_emb_aug_path} "
        f"(shape={all_img_embeds.shape})"
    )
    print(
        f"[save] RAW+aug text embeds  → {txt_emb_aug_path} "
        f"(shape={all_txt_embeds.shape})"
    )

    # 9) Update metadata JSON/CSV with augmentation info + raw index
    for i, rec in enumerate(results):
        rec["raw_embed_index"] = RAW_INDEX
        rec["image_augmentations"] = img_aug_types_per_sample[i]
        rec["text_augmentations"] = txt_aug_types_per_sample[i]

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[save] UPDATED JSON (with augmentation info) → {json_path}")

    # Extended CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "image_path",
            "caption",
            "caption_time_sec",
            "raw_embed_index",
            "image_augmentations",
            "text_augmentations",
        ]
        writer.writerow(header)
        for rec in results:
            writer.writerow(
                [
                    rec["image_path"],
                    rec["caption"],
                    f"{rec['caption_time_sec']:.6f}",
                    rec["raw_embed_index"],
                    ";".join(rec["image_augmentations"]),
                    ";".join(rec["text_augmentations"]),
                ]
            )
    print(f"[save] UPDATED CSV (with augmentation info) → {csv_path}")


if __name__ == "__main__":
    main()
