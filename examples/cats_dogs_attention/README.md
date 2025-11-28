# Cats & Dogs Attention Example

End-to-end demo that builds a tiny cats/dogs dataset from CIFAR-10, captions it with BLIP, creates image/text embeddings, and trains a GlobalWorkspaceFusion model on those embeddings. The notebook `cats_dogs_fetch.ipynb` walks through the same flow and visualizes results.

## Prerequisites
- Python 3.10+ recommended
- GPU strongly recommended (BLIP captioning/embedding is slow on CPU)
- Network access to download CIFAR-10 and BLIP weights on first run

## Setup
```bash
cd shimmer
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install torch torchvision lightning transformers pillow tqdm matplotlib jupyter
```

## 1) Generate images, captions, and embeddings
Run from repo root. This will:
- download CIFAR-10 and extract 500 cats + 500 dogs if images are missing
- download BLIP, caption images, and save all augmented embeddings/metadata under `examples/cats_dogs_attention/data/cats_dogs/`
```bash
python examples/cats_dogs_attention/captioning_test.py
```
You can override the output dir with `CATS_DOGS_BLIP_DIR=/path/to/data`.

## 2) (Optional) Train the GW model on the embeddings
Produces checkpoints under `examples/cats_dogs_attention/checkpoints_cats_dogs_gw/`.
```bash
python examples/cats_dogs_attention/replicate_nb.py
```

## 3) Run the notebook
Launch Jupyter and open `examples/cats_dogs_attention/cats_dogs_fetch.ipynb`.
- It expects the embedding files from step 1 in `data/cats_dogs/`.
- It will download CIFAR-10 into `data/cifar10/` for quick visual checks if not present.
- If you trained in step 2, point the notebook to your checkpoint folder; otherwise it will train from scratch inside the notebook.

You can also view the rendered `cats_dogs_fetch.html` if you only need the walkthrough without executing cells.
