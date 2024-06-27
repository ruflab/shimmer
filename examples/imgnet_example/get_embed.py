import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import os

class CaptionsDataset():
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.captions = self.df['Caption'].tolist()

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        return self.captions[idx]

def generate_embeddings(captions, model, batch_size=32, device='cuda'):
    # Prepare the device
    model.to(device)

    embeddings = []
    for i in tqdm(range(0, len(captions), batch_size), desc="Generating Embeddings"):
        batch_captions = captions[i:i + batch_size]
        batch_embeddings = model.encode(batch_captions, convert_to_tensor=True, batch_size=batch_size, device=device)
        embeddings.append(batch_embeddings.cpu().numpy())  # Move embeddings to CPU and convert to numpy

    # Concatenate all batch embeddings
    all_embeddings = np.vstack(embeddings)
    return all_embeddings
    
def normalize_embeddings(embeddings):
    mean = np.mean(embeddings)
    std = np.std(embeddings)
    normalized_embeddings = (embeddings - mean) / std

    return normalized_embeddings, mean, std

# Parameters
csv_file = "new_captions_only.csv"
model_name = "BAAI/bge-small-en-v1.5"
output_embeddings_file = "bge_fullsized_captions.npy"
output_normalized_embeddings_file = "bge_fullsized_captions_norm.npy"

# Load the captions
captions_dataset = CaptionsDataset(csv_file)

# Initialize the BGE model
bge_model = SentenceTransformer(model_name)

# Generate embeddings for all captions
print("Generating embeddings...")
captions_embeddings = generate_embeddings(captions_dataset.captions, bge_model, batch_size=32, device='cuda')

# Save embeddings to file
np.save(output_embeddings_file, captions_embeddings)
print(f"Saved embeddings to {output_embeddings_file}")

# Normalize the embeddings
print("Normalizing embeddings...")
normalized_embeddings, embeddings_mean, embeddings_std = normalize_embeddings(captions_embeddings)

# Save normalized embeddings
np.save(output_normalized_embeddings_file, normalized_embeddings)
print(f"Saved normalized embeddings to {output_normalized_embeddings_file}")

# Print stats
print("Original Mean:", embeddings_mean)
print("Original Std:", embeddings_std)
print("Normalized Mean:", np.mean(normalized_embeddings, axis=0))
print("Normalized Std:", np.std(normalized_embeddings, axis=0))
