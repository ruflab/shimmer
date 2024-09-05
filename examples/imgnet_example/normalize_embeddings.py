import numpy as np
import os

# Path to the directory where the embeddings file is stored and the filename
root_dir = ''  # Change this to your directory path
embeddings_file = '/home/rbertin/pyt_scripts/BLIP_TEST/gemma/gemma_bge_captions_val.npy'
file_path = os.path.join(root_dir, embeddings_file)

# Load embeddings
embeddings = np.load(file_path)

# Print original mean and std across all dimensions
original_mean = np.mean(embeddings)
original_std = np.std(embeddings)
print("Original Mean:", original_mean)
print("Original Std:", original_std)

# Normalize the embeddings using single mean and std across all dimensions
normalized_embeddings = (embeddings - original_mean) / original_std

# Print normalized mean and std
normalized_mean = np.mean(normalized_embeddings)
normalized_std = np.std(normalized_embeddings)
print("Normalized Mean:", normalized_mean)
print("Normalized Std:", normalized_std)

# Save normalized embeddings
normalized_file_path = "/home/rbertin/pyt_scripts/BLIP_TEST/gemma/gemma_norm_bge_captions_val.npy"
np.save(normalized_file_path, normalized_embeddings)
print("Saved normalized embeddings to", normalized_file_path)
