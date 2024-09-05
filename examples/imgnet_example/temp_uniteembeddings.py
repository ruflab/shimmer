import numpy as np

# Function to normalize embeddings to N(0,1) using a single mean and std for all dimensions
def standardize_embeddings_global(embeddings):
    mean = np.mean(embeddings)
    std = np.std(embeddings)

    print("standardize based on ", mean, std)
    return (embeddings - mean) / std

# Load the first file
file1 = '/home/rbertin/cleaned/git_synced/shimmer/examples/imgnet_example/sd_image_embeddings/image_embeddings_torch.Size([256, 4, 16, 16])_sd.npy'
embeddings1 = np.load(file1)

# Load the second file
file2 = '/home/rbertin/cleaned/git_synced/shimmer/examples/imgnet_example/sd_image_embeddings/image_embeddings_torch.Size([143, 4, 16, 16])_sd.npy'
embeddings2 = np.load(file2)



# Standardize the embeddings
standardized_embeddings1 = standardize_embeddings_global(embeddings1)
standardized_embeddings2 = standardize_embeddings_global(embeddings2)

# Concatenate the standardized embeddings
combined_embeddings = np.concatenate((standardized_embeddings1, standardized_embeddings2), axis=0)
combined_embeddings = combined_embeddings.reshape(combined_embeddings.shape[0], -1)

print("combined embeddings ", combined_embeddings.mean(),combined_embeddings.std(), combined_embeddings.shape)

# Save the combined standardized embeddings to a new file
output_file = 'sd_image_embeddings/train_united.npy'
np.save(output_file, combined_embeddings)

print(f'Combined standardized embeddings saved to {output_file}')