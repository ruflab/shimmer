import pandas as pd

# Define the path to the captions CSV file
val_caption_csv = "captions_fullimgnet_val_noshuffle.csv"

# Load the captions dataset
df = pd.read_csv(val_caption_csv)

# Print the captions for indices 1492 and 1466
caption_1492 = df.iloc[1492]['Caption']
caption_1466 = df.iloc[1466]['Caption']

print(f"Caption at index 1492: {caption_1492}")
print(f"Caption at index 1466: {caption_1466}")
