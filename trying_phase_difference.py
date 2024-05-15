from dataset import AudioDataset, phase_differece_feature
import torch
import pandas as pd

# Using the dataloader to load the dataset,
# Load in the first batch of data,
# then extract the phase difference feature from the first batch of data.

# Load the dataset
annotations = pd.read_csv("annotations.csv")
dataset = AudioDataset(annotations, '/home/hf1/Documents/soundfiles/SDDS_segmented_Allfiles')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

# Load the first batch of data
x, y = next(iter(dataloader))
print(x.shape)

# Extract the phase difference feature
feature = phase_differece_feature(x, y)
print(feature.shape)