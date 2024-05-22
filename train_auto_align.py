import os
import glob
import torch
import torchaudio
import numpy as np
import pandas as pd
import diff_apf_pytorch
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from tqdm import tqdm
from typing import List

from losses import MultiResolutionSTFTLoss, MSELoss
import diff_apf_pytorch.modules
from phase_difference_model import ParameterNetwork
from dataset import AudioDataset, phase_differece_feature

FS = 44100
MODEL_TYPE = "MSE_PhaseFeature"
gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the parametric all-pass instance
equalizer = diff_apf_pytorch.modules.ParametricEQ(FS)

# Create the parameter estimation network
net = ParameterNetwork(equalizer.num_params).to(gpu)

annotations = pd.read_csv("annotations.csv")
audio_dir = "/home/hf1/Documents/soundfiles/SDDS_segmented_Allfiles"
lr = 1e-4
batch_size = 128
num_epochs = 1000
log_dir = f"outputs/{MODEL_TYPE}_diff_apf_3"

# Create the log directory
os.makedirs(log_dir, exist_ok=True)

# Gradient clipping value
max_grad_norm = 1.0

# Create the optimizer
optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)

# Learning rate scheduler
epoch_scheduler = ExponentialLR(optimizer, gamma=0.95)
plateau_scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

# Using the custom SDDS dataset which returns the input and target pairs
train_dataset = AudioDataset(annotations, audio_dir=audio_dir, fs=FS)
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

# Create a SummaryWriter to log the training process to TensorBoard
writer = SummaryWriter(log_dir)

# Try using the MSE loss function instead
criterion = MSELoss(reduction='mean').to(gpu)

print("****** Training ******")

# Initialize the number of steps
steps = 0

# Initialize the loss history
epoch_loss_history = []
# Iterate over the epochs in the training process
for epoch in range(num_epochs):
    net.train()
    print(f"Epoch: {epoch + 1}/{num_epochs}")

    batch_loss_history = []
    pbar = tqdm(dataloader)
    for batch, data in enumerate(pbar):
        input_x, target_y = data[0].to(gpu, non_blocking=True), data[1].to(gpu, non_blocking=True)

        # Predicting the parameters of the all-pass filters using the phase difference feature
        feature = phase_differece_feature(input_x, target_y)
        p_hat = net(feature)

        # Apply the estimated parameters to the input signal to align the phase
        x_hat = equalizer.process_normalized(input_x, p_hat)
        x_hat = torch.tanh(x_hat)

        # Calculate the loss between the output and the target
        loss = criterion(x_hat, target_y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
        optimizer.step()

        writer.add_scalar("loss/loss pr. step", loss.item(), steps)
        steps += 1

        batch_loss_history.append(loss.item())
        pbar.set_description(f"Loss: {np.mean(batch_loss_history):.4e}")

    epoch_loss_history.append(np.mean(batch_loss_history))

    if (epoch + 1) % 10 == 0:
        checkpoint_path = os.path.join(log_dir, 'models')
        os.makedirs(checkpoint_path, exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item()
        }, os.path.join(checkpoint_path, f"{MODEL_TYPE}_epoch_{epoch + 1}_loss_{loss:.4e}.pth"))

    epoch_scheduler.step()
    plateau_scheduler.step(np.mean(batch_loss_history))

print("******** Done ********")
