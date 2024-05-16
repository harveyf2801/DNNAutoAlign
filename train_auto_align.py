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
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm
from typing import List

from losses import MultiResolutionSTFTLoss, MSELoss
import diff_apf_pytorch.modules
from phase_difference_model import ParameterNetwork
from dataset import AudioDataset, phase_differece_feature

FS = 44100
MODEL_TYPE = "MSE_PhaseFeature"

############################################
# Setting up multiple gpu cores
# import idr_torch

#####################

# import torch.distributed as dist
 
# from torch.nn.parallel import DistributedDataParallel as DDP

# dist.init_process_group(backend='nccl',
#                         world_size=idr_torch.world_size,
#                         rank=idr_torch.rank)

# torch.cuda.set_device(idr_torch.local_rank)
gpu = torch.device("cuda")

# Create the parametric all-pass instance
equalizer = diff_apf_pytorch.modules.ParametricEQ(FS)

# Create the parameter estimation network
net = ParameterNetwork(equalizer.num_params)
# ddp_model = DDP(net)


############################################

# In this example we will train a neural network to perform automatic phase alignment.
# We train the network to estimate the parameters of 6 cascaded all-pass filters.
# Using the SDDS dataset, we pass in a target and train the network to align the input signal with the target / reference signal.

annotations = pd.read_csv("annotations.csv")
audio_dir = "/home/hf1/Documents/soundfiles/SDDS_segmented_Allfiles"
lr = 2e-3 # 1e-5 # 2e-3
batch_size = 256 # 16 # 512
num_epochs = 1000 # 1000
log_dir = f"outputs/{MODEL_TYPE}_diff_apf"

# Create the log directory
os.makedirs(log_dir, exist_ok=True)

# Gradient clipping value
max_grad_norm = 1.0

# Create the optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)

# Learning rate scheduler based on epoch count
epoch_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# Learning rate scheduler
plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

# Using the custom SDDS dataset which returns the input and target pairs
train_dataset = AudioDataset(annotations, audio_dir=audio_dir, fs=FS)
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

# Create a folder to store the event files
# increment the folder name if it already exists
folder = Path('runs', f'{MODEL_TYPE}_diff_apf')
i = 1
while folder.exists():
    folder = Path('runs', f'{MODEL_TYPE}_diff_apf_1')
    i += 1

# Create a SummaryWriter to log the training process to TensorBoard
writer = SummaryWriter(folder)

# Create the MR-STFT loss function (using Abargum (2023) version)
# criterion = MultiResolutionSTFTLoss(
#     fft_sizes=[1024, 512, 2048],
#     hop_sizes=[120, 50, 240],
#     win_lengths=[600, 240, 1200]
# )
# Try using the MSE loss function instead
criterion = MSELoss(reduction='mean')

# Move the network and loss function to the GPU if available
net = net.to(gpu)
if gpu is not None:
    criterion.cuda()

print("****** Training ******")

# Initialize the number of steps
steps = 0

# Initialize the loss history
epoch_loss_history = []
# Iterate over the epochs in the training process
for epoch in range(num_epochs):
    # Set the network to training mode
    net.train()
    
    print("Epoch:", epoch + 1)
    # Initialize the batch loss history
    batch_loss_history = []
    # Iterate over the batches in the dataset
    pbar = tqdm(dataloader)
    for batch, data in enumerate(pbar):
        # Get the input and target pairs
        input_x, target_y = data[0], data[1]

        # Move the input and target pairs to the GPU if available
        input_x = input_x.to(gpu, non_blocking=True)
        target_y = target_y.cuda(gpu, non_blocking=True)
        
        # RATHER THAN CREATING A FEATURE, WE PASS IN THE INPUT DIRECTLY

        # Predicting the parameters of the all-pass filters using
        # the phase difference analysis feature as an input to the network
        feature = phase_differece_feature(input_x, target_y)
        p_hat = net(feature)

        # Apply the estimated parameters to the input signal
        # to align the phase
        x_hat = equalizer.process_normalized(input_x, p_hat)
        # Apply the tanh function to the output
        # to ensure the output is in the range [-1, 1]
        x_hat = torch.tanh(x_hat)

        # Calculate the loss between the output and the target
        loss = criterion(x_hat, target_y)

        # Zero the gradients, backward pass, and update the weights
        # using the optimizer
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
        optimizer.step()

        # Log the loss to TensorBoard and update the steps
        writer.add_scalar("loss/loss pr. step", loss.item(), steps)
        steps += 1

        # Append the loss to the batch loss history and update the progress bar
        batch_loss_history.append(loss.item())
        pbar.set_description(f"loss: {np.mean(batch_loss_history):.4e}")

    # Append the mean loss to the epoch loss history
    epoch_loss_history.append(np.mean(batch_loss_history))

    # Plot the loss history
    # plotting.plot_loss(log_dir, epoch_loss_history)

    # Validate the network every 4 epochs
    if (epoch + 1) % 2 == 0:

        # Save the network and optimizer state checkpoints
        checkpoint_path = os.path.join(log_dir, 'models')
        os.makedirs(checkpoint_path, exist_ok=True)
        torch.save({
        'epoch': epoch + 1,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss},
        os.path.join(checkpoint_path, f"{MODEL_TYPE}_diff_apf_epoch_{epoch + 1}_loss_{loss}.pth"))
    
    # Step the epoch-based learning rate scheduler
    epoch_scheduler.step()

    # Step the learning rate scheduler based on the average loss of the epoch
    plateau_scheduler.step(loss, epoch=epoch)

print("******** Done ********")