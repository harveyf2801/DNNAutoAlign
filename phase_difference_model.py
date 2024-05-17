import torch
from dataset import phase_differece_feature

import torch.nn as nn
import torch.nn.functional as F


class ParameterNetwork(torch.nn.Module):
    '''
    A neural network to estimate the parameters of cascaded all-pass filters.
    
    Parameters:
        num_control_params (int): number of control parameters to estimate
    '''
    def __init__(self, num_control_params: int, input_size: int = (128, 345)) -> None:
        super().__init__()
        self.num_control_params = num_control_params
        self.input_size = input_size
        
        # Define convolutional layers
        self.conv1 = nn.Conv1d(in_channels=self.input_size[0], out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        
        # Define max-pooling layer
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Define transformer layer
        self.transformer = nn.Transformer(d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6)
        
        # Define fully connected layers
        self.fc1 = nn.Linear(256 * (self.input_size[1] // 8), 512)  # Adjusted input size based on max-pooling
        self.fc2 = nn.Linear(512, self.num_control_params)

    def forward(self, feature: torch.Tensor):
        # Convolutional layers with ReLU activation
        feature = F.relu(self.conv1(feature))
        feature = self.pool(feature)  # Max-pooling layer to down-sample the spatial dimensions
        feature = F.relu(self.conv2(feature))
        feature = self.pool(feature)
        feature = F.relu(self.conv3(feature))
        feature = self.pool(feature)

        # Flatten the tensor to prepare for transformer layer
        feature = feature.view(-1, 256, (self.input_size[1] // 8))

        # Apply transformer layer
        feature = self.transformer(feature, feature)

        # Flatten the tensor to prepare for fully connected layers
        feature = feature.view(-1, 256 * (self.input_size[1] // 8))

        # Fully connected layers with ReLU activation
        feature = F.relu(self.fc1(feature))
        feature = self.fc2(feature)

        return torch.sigmoid(feature)