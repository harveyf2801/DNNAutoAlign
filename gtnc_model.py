import torch


class GatedTemporalConvBlock(torch.nn.Module):
    '''
    A gated temporal convolution block with batch normalization, dropout, PReLU, and dilated causal convolution.
    
    Parameters:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolutional kernel
        dilation (int): dilation factor for the convolutional kernel
        dropout (float): dropout probability
    '''
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        dropout: float = 0.5,
    ):
        super().__init__()
        # Define the first gated convolutional layer
        self.conv1 = torch.nn.Conv1d(
            in_channels,
            out_channels * 2,  # double output channels for gated convolution
            kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation,  # dilated causal convolution
        )
        # Batch normalization for the first gated convolutional layer
        self.bn1 = torch.nn.BatchNorm1d(out_channels)
        # PReLU activation
        self.prelu1 = torch.nn.PReLU(out_channels)
        # Dropout layer to prevent overfitting
        self.dropout = torch.nn.Dropout(dropout)
        # Define the second gated convolutional layer
        self.conv2 = torch.nn.Conv1d(
            out_channels,
            out_channels * 2,
            kernel_size,
            dilation=1,
            padding=(kernel_size - 1) // 2,
        )
        # Batch normalization for the second gated convolutional layer
        self.bn2 = torch.nn.BatchNorm1d(out_channels)
        # PReLU activation
        self.prelu2 = torch.nn.PReLU(out_channels)

    def forward(self, x: torch.Tensor):
        ''' Forward pass of the Gated Temporal Convolution Block. '''
        # Apply gated convolutional layers with batch normalization, PReLU, and dropout
        # First convolutional layer
        x_conv = self.conv1(x)
        x1, x2 = torch.chunk(x_conv, chunks=2, dim=1)  # Split the output into two parts for gating
        x = torch.sigmoid(x1) * torch.tanh(x2)  # Apply gating mechanism
        x = self.bn1(x)  # Apply batch normalization
        x = self.prelu1(x)  # Apply PReLU activation
        x = self.dropout(x)  # Apply dropout

        # Second convolutional layer
        x_conv = self.conv2(x)
        x1, x2 = torch.chunk(x_conv, chunks=2, dim=1)  # Split the output into two parts for gating
        x = torch.sigmoid(x1) * torch.tanh(x2)  # Apply gating mechanism
        x = self.bn2(x)  # Apply batch normalization
        x = self.prelu2(x)  # Apply PReLU activation
        x = self.dropout(x)  # Apply dropout

        return x
