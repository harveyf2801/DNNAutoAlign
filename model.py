import torch

# In this example we will train a neural network to perform automatic phase alignment.
# We train the network to estimate the parameters of 6 cascaded all-pass filters.
# Using the SDDS dataset, we pass in a target and train the network to align the input signal with the target / reference signal.


class TCNBlock(torch.nn.Module):
    '''
    A simple TCN block with a dilated convolution and batch normalization.
    
    Parameters:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolutional kernel
        dilation (int): dilation factor for the convolutional kernel
    '''
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            stride=2,  # Adjusted stride for downsampling
            padding=(kernel_size - 1) // 2 * dilation,
        )
        self.relu1 = torch.nn.PReLU(out_channels)
        self.bn1 = torch.nn.BatchNorm1d(out_channels)
        self.conv2 = torch.nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            dilation=1,
            padding=(kernel_size - 1) // 2,
        )
        self.relu2 = torch.nn.PReLU(out_channels)
        self.bn2 = torch.nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor):
        ''' Forward pass of the TCN block. '''
        # apply convolutional layers
        x = self.bn1(self.relu1(self.conv1(x)))
        x = self.bn2(self.relu2(self.conv2(x)))
        return x


class ParameterNetwork(torch.nn.Module):
    '''
    A neural network to estimate the parameters of cascaded all-pass filters.
    
    Parameters:
        num_control_params (int): number of control parameters to estimate
        ch_dim (int): number of channels in the TCN blocks
    '''
    def __init__(self, num_control_params: int, ch_dim: int = 256) -> None:
        super().__init__()
        self.num_control_params = num_control_params
        
        # A TCN with 10 blocks is used to estimate the parameters
        # of the all-pass filters.
        self.x_blocks = torch.nn.ModuleList()
        self.x_blocks.append(TCNBlock(2, ch_dim, 7, dilation=1))
        self.x_blocks.append(TCNBlock(ch_dim, ch_dim, 7, dilation=2))
        self.x_blocks.append(TCNBlock(ch_dim, ch_dim, 7, dilation=4))
        self.x_blocks.append(TCNBlock(ch_dim, ch_dim, 7, dilation=8))
        self.x_blocks.append(TCNBlock(ch_dim, ch_dim, 7, dilation=16))
        self.x_blocks.append(TCNBlock(ch_dim, ch_dim, 7, dilation=1))
        self.x_blocks.append(TCNBlock(ch_dim, ch_dim, 7, dilation=2))
        self.x_blocks.append(TCNBlock(ch_dim, ch_dim, 7, dilation=4))
        self.x_blocks.append(TCNBlock(ch_dim, ch_dim, 7, dilation=8))
        self.x_blocks.append(TCNBlock(ch_dim, ch_dim, 7, dilation=16))

        # A simple MLP is used to map the output of the TCN to the parameters.
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(ch_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_control_params),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # Sum the input and target signal, then normalise them
        # input_data = x + y
        # input_data = input_data / torch.max(torch.abs(input_data))

        # Concatenate the input and target signal
        input_data = torch.cat([x, y], dim=1)

        # Apply the TCN blocks
        for block in self.x_blocks:
            input_data = block(input_data)

        # Average / aggregate over time
        input_data = input_data.mean(dim=-1)

        # Apply the MLP to map to the parameters
        return torch.sigmoid(self.mlp(input_data))