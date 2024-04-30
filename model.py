import torch
from gtnc_model import TemporalConvBlock


class FiLM(torch.nn.Module):
    '''
    Feature-wise Linear Modulation (FiLM) layer.
    
    Parameters:
        input_size (int): Size of the input features
        modulation_size (int): Size of the modulation vector
    '''
    def __init__(self, input_size: int, modulation_size: int):
        super().__init__()
        # A linear layer is used to estimate the gamma and beta parameters
        self.linear = torch.nn.Linear(modulation_size, input_size * 2)

    def forward(self, x: torch.Tensor, modulation: torch.Tensor):
        # Apply FiLM modulation
        gamma, beta = torch.chunk(self.linear(modulation), chunks=2, dim=-1)
        return x * gamma.unsqueeze(-1) + beta.unsqueeze(-1)
    

class ParameterNetwork(torch.nn.Module):
    '''
    A neural network to estimate the parameters of cascaded all-pass filters.
    
    Parameters:
        num_control_params (int): number of control parameters to estimate
        ch_dim (int): number of channels in the TCN blocks
    '''
    def __init__(self, num_control_params: int, ch_dim: int = 256, modulation_size: int = 128, num_heads: int = 8) -> None:
        super().__init__()
        self.num_control_params = num_control_params
        self.num_heads = num_heads # was used to trial self attention based network
        
        # Gated Temporal Convolutional Network (GTCN)
        self.tcn_blocks = torch.nn.ModuleList()
        self.tcn_blocks.append(TemporalConvBlock(2, ch_dim, 7, dilation=1))
        self.tcn_blocks.append(TemporalConvBlock(ch_dim, ch_dim, 7, dilation=2))
        self.tcn_blocks.append(TemporalConvBlock(ch_dim, ch_dim, 7, dilation=4))
        self.tcn_blocks.append(TemporalConvBlock(ch_dim, ch_dim, 7, dilation=8))
        self.tcn_blocks.append(TemporalConvBlock(ch_dim, ch_dim, 7, dilation=16))
        self.tcn_blocks.append(TemporalConvBlock(ch_dim, ch_dim, 7, dilation=1))
        self.tcn_blocks.append(TemporalConvBlock(ch_dim, ch_dim, 7, dilation=2))
        self.tcn_blocks.append(TemporalConvBlock(ch_dim, ch_dim, 7, dilation=4))
        self.tcn_blocks.append(TemporalConvBlock(ch_dim, ch_dim, 7, dilation=8))
        self.tcn_blocks.append(TemporalConvBlock(ch_dim, ch_dim, 7, dilation=16))

        # MLP to map GTCN output to modulation vector
        self.mlp_gtcn = torch.nn.Sequential(
            torch.nn.Linear(ch_dim, modulation_size),
            torch.nn.ReLU(),
            torch.nn.Linear(modulation_size, modulation_size),
            torch.nn.ReLU(),
        )

        # FiLM layer
        self.film = FiLM(input_size=ch_dim, modulation_size=modulation_size)

        # MLP to map modulation vector to control parameters
        self.mlp_film = torch.nn.Sequential(
            torch.nn.Linear(modulation_size, ch_dim*2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5), 
            torch.nn.Linear(ch_dim*2, ch_dim*2),
            torch.nn.ReLU(),
            torch.nn.Linear(ch_dim*2, num_control_params),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # Concatenate input and target signal
        input_data = torch.cat([x, y], dim=1)

        # Apply GTCN blocks
        for block in self.tcn_blocks:
            input_data = block(input_data)

        # Apply MLP to get modulation vector
        modulation = self.mlp_gtcn(torch.mean(input_data, dim=-1))

        # Apply FiLM modulation
        input_data = self.film(input_data, modulation)

        # Apply MLP to get control parameters
        return torch.sigmoid(self.mlp_film(modulation))


if __name__ == "__main__":
    import numpy as np
    from torchsummary import summary
    from torchviz import make_dot

    # Create a network instance
    net = ParameterNetwork(num_control_params=6)
    
    # Print the network architecture
    summary(net, input_size=[(1, 44100), (1, 44100)])

    # Visualize the network architecture
    # x = torch.zeros(1, 1, 44100, dtype=torch.float, requires_grad=True)
    # y = torch.zeros(1, 1, 44100, dtype=torch.float, requires_grad=True)
    # out = net(x, y)
    # g = make_dot(out, params=dict(net.named_parameters())).render("attached", format="png")

    # Print number of parameters
    # n_params = 0
    # for name, param in net.named_parameters():
    #         n_params += np.prod(param.size())
    # print(f'The model contains a total of {n_params:,} parameters.')