import torch
import torchaudio
from torch.nn import Module
from torch import Tensor
import torch.nn.functional as F
from torch.nn import MSELoss
import numpy as np

"""STFT and Multi Resolution STFT Loss Modules.
    See [Steinmetz & Reiss, 2020]('auraloss: Audio-focused loss functions in PyTorch')"""
class STFTLoss(torch.nn.Module):
    def __init__(
        self,
        fft_size=1024,
        hop_size=256,
        win_length=1024,
        window="hann_window",
        eps=1e-8,
    ):
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)
        self.eps = eps
        self.loss = MSELoss(reduction='mean')

    def stft(self, x):
        x_stft = torch.stft(
            x,
            self.fft_size,
            self.hop_size,
            self.win_length,
            self.window,
            return_complex=True,
        )
        
        x_mag = torch.sqrt(
            torch.clamp((x_stft.real ** 2) + (x_stft.imag ** 2), min=self.eps)
        )
        
        return x_mag[:, :(self.fft_size//8), :]
    
    def forward(self, x, y):
        self.window = self.window.to(x.device)
        
        x_mag = self.stft(x.view(-1, x.size(-1)))
        y_mag = self.stft(y.view(-1, y.size(-1)))

        mag_1 = self.stft(x.view(-1, x.size(-1)) + y.view(-1, y.size(-1)))
        mag_2 = x_mag + y_mag
        
        stft_loss = self.loss(mag_2, mag_1)
        loss = stft_loss
        return loss

class MultiResolutionSTFTLoss(torch.nn.Module):
    def __init__(
        self,
        fft_sizes=[1024, 512, 2048],
        hop_sizes=[120, 50, 240],
        win_lengths=[600, 240, 1200],
        #fft_sizes=[128, 512, 256],
        #hop_sizes=[13, 50, 25],
        #win_lengths=[60, 240, 120],
        window="hann_window",
        **kwargs,
    ):
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)  # must define all
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [
                STFTLoss(
                    fs,
                    ss,
                    wl,
                    window,
                    **kwargs,
                )
            ]

    def forward(self, x, y):
        mrstft_loss = 0.0

        for f in self.stft_losses:
                mrstft_loss += f(x, y)
        mrstft_loss /= len(self.stft_losses)
        
        return mrstft_loss