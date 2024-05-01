import torch
import numpy as np
import pandas as pd
import torchaudio
import torch.nn.functional as F
from pathlib import Path

def unwrap(phi, dim=1):
    """
    Unwrap phase values.

    Parameters:
    phi (torch.Tensor): Phase tensor.
    dim (int): Dimension to unwrap.

    Returns:
    torch.Tensor: Unwrapped phase tensor.
    """
    dphi = torch.diff(phi, dim=dim)
    dphi = F.pad(dphi, (0, 0, 1, 0))
    
    dphi_m = ((dphi + np.pi) % (2 * np.pi)) - np.pi
    dphi_m[(dphi_m == -np.pi) & (dphi > 0)] = np.pi
    phi_adj = dphi_m - dphi
    phi_adj[dphi.abs() < np.pi] = 0
    
    return phi + torch.cumsum(phi_adj, dim=dim)


def stft(x: torch.Tensor,
        fft_size: int = 1024,
        hop_size: int = 64,
        win_length: int = 1024,
        window: torch.Tensor = "hann_window") -> torch.Tensor:
    """
    Compute the short-time Fourier transform of a signal.

    Parameters:
    x (torch.Tensor): Input tensor.
    fft_size (int): FFT size.
    hop_size (int): Hop size.
    win_length (int): Window length.
    window (str): Window type.

    Returns:
    torch.Tensor: Short-time Fourier transform.
    """
    window = getattr(torch, window)(win_length)
    x_stft = torch.stft(
                    x,
                    fft_size,
                    hop_size,
                    win_length,
                    window,
                    return_complex=True)
            
    x_phs = x_stft[:, :(fft_size//8), :]
    return x_phs

def phase_differece_feature(input_x: torch.Tensor, target_y: torch.Tensor) -> torch.Tensor:
    """
    Compute the phase difference feature between input and target pairs.

    Parameters:
    input_x (torch.Tensor): Input tensor.
    target_y (torch.Tensor): Target tensor.

    Returns:
    torch.Tensor: Phase difference feature.
    """
    # compute the phase difference between input and target
    x_phs = stft(input_x.view(-1, input_x.size(-1)))
    y_phs = stft(target_y.view(-1, target_y.size(-1)))
    phase_diff = torch.real(y_phs - x_phs)
    return phase_diff


class AudioDataset(torch.nn.Module):
    """
    Audio dataset class.
    
    Parameters:
    annotations (pd.DataFrame): Annotations dataframe.
    audio_dir (str): Audio directory.
    fs (int): Sample rate.
    """
    def __init__(
        self,
        annotations: pd.DataFrame,
        audio_dir: str = "soundfiles",
        fs: int = 44100,
    ) -> None:
        super().__init__()
        self.annotations = annotations
        self.audio_dir = audio_dir
        self.fs = fs
        self.tmp_ann = self.annotations.copy()

    def __len__(self):
        return len(self.annotations)

    def load_audio(self, filename: str):
        # read segment of audio from file
        x, sr = torchaudio.load(
            Path(self.audio_dir, filename)
        )

        # resample if necessary
        if sr != self.fs:
            x = torchaudio.transforms.Resample(sr, self.fs)(x)

        # convert stereo to mono
        if x.size(0) == 2:
            x = torch.mean(x, dim=0, keepdim=True)

        # normalize audio
        x = x / x.abs().max()

        # clamp to [-1,1] to ensure within range
        x = torch.clamp(x, -1, 1)

        return x

    def __getitem__(self, idx: int):
        # Selecting a random class
        class_id = np.random.choice(self.annotations['ClassID'].unique())

        # Selecting a random top and bottom snare record from annotations
        input_df = self.annotations.query(f'(ClassID == {class_id}) & (Position == "TP")').sample(n=1)
        target_df = self.annotations.query(f'(ClassID == {class_id}) & (Position == "BTM")').sample(n=1)

        # Load audio
        input_audio = self.load_audio(Path(self.audio_dir, input_df['FileName'].values[0]))
        target_audio = self.load_audio(Path(self.audio_dir, target_df['FileName'].values[0]))

        return input_audio, target_audio


class TestAudioDataset(AudioDataset):
    """
    Audio dataset class.
    
    Parameters:
    annotations (pd.DataFrame): Annotations dataframe.
    audio_dir (str): Audio directory.
    fs (int): Sample rate.
    """
    def __init__(
        self,
        annotations: pd.DataFrame,
        audio_dir: str = "soundfiles",
        fs: int = 44100,
    ) -> None:
        super().__init__(annotations, audio_dir, fs)

    def __len__(self):
        return len(self.annotations.query(f'Position == "SHL"'))

    def __getitem__(self, idx: int):
        # Selecting a random class
        class_id = np.random.choice(self.annotations['ClassID'].unique())

        # Selecting a random top and bottom snare record from annotations
        df = self.annotations.query(f'(ClassID == {class_id}) & (Position == "SHL")').sample(n=2)
        input_df, target_df = df.iloc[0], df.iloc[1]

        # Load audio
        input_audio = self.load_audio(Path(self.audio_dir, input_df['FileName']))
        target_audio = self.load_audio(Path(self.audio_dir, target_df['FileName']))

        return input_audio, target_audio
    
    
if __name__ == '__main__':
    ann = pd.read_csv('/home/hf1/Documents/soundfiles/annotations.csv')
    ds = TestAudioDataset(ann, '/home/hf1/Documents/soundfiles/SDDS_segmented_Allfiles/', 44100)