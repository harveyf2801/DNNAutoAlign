import torch
import numpy as np
import scipy.signal
import diff_apf_pytorch.signals as signals

from functools import partial
from typing import Dict, List


def parametric_eq(
    x: torch.Tensor,
    sample_rate: float,
    band0_cutoff_freq: torch.Tensor,
    band0_q_factor: torch.Tensor,
    band1_cutoff_freq: torch.Tensor,
    band1_q_factor: torch.Tensor,
    band2_cutoff_freq: torch.Tensor,
    band2_q_factor: torch.Tensor,
    band3_cutoff_freq: torch.Tensor,
    band3_q_factor: torch.Tensor,
    band4_cutoff_freq: torch.Tensor,
    band4_q_factor: torch.Tensor,
    band5_cutoff_freq: torch.Tensor,
    band5_q_factor: torch.Tensor
):
    """Six-band Parametric Equalizer.

    Low-shelf -> Band 1 -> Band 2 -> Band 3 -> Band 4 -> High-shelf

    [1] Välimäki, Vesa, and Joshua D. Reiss.
        "All about audio equalization: Solutions and frontiers."
        Applied Sciences 6.5 (2016): 129.

    [2] Nercessian, Shahan.
        "Neural parametric equalizer matching using differentiable biquads."
        Proc. Int. Conf. Digital Audio Effects (eDAFx-20). 2020.

    [3] Colonel, Joseph T., Christian J. Steinmetz, et al.
        "Direct design of biquad filter cascades with deep learning by sampling random polynomials."
        IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022.

    [4] Steinmetz, Christian J., Nicholas J. Bryan, and Joshua D. Reiss.
        "Style Transfer of Audio Effects with Differentiable Signal Processing."
        Journal of the Audio Engineering Society. Vol. 70, Issue 9, 2022, pp. 708-721.

    Args:
        x (torch.Tensor): Time domain tensor with shape (bs, chs, seq_len)
        sample_rate (float): Audio sample rate.
        low_shelf_gain_db (torch.Tensor): Low-shelf filter gain in dB.
        low_shelf_cutoff_freq (torch.Tensor): Low-shelf filter cutoff frequency in Hz.
        low_shelf_q_factor (torch.Tensor): Low-shelf filter Q-factor.
        band0_gain_db (torch.Tensor): Band 1 filter gain in dB.
        band0_cutoff_freq (torch.Tensor): Band 1 filter cutoff frequency in Hz.
        band0_q_factor (torch.Tensor): Band 1 filter Q-factor.
        band1_gain_db (torch.Tensor): Band 2 filter gain in dB.
        band1_cutoff_freq (torch.Tensor): Band 2 filter cutoff frequency in Hz.
        band1_q_factor (torch.Tensor): Band 2 filter Q-factor.
        band2_gain_db (torch.Tensor): Band 3 filter gain in dB.
        band2_cutoff_freq (torch.Tensor): Band 3 filter cutoff frequency in Hz.
        band2_q_factor (torch.Tensor): Band 3 filter Q-factor.
        band3_gain_db (torch.Tensor): Band 4 filter gain in dB.
        band3_cutoff_freq (torch.Tensor): Band 4 filter cutoff frequency in Hz.
        band3_q_factor (torch.Tensor): Band 4 filter Q-factor.
        high_shelf_gain_db (torch.Tensor): High-shelf filter gain in dB.
        high_shelf_cutoff_freq (torch.Tensor): High-shelf filter cutoff frequency in Hz.
        high_shelf_q_factor (torch.Tensor): High-shelf filter Q-factor.

    Returns:
        y (torch.Tensor): Filtered signal.
    """
    bs, chs, seq_len = x.size()

    # reshape to move everything to batch dim
    # x = x.view(-1, 1, seq_len)
    band0_cutoff_freq = band0_cutoff_freq.view(-1, 1, 1)
    band0_q_factor = band0_q_factor.view(-1, 1, 1)
    band1_cutoff_freq = band1_cutoff_freq.view(-1, 1, 1)
    band1_q_factor = band1_q_factor.view(-1, 1, 1)
    band2_cutoff_freq = band2_cutoff_freq.view(-1, 1, 1)
    band2_q_factor = band2_q_factor.view(-1, 1, 1)
    band3_cutoff_freq = band3_cutoff_freq.view(-1, 1, 1)
    band3_q_factor = band3_q_factor.view(-1, 1, 1)
    band4_cutoff_freq = band4_cutoff_freq.view(-1, 1, 1)
    band4_q_factor = band4_q_factor.view(-1, 1, 1)
    band5_cutoff_freq = band5_cutoff_freq.view(-1, 1, 1)
    band5_q_factor = band5_q_factor.view(-1, 1, 1)

    eff_bs = x.size(0)

    # six second order sections
    sos = torch.zeros(eff_bs, 6, 6).type_as(band0_cutoff_freq)

    # ------------ band0 ------------
    b, a = signals.calculate_allpass_coefficients(
        band0_cutoff_freq,
        band0_q_factor,
        sample_rate,
    )
    sos[:, 0, :] = torch.cat((b, a), dim=-1)
    # ------------ band1 ------------
    b, a = signals.calculate_allpass_coefficients(
        band1_cutoff_freq,
        band1_q_factor,
        sample_rate,
    )
    sos[:, 1, :] = torch.cat((b, a), dim=-1)
    # ------------ band2 ------------
    b, a = signals.calculate_allpass_coefficients(
        band2_cutoff_freq,
        band2_q_factor,
        sample_rate,
    )
    sos[:, 2, :] = torch.cat((b, a), dim=-1)
    # ------------ band3 ------------
    b, a = signals.calculate_allpass_coefficients(
        band3_cutoff_freq,
        band3_q_factor,
        sample_rate,
    )
    sos[:, 3, :] = torch.cat((b, a), dim=-1)
    # ------------ band4 ------------
    b, a = signals.calculate_allpass_coefficients(
        band4_cutoff_freq,
        band4_q_factor,
        sample_rate,
    )
    sos[:, 4, :] = torch.cat((b, a), dim=-1)
    # ------------ band5 ------------
    b, a = signals.calculate_allpass_coefficients(
        band5_cutoff_freq,
        band5_q_factor,
        sample_rate,
    )
    sos[:, 5, :] = torch.cat((b, a), dim=-1)

    x_out = signals.sosfilt_via_fsm(sos, x)

    # move channels back
    x_out = x_out.view(bs, chs, seq_len)

    return x_out