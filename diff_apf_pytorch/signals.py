import torch


def fft_freqz(b, a, n_fft: int = 512):
    B = torch.fft.rfft(b, n_fft)
    A = torch.fft.rfft(a, n_fft)
    H = B / A
    return H


def fft_sosfreqz(sos: torch.Tensor, n_fft: int = 512):
    """Compute the complex frequency response via FFT of cascade of biquads

    Args:
        sos (torch.Tensor): Second order filter sections with shape (bs, n_sections, 6)
        n_fft (int): FFT size. Default: 512
    Returns:
        H (torch.Tensor): Overall complex frequency response with shape (bs, n_bins)
    """
    bs, n_sections, n_coeffs = sos.size()
    assert n_coeffs == 6  # must be second order
    for section_idx in range(n_sections):
        b = sos[:, section_idx, :3]
        a = sos[:, section_idx, 3:]
        if section_idx == 0:
            H = fft_freqz(b, a, n_fft=n_fft)
        else:
            H *= fft_freqz(b, a, n_fft=n_fft)
    return H


def freqdomain_fir(x, H, n_fft):
    X = torch.fft.rfft(x, n_fft)
    Y = X * H.type_as(X)
    y = torch.fft.irfft(Y, n_fft)
    return y


def sosfilt_via_fsm(sos: torch.Tensor, x: torch.Tensor):
    """Use the frequency sampling method to approximate a cascade of second order IIR filters.

    The filter will be applied along the final dimension of x.
    Parameters:
        sos (torch.Tensor): Tensor of coefficients with shape (bs, n_sections, 6).
        x (torch.Tensor): Time domain signal with shape (bs, ... , timesteps)

    Returns:
        y (torch.Tensor): Filtered time domain signal with shape (bs, ..., timesteps)
    """
    bs = x.size(0)

    # round up to nearest power of 2 for FFT
    n_fft = 2 ** torch.ceil(torch.log2(torch.tensor(x.shape[-1] + x.shape[-1] - 1)))
    n_fft = n_fft.int()

    # compute complex response as ratio of polynomials
    H = fft_sosfreqz(sos, n_fft=n_fft)

    # add extra dims to broadcast filter across
    for _ in range(x.ndim - 2):
        H = H.unsqueeze(1)

    # apply as a FIR filter in the frequency domain
    y = freqdomain_fir(x, H, n_fft)

    # crop
    y = y[..., : x.shape[-1]]

    return y


def calculate_allpass_coefficients(cutoff_freq: torch.Tensor,
                                    q_factor: torch.Tensor,
                                    sample_rate: float,):
    '''
    CALCULATE COEFFICIENTS Outputs IIR coeffs b and a for a standard all-pass filter band.

    Parameters:
    - cutoff_freq: Center frequency of the all-pass filter
                    (can range from 1Hz to nyquist fs/2)
    - q_factor: Quality factor, which determines the shape of the filter
                at the center frequency (q can range from 0.1 to 40)
    - sample_rate: Sampling rate of the audio to apply the filter on

    Returns:
    - b: Numerator coefficients of the all-pass filter
    - a: Denominator coefficients of the all-pass filter
    '''

    # Reshape params
    bs = cutoff_freq.size(0)
    cutoff_freq = cutoff_freq.view(bs, -1)
    q_factor = q_factor.view(bs, -1)

    omega = 2 * torch.pi * cutoff_freq / sample_rate
    alpha = torch.sin(omega) / (2 * q_factor)

    b0 = 1 - alpha
    b1 = -2 * torch.cos(omega)
    b2 = 1 + alpha
    a0 = 1 + alpha
    a1 = -2 * torch.cos(omega)
    a2 = 1 - alpha

    b = torch.stack([b0, b1, b2], dim=1).view(bs, -1)
    a = torch.stack([a0, a1, a2], dim=1).view(bs, -1)

    # normalize
    b = b.type_as(cutoff_freq) / a0
    a = a.type_as(cutoff_freq) / a0

    return b, a