import numpy as np
from scipy.signal import blackmanharris
from numpy.fft import rfft, irfft

# Helper functions

def calc_db(input_signal, reference=1.0):
    '''
    Returns the signal in decibels.
    '''
    return 20 * np.log10(input_signal / reference)

def rms_flat(a):
    """
    Return the root mean square of all the elements of *a*, flattened out.
    """
    return np.sqrt(np.mean(np.absolute(a)**2))


def find_range(f, x):
    """
    Find range between nearest local minima from peak at index x
    """
    lowermin = 0
    uppermin = 0
    for i in np.arange(x+1, len(f)):
        if f[i+1] >= f[i]:
            uppermin = i
            break
    for i in np.arange(x-1, 0, -1):
        if f[i] <= f[i-1]:
            lowermin = i + 1
            break
    return (lowermin, uppermin)

# Quality evaluation metrics
# Used to compare the input signal with the output signal to check
# if there's any artifacts introduced by the filter

def snr(input_signal, axis=0, ddof=0):
    """
    Signal to Noise Ratio (SNR)
    """
    a = np.asanyarray(input_signal)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return 20*np.log10(abs(np.where(sd == 0, 0, m/sd)))

def thdn(input_signal, samplerate=44_100):
    """
    Total Harmonic Distortion (THD)

    Adapted from https://gist.github.com/endolith/246092
    """
    # Get rid of DC and window the signal
    input_signal -= np.mean(input_signal)
    windowed = input_signal * blackmanharris(len(input_signal))

    # Measure the total signal before filtering but after windowing
    total_rms = rms_flat(windowed)

    # Find the peak of the frequency spectrum (fundamental frequency), and
    # filter the signal by throwing away values between the nearest local
    # minima
    f = rfft(windowed)
    i = np.argmax(abs(f))

    # Not exact
    lowermin, uppermin = find_range(abs(f), i)
    f[lowermin: uppermin] = 0

    # Transform noise back into the signal domain and measure it
    noise = irfft(f)
    THDN = rms_flat(noise) / total_rms

    return (THDN*100)

# Loudness evaluation metrics
# Used to compare the loudness of the input signal with the output signal

def db_rms(input_signal):
    '''
    Returns the root mean square (RMS) level of the signal in decibels.
    '''
    # Calculate the root mean square (RMS) level of the signal
    rms = rms_flat(input_signal)

    # Calculate decibel level
    return calc_db(rms)

def dB_peak(input_signal):
    """
    Peak loudness
    """
    return calc_db(np.max(np.abs(input_signal)))

# Loss evaluation metrics

def mse_loss(input_signal, output_signal):
    '''
    Mean Squared Error (MSE) loss function.
    '''
    return np.mean((input_signal - output_signal) ** 2)