#This code contains the denoising of ECG signals

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

# Implementing a band-pass filter to remove frequencies out side the tyoical ECG range
def bandpass_filter(signal, fs, lowcut=0.5, highcut=40):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, signal)

# Removes power line interference (typically 50Hz or 60Hz)
def notch_filter(signal, fs, freq=50, Q=30):
    nyq = 0.5 * fs
    freq = freq / nyq
    b, a = iirnotch(freq, Q)
    return filtfilt(b, a, signal)

# Removes baseline wander using a moving average approach
def remove_baseline(signal, fs, window_size=0.2):
    window_samples = int(window_size * fs)
    baseline = np.convolve(signal, np.ones(window_samples)/window_samples, mode='same')
    return signal - baseline