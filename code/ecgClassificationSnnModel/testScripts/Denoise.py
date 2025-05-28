# denoise.py
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

def bandpass_filter(signal, fs, lowcut=0.5, highcut=40):
    """
    Apply Butterworth bandpass filter to remove low and high frequency noise.
    
    Args:
        signal (np.array): Raw ECG signal
        fs (float): Sampling frequency in Hz
        lowcut (float): Lower cutoff frequency (Hz)
        highcut (float): Upper cutoff frequency (Hz)
        
    Returns:
        np.array: Bandpass-filtered signal
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, signal)

def notch_filter(signal, fs, freq=50, Q=30):
    """
    Remove power line interference using IIR notch filter.
    
    Args:
        signal (np.array): Input ECG signal
        fs (float): Sampling frequency in Hz
        freq (float): Frequency to remove (typically 50/60 Hz)
        Q (float): Quality factor controlling bandwidth
        
    Returns:
        np.array: Notch-filtered signal
    """
    nyq = 0.5 * fs
    freq_normalized = freq / nyq
    b, a = iirnotch(freq_normalized, Q)
    return filtfilt(b, a, signal)

def remove_baseline(signal, fs, window_size=0.2):
    """
    Remove baseline wander using moving average subtraction.
    
    Args:
        signal (np.array): Input ECG signal
        fs (float): Sampling frequency in Hz
        window_size (float): Window size in seconds for moving average
        
    Returns:
        np.array: Baseline-corrected signal
    """
    window_samples = int(window_size * fs)
    baseline = np.convolve(signal, np.ones(window_samples)/window_samples, mode='same')
    return signal - baseline
