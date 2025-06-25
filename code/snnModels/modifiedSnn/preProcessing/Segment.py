# Contains the code for segmenting the ECG signal 

# NeuroKit2 is a Python toolbox for neurophysiological signal 
# processing (such as ECG, EDA, EMG, PPG, and more). It provides functions for: -> ECG processing (R-peak detection, 
# HRV analysis) -> EDA (electrodermal activity) analysis, -> Respiration signal processing

import neurokit2 as nk
import numpy as np
from scipy.signal import find_peaks

def extract_heartbeats(signal, fs, annotation_rpeaks=None, before=0.25, after=0.4, fixed_length=250):
    """
    Extract fixed-length heartbeats centered at R-peaks
    
    Args:
        signal: ECG signal
        fs: Sampling frequency (Hz)
        annotation_rpeaks: Optional pre-annotated R-peaks
        before: Seconds before R-peak (default 0.25)
        after: Seconds after R-peak (default 0.4)
        fixed_length: Target samples per beat (default 250)
        
    Returns:
        beats: Array of fixed-length beats (n_beats, fixed_length)
        valid_rpeaks: Array of used R-peak positions
    """
    # Clean signal and detect R-peaks using neurokit2 library
    cleaned = nk.ecg_clean(signal, sampling_rate=fs)
    rpeaks = annotation_rpeaks if annotation_rpeaks is not None else \
             nk.ecg_findpeaks(cleaned, sampling_rate=fs)['ECG_R_Peaks']
    
    # Detect R peaks using Pan-Tompkins algorithm
    # if annotation_rpeaks is not None :
    #     rpeaks = annotation_rpeaks
    # else:
    #     rpeaks = pan_tompkins_rpeak_detection(signal, fs)
    
    beats = []
    valid_rpeaks = []
    
    for peak in rpeaks:
        # Calculate fixed window around R-peak
        center = int(peak)
        start = center - fixed_length//2
        end = center + fixed_length//2
        
        # Handle edge cases
        if start < 0:
            start, end = 0, fixed_length
        elif end > len(signal):
            start, end = len(signal)-fixed_length, len(signal)
        
        beat = signal[start:end]
        
        # Ensure exact length (in case of sampling rate rounding)
        if len(beat) < fixed_length:
            beat = np.pad(beat, (0, fixed_length - len(beat)))
        elif len(beat) > fixed_length:
            beat = beat[:fixed_length]
        
        beats.append(beat)
        valid_rpeaks.append(peak)
    
    # Here sample length has taken as 250 samples, because R peak lenght is about 0.694 seconds
    # Returns beats -> An array of fixed length heart beats (Segments), valid_rpeaks -> An array containing the sample position 
    # of R peaks. Eg : valid_rpeaks = [10, 234, 565] 
    return np.array(beats), np.array(valid_rpeaks)


# Implementing Pan-Tompkins algorithm to detect R peak

def pan_tompkins_rpeak_detection(signal, fs):
    """
    Pan-Tompkins algorithm to detect R-peaks in ECG signal.
    Args:
        signal: preprocessed ECG signal (1D numpy array)
        fs: sampling frequency in Hz
    Returns:
        rpeaks: numpy array of detected R-peak sample indices
    """

    # 1. Derivative filter (5-point derivative)
    derivative_kernel = np.array([1, 2, 0, -2, -1]) * (1/(8/fs))
    differentiated = np.convolve(signal, derivative_kernel, mode='same')

    # 2. Squaring
    squared = differentiated ** 2

    # 3. Moving window integration (150 ms window)
    window_size = int(0.15 * fs)
    integrated = np.convolve(squared, np.ones(window_size)/window_size, mode='same')

    # 4. Peak detection with minimum distance of 200 ms (refractory period)
    min_distance = int(0.2 * fs)
    threshold = 0.5 * np.max(integrated)  
    peaks, _ = find_peaks(integrated, distance=min_distance, height=threshold)

    # 5. Refine R-peak locations: find max in original signal ±50 ms around detected peaks
    rpeaks = []
    search_radius = int(0.05 * fs)
    for peak in peaks:
        start = max(peak - search_radius, 0)
        end = min(peak + search_radius, len(signal))
        local_max = np.argmax(signal[start:end]) + start
        rpeaks.append(local_max)

    return np.array(rpeaks)
