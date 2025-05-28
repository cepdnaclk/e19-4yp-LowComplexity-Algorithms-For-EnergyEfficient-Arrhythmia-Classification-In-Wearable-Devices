# segment.py

import neurokit2 as nk
import numpy as np

def extract_heartbeats(signal, fs, annotation_rpeaks=None, before=0.25, after=0.4, fixed_length=250):
    """
    Extract fixed-length heartbeats centered at R-peaks from an ECG signal.

    Parameters:
    -----------
    signal : array-like
        Raw ECG signal (1D numpy array or list).
    fs : float
        Sampling frequency of the ECG signal in Hz.
    annotation_rpeaks : array-like or None, optional
        Pre-annotated R-peak indices (sample positions). If None, R-peaks are detected automatically.
    before : float, optional
        Time in seconds before the R-peak to include in the segment (default is 0.25s).
    after : float, optional
        Time in seconds after the R-peak to include in the segment (default is 0.4s).
    fixed_length : int, optional
        Fixed number of samples per heartbeat segment (default is 250).

    Returns:
    --------
    beats : np.ndarray
        Array of shape (n_beats, fixed_length) containing fixed-length heartbeat segments.
    valid_rpeaks : np.ndarray
        Array of sample indices corresponding to the R-peaks used for segmentation.

    Notes:
    ------
    - If `annotation_rpeaks` is not provided, R-peaks are detected using NeuroKit2's ecg_findpeaks.
    - Segments near the start or end of the signal that cannot be centered properly are padded or truncated.
    - The fixed_length is used to ensure uniform segment size for downstream processing.
    """
    # Clean the ECG signal for better peak detection
    cleaned_signal = nk.ecg_clean(signal, sampling_rate=fs)

    # Detect R-peaks if not provided
    if annotation_rpeaks is not None:
        rpeaks = annotation_rpeaks
    else:
        peaks_dict = nk.ecg_findpeaks(cleaned_signal, sampling_rate=fs)
        rpeaks = peaks_dict['ECG_R_Peaks']

    beats = []
    valid_rpeaks = []

    half_length = fixed_length // 2

    for peak in rpeaks:
        center = int(peak)
        start = center - half_length
        end = center + half_length

        # Handle edge cases at signal boundaries
        if start < 0:
            start = 0
            end = fixed_length
        elif end > len(signal):
            end = len(signal)
            start = end - fixed_length

        # Extract segment
        segment = signal[start:end]

        # Pad if segment is shorter than fixed_length (can happen near edges)
        if len(segment) < fixed_length:
            segment = np.pad(segment, (0, fixed_length - len(segment)), 'constant')

        beats.append(segment)
        valid_rpeaks.append(center)

    beats = np.array(beats)
    valid_rpeaks = np.array(valid_rpeaks)

    return beats, valid_rpeaks