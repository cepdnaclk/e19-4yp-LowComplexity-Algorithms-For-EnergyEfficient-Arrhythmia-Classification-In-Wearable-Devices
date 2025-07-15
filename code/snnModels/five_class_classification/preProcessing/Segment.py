# Contains the code for segmenting the ECG signal 

# NeuroKit2 is a Python toolbox for neurophysiological signal 
# processing (such as ECG, EDA, EMG, PPG, and more). It provides functions for: -> ECG processing (R-peak detection, 
# HRV analysis) -> EDA (electrodermal activity) analysis, -> Respiration signal processing

import neurokit2 as nk
import numpy as np
from scipy.signal import find_peaks
from Load import load_ecg
import pandas as pd

def extract_heartbeats(signal, fs, annotation_rpeaks=None, before=0.25, after=0.4, fixed_length=300):
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
        # start = center - 210
        # end = center + 90
        
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

# Below code is to determine the performances of Nerokit and Pantompkins

def count_true_beats(ann):
    """
    Count true beats from annotations, excluding non-beat symbols.
    
    Args:
        ann: WFDB annotation object
    Returns:
        int: Number of true beats
    """
    beat_symbols = ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q']
    if isinstance(ann.symbol, str):
        return 1 if ann.symbol in beat_symbols else 0
    return sum(1 for symbol in ann.symbol if symbol in beat_symbols)

def compare_rpeak_detection(data_dir):
    """
    Compare R-peak detection methods and true beat counts for all MIT-BIH records.
    
    Args:
        data_dir (str): Path to MIT-BIH data directory
    Returns:
        pandas.DataFrame: Table with record ID, true beats, peak counts, and error metrics
    """
    # List of all MIT-BIH record IDs
    record_ids = [
        '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
        '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
        '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
        '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
        '222', '223', '228', '230', '231', '232', '233', '234'
    ]
    
    results = []
    for record_id in record_ids:
        # Load ECG data
        result = load_ecg(record_id, data_dir)
        if result is None:
            print(f"Skipping record {record_id} due to loading error")
            results.append({
                'Record ID': record_id,
                'True Beats': 0,
                'NeuroKit2 Peaks': 0,
                'Pan-Tompkins Peaks': 0,
                'NeuroKit2 Error Rate (%)': 0.0,
                'Pan-Tompkins Error Rate (%)': 0.0,
                'NeuroKit2 Difference': 0,
                'Pan-Tompkins Difference': 0
            })
            continue
        
        signal, true_rpeaks, fs, ann = result
        
        # Detect R-peaks with NeuroKit2
        cleaned = nk.ecg_clean(signal, sampling_rate=fs)
        nk_rpeaks = nk.ecg_findpeaks(cleaned, sampling_rate=fs)['ECG_R_Peaks']
        
        # Detect R-peaks with Pan-Tompkins
        pt_rpeaks = pan_tompkins_rpeak_detection(signal, fs)
        
        # Count true beats from annotations
        true_beat_count = count_true_beats(ann)
        
        # Count detected peaks
        nk_peak_count = len(nk_rpeaks)
        pt_peak_count = len(pt_rpeaks)
        
        # Calculate error rates and differences
        nk_diff = abs(true_beat_count - nk_peak_count)
        pt_diff = abs(true_beat_count - pt_peak_count)
        nk_error_rate = (nk_diff / true_beat_count * 100) if true_beat_count > 0 else 0.0
        pt_error_rate = (pt_diff / true_beat_count * 100) if true_beat_count > 0 else 0.0
        
        # Store results
        results.append({
            'Record ID': record_id,
            'True Beats': true_beat_count,
            'NeuroKit2 Peaks': nk_peak_count,
            'Pan-Tompkins Peaks': pt_peak_count,
            'NeuroKit2 Error Rate (%)': round(nk_error_rate, 2),
            'Pan-Tompkins Error Rate (%)': round(pt_error_rate, 2),
            'NeuroKit2 Difference': nk_diff,
            'Pan-Tompkins Difference': pt_diff
        })
    
    # Create DataFrame for results
    df = pd.DataFrame(results)
    
    # Save to CSV
    output_file = 'rpeak_detection_comparison.csv'
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to '{output_file}'")
    
    return df

# Example usage
if __name__ == "__main__":
    data_dir = 'data/mitdb'  
    results_df = compare_rpeak_detection(data_dir)
    print("\nR-Peak Detection Comparison for All MIT-BIH Records:")
    print(results_df)