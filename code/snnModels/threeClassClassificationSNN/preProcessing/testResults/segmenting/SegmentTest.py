import neurokit2 as nk
import numpy as np
from Load import load_ecg
import pandas as pd
from Denoise import bandpass_filter, notch_filter, remove_baseline
from Segment import pan_tompkins_rpeak_detection

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
        
        # apply denosing
        bp_signal = bandpass_filter(signal, fs);
        nt_signal = notch_filter(bp_signal, fs);
        br_signal = remove_baseline(nt_signal, fs);
        
        # Detect R-peaks with NeuroKit2
        cleaned = nk.ecg_clean(br_signal, sampling_rate=fs)
        nk_rpeaks = nk.ecg_findpeaks(cleaned, sampling_rate=fs)['ECG_R_Peaks']
        
        # Detect R-peaks with Pan-Tompkins
        pt_rpeaks = pan_tompkins_rpeak_detection(br_signal, fs)
        
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
