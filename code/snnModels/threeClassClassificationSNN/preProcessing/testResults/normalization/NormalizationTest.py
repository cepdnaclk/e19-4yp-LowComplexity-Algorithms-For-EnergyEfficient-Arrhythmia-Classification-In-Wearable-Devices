import numpy as np
from Normalization import normalize_beats
from Denoise import bandpass_filter, notch_filter, remove_baseline
from Segment import extract_heartbeats
from Load import load_ecg
import pandas as pd

def test_normalization(beats, record_id):
    """
    Test the quality of normalized ECG beats.
    Args:
        beats: Array of normalized beats (n_beats, fixed_length)
        record_id: MIT-BIH record ID
    Returns:
        dict: Test results including range check, statistics, and issues
    """
    test_results = {
        'Record ID': record_id,
        'Within Range [-1, 1]': True,
        'Mean Absolute Value': 0.0,
        'Std Dev': 0.0,
        'Min Value': 0.0,
        'Max Value': 0.0,
        'Issues': []
    }
    
    # Check if all values are within [-1, 1]
    if np.any(beats < -1) or np.any(beats > 1):
        test_results['Within Range [-1, 1]'] = False
        test_results['Issues'].append("Values outside [-1, 1]")
    
    # Calculate statistics across all beats
    test_results['Min Value'] = np.min(beats)
    test_results['Max Value'] = np.max(beats)
    test_results['Mean Absolute Value'] = np.mean(np.abs(beats))
    test_results['Std Dev'] = np.std(beats)
    
    # Check for low standard deviation (indicating potential signal compression)
    if test_results['Std Dev'] < 0.1:
        test_results['Issues'].append("Low standard deviation, possible signal compression")
    
    # Check for mean far from zero (indicating potential bias)
    if abs(test_results['Mean Absolute Value']) > 0.5:
        test_results['Issues'].append("Mean far from zero, potential normalization bias")
    
    # Check for zero or near-zero beats (indicating potential normalization failure)
    if np.any(np.all(np.abs(beats) < 1e-6, axis=1)):
        test_results['Issues'].append("Zero or near-zero beats detected")
    
    return test_results

def process_and_test_record(record_id, data_dir):
    """
    Process an ECG record and test normalization.
    Args:
        record_id: MIT-BIH record ID
        data_dir: Path to MIT-BIH data directory
    Returns:
        dict: Results including normalized beats, valid rpeaks, and test results
    """
    signal, rpeaks, fs, ann = load_ecg(record_id, data_dir)
    print(f"Record {record_id}: Total annotations: {len(ann.sample)}")
    
    signal = bandpass_filter(signal, fs)
    signal = notch_filter(signal, fs)
    signal = remove_baseline(signal, fs)
    
    beats, valid_rpeaks = extract_heartbeats(signal, fs, ann.sample)
    print(f"Record {record_id}: Extracted {len(beats)} valid beats")
    
    normalized_beats = normalize_beats(beats)
    
    # Test normalization
    test_results = test_normalization(normalized_beats, record_id)
    print(f"Record {record_id}: Normalization Test - "
          f"Within Range: {test_results['Within Range [-1, 1]']}, "
          f"Mean: {test_results['Mean Absolute Value']:.4f}, "
          f"Std: {test_results['Std Dev']:.4f}, "
          f"Min: {test_results['Min Value']:.4f}, "
          f"Max: {test_results['Max Value']:.4f}, "
          f"Issues: {test_results['Issues'] if test_results['Issues'] else 'None'}")
    
    return {
        'normalized_beats': normalized_beats,
        'valid_rpeaks': valid_rpeaks,
        'test_results': test_results
    }

if __name__ == "__main__":
    data_dir = 'data/mitdb'
    # Test on a subset of MIT-BIH records, including problematic ones
    record_ids =  [
        '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
        '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
        '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
        '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
        '222', '223', '228', '230', '231', '232', '233', '234'
    ]
    all_results = []
    
    for record_id in record_ids:
        result = process_and_test_record(record_id, data_dir)
        all_results.append(result['test_results'])
    
    # Save test results to CSV
    df = pd.DataFrame(all_results)
    output_file = 'normalization_test_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\nNormalization test results saved to '{output_file}'")