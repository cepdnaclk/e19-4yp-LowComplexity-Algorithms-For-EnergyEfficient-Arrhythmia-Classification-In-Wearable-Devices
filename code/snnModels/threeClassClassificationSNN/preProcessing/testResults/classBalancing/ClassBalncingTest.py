import numpy as np
from collections import Counter
import pandas as pd
from ClassBalancing import balance_classes
from Load import load_ecg
from Denoise import bandpass_filter, notch_filter, remove_baseline
from Segment import extract_heartbeats
from Normalization import normalize_beats
from Labels import create_labels

def extract_all_segments(record_ids, data_dir):
    """
    Extract beats and labels from all records.
    Args:
        record_ids: List of MIT-BIH record IDs
        data_dir: Path to MIT-BIH data directory
    Returns:
        tuple: (all_beats, all_labels, record_mapping)
            - all_beats: Array of all normalized beats
            - all_labels: Array of all labels
            - record_mapping: List of (record_id, start_idx, end_idx) for each record
    """
    all_beats = []
    all_labels = []
    record_mapping = []
    current_idx = 0
    
    for record_id in record_ids:
        # Load ECG data
        signal, true_rpeaks, fs, ann = load_ecg(record_id, data_dir)
        print(f"Record {record_id}: Total annotations: {len(ann.sample)}")
        
        # Apply filtering
        signal = bandpass_filter(signal, fs)
        signal = notch_filter(signal, fs)
        signal = remove_baseline(signal, fs)
        
        # Extract heartbeats
        beats, valid_rpeaks = extract_heartbeats(signal, fs, ann.sample)
        print(f"Record {record_id}: Extracted {len(beats)} valid beats")
        
        # Normalize beats
        beats = normalize_beats(beats)
        
        # Create labels
        labels = create_labels(valid_rpeaks, ann)
        
        # Store beats and labels
        all_beats.append(beats)
        all_labels.append(labels)
        
        # Record mapping (start and end indices for this record's beats)
        record_mapping.append((record_id, current_idx, current_idx + len(beats)))
        current_idx += len(beats)
    
    # Concatenate all beats and labels
    all_beats = np.vstack(all_beats) if all_beats else np.array([])
    all_labels = np.hstack(all_labels) if all_labels else np.array([])
    
    return all_beats, all_labels, record_mapping

def test_global_class_balancing(record_ids, data_dir):
    """
    Test class balancing on all segments from all records.
    Args:
        record_ids: List of MIT-BIH record IDs
        data_dir: Path to MIT-BIH data directory
    Returns:
        dict: Test results including global and per-record class counts, balancing method, and issues
    """
    # Extract all segments
    all_beats, all_labels, record_mapping = extract_all_segments(record_ids, data_dir)
    
    # Count global class distribution before balancing
    global_counts_before = Counter(all_labels)
    
    # Balance classes globally
    balanced_beats, balanced_labels = balance_classes(all_beats, all_labels)
    
    # Count global class distribution after balancing
    global_counts_after = Counter(balanced_labels)
    
    # Count per-record class distribution after balancing
    per_record_counts = []
    for record_id, start_idx, end_idx in record_mapping:
        # Get original labels for this record
        original_labels = all_labels[start_idx:end_idx]
        original_counts = Counter(original_labels)
        
        # Estimate post-balancing counts (approximate, as SMOTE may mix samples)
        # For simplicity, we assume balanced labels are distributed proportionally
        record_beats = all_beats[start_idx:end_idx]
        if len(record_beats) == 0:
            per_record_counts.append({
                'Record ID': record_id,
                'Class 0 Count Before': original_counts.get(0, 0),
                'Class 1 Count Before': original_counts.get(1, 0),
                'Class 2 Count Before': original_counts.get(2, 0),
                'Class 0 Count After': 0,
                'Class 1 Count After': 0,
                'Class 2 Count After': 0
            })
            continue
        
        # Approximate post-balancing counts based on global proportions
        total_original = sum(original_counts.values())
        total_balanced = sum(global_counts_after.values())
        if total_balanced > 0:
            scale_factor = len(balanced_beats) / len(all_beats) * (len(record_beats) / total_original)
            post_counts = {k: int(v * scale_factor) for k, v in global_counts_after.items()}
        else:
            post_counts = {0: 0, 1: 0, 2: 0}
        
        per_record_counts.append({
            'Record ID': record_id,
            'Class 0 Count Before': original_counts.get(0, 0),
            'Class 1 Count Before': original_counts.get(1, 0),
            'Class 2 Count Before': original_counts.get(2, 0),
            'Class 0 Count After': post_counts.get(0, 0),
            'Class 1 Count After': post_counts.get(1, 0),
            'Class 2 Count After': post_counts.get(2, 0)
        })
    
    # Test results
    test_results = {
        'Total Beats Before': len(all_beats),
        'Global Class 0 Count Before': global_counts_before.get(0, 0),
        'Global Class 1 Count Before': global_counts_before.get(1, 0),
        'Global Class 2 Count Before': global_counts_before.get(2, 0),
        'Total Beats After': len(balanced_beats),
        'Global Class 0 Count After': global_counts_after.get(0, 0),
        'Global Class 1 Count After': global_counts_after.get(1, 0),
        'Global Class 2 Count After': global_counts_after.get(2, 0),
        'Balancing Method': 'RandomOverSampler' if min(global_counts_before.values()) < 2 else 'SMOTE',
        'Issues': []
    }
    
    # Check for valid labels
    if np.any(balanced_labels < 0) or np.any(balanced_labels > 2):
        test_results['Issues'].append("Invalid class labels detected (outside [0, 2])")
    
    # Check if all classes are present after balancing
    missing_classes = [class_idx for class_idx in range(3) if global_counts_after.get(class_idx, 0) == 0]
    if missing_classes:
        test_results['Issues'].append(f"Missing classes after balancing: {missing_classes}")
    
    # Check if balancing achieved equal or near-equal counts
    max_count = max(global_counts_after.values())
    min_count = min(global_counts_after.values())
    if max_count > 0 and (max_count - min_count) / max_count > 0.1:  # Allow 10% deviation
        test_results['Issues'].append("Classes not sufficiently balanced globally")
    
    # Check data integrity (beat shape)
    expected_shape = all_beats.shape[1] if len(all_beats.shape) > 1 else all_beats.shape[0]
    if len(balanced_beats) > 0 and balanced_beats.shape[1] != expected_shape:
        test_results['Issues'].append("Balanced beats have incorrect shape")
    
    # Check for empty output
    if len(balanced_beats) == 0 or len(balanced_labels) == 0:
        test_results['Issues'].append("Empty output after balancing")
    
    print(f"Global Balancing Test - "
          f"Before: {global_counts_before}, After: {global_counts_after}, "
          f"Method: {test_results['Balancing Method']}, "
          f"Issues: {test_results['Issues'] if test_results['Issues'] else 'None'}")
    
    # Combine global and per-record results
    global_results = [test_results]
    results_df = pd.DataFrame(per_record_counts)
    global_df = pd.DataFrame(global_results)
    
    return global_results, per_record_counts

if __name__ == "__main__":
    data_dir = 'data/mitdb'
    record_ids = [
        '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
        '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
        '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
        '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
        '222', '223', '228', '230', '231', '232', '233', '234'
    ]
    
    global_results, per_record_counts = test_global_class_balancing(record_ids, data_dir)
    
    # Save results to CSV
    global_df = pd.DataFrame(global_results)
    per_record_df = pd.DataFrame(per_record_counts)
    global_df.to_csv('global_class_balancing_test_results.csv', index=False)
    per_record_df.to_csv('per_record_class_balancing_test_results.csv', index=False)
    print(f"\nGlobal balancing test results saved to 'global_class_balancing_test_results.csv'")
    print(f"Per-record balancing test results saved to 'per_record_class_balancing_test_results.csv'")