import numpy as np
from Labels import create_labels
from Load import load_ecg   
from Denoise import bandpass_filter, notch_filter, remove_baseline
from Segment import extract_heartbeats
import pandas as pd 

AAMI_classes = {
    0: ['N', 'L', 'R', 'e', 'j'],      
    1: ['A', 'a', 'J', 'S', 'V', 'E', 'F'],          
    2: ['F', 'P', '/', 'f', 'u']          
}

def test_label_creation(record_id, data_dir):
    """
    Test the create_labels function by counting labels per class and checking for unmapped symbols.
    Args:
        record_id: MIT-BIH record ID
        data_dir: Path to MIT-BIH data directory
    Returns:
        dict: Test results including label counts, unmapped symbols, and issues
    """
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
    
    # Create labels
    labels = create_labels(valid_rpeaks, ann)
    
    # Count labels per class
    class_counts = {
        0: np.sum(labels == 0),  # Normal
        1: np.sum(labels == 1),  # Supraventricular/Ventricular ectopic
        2: np.sum(labels == 2)   # Ventricular fusion/Paced/Other
    }
    
    # Check for unmapped symbols (assigned to class 2)
    unmapped_symbols = []
    valid_symbols = set(sum(AAMI_classes.values(), []))  # All valid AAMI symbols
    for peak, symbol in zip(valid_rpeaks, ann.symbol):
        if symbol not in valid_symbols:
            unmapped_symbols.append(symbol)
    
    # Test results
    test_results = {
        'Record ID': record_id,
        'Total Beats': len(beats),
        'Class 0 Count (Normal)': class_counts[0],
        'Class 1 Count (SVEB/VEB)': class_counts[1],
        'Class 2 Count (Fusion/Paced/Other)': class_counts[2],
        'Unmapped Symbols': list(set(unmapped_symbols)),
        'Issues': []
    }
    
    # Check for mismatches between beats and labels
    if len(beats) != len(labels):
        test_results['Issues'].append(f"Mismatch: {len(beats)} beats vs {len(labels)} labels")
    
    # Check if all beats have valid labels
    if np.any(labels < 0) or np.any(labels > 2):
        test_results['Issues'].append("Invalid class labels detected (outside [0, 2])")
    
    # Check for unexpected unmapped symbols
    if unmapped_symbols:
        test_results['Issues'].append(f"Unmapped symbols found: {set(unmapped_symbols)}")
    
    # Check for class imbalance
    total_labels = sum(class_counts.values())
    if total_labels > 0:
        class_ratios = {k: v/total_labels for k, v in class_counts.items()}
        if any(ratio > 0.9 for ratio in class_ratios.values()):
            test_results['Issues'].append("Severe class imbalance detected")
    
    print(f"Record {record_id}: Label Test - "
          f"Class 0: {class_counts[0]}, Class 1: {class_counts[1]}, Class 2: {class_counts[2]}, "
          f"Unmapped Symbols: {test_results['Unmapped Symbols']}, "
          f"Issues: {test_results['Issues'] if test_results['Issues'] else 'None'}")
    
    return test_results

if __name__ == "__main__":
    data_dir = 'data/mitdb'
    record_ids = [
        '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
        '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
        '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
        '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
        '222', '223', '228', '230', '231', '232', '233', '234'
    ]
    all_results = []
    
    for record_id in record_ids:
        result = test_label_creation(record_id, data_dir)
        all_results.append(result)
    
    # Save test results to CSV
    df = pd.DataFrame(all_results)
    output_file = 'label_creation_test_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\nLabel creation test results saved to '{output_file}'")