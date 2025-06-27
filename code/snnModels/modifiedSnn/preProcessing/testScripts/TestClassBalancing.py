import numpy as np
from preProcessing.Denoise import bandpass_filter, notch_filter, remove_baseline
from preProcessing.Segment import extract_heartbeats
from preProcessing.Normalization import normalize_beats
from preProcessing.ClassBalancing import balance_classes
from preProcessing.Load import load_ecg
from preProcessing.Labels import create_labels
from collections import Counter

def extract_beats_and_labels(record_id, data_dir, verbose=True):
    signal, rpeaks, fs, ann = load_ecg(record_id, data_dir)
    if verbose:
        print(f"Record {record_id}: Total annotations : {len(ann.sample)}")
    
    signal = bandpass_filter(signal, fs)
    signal = notch_filter(signal, fs)
    signal = remove_baseline(signal, fs)
    
    beats, valid_rpeaks = extract_heartbeats(signal, fs, ann.sample)
    if verbose:
        print(f"Record {record_id}: Extracted {len(beats)} valid beats")
    
    beats = normalize_beats(beats)
    labels = create_labels(valid_rpeaks, ann)
    
    return beats, labels

def sum_samples_per_class(labels, num_classes=5):
    labels_array = np.array(labels)
    counts = {}
    for cls in range(num_classes):
        counts[cls] = np.count_nonzero(labels_array == cls)
    return counts

def process_multiple_records_combined(record_list, data_dir, apply_balancing=True):
    all_beats = []
    all_labels = []
    
    # Step 1: Extract beats and labels from all records and accumulate
    for record_id in record_list:
        print(f"\n{'='*40}\nProcessing record {record_id}\n{'='*40}")
        beats, labels = extract_beats_and_labels(record_id, data_dir, verbose=True)
        
        # Print class distribution for this record
        label_counts = Counter(labels)
        print(f"\nClass distribution for record {record_id}:")
        for class_label, count in label_counts.items():
            print(f"{class_label}\t{count} samples")
        
        all_beats.append(beats)
        all_labels.extend(labels)
    
    # Concatenate all beats into a single numpy array
    all_beats = np.concatenate(all_beats, axis=0)
    
    print(f"\nTotal beats extracted from all records: {len(all_beats)}")
    
    # Step 2: Print class distribution before balancing
    label_counts_before = Counter(all_labels)
    print("\nClass distribution before balancing (combined):")
    for class_label, count in label_counts_before.items():
        print(f"{class_label}\t{count} samples")
    
    # Step 3: Apply balancing on combined data if requested
    if apply_balancing:
        print("\nApplying class balancing using SMOTE on combined dataset...")
        beats_balanced, labels_balanced = balance_classes(all_beats, all_labels)
        
        label_counts_after = Counter(labels_balanced)
        print("\nClass distribution after balancing (combined):")
        for class_label, count in label_counts_after.items():
            print(f"{class_label}\t{count} samples")
        
        return beats_balanced, labels_balanced
    else:
        return all_beats, all_labels


if __name__ == "__main__":
    data_dir = 'data/mitdb'
    
    DS1_train = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220]
    # DS1_train = [101, 106, 108]
    DS2_test = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
    
    print("\nProcessing DS1_train records combined")
    beats_balanced, labels_balanced = process_multiple_records_combined(DS1_train, data_dir, apply_balancing=True)
    
    # print("\nProcessing DS2_test records")
    # process_multiple_records(DS2_test, data_dir, apply_balancing=True)
