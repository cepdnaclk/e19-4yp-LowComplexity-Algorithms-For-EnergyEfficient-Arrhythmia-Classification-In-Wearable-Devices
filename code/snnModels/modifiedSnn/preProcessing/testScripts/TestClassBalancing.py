# This code contains the code for testing the class balancing with the SMOTE method

import numpy as np
from preProcessing.Denoise import bandpass_filter, notch_filter, remove_baseline
from preProcessing.Segment import extract_heartbeats
from preProcessing.Normalization import normalize_beats
from preProcessing.ClassBalancing import balance_classes
from preProcessing.Load import load_ecg
from preProcessing.Labels import create_labels
from preProcessing.Labels import AAMI_classes
from collections import Counter

def process_record_for_class_balance_test(record_id, data_dir, apply_balancing=True):
    
    signal, rpeaks, fs, ann = load_ecg(record_id, data_dir)
    print(f"Total annotations : {len(ann.sample)}")
    
    signal = bandpass_filter(signal, fs)
    signal = notch_filter(signal, fs)
    signal = remove_baseline(signal, fs)
    
    beats, valid_rpeaks = extract_heartbeats(signal, fs, ann.sample)
    print(f"Extracted {len(beats)} valid beats")
    
    beats = normalize_beats(beats)
    labels = create_labels(valid_rpeaks, ann)
    
    # Count the number of samples per class
    label_counts = Counter(labels)
    print("Labels Counts : ", len(label_counts))
    
    print(f"\nClass distribution for record {record_id}:")
    for class_label, count in label_counts.items():
        print(f"{class_label}\t{count} samples")
    
    # Apply class balancing only if more than one class exists and apply_balancing is True
    unique_classes = len(label_counts)
    if apply_balancing and unique_classes > 1:
        print("\nApplying class balancing using SMOTE/RandomOverSampler...")
        beats_balanced, labels_balanced = balance_classes(beats, labels)
        
        # Count the number of samples per class after balancing
        balanced_label_counts = Counter(labels_balanced)
        print(f"\nClass distribution after balancing for record {record_id}:")
        for class_label, count in balanced_label_counts.items():
            print(f"{class_label}\t{count} samples")
        
        return beats_balanced, labels_balanced
    else:
        if unique_classes <= 1:
            print("\nOnly one class present in the record. Skipping class balancing.")
        return beats, labels

if __name__ == "__main__":
    data_dir = 'data/mitdb'
    record_id = 100
    
    beats, labels = process_record_for_class_balance_test(record_id, data_dir, apply_balancing=True)
    print("length of beats : ", len(beats))
