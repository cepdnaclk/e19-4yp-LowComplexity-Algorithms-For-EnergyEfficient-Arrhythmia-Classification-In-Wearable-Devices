import numpy as np
from collections import Counter
from preProcessing.Denoise import bandpass_filter, notch_filter, remove_baseline
from preProcessing.Segment import extract_heartbeats
from preProcessing.Normalization import normalize_beats
from preProcessing.ClassBalancing import balance_classes
from preProcessing.Load import load_ecg
from preProcessing.Labels import create_labels
from snnModel.Train import train_model
from snnModel.Evaluate import evaluate_model
from snnModel.DeltaModulation import delta_modulation
from preProcessing.Labels import AAMI_classes
import torch

def process_record(record_id, data_dir):
    signal, rpeaks, fs, ann = load_ecg(record_id, data_dir)
    print(f"Record {record_id}: Total annotations: {len(ann.sample)}")
    
    signal = bandpass_filter(signal, fs)
    signal = notch_filter(signal, fs)
    signal = remove_baseline(signal, fs)
    
    beats, valid_rpeaks = extract_heartbeats(signal, fs, ann.sample)
    print(f"Record {record_id}: Extracted {len(beats)} valid beats")
    
    beats = normalize_beats(beats)
    labels = create_labels(valid_rpeaks, ann)
    beats_spikes = delta_modulation(beats)
    
    if len(labels) != len(beats_spikes):
        print(f"Warning: Label count ({len(labels)}) != beats count ({len(beats_spikes)}) for record {record_id}. Filtering...")
        labeled_beats = []
        labeled_valid_rpeaks = []
        labeled_labels = []
        for i, rpeak in enumerate(valid_rpeaks):
            idx = np.where(ann.sample == rpeak)[0]
            if len(idx) > 0:
                symbol = ann.symbol[idx[0]]
                if symbol in AAMI_classes:
                    labeled_beats.append(beats_spikes[i])
                    labeled_valid_rpeaks.append(rpeak)
                    labeled_labels.append(AAMI_classes[symbol])
        beats_spikes = np.array(labeled_beats)
        valid_rpeaks = np.array(labeled_valid_rpeaks)
        labels = np.array(labeled_labels)
        print(f"After filtering: {len(beats_spikes)} beats, {len(labels)} labels")
    
    if len(beats_spikes) == 0:
        print(f"No valid beats with labels for record {record_id}. Skipping.")
        return np.array([]), np.array([])
    
    return beats_spikes, labels

def extract_all_beats_labels(record_ids, data_dir):
    all_beats = []
    all_labels = []
    for record_id in record_ids:
        X, y = process_record(str(record_id), data_dir)
        if X.shape[0] > 0:
            all_beats.append(X)
            all_labels.append(y)
        else:
            print(f"Skipping record {record_id} due to no valid data.")
    if all_beats:
        X_all = np.concatenate(all_beats, axis=0)
        y_all = np.concatenate(all_labels, axis=0)
        print(f"Extracted total {X_all.shape[0]} beats from {len(record_ids)} records.")
        return X_all, y_all
    else:
        return np.array([]), np.array([])

def balance_dataset(X, y):
    unique_classes = np.unique(y)
    if len(unique_classes) > 1:
        try:
            X_balanced, y_balanced = balance_classes(X, y)
            print(f"Balanced dataset: original {len(X)}, balanced {len(X_balanced)}")
            return X_balanced, y_balanced
        except ValueError as e:
            print(f"Balancing error: {e}. Using original data.")
            return X, y
    else:
        print(f"Only one class ({unique_classes[0]}) present. Skipping balancing.")
        return X, y

if __name__ == "__main__":
    data_dir = 'data/mitdb'

    DS1_train = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205]
    DS1_val = [207, 208, 209, 215, 220]
    DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

    # Load and balance training data
    X_train, y_train = extract_all_beats_labels(DS1_train, data_dir)
    X_train, y_train = balance_dataset(X_train, y_train)

    # Load validation and test data (no balancing)
    X_val, y_val = extract_all_beats_labels(DS1_val, data_dir)
    X_test, y_test = extract_all_beats_labels(DS2, data_dir)

    print("\n ---------------Getting lengths of dataset--------------------")
    print("Total number of training beats : ", len(X_train))
    print("Total number of training labels : ", len(y_train))
    print("Total number of validation beats : ", len(X_val))
    print("Total number of validation labels : ", len(y_val))
    print("Total number of test beats : ", len(X_test))
    print("Total number of test labels : ", len(y_test))

    if X_train.shape[0] > 0 and X_val.shape[0] > 0 and X_test.shape[0] > 0:
        print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}, Test samples: {X_test.shape[0]}")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        model, history = train_model(
            X_train, y_train, X_val, y_val, X_test, y_test,
            batch_size=64, num_epochs=10, device=device
        )
        evaluate_model(model, X_val, y_val, X_test, y_test, device=device)
    else:
        print("Insufficient data loaded for training, validation, or testing.")
