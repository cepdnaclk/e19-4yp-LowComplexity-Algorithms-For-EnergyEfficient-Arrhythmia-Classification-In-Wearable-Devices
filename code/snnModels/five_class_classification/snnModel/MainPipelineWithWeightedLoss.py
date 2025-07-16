# Try " python -m models.MainPipeLine to run the " to run the script 

import numpy as np
from preProcessing.Denoise import bandpass_filter, notch_filter, remove_baseline
from preProcessing.Segment import extract_heartbeats
from preProcessing.ClassBalancing import balance_classes
from preProcessing.Normalization import normalize_beats
from preProcessing.Load import load_ecg
from preProcessing.Labels import create_labels
from snnModel.Train import train_model
from snnModel.Evaluate import evaluate_model
from snnModel.DeltaModulation import delta_modulation
from preProcessing.Labels import AAMI_classes
import torch
from sklearn.utils.class_weight import compute_class_weight

def process_record(record_id, data_dir, balance=True):
    
    signal, rpeaks, fs, ann = load_ecg(record_id, data_dir)
    print(f"Total annotations: {len(ann.sample)}")
    
    signal = bandpass_filter(signal, fs)
    signal = notch_filter(signal, fs)
    signal = remove_baseline(signal, fs)
    
    beats, valid_rpeaks = extract_heartbeats(signal, fs, ann.sample)
    print(f"Extracted {len(beats)} valid beats")
    
    beats = normalize_beats(beats)
    beats_spikes = delta_modulation(beats)
    labels = create_labels(valid_rpeaks, ann)
    
    if len(labels) != len(beats_spikes):
        print(f"Warning: Number of labels ({len(labels)}) does not match number of beats ({len(beats_spikes)}) for record {record_id}.")
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
        print(f"After filtering for labels: {len(beats_spikes)} beats, {len(labels)} labels.")

    if len(beats_spikes) == 0:
        print(f"No beats with valid labels extracted for record {record_id}.")
        return np.array([]), np.array([])

    return beats_spikes, labels

def load_dataset(record_ids, data_dir):
    all_beats = []
    all_labels = []
    for record_id in record_ids:
        X, y = process_record(str(record_id), data_dir)
        if X.shape[0] > 0:
            all_beats.append(X)
            all_labels.append(y)
        else:
            print(f"Skipping record {record_id} due to processing issues or no valid beats.")

    if all_beats:
        X_all = np.concatenate(all_beats, axis=0)
        y_all = np.concatenate(all_labels, axis=0)
        print(f"Loaded {len(record_ids)} records: total samples = {X_all.shape[0]}")
        return X_all, y_all  # <-- Add this return statement here

    else:
        print(f"No valid data loaded from {len(record_ids)} records.")
        return np.array([]), np.array([])

    
if __name__ == "__main__":
    data_dir = 'data/mitdb'
        
    DS1_train = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205]
    DS1_val = [207, 208, 209, 215, 220]
    DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

    # Load without balancing
    X_train, y_train = load_dataset(DS1_train, data_dir)
    X_val, y_val = load_dataset(DS1_val, data_dir)
    X_test, y_test = load_dataset(DS2, data_dir)
    
    print("\n ---------------Getting lengths of dataset--------------------")
    print("Total number of training beats : ", len(X_train))
    print("Total number of training labels : ", len(y_train))
    print("Total number of validation beats : ", len(X_val))
    print("Total number of validation labels : ", len(y_val))
    print("Total number of test beats : ", len(X_test))
    print("Total number of test labels : ", len(y_test))

    if X_train.shape[0] > 0 and X_val.shape[0] > 0 and X_test.shape[0] > 0:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        classes = np.unique(y_train)
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        print(f"Class weights: {class_weights}")

        model, history = train_model(X_train, y_train, X_val, y_val, X_test, y_test,
                                     batch_size=64, num_epochs=10, device=device,
                                     class_weights=class_weights_tensor)
        evaluate_model(model, X_val, y_val, X_test, y_test, device=device)
    else:
        print("Insufficient data loaded for training, validation, or testing.")
