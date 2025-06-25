# Try " python -m models.MainPipeLine to run the " to run the script 

import numpy as np
from preProcessing.Denoise import bandpass_filter, notch_filter, remove_baseline
from preProcessing.Segment import extract_heartbeats
from preProcessing.ClassBalancing import balance_classes
from preProcessing.Normalization import normalize_beats
from preProcessing.Load import load_ecg
from preProcessing.Labels import create_labels
from cnnModel.Train import train_model
from cnnModel.Evaluate import evaluate_model, plot_metrics

def process_record(record_id, data_dir, balance=True):
    
    signal, rpeaks, fs, ann = load_ecg(record_id, data_dir)
    print(f"Total annotations: {len(ann.sample)}")
    
    signal = bandpass_filter(signal, fs)
    signal = notch_filter(signal, fs)
    signal = remove_baseline(signal, fs)
    
    beats, valid_rpeaks = extract_heartbeats(signal, fs, ann.sample)
    print(f"Extracted {len(beats)} valid beats")
    
    beats = normalize_beats(beats)
    labels = create_labels(valid_rpeaks, ann)
    
    beats_flat = beats.reshape(beats.shape[0], -1)

    if balance:
        unique_classes = np.unique(labels)
        if len(unique_classes) > 1:
            X_balanced, y_balanced = balance_classes(beats_flat, labels)
        else:
            print(f"Only one class ({unique_classes[0]}) present in record {record_id}, skipping balancing.")
            X_balanced, y_balanced = beats_flat, labels
    else:
        X_balanced, y_balanced = beats_flat, labels

    X_balanced = X_balanced.reshape(-1, beats.shape[1], 1)
    
    return X_balanced, y_balanced

def load_dataset(record_ids, data_dir, balance=True):
    all_beats = []
    all_labels = []
    for record_id in record_ids:
        print(f"Processing record {record_id}...")
        X, y = process_record(str(record_id), data_dir, balance=balance)
        all_beats.append(X)
        all_labels.append(y)
    X_all = np.concatenate(all_beats, axis=0)
    y_all = np.concatenate(all_labels, axis=0)
    print(f"Loaded {len(record_ids)} records: total samples = {X_all.shape[0]}")
    return X_all, y_all


if __name__ == "__main__":
    data_dir = 'data/mitdb'
    
    DS_1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 
            201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
    DS_2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 
            213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
    
    # Load training data from DS_1
    X_train, y_train = load_dataset(DS_1, data_dir, balance=True)
    
    # Load testing data from DS_2
    X_test, y_test = load_dataset(DS_2, data_dir, balance=False)
    
    print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
    
    # print("///////////////////////////////////")
    # print("Total Training segments : ", len(X_train))
    # print("Total training labels : ", len(y_train))
    
    # print("++++++++++++++++++++++++++++++++++++")
    # print("Total testing segments : ", len(X_test))
    # print("Total testing labels : ", len(y_test)) 
    # print("-----------------------------------")
    
    # Train model explicitly on train set, validate on test set
    model, history = train_model(X_train, y_train, X_test, y_test)
    
    plot_metrics(history)
    evaluate_model(model, X_test, y_test)
    
    
    # Clarifying the dataset provided for training the model
    # print("First 10 X_train data(segments) : ", X_train[:10]) # This contains an array of segments (One segment is  
    # 250 samples long) -> there are large number of 250 sampled arrays in this X_train array
    # print("First 10 y train data (Labels) : ", y_train[:30])
    # print("Shape of the dataset : ", X_train.shape)
