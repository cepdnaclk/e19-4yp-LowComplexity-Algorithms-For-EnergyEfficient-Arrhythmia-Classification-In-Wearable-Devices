import numpy as np
import matplotlib.pyplot as plt
from preProcessing.ClassBalancing import balance_classes
from preProcessing.Denoise import bandpass_filter, notch_filter, remove_baseline
from preProcessing.Segment import extract_heartbeats
from preProcessing.Normalization import normalize_beats
from preProcessing.Load import load_ecg
from preProcessing.Labels import create_labels
from collections import Counter

def extract_beats_and_labels(record_id, data_dir):
    signal, rpeaks, fs, ann = load_ecg(record_id, data_dir)
    
    signal = bandpass_filter(signal, fs)
    signal = notch_filter(signal, fs)
    signal = remove_baseline(signal, fs)
    
    beats, valid_rpeaks = extract_heartbeats(signal, fs, ann.sample)
    
    beats = normalize_beats(beats)
    labels = create_labels(valid_rpeaks, ann)
    return beats, labels

def plot_original_and_synthetic_beat(X, y):
    X_balanced, y_balanced = balance_classes(X, y)
    
    # Identify synthetic samples (beyond original samples)
    n_original = X.shape[0]
    
    # Find indices of synthetic samples
    synthetic_indices = range(n_original, len(X_balanced))
    if len(synthetic_indices) == 0:
        print("No synthetic samples generated.")
        return
    
    # Plot first original beat and first synthetic beat
    plt.figure(figsize=(12,5))
    
    plt.subplot(1, 2, 1)
    plt.plot(X[0])
    plt.title('Original Beat (Class {})'.format(y[0]))
    plt.xlabel('Time samples')
    plt.ylabel('Amplitude')
    
    plt.subplot(1, 2, 2)
    plt.plot(X_balanced[synthetic_indices[0]])
    plt.title('Synthetic Beat (Class {})'.format(y_balanced[synthetic_indices[0]]))
    plt.xlabel('Time samples')
    plt.ylabel('Amplitude')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data_dir = 'data/mitdb'
    record_id = 100  # example record
    
    X, y = extract_beats_and_labels(record_id, data_dir)
    
    print("Class distribution before balancing:", Counter(y))
    
    plot_original_and_synthetic_beat(X, y)
