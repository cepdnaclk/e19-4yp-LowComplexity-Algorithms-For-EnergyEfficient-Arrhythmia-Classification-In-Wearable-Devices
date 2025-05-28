# This code contains the code for Feature extraction --> IMPLEMEMNTED IN ELSE WHERE(MainPipeLine.py), REMOVE LATER

from Denoise import bandpass_filter, notch_filter, remove_baseline
from Segment import extract_heartbeats
from ClassBalancing import balance_classes
from Normalization import normalize_beats
from Load import load_ecg
from Labels import create_labels


def extract_waveform_features(record_id, data_dir):
    
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
    X_balanced, y_balanced = balance_classes(beats_flat, labels)
    X_balanced = X_balanced.reshape(-1, beats.shape[1], 1)
    
    return X_balanced, y_balanced