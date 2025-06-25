import wfdb
import numpy as np
import os
from Denoise import bandpass_filter, notch_filter, remove_baseline
from segment import extract_heartbeats
from classBalancing import balance_classes
from normalization import zscore_normalize
from sklearn.model_selection import train_test_split

def load_mitbih_records(record_numbers, fs=360, segment_length=200):
    X = []
    y = []

    for record_num in record_numbers:
        print(f"Processing record {record_num}...")
        # record= wfdb.dl_database('mitdb', dl_dir='./data/mitdb')
        record = wfdb.rdrecord(record_num, pn_dir='mitdb')
        annotation = wfdb.rdann(record_num, 'atr', pn_dir='mitdb')

        signal = record.p_signal[:, 0]  # MLII lead

        # Denoise entire signal first
        signal = bandpass_filter(signal, fs)
        signal = notch_filter(signal, fs)
        signal = remove_baseline(signal, fs)
    

        # Extract fixed-length heartbeats centered at R-peaks
        beats, rpeaks = extract_heartbeats(signal, fs, annotation_rpeaks=annotation.sample, fixed_length=segment_length)

        # Assign labels: Normal 'N' = 0, others = 1
        labels = np.array([0 if sym == 'N' else 1 for sym in annotation.symbol])

        # Filter beats and labels to match valid rpeaks used in segmentation
        # Note: rpeaks may be fewer than annotation.sample due to edge trimming
        valid_indices = [np.where(annotation.sample == rp)[0][0] for rp in rpeaks]
        labels = labels[valid_indices]

        X.extend(beats)
        y.extend(labels)

    X = np.array(X)
    y = np.array(y)

    print(f"Total segments before balancing: {len(X)}, Normal: {(y==0).sum()}, Abnormal: {(y==1).sum()}")

    # Balance classes
    X_bal, y_bal = balance_classes(X, y, method='smote')
    print(f"Total segments after balancing: {len(X_bal)}, Normal: {(y_bal==0).sum()}, Abnormal: {(y_bal==1).sum()}")

    # Normalize
    X_norm = zscore_normalize(X_bal)
    print(f"Data normalized. Shape: {X_norm.shape}")
    
    return X_norm, y_bal

def main():
        # DS_1 contains training records
    DS_1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 124,
            201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
        # DS_2 contains test records
    DS_2 = [100, 103, 105, 111, 113, 121, 200, 202, 210, 212,
            213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

    fs = 360
    segment_length = 200

    # Prepare training data from DS_1
    X_train, y_train = load_mitbih_records([str(r) for r in DS_1], fs, segment_length)
    print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")

    # Prepare testing data from DS_2
    X_test, y_test = load_mitbih_records([str(r) for r in DS_2], fs, segment_length)
    print(f"Testing data shape: {X_test.shape}, Testing labels shape: {y_test.shape}")

    # Save data
    os.makedirs('./data', exist_ok=True)
    np.save('./data/train_data.npy', X_train)
    np.save('./data/train_labels.npy', y_train)
    np.save('./data/test_data.npy', X_test)
    np.save('./data/test_labels.npy', y_test)

    print("Saved processed data to 'data/' folder.")

if __name__ == "__main__":
    main()
