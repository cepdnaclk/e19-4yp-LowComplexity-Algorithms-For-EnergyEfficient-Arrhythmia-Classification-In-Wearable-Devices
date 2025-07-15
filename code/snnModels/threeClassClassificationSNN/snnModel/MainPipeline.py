import numpy as np
from collections import Counter
from preProcessing.Denoise import bandpass_filter, notch_filter, remove_baseline
from preProcessing.SegmentMod import extract_heartbeats
from preProcessing.Normalization import normalize_beats
from preProcessing.ClassBalancing import balance_classes
from preProcessing.Load import load_ecg
from preProcessing.Labels import create_labels
from snnModel.Train import train_model
from snnModel.Evaluate import evaluate_model, evaluate_model_epochs
from snnModel.DeltaModulation import delta_modulation
from preProcessing.Labels import AAMI_classes
import torch
from sklearn.model_selection import KFold

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
    
    # if len(labels) != len(beats_spikes):
    #     print(f"Warning: Label count ({len(labels)}) != beats count ({len(beats_spikes)}) for record {record_id}. Filtering...")
    #     labeled_beats = []
    #     labeled_valid_rpeaks = []
    #     labeled_labels = []
    #     for i, rpeak in enumerate(valid_rpeaks):
    #         idx = np.where(ann.sample == rpeak)[0]
    #         if len(idx) > 0:
    #             symbol = ann.symbol[idx[0]]
    #             if symbol in AAMI_classes:
    #                 labeled_beats.append(beats_spikes[i])
    #                 labeled_valid_rpeaks.append(rpeak)
    #                 labeled_labels.append(AAMI_classes[symbol])
    #     beats_spikes = np.array(labeled_beats)
    #     valid_rpeaks = np.array(labeled_valid_rpeaks)
    #     labels = np.array(labeled_labels)
    #     print(f"After filtering: {len(beats_spikes)} beats, {len(labels)} labels")
    
    # if len(beats_spikes) == 0:
    #     print(f"No valid beats with labels for record {record_id}. Skipping.")
    #     return np.array([]), np.array([])
    
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Combine all MIT-BIH records
    all_records = [
        '100', '101', '103', '105', '106', '108', '109', '111', '112', '113',
        '114', '115', '116', '117', '118', '119', '121', '122', '123', '124',
        '200', '201', '202', '203', '205', '207', '208', '209', '210', '212',
        '213', '214', '215', '217', '219', '220', '221', '222', '223', '228',
        '230', '231', '232', '233', '234'
    ]

    # Load all data
    X_all, y_all = extract_all_beats_labels(all_records, data_dir)

    # Convert to tensors for GPU processing
    X_all = torch.tensor(X_all, dtype=torch.float, device=device)
    y_all = torch.tensor(y_all, dtype=torch.long, device=device)
    
    # Print class distribution before splitting
    # print("\n---------------Class distribution before splitting--------------------")
    # class_counts_before = Counter(y_all.cpu().numpy())
    # for class_idx in range(3):
    #     count = class_counts_before.get(class_idx, 0)
    #     print(f"Class {class_idx} ({['Normal', 'SVEB/VEB', 'Fusion/Paced/Other'][class_idx]}): {count} samples")

    # Stratified 80/20 split per class
    train_beats = []
    train_labels = []
    test_beats = []
    test_labels = []
    for class_idx in range(3):
        class_mask = y_all == class_idx
        class_beats = X_all[class_mask]
        class_labels = y_all[class_mask]
        
        if class_beats.shape[0] == 0:
            print(f"Warning: No samples for class {class_idx}. Skipping.")
            continue
        
        n_samples = class_beats.shape[0]
        n_train = int(0.8 * n_samples)
        indices = torch.randperm(n_samples, device=device)
        
        train_beats.append(class_beats[indices[:n_train]])
        train_labels.append(class_labels[indices[:n_train]])
        test_beats.append(class_beats[indices[n_train:]])
        test_labels.append(class_labels[indices[n_train:]])

    # Concatenate training and test sets
    X_train_all = torch.cat(train_beats, dim=0) if train_beats else torch.tensor([], device=device)
    y_train_all = torch.cat(train_labels, dim=0) if train_labels else torch.tensor([], device=device)
    X_test = torch.cat(test_beats, dim=0) if test_beats else torch.tensor([], device=device)
    y_test = torch.cat(test_labels, dim=0) if test_labels else torch.tensor([], device=device)
    
    # Print class distribution after splitting
    # print("\n---------------Class distribution after splitting--------------------")
    # print("Training set:")
    # train_class_counts = Counter(y_train_all.cpu().numpy())
    # for class_idx in range(3):
    #     count = train_class_counts.get(class_idx, 0)
    #     print(f"Class {class_idx} ({['Normal', 'SVEB/VEB', 'Fusion/Paced/Other'][class_idx]}): {count} samples")
    # print("Test set:")
    # test_class_counts = Counter(y_test.cpu().numpy())
    # for class_idx in range(3):
    #     count = test_class_counts.get(class_idx, 0)
    #     print(f"Class {class_idx} ({['Normal', 'SVEB/VEB', 'Fusion/Paced/Other'][class_idx]}): {count} samples")

    # Balance training data
    X_train_all, y_train_all = balance_dataset(X_train_all, y_train_all)

    print("\n---------------Getting lengths of dataset--------------------")
    print("Total number of training beats : ", len(X_train_all))
    print("Total number of training labels : ", len(y_train_all))
    print("Total number of test beats : ", len(X_test))
    print("Total number of test labels : ", len(y_test))
    # print(f"Training class distribution: {Counter(y_train_all.cpu().numpy())}")
    print(f"Test class distribution: {Counter(y_test.cpu().numpy())}")

    if X_train_all.shape[0] > 0 and X_test.shape[0] > 0:
        # Set up k-fold cross-validation
        n_splits = 5
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold = 1
        best_val_acc = 0
        best_model = None
        history_all = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

        print(f"\nStarting {n_splits}-fold cross-validation...")
        for train_idx, val_idx in kf.split(X_train_all):
            print(f"\nFold {fold}/{n_splits}")
            X_train_fold = X_train_all[train_idx]
            y_train_fold = y_train_all[train_idx]
            X_val_fold = X_train_all[val_idx]
            y_val_fold = y_train_all[val_idx]

            print(f"Training fold samples: {X_train_fold.shape[0]}, Validation fold samples: {X_val_fold.shape[0]}")

            # Train model for this fold
            model, history = train_model(
                X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                batch_size=64, num_epochs=10, device=device
            )

            # Evaluate model on validation set to determine best model
            model.eval()
            X_val_tensor = torch.FloatTensor(X_val_fold).to(device)
            y_val_tensor = torch.LongTensor(y_val_fold).to(device)
            with torch.no_grad():
                outputs = model(X_val_tensor)
                _, predicted = torch.max(outputs, 1)
                val_acc = (predicted == y_val_tensor).sum().item() / y_val_tensor.size(0)

            # Save best model based on validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = model
                print(f"New best model found in fold {fold} with validation accuracy: {val_acc:.4f}")

            # Aggregate history
            history_all['train_loss'].append(history['train_loss'])
            history_all['train_acc'].append(history['train_acc'])
            history_all['val_loss'].append(history['val_loss'])
            history_all['val_acc'].append(history['val_acc'])

            fold += 1

        # Average metrics across folds
        avg_history = {
            'train_loss': np.mean(history_all['train_loss'], axis=0).tolist(),
            'train_acc': np.mean(history_all['train_acc'], axis=0).tolist(),
            'val_loss': np.mean(history_all['val_loss'], axis=0).tolist(),
            'val_acc': np.mean(history_all['val_acc'], axis=0).tolist()
        }

        # Plot averaged cross-validation metrics
        from snnModel.Evaluate import plot_metrics
        plot_metrics(avg_history)

        # Evaluate the best model on the test set for 10 epochs
        print("\nEvaluating best model on test set for 10 epochs...")
        test_history = evaluate_model_epochs(best_model, X_test, y_test, num_epochs=10, device=device)

        # Plot test evaluation metrics
        print("\nPlotting test evaluation metrics...")
        plot_metrics(test_history)

        # Final single-pass evaluation for comparison
        print("\nFinal single-pass evaluation of best model...")
        evaluate_model(best_model, X_train_all, y_train_all, X_test, y_test, device=device)
    else:
        print("Insufficient data loaded for training or testing.")