import numpy as np
from collections import Counter
from preProcessing.Denoise import bandpass_filter, notch_filter, remove_baseline
from preProcessing.SegmentMod import extract_heartbeats
from preProcessing.Normalization import normalize_beats
from preProcessing.ClassBalancing import balance_classes
from preProcessing.Load import load_ecg
from preProcessing.Labels import create_labels, AAMI_classes
from snnModel.Train import train_model
from snnModel.Evaluate import evaluate_model, evaluate_model_epochs, plot_metrics
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from torch.utils.data import TensorDataset, DataLoader

try:
    from snnModel.DeltaModulation import delta_modulation
except ImportError:
    print("Warning: DeltaModulation module not found. Using placeholder delta_modulation.")
    def delta_modulation(beats):
        print(f"Placeholder delta modulation applied to beats of shape {beats.shape}")
        return beats  # Temporary return; replace with actual spike conversion logic

def process_record(record_id, data_dir):
    signal, rpeaks, fs, ann = load_ecg(record_id, data_dir)
    print(f"Record {record_id}: Total annotations: {len(ann.sample)}")
    
    signal = bandpass_filter(signal, fs)
    signal = notch_filter(signal, fs)
    signal = remove_baseline(signal, fs)
    
    beats, valid_rpeaks = extract_heartbeats(signal, fs, ann.sample)
    print(f"Record {record_id}: Extracted {len(beats)} valid beats, shape: {beats.shape if beats.size else 'empty'}")
    
    beats = normalize_beats(beats)
    labels = create_labels(valid_rpeaks, ann)  # Now returns binary labels (0: Normal, 1: Abnormal)
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
        print(f"After filtering: {len(beats_spikes)} beats, shape: {beats_spikes.shape}")
    
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
        print(f"Extracted total {X_all.shape[0]} beats from {len(record_ids)} records, shape: {X_all.shape}")
        return X_all, y_all
    else:
        return np.array([]), np.array([])

def balance_dataset(X, y):
    unique_classes = np.unique(y)
    if len(unique_classes) > 1:
        try:
            X_balanced, y_balanced = balance_classes(X, y)
            print(f"Balanced dataset: original {len(X)} samples, shape {X.shape}, balanced {len(X_balanced)} samples, shape {X_balanced.shape}")
            return X_balanced, y_balanced
        except ValueError as e:
            print(f"Balancing error: {e}. Using original data.")
            return X, y
    else:
        print(f"Only one class ({unique_classes[0]}) present. Skipping balancing.")
        return X, y

def evaluate_model(model, X_train, y_train, X_test, y_test, device='cuda'):
    model.eval()
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    with torch.no_grad():
        # Training set evaluation
        train_outputs = model(X_train_tensor)
        _, train_pred = torch.max(train_outputs, 1)
        train_acc = (train_pred == y_train_tensor).sum().item() / y_train_tensor.size(0)
        train_f1 = f1_score(y_train, train_pred.cpu().numpy(), average='binary')
        
        # Test set evaluation
        test_outputs = model(X_test_tensor)
        _, test_pred = torch.max(test_outputs, 1)
        test_acc = (test_pred == y_test_tensor).sum().item() / y_test_tensor.size(0)
        test_f1 = f1_score(y_test, test_pred.cpu().numpy(), average='binary')
    
    print(f"Final Evaluation - Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")
    return {'train_acc': train_acc, 'train_f1': train_f1, 'test_acc': test_acc, 'test_f1': test_f1}

def evaluate_model_epochs(model, X_test, y_test, num_epochs=10, device='cuda'):
    history = {'test_loss': [], 'test_acc': [], 'test_f1': []}
    criterion = nn.CrossEntropyLoss()
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print("Evaluating model on test set for 10 epochs...")
    for epoch in range(num_epochs):
        model.eval()
        running_test_loss = 0
        correct_test = 0
        total_test = 0
        all_test_preds = []
        all_test_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
                all_test_preds.extend(predicted.cpu().numpy())
                all_test_labels.extend(labels.cpu().numpy())
        
        test_loss = running_test_loss / total_test
        test_acc = correct_test / total_test
        test_f1 = f1_score(all_test_labels, all_test_preds, average='binary')
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['test_f1'].append(test_f1)
        
        print(f"Test Epoch {epoch+1}/{num_epochs}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")
    
    return history

if __name__ == "__main__":
    data_dir = 'data/mitdb'

    DS1_train = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220]
    DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

    # Load and balance training data (DS1_train)
    X_train_all, y_train_all = extract_all_beats_labels(DS1_train, data_dir)
    X_train_all, y_train_all = balance_dataset(X_train_all, y_train_all)

    # Load test data (DS2)
    X_test, y_test = extract_all_beats_labels(DS2, data_dir)

    print("\n---------------Getting lengths of dataset--------------------")
    print("Total number of training beats : ", len(X_train_all))
    print("Total number of training labels : ", len(y_train_all))
    print("Total number of test beats : ", len(X_test))
    print("Total number of test labels : ", len(y_test))

    if X_train_all.shape[0] > 0 and X_test.shape[0] > 0:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        # Compute class weights for binary classification
        class_counts = Counter(y_train_all)
        n_samples = len(y_train_all)
        class_weights = torch.FloatTensor([n_samples / (count * 1.5) if count < max(class_counts.values()) else n_samples / count for count in class_counts.values()]).to(device)

        # Set up k-fold cross-validation
        n_splits = 5
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold = 1
        best_val_acc = 0
        best_model = None
        history_all = {'train_loss': [], 'train_acc': [], 'train_f1': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

        print(f"\nStarting {n_splits}-fold cross-validation...")
        for train_idx, val_idx in kf.split(X_train_all):
            print(f"\nFold {fold}/{n_splits}")
            X_train_fold = X_train_all[train_idx]
            y_train_fold = y_train_all[train_idx]
            X_val_fold = X_train_all[val_idx]
            y_val_fold = y_train_all[val_idx]

            print(f"Training fold samples: {X_train_fold.shape[0]}, shape: {X_train_fold.shape}")
            print(f"Validation fold samples: {X_val_fold.shape[0]}, shape: {X_val_fold.shape}")

            # Train model for this fold
            model, history = train_model(
                X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                batch_size=64, num_epochs=10, device=device, class_weights=class_weights
            )

            # Evaluate model on validation set to determine best model
            model.eval()
            X_val_tensor = torch.FloatTensor(X_val_fold).to(device)
            y_val_tensor = torch.LongTensor(y_val_fold).to(device)
            with torch.no_grad():
                outputs = model(X_val_tensor)
                _, predicted = torch.max(outputs, 1)
                val_acc = (predicted == y_val_tensor).sum().item() / y_val_tensor.size(0)
                val_f1 = f1_score(y_val_fold, predicted.cpu().numpy(), average='binary')

            # Save best model based on validation F1
            if val_f1 > best_val_acc:
                best_val_acc = val_f1
                best_model = model
                print(f"New best model found in fold {fold} with validation F1: {val_f1:.4f}")

            # Aggregate history
            history_all['train_loss'].append(history['train_loss'])
            history_all['train_acc'].append(history['train_acc'])
            history_all['train_f1'].append(history['train_f1'])
            history_all['val_loss'].append(history['val_loss'])
            history_all['val_acc'].append(history['val_acc'])
            history_all['val_f1'].append(history['val_f1'])

            fold += 1

        # Average metrics across folds
        avg_history = {
            'train_loss': np.mean(history_all['train_loss'], axis=0).tolist(),
            'train_acc': np.mean(history_all['train_acc'], axis=0).tolist(),
            'train_f1': np.mean(history_all['train_f1'], axis=0).tolist(),
            'val_loss': np.mean(history_all['val_loss'], axis=0).tolist(),
            'val_acc': np.mean(history_all['val_acc'], axis=0).tolist(),
            'val_f1': np.mean(history_all['val_f1'], axis=0).tolist()
        }

        # Plot averaged cross-validation metrics
        plot_metrics(avg_history)

        # Evaluate the best model on the test set
        print("\nEvaluating best model on test set for 10 epochs...")
        test_history = evaluate_model_epochs(best_model, X_test, y_test, num_epochs=10, device=device)

        # Plot test evaluation metrics
        print("\nPlotting test evaluation metrics...")
        plot_metrics(test_history)

        # Final single-pass evaluation
        print("\nFinal single-pass evaluation of best model...")
        evaluate_model(best_model, X_train_all, y_train_all, X_test, y_test, device=device)
    else:
        print("Insufficient data loaded for training or testing.")