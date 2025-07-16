import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from preProcessing.Denoise import bandpass_filter, notch_filter, remove_baseline
from preProcessing.SegmentMod import extract_heartbeats
from preProcessing.Normalization import normalize_beats
from preProcessing.Load import load_ecg
from snnModel.Train import train_model
from snnModel.Evaluate import evaluate_model, evaluate_model_epochs
from snnModel.DeltaModulation import delta_modulation
import torch
from sklearn.model_selection import train_test_split, KFold
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, precision_score, f1_score
import seaborn as sns
import os
import shutil

# Five-class AAMI classes
AAMI_classes = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  # Normal
    'A': 1, 'a': 1, 'J': 1, 'S': 1,         # Supraventricular
    'V': 2, 'E': 2,                         # Ventricular
    'F': 3,                                 # Fusion
    'P': 4, '/': 4, 'f': 4, 'u': 4          # Paced/Unknown
}

def create_labels(rpeaks, ann):
    """
    Create five-class labels for ECG beats based on annotations.
    Args:
        rpeaks: Array of R-peak sample indices
        ann: Annotation object with 'sample' and 'symbol' attributes
    Returns:
        labels: Array of labels (0: Normal, 1: Supraventricular, 2: Ventricular, 3: Fusion, 4: Paced/Unknown)
    """
    labels = []
    for rpeak in rpeaks:
        idx = np.where(ann.sample == rpeak)[0]
        if len(idx) > 0:
            symbol = ann.symbol[idx[0]]
            labels.append(AAMI_classes.get(symbol, 4))  # Default to Paced/Unknown
        else:
            labels.append(4)
    return np.array(labels)

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
    print("Original class distribution:", Counter(y))
    smote = SMOTE(random_state=42)
    try:
        X_balanced, y_balanced = smote.fit_resample(X, y)
        print("Balanced class distribution:", Counter(y_balanced))
        return X_balanced, y_balanced
    except ValueError as e:
        print(f"Balancing error: {e}. Using original data.")
        return X, y

def plot_metrics(history, fold=None, is_test=False, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)
    title_suffix = f"Fold {fold}" if fold is not None else "Test Set"
    
    plt.figure(figsize=(10, 5))
    if is_test:
        plt.plot(history['test_loss'], label='Test Loss', color='purple')
    else:
        plt.plot(history['train_loss'], label='Training Loss', color='blue')
        plt.plot(history['val_loss'], label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss vs. Epoch - {title_suffix}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'loss_{title_suffix.lower().replace(" ", "_")}.png'))
    plt.show()  # Display interactively
    plt.close()
    
    plt.figure(figsize=(10, 5))
    if is_test:
        plt.plot(history['test_acc'], label='Test Accuracy', color='purple')
    else:
        plt.plot(history['train_acc'], label='Training Accuracy', color='blue')
        plt.plot(history['val_acc'], label='Validation Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs. Epoch - {title_suffix}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'accuracy_{title_suffix.lower().replace(" ", "_")}.png'))
    plt.show()  # Display interactively
    plt.close()

def plot_confusion_matrix(y_true, y_pred, fold=None, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)
    title_suffix = f"Fold {fold}" if fold is not None else "Test Set"
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4], normalize='true')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Paced/Unknown'],
                yticklabels=['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Paced/Unknown'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {title_suffix}')
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_{title_suffix.lower().replace(" ", "_")}.png'))
    plt.show()  # Display interactively
    plt.close()
    
    # Print confusion matrix to console
    print(f"\nConfusion Matrix - {title_suffix}")
    print(cm)

if __name__ == "__main__":
    data_dir = 'data/mitdb'
    
    # Combined dataset
    all_records = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 
                   207, 208, 209, 215, 220, 100, 103, 105, 111, 113, 117, 121, 123, 200, 
                   202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
    
    # Extract all beats and labels
    X_all, y_all = extract_all_beats_labels(all_records, data_dir)
    
    if X_all.shape[0] > 0:
        # Stratified 80:20 split
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
        )
        print("Training set class distribution:", Counter(y_train))
        print("Test set class distribution:", Counter(y_test))
        
        # Set up 5-fold cross-validation
        n_splits = 5
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold = 1
        best_val_acc = 0
        best_model = None
        history_all = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        print(f"\nStarting {n_splits}-fold cross-validation...")
        for train_idx, val_idx in kf.split(X_train):
            print(f"\nFold {fold}/{n_splits}")
            X_train_fold = X_train[train_idx]
            y_train_fold = y_train[train_idx]
            X_val_fold = X_train[val_idx]
            y_val_fold = y_train[val_idx]
            
            # Balance training fold
            X_train_fold, y_train_fold = balance_dataset(X_train_fold, y_train_fold)
            
            print(f"Training fold samples: {X_train_fold.shape[0]}, Validation fold samples: {X_val_fold.shape[0]}")
            
            # Train model for this fold
            model, history = train_model(
                X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                batch_size=64, num_epochs=10, device=device
            )
            
            # Evaluate model on validation set
            model.eval()
            X_val_tensor = torch.FloatTensor(X_val_fold).to(device)
            y_val_tensor = torch.LongTensor(y_val_fold).to(device)
            with torch.no_grad():
                outputs = model(X_val_tensor)
                _, predicted = torch.max(outputs, 1)
                val_acc = (predicted == y_val_tensor).sum().item() / y_val_tensor.size(0)
                val_precision = precision_score(y_val_fold, predicted.cpu().numpy(), average='macro', zero_division=0)
                val_f1 = f1_score(y_val_fold, predicted.cpu().numpy(), average='macro', zero_division=0)
                print(f"Fold {fold} Validation Metrics:")
                print(f"  Accuracy: {val_acc:.4f}")
                print(f"  Precision (Macro): {val_precision:.4f}")
                print(f"  F1 Score (Macro): {val_f1:.4f}")
                plot_confusion_matrix(y_val_fold, predicted.cpu().numpy(), fold=fold)
            
            # Save best model based on validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = model
                print(f"New best model found in fold {fold} with validation accuracy: {val_acc:.4f}")
            
            # Store and plot fold metrics
            history_all['train_loss'].append(history['train_loss'])
            history_all['train_acc'].append(history['train_acc'])
            history_all['val_loss'].append(history['val_loss'])
            history_all['val_acc'].append(history['val_acc'])
            plot_metrics(history, fold=fold, is_test=False)
            
            fold += 1
        
        # Average metrics across folds
        avg_history = {
            'train_loss': np.mean(history_all['train_loss'], axis=0).tolist(),
            'train_acc': np.mean(history_all['train_acc'], axis=0).tolist(),
            'val_loss': np.mean(history_all['val_loss'], axis=0).tolist(),
            'val_acc': np.mean(history_all['val_acc'], axis=0).tolist()
        }
        print("\nAverage Cross-Validation Metrics:")
        print(f"  Avg Validation Loss: {np.mean(avg_history['val_loss']):.4f}")
        print(f"  Avg Validation Accuracy: {np.mean(avg_history['val_acc']):.4f}")
        plot_metrics(avg_history, fold=None, is_test=False)
        
        # Evaluate best model on test set for 10 epochs
        print("\nEvaluating best model on test set for 10 epochs...")
        test_history = evaluate_model_epochs(best_model, X_test, y_test, num_epochs=10, device=device)
        print(f"Average Test Loss: {np.mean(test_history['test_loss']):.4f}")
        print(f"Average Test Accuracy: {np.mean(test_history['test_acc']):.4f}")
        plot_metrics(test_history, fold='Test Set', is_test=True)
        
        # Compute test set confusion matrix and metrics
        best_model.eval()
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_test_tensor = torch.LongTensor(y_test).to(device)
        with torch.no_grad():
            outputs = best_model(X_test_tensor)
            _, predicted = torch.max(outputs, 1)
            test_precision = precision_score(y_test, predicted.cpu().numpy(), average='macro', zero_division=0)
            test_f1 = f1_score(y_test, predicted.cpu().numpy(), average='macro', zero_division=0)
            test_acc = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
            print("\nTest Set Metrics:")
            print(f"  Accuracy: {test_acc:.4f}")
            print(f"  Precision (Macro): {test_precision:.4f}")
            print(f"  F1 Score (Macro): {test_f1:.4f}")
            plot_confusion_matrix(y_test, predicted.cpu().numpy(), fold=None)
        
        print("\n---------------Dataset Summary--------------------")
        print(f"Total number of training beats: {len(X_train)}")
        print(f"Total number of training labels: {len(y_train)}")
        print(f"Total number of test beats: {len(X_test)}")
        print(f"Total number of test labels: {len(y_test)}")
    else:
        print("Insufficient data loaded for training or testing.")