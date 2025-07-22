
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from preProcessing.Denoise import bandpass_filter, notch_filter, remove_baseline
from preProcessing.SegmentMod import extract_heartbeats
from preProcessing.Normalization import normalize_beats
from preProcessing.Load import load_ecg
from snnModel.Train import train_model, validate_model
from snnModel.Evaluate import evaluate_model_epochs
from snnModel.SnnModel import CSNN
import torch
from sklearn.model_selection import train_test_split, KFold
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, precision_score, f1_score
import seaborn as sns
import os
import shutil
import snntorch.functional as SF
import snntorch.spikeplot as splt

# AAMI classes
AAMI_classes = {
    0: ['N', 'L', 'R', 'e', 'j'],  # Normal
    1: ['A', 'a', 'J', 'S'],       # Supraventricular ectopic beats
    2: ['V', 'E'],                 # Ventricular ectopic beats
    3: ['F'],                      # Fusion beats
    4: ['P', '/', 'f', 'u']        # Unknown / unclassified beats
}

def get_class_from_symbol(symbol):
    for class_idx, symbols in AAMI_classes.items():
        if symbol in symbols:
            return class_idx
    return 4

def poisson_encoding(beats, time_steps=250, max_rate=100.0):
    n_beats, n_samples = beats.shape
    spike_trains = np.zeros((n_beats, time_steps), dtype=np.int32)
    for i in range(n_beats):
        beat = beats[i]
        interp_indices = np.linspace(0, n_samples-1, time_steps).astype(int)
        interpolated_beat = beat[interp_indices]
        rates = interpolated_beat * max_rate
        for t in range(time_steps):
            if np.random.random() < rates[t] / max_rate:
                spike_trains[i, t] = 1
    return spike_trains

def create_labels(rpeaks, ann):
    labels = []
    beat_symbols = ann.symbol
    annotation_samples = ann.sample
    for peak in rpeaks:
        idx = np.argmin(np.abs(annotation_samples - peak))
        symbol = beat_symbols[idx]
        class_idx = get_class_from_symbol(symbol)
        labels.append(class_idx)
    return np.array(labels)

def process_record(record_id, data_dir, use_poisson=True, time_steps=250):
    signal, rpeaks, fs, ann = load_ecg(record_id, data_dir)
    print(f"Record {record_id}: Total annotations: {len(ann.sample)}, R-peaks: {len(rpeaks)}")
    
    signal = bandpass_filter(signal, fs)
    signal = notch_filter(signal, fs)
    signal = remove_baseline(signal, fs)
    
    beats, valid_rpeaks = extract_heartbeats(signal, fs, annotation_rpeaks=rpeaks)
    print(f"Record {record_id}: Extracted {len(beats)} valid beats")
    
    beats = normalize_beats(beats)
    
    if use_poisson:
        beats_spikes = poisson_encoding(beats, time_steps=time_steps)
    else:
        beats_spikes = beats
    
    labels = create_labels(valid_rpeaks, ann)
    
    if len(labels) != len(beats_spikes):
        print(f"Warning: Label count ({len(labels)}) != beats count ({len(beats_spikes)}) for record {record_id}. Filtering...")
        labeled_beats = []
        labeled_labels = []
        for i, rpeak in enumerate(valid_rpeaks):
            idx = np.argmin(np.abs(ann.sample - rpeak))
            symbol = ann.symbol[idx]
            if symbol in sum(AAMI_classes.values(), []):
                labeled_beats.append(beats_spikes[i])
                labeled_labels.append(get_class_from_symbol(symbol))
        beats_spikes = np.array(labeled_beats)
        labels = np.array(labeled_labels)
        print(f"After filtering: {len(beats_spikes)} beats, {len(labels)} labels")
    
    if len(beats_spikes) == 0:
        print(f"No valid beats with labels for record {record_id}. Skipping.")
        return np.array([]), np.array([]), Counter()
    
    print(f"Record {record_id}: Beats shape: {beats_spikes.shape}")
    return beats_spikes, labels, Counter(labels)

def plot_spike_trains(spk_rec, y_true, idx=0, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 7))
    labels = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Paced/Unknown']
    splt.spike_count(spk_rec[:, idx].detach().cpu(), fig, ax, labels=labels, animate=False)
    plt.title(f'Spike Trains for Sample {idx} (True Label: {labels[y_true[idx]]})')
    plt.savefig(os.path.join(save_dir, f'spike_trains_sample_{idx}.png'))
    plt.close()

def extract_all_beats_labels(record_ids, data_dir, use_poisson=True, time_steps=250):
    all_beats = []
    all_labels = []
    record_distributions = {}
    for record_id in record_ids:
        X, y, class_dist = process_record(str(record_id), data_dir, use_poisson, time_steps)
        if X.shape[0] > 0:
            all_beats.append(X)
            all_labels.append(y)
            record_distributions[record_id] = class_dist
        else:
            print(f"Skipping record {record_id} due to no valid data.")
    if all_beats:
        X_all = np.concatenate(all_beats, axis=0)
        y_all = np.concatenate(all_labels, axis=0)
        print(f"Extracted total {X_all.shape[0]} beats from {len(record_ids)} records.")
        return X_all, y_all, record_distributions
    else:
        return np.array([]), np.array([]), {}

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
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss vs. Epoch - {title_suffix}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'loss_{title_suffix.lower().replace(" ", "_")}.png'))
    plt.close()
    
    plt.figure(figsize=(10, 5))
    if is_test:
        plt.plot(history['test_acc'], label='Test Accuracy', color='purple')
    else:
        plt.plot(history['train_acc'], label='Training Accuracy', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs. Epoch - {title_suffix}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'accuracy_{title_suffix.lower().replace(" ", "_")}.png'))
    plt.close()
    
    plt.figure(figsize=(10, 5))
    if is_test:
        plt.plot(history['time_per_batch'], label='Time per Batch', color='green')
    else:
        plt.plot(history['train_time_per_batch'], label='Train Time per Batch', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    plt.title(f'Time per Batch vs. Epoch - {title_suffix}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'time_{title_suffix.lower().replace(" ", "_")}.png'))
    plt.close()
    
    plt.figure(figsize=(10, 5))
    if is_test:
        plt.plot(history['memory_peak_mb'], label='Peak Memory', color='red')
    else:
        plt.plot(history['train_memory_peak_mb'], label='Train Peak Memory', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Memory (MB)')
    plt.title(f'Peak Memory vs. Epoch - {title_suffix}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'memory_{title_suffix.lower().replace(" ", "_")}.png'))
    plt.close()
    
    if not is_test:
        plt.figure(figsize=(10, 5))
        plt.plot(history['model_file_size_mb'], label='Model File Size', color='purple')
        plt.xlabel('Epoch')
        plt.ylabel('Size (MB)')
        plt.title(f'Model File Size vs. Epoch - {title_suffix}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'model_size_{title_suffix.lower().replace(" ", "_")}.png'))
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
    plt.close()
    
    print(f"\nConfusion Matrix - {title_suffix}")
    print(cm)

if __name__ == "__main__":
    data_dir = 'data/mitdb'
    
    all_records = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 
                   207, 208, 209, 215, 220, 100, 103, 105, 111, 113, 117, 121, 123, 200, 
                   202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
    
    train_records, test_records = train_test_split(
        all_records, test_size=0.2, random_state=42
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Extract data
    print(f"\nTraining records ({len(train_records)}):")
    X_train, y_train, train_distributions = extract_all_beats_labels(train_records, data_dir, use_poisson=True, time_steps=250)
    print(f"\nTest records ({len(test_records)}):")
    X_test, y_test, test_distributions = extract_all_beats_labels(test_records, data_dir, use_poisson=True, time_steps=250)
    
    if X_train.shape[0] > 0 and X_test.shape[0] > 0:
        print("\nTraining set class distribution:", Counter(y_train))
        print("Test set class distribution:", Counter(y_test))
        
        # Train and evaluate CSNN
        n_splits = 5
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold = 1
        best_val_acc = 0
        best_model = None
        best_model_path = os.path.join('checkpoints', 'best_model.pt')
        history_all = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'train_time_per_batch': [], 'val_time_per_batch': [],
            'train_memory_peak_mb': [], 'val_memory_peak_mb': [],
            'model_file_size_mb': []
        }
        
        print(f"\nStarting {n_splits}-fold cross-validation for CSNN...")
        for train_idx, val_idx in kf.split(X_train):
            print(f"\nFold {fold}/{n_splits}")
            X_train_fold = X_train[train_idx]
            y_train_fold = y_train[train_idx]
            X_val_fold = X_train[val_idx]
            y_val_fold = y_train[val_idx]
            
            X_train_fold, y_train_fold = balance_dataset(X_train_fold, y_train_fold)
            
            print(f"Training fold samples: {X_train_fold.shape[0]}, Validation fold samples: {X_val_fold.shape[0]}")
            
            # Train model
            model, train_history = train_model(
                X_train_fold, y_train_fold,
                batch_size=64, num_epochs=20, device=device, num_steps=10, beta=0.5
            )
            
            # Validate model
            val_loss, val_acc, val_time, val_memory, y_pred = validate_model(
                model, X_val_fold, y_val_fold, batch_size=64, device=device, num_steps=50
            )
            val_precision = precision_score(y_val_fold, y_pred, average='macro', zero_division=0)
            val_f1 = f1_score(y_val_fold, y_pred, average='macro', zero_division=0)
            print(f"Fold {fold} Validation Metrics:")
            print(f"  Validation Loss: {val_loss:.4f}")
            print(f"  Accuracy: {val_acc:.4f}")
            print(f"  Precision (Macro): {val_precision:.4f}")
            print(f"  F1 Score (Macro): {val_f1:.4f}")
            print(f"  Validation Time/Batch: {val_time:.4f}s")
            print(f"  Validation Memory: {val_memory:.2f}MB")
            plot_confusion_matrix(y_val_fold, y_pred, fold=fold)
            
            # Plot spike trains for one sample
            model.eval()
            X_val_tensor = torch.FloatTensor(X_val_fold.reshape(-1, 1, 250)).to(device)
            with torch.no_grad():
                spk_rec, _ = model(X_val_tensor)
            plot_spike_trains(spk_rec, y_val_fold, idx=0, save_dir=f'plots/fold_{fold}')
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = model
                try:
                    torch.save(model.state_dict(), best_model_path)
                    best_model_size_mb = os.path.getsize(best_model_path) / 1e6
                    print(f"New best model saved: {best_model_path}, Size: {best_model_size_mb:.2f}MB")
                except OSError as e:
                    print(f"Error saving best model: {e}")
            
            # Combine training and validation metrics
            history = {
                'train_loss': train_history['train_loss'],
                'train_acc': train_history['train_acc'],
                'train_time_per_batch': train_history['train_time_per_batch'],
                'train_memory_peak_mb': train_history['train_memory_peak_mb'],
                'model_file_size_mb': train_history['model_file_size_mb'],
                'val_loss': [val_loss],
                'val_acc': [val_acc],
                'val_time_per_batch': [val_time],
                'val_memory_peak_mb': [val_memory]
            }
            history_all['train_loss'].append(history['train_loss'])
            history_all['train_acc'].append(history['train_acc'])
            history_all['val_loss'].append(history['val_loss'])
            history_all['val_acc'].append(history['val_acc'])
            history_all['train_time_per_batch'].append(history['train_time_per_batch'])
            history_all['val_time_per_batch'].append(history['val_time_per_batch'])
            history_all['train_memory_peak_mb'].append(history['train_memory_peak_mb'])
            history_all['val_memory_peak_mb'].append(history['val_memory_peak_mb'])
            history_all['model_file_size_mb'].append(history['model_file_size_mb'])
            plot_metrics(history, fold=fold, is_test=False)
            
            fold += 1
        
        avg_history = {
            'train_loss': np.mean(history_all['train_loss'], axis=0).tolist(),
            'train_acc': np.mean(history_all['train_acc'], axis=0).tolist(),
            'val_loss': np.mean(history_all['val_loss'], axis=0).tolist(),
            'val_acc': np.mean(history_all['val_acc'], axis=0).tolist(),
            'train_time_per_batch': np.mean(history_all['train_time_per_batch'], axis=0).tolist(),
            'val_time_per_batch': np.mean(history_all['val_time_per_batch'], axis=0).tolist(),
            'train_memory_peak_mb': np.mean(history_all['train_memory_peak_mb'], axis=0).tolist(),
            'val_memory_peak_mb': np.mean(history_all['val_memory_peak_mb'], axis=0).tolist(),
            'model_file_size_mb': np.mean(history_all['model_file_size_mb'], axis=0).tolist()
        }
        print("\nAverage Cross-Validation Metrics (CSNN):")
        print(f"  Avg Validation Loss: {np.mean(avg_history['val_loss']):.4f}")
        print(f"  Avg Validation Accuracy: {np.mean(avg_history['val_acc']):.4f}")
        print(f"  Avg Train Time per Batch: {np.mean(avg_history['train_time_per_batch']):.4f}s")
        print(f"  Avg Val Time per Batch: {np.mean(avg_history['val_time_per_batch']):.4f}s")
        print(f"  Avg Train Memory: {np.mean(avg_history['train_memory_peak_mb']):.2f}MB")
        print(f"  Avg Val Memory: {np.mean(avg_history['val_memory_peak_mb']):.2f}MB")
        print(f"  Avg Model File Size: {np.mean(avg_history['model_file_size_mb']):.2f}MB")
        plot_metrics(avg_history, fold='Average', is_test=False)
        
        # Evaluate CSNN on test set
        print("\nEvaluating CSNN on test set...")
        csnn_history = evaluate_model_epochs(best_model, X_test, y_test, num_epochs=10, num_steps=50, device=device, batch_size=64)
        try:
            test_model_path = os.path.join('checkpoints', 'test_model.pt')
            torch.save(best_model.state_dict(), test_model_path)
            test_model_size_mb = os.path.getsize(test_model_path) / 1e6
            print(f"Test model saved: {test_model_path}, Size: {test_model_size_mb:.2f}MB")
        except OSError as e:
            print(f"Error saving test model: {e}")
        print(f"CSNN Average Test Loss: {np.mean(csnn_history['test_loss']):.4f}")
        print(f"CSNN Average Test Accuracy: {np.mean(csnn_history['test_acc']):.4f}")
        print(f"CSNN Average Time per Batch: {np.mean(csnn_history['time_per_batch']):.4f}s")
        print(f"CSNN Average Peak Memory: {np.mean(csnn_history['memory_peak_mb']):.2f}MB")
        print(f"CSNN Test Model File Size: {test_model_size_mb:.2f}MB")
        plot_metrics(csnn_history, fold='Test Set CSNN', is_test=True)
        
        # Test CSNN adjustments
        adjustments = [
            {'num_steps': 10, 'beta': 0.5, 'time_steps': 250, 'name': 'CSNN_num_steps_10'},
            {'num_steps': 100, 'beta': 0.5, 'time_steps': 250, 'name': 'CSNN_num_steps_100'},
            {'num_steps': 50, 'beta': 0.9, 'time_steps': 250, 'name': 'CSNN_beta_0.9'},
            {'num_steps': 50, 'beta': 0.5, 'time_steps': 100, 'name': 'CSNN_time_steps_100'}
        ]
        
        for adj in adjustments:
            print(f"\nEvaluating CSNN with {adj['name']}...")
            if adj['time_steps'] != 250:
                X_test_adj, y_test_adj, _ = extract_all_beats_labels(test_records, data_dir, use_poisson=True, time_steps=adj['time_steps'])
            else:
                X_test_adj, y_test_adj = X_test, y_test
            model = CSNN(num_inputs=adj['time_steps'], num_outputs=5, num_steps=adj['num_steps'], beta=adj['beta'], device=device).to(device)
            if adj['time_steps'] != 250:
                linear_size = 64 * ((adj['time_steps'] - 5 + 1) // 2 - 5 + 1) // 2
                model.net[-2] = nn.Linear(linear_size, 5).to(device)
            history = evaluate_model_epochs(model, X_test_adj, y_test_adj, num_epochs=10, num_steps=adj['num_steps'], device=device, batch_size=64)
            try:
                adj_model_path = os.path.join('checkpoints', f"{adj['name']}_model.pt")
                torch.save(model.state_dict(), adj_model_path)
                adj_model_size_mb = os.path.getsize(adj_model_path) / 1e6
                print(f"{adj['name']} model saved: {adj_model_path}, Size: {adj_model_size_mb:.2f}MB")
            except OSError as e:
                print(f"Error saving {adj['name']} model: {e}")
            print(f"{adj['name']} Average Test Loss: {np.mean(history['test_loss']):.4f}")
            print(f"{adj['name']} Average Test Accuracy: {np.mean(history['test_acc']):.4f}")
            print(f"{adj['name']} Average Time per Batch: {np.mean(history['time_per_batch']):.4f}s")
            print(f"{adj['name']} Average Peak Memory: {np.mean(history['memory_peak_mb']):.2f}MB")
            print(f"{adj['name']} Model File Size: {adj_model_size_mb:.2f}MB")
            plot_metrics(history, fold=f"Test Set {adj['name']}", is_test=True)
        
        # Final test evaluation
        best_model.eval()
        X_test_tensor = torch.FloatTensor(X_test.reshape(-1, 1, 250)).to(device)
        y_test_tensor = torch.LongTensor(y_test).to(device)
        with torch.no_grad():
            spk_rec, _ = best_model(X_test_tensor)
            y_pred = torch.argmax(spk_rec.sum(dim=0), dim=1).cpu().numpy()
            test_acc = SF.accuracy_rate(spk_rec, y_test_tensor)
            test_precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
            test_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            print("\nTest Set Metrics (CSNN):")
            print(f"  Accuracy: {test_acc:.4f}")
            print(f"  Precision (Macro): {test_precision:.4f}")
            print(f"  F1 Score (Macro): {test_f1:.4f}")
            plot_confusion_matrix(y_test, y_pred, fold=None)
            plot_spike_trains(spk_rec, y_test, idx=0, save_dir='plots/test_set')
        
        print("\n---------------Dataset Summary--------------------")
        print(f"Total number of training beats: {len(X_train)}")
        print(f"Total number of training labels: {len(y_train)}")
        print(f"Total number of test beats: {len(X_test)}")
        print(f"Total number of test labels: {len(y_test)}")
    else:
        print("Insufficient data loaded for training or testing.")
