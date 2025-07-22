
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import shutil
import snntorch as snn
import snntorch.functional as SF
import time
import tracemalloc
import numpy as np
from snnModel.SnnModel import CSNN

def train_model(X_train, y_train, batch_size=64, num_epochs=20, device='cuda', num_steps=50, beta=0.5):
    """
    Train a CSNN model for all epochs, measuring time, memory, and model file size.
    Args:
        X_train: Training inputs (shape: [n_samples, num_inputs])
        y_train: Training labels
        batch_size: Batch size
        num_epochs: Number of epochs
        device: Device to run model on
        num_steps: Number of time steps
        beta: Decay rate
    Returns:
        model: Trained model
        history: Training metrics
    """
    model = CSNN(num_inputs=250, num_outputs=5, num_steps=num_steps, beta=beta, device=device).to(device)
    loss_fn = SF.ce_rate_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.9, 0.999))

    # Reshape inputs
    X_train = X_train.reshape(-1, 1, 250)

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)

    # Create data loader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    history = {
        'train_loss': [], 'train_acc': [],
        'train_time_per_batch': [], 'train_memory_peak_mb': [],
        'model_file_size_mb': []
    }
    
    checkpoint_dir = os.path.abspath('checkpoints')
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Created checkpoint directory: {checkpoint_dir}")
    except OSError as e:
        print(f"Error creating checkpoint directory: {e}")
        raise

    checkpoint_paths = []

    print("Training CSNN phase...")
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0
        correct_train = 0
        total_train = 0
        train_batch_times = []
        train_batch_memories = []
        
        tracemalloc.start()
        for data, labels in train_loader:
            start_time = time.perf_counter()
            optimizer.zero_grad()
            spk_rec, _ = model(data)
            loss = loss_fn(spk_rec, labels)
            acc = SF.accuracy_rate(spk_rec, labels)
            
            loss.backward()
            optimizer.step()
            
            end_time = time.perf_counter()
            train_batch_times.append(end_time - start_time)
            
            current, peak = tracemalloc.get_traced_memory()
            train_batch_memories.append(peak / 1e6)  # Convert to MB
            tracemalloc.reset_peak()
            
            running_train_loss += loss.item() * data.size(0)
            correct_train += acc * data.size(0)
            total_train += data.size(0)
        
        tracemalloc.stop()
        
        train_loss = running_train_loss / total_train
        train_acc = correct_train / total_train
        train_time = np.mean(train_batch_times)
        train_memory = np.mean(train_batch_memories)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_time_per_batch'].append(train_time)
        history['train_memory_peak_mb'].append(train_memory)

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
        try:
            torch.save(model.state_dict(), checkpoint_path)
            checkpoint_paths.append(checkpoint_path)
            file_size_mb = os.path.getsize(checkpoint_path) / 1e6  # Convert to MB
            history['model_file_size_mb'].append(file_size_mb)
            print(f"Saved checkpoint: {checkpoint_path}, Size: {file_size_mb:.2f}MB")
        except OSError as e:
            print(f"Error saving checkpoint {checkpoint_path}: {e}")
            raise

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Train Time/Batch: {train_time:.4f}s, Train Memory: {train_memory:.2f}MB, "
              f"Model Size: {file_size_mb:.2f}MB")

    # Clean up checkpoints
    try:
        for checkpoint_path in checkpoint_paths:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        print(f"Cleaned up checkpoint directory: {checkpoint_dir}")
    except OSError as e:
        print(f"Warning: Failed to clean up checkpoints: {e}")

    return model, history

def validate_model(model, X_val, y_val, batch_size=64, device='cuda', num_steps=50):
    """
    Validate a trained CSNN model, measuring time and memory.
    Args:
        model: Trained CSNN model
        X_val: Validation inputs
        y_val: Validation labels
        batch_size: Batch size
        device: Device to run model on
        num_steps: Number of time steps
    Returns:
        val_loss: Validation loss
        val_acc: Validation accuracy
        val_time_per_batch: Average time per batch (s)
        val_memory_peak_mb: Average peak memory (MB)
        y_pred: Predicted labels
    """
    model.eval()
    X_val = X_val.reshape(-1, 1, 250)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    loss_fn = SF.ce_rate_loss()
    running_val_loss = 0
    correct_val = 0
    total_val = 0
    val_batch_times = []
    val_batch_memories = []
    y_pred_all = []
    
    tracemalloc.start()
    with torch.no_grad():
        for data, labels in val_loader:
            start_time = time.perf_counter()
            spk_rec, _ = model(data)
            loss = loss_fn(spk_rec, labels)
            acc = SF.accuracy_rate(spk_rec, labels)
            y_pred = torch.argmax(spk_rec.sum(dim=0), dim=1).cpu().numpy()
            
            end_time = time.perf_counter()
            val_batch_times.append(end_time - start_time)
            
            current, peak = tracemalloc.get_traced_memory()
            val_batch_memories.append(peak / 1e6)
            tracemalloc.reset_peak()
            
            running_val_loss += loss.item() * data.size(0)
            correct_val += acc * data.size(0)
            total_val += data.size(0)
            y_pred_all.extend(y_pred)
    
    tracemalloc.stop()
    
    val_loss = running_val_loss / total_val
    val_acc = correct_val / total_val
    val_time = np.mean(val_batch_times)
    val_memory = np.mean(val_batch_memories)
    
    return val_loss, val_acc, val_time, val_memory, np.array(y_pred_all)
