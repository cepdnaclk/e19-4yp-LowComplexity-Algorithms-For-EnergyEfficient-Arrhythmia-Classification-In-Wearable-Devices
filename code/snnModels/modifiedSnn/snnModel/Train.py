from snnModel.SnnModel import SNN
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
from snnModel.Evaluate import plot_metrics
from sklearn.metrics import f1_score

def train_model(X_train, y_train, X_val, y_val, batch_size=64, num_epochs=10, device='cuda', class_weights=None):
    print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")
    model = SNN(num_inputs=X_train.shape[1], num_outputs=5).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    history = {'train_loss': [], 'train_acc': [], 'train_f1': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_paths = []
    best_val_f1 = 0

    print("Training phase (10 epochs)...")
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0
        correct_train = 0
        total_train = 0
        all_train_preds = []
        all_train_labels = []
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

        train_loss = running_train_loss / total_train
        train_acc = correct_train / total_train
        train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted')
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)

        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_train_epoch_{epoch+1}.pt')
        torch.save(model.state_dict(), checkpoint_path)
        checkpoint_paths.append(checkpoint_path)

        print(f"Train Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")

    print("Validation phase (10 epochs)...")
    for epoch in range(num_epochs):
        model.eval()
        running_val_loss = 0
        correct_val = 0
        total_val = 0
        all_val_preds = []
        all_val_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        val_loss = running_val_loss / total_val
        val_acc = correct_val / total_val
        val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted')
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        scheduler.step(val_loss)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_checkpoint_path = os.path.join(checkpoint_dir, f'best_model.pt')
            torch.save(model.state_dict(), best_checkpoint_path)
        else:
            if epoch >= 2 and val_f1 < best_val_f1:
                print(f"Early stopping triggered at Val Epoch {epoch+1}/{num_epochs} with Val F1: {val_f1:.4f}")
                break

        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_val_epoch_{epoch+1}.pt')
        torch.save(model.state_dict(), checkpoint_path)
        checkpoint_paths.append(checkpoint_path)

        print(f"Val Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

    if best_val_f1 > 0:
        model.load_state_dict(torch.load(best_checkpoint_path))

    for checkpoint_path in checkpoint_paths:
        if checkpoint_path != best_checkpoint_path:
            os.remove(checkpoint_path)
    if os.path.exists(best_checkpoint_path):
        os.remove(best_checkpoint_path)
    os.rmdir(checkpoint_dir)

    return model, history