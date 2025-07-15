import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset

def evaluate_model(model, X_val, y_val, X_test, y_test, device='cuda'):
    """Evaluate the model on validation and test sets for 3-class classification."""
    def compute_metrics(X, y, dataset_name):
        # Ensure inputs are tensors on the correct device
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float, device=device)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long, device=device)
        
        # Validate labels
        if torch.any(y < 0) or torch.any(y > 2):
            raise ValueError(f"Invalid labels in {dataset_name} set: must be in [0, 2], found {torch.unique(y)}")
        
        model.eval()
        with torch.no_grad():
            outputs = model(X)
            loss = nn.CrossEntropyLoss()(outputs, y).item()
            _, predicted = torch.max(outputs, 1)
            y_np = y.cpu().numpy()
            predicted_np = predicted.cpu().numpy()

            accuracy = accuracy_score(y_np, predicted_np)
            precision = precision_score(y_np, predicted_np, average='macro')
            recall = recall_score(y_np, predicted_np, average='macro')
            f1 = f1_score(y_np, predicted_np, average='macro')
            cm = confusion_matrix(y_np, predicted_np)

            print(f"\n{dataset_name} Metrics:")
            print(f"  Loss: {loss:.4f}")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  Confusion Matrix:\n{cm}")

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Normal', 'SVEB/VEB', 'Fusion/Paced/Other'],
                        yticklabels=['Normal', 'SVEB/VEB', 'Fusion/Paced/Other'])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'{dataset_name} Confusion Matrix')
            plt.savefig(f'{dataset_name.lower()}_confusion_matrix.png')
            plt.close()

            return {
                'loss': loss,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': cm
            }

    val_metrics = compute_metrics(X_val, y_val, 'Validation')
    test_metrics = compute_metrics(X_test, y_test, 'Test')
    return val_metrics, test_metrics

def evaluate_model_epochs(model, X_test, y_test, num_epochs=10, device='cuda'):
    """Evaluate the model on the test set over multiple epochs for 3-class classification."""
    if not isinstance(X_test, torch.Tensor):
        X_test = torch.tensor(X_test, dtype=torch.float, device=device)
    if not isinstance(y_test, torch.Tensor):
        y_test = torch.tensor(y_test, dtype=torch.long, device=device)
    
    # Validate labels
    if torch.any(y_test < 0) or torch.any(y_test > 2):
        raise ValueError(f"Invalid test labels: must be in [0, 2], found {torch.unique(y_test)}")
    
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    history = {
        'test_loss': [],
        'test_acc': []
    }

    print("Testing phase...")
    for epoch in range(num_epochs):
        model.eval()
        running_test_loss = 0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                running_test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_loss = running_test_loss / total_test
        test_acc = correct_test / total_test
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        print(f"Test Epoch {epoch+1}/{num_epochs}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    avg_test_loss = np.mean(history['test_loss'])
    avg_test_acc = np.mean(history['test_acc'])
    print(f"\nAverage Test Metrics over {num_epochs} epochs:")
    print(f"  Average Test Loss: {avg_test_loss:.4f}")
    print(f"  Average Test Accuracy: {avg_test_acc:.4f}")

    return history

def plot_metrics(history):
    """Plot loss and accuracy curves."""
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    if 'train_loss' in history:
        plt.plot(history['train_loss'], label='Train Loss', color='#1f77b4')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss', color='#ff7f0e')
    if 'test_loss' in history:
        plt.plot(history['test_loss'], label='Test Loss', color='#2ca02c')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()

    plt.subplot(2, 1, 2)
    if 'train_acc' in history:
        plt.plot(history['train_acc'], label='Train Accuracy', color='#1f77b4')
    if 'val_acc' in history:
        plt.plot(history['val_acc'], label='Validation Accuracy', color='#ff7f0e')
    if 'test_acc' in history:
        plt.plot(history['test_acc'], label='Test Accuracy', color='#2ca02c')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curves')
    plt.legend()

    plt.tight_layout()
    plt.savefig('metrics.png')
    plt.close()