import numpy as np
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset

def evaluate_model(model, X_val, y_val, X_test, y_test, device='cuda'):
    def compute_metrics(X, y, dataset_name):
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.LongTensor(y).to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(X_tensor)
            loss = nn.CrossEntropyLoss()(outputs, y_tensor).item()
            _, predicted = torch.max(outputs, 1)
            y_np = y_tensor.cpu().numpy()
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
                        xticklabels=['Normal', 'SVEB', 'VEB', 'Fusion', 'Unknown'],
                        yticklabels=['Normal', 'SVEB', 'VEB', 'Fusion', 'Unknown'])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'{dataset_name} Confusion Matrix')
            plt.savefig(f'{dataset_name.lower()}_confusion_matrix.png')
            plt.close()

    compute_metrics(X_val, y_val, 'Validation')
    compute_metrics(X_test, y_test, 'Test')

def evaluate_model_epochs(model, X_test, y_test, num_epochs=10, device='cuda'):
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
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

    # Compute average metrics
    avg_test_loss = np.mean(history['test_loss'])
    avg_test_acc = np.mean(history['test_acc'])
    print(f"\nAverage Test Metrics over {num_epochs} epochs:")
    print(f"  Average Test Loss: {avg_test_loss:.4f}")
    print(f"  Average Test Accuracy: {avg_test_acc:.4f}")

    return history

def plot_metrics(history):
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