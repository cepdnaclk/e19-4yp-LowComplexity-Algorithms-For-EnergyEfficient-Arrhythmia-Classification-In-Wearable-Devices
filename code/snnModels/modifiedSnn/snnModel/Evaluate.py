# Evaluation Function
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

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

# Plot Metrics
def plot_metrics(history):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='#1f77b4')
    plt.plot(history['val_loss'], label='Validation Loss', color='#ff7f0e')
    plt.plot(history['test_loss'], label='Test Loss', color='#2ca02c')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training, Validation, and Test Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(history['train_acc'], label='Train Accuracy', color='#1f77b4')
    plt.plot(history['val_acc'], label='Validation Accuracy', color='#ff7f0e')
    plt.plot(history['test_acc'], label='Test Accuracy', color='#2ca02c')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training, Validation, and Test Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('metrics.png')
    plt.close()