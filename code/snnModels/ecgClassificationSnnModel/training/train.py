"""
train.py
--------
Training loop for the SNN binary classification model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from training.utils import accuracy_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

def train(model, X_train, y_train, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    model.train()

    accuracy_history = []

    for epoch in range(1, config.epochs + 1):
        epoch_loss = 0.0
        epoch_acc = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
            epoch_acc += acc

        avg_loss = epoch_loss / len(dataloader)
        avg_acc = epoch_acc / len(dataloader)
        accuracy_history.append(avg_acc)
        print(f"Epoch {epoch}/{config.epochs} - Loss: {avg_loss:.4f} - Accuracy: {avg_acc:.4f}")

    return accuracy_history


def evaluate(model, X_test, y_test, batch_size=128):
    print("Entered evaluate function")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    dataset = TensorDataset(X_test, y_test)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    all_preds, all_labels, all_scores = [], [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            scores = torch.sigmoid(outputs)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_scores.append(scores.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_scores = torch.cat(all_scores).numpy()
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Number of test samples: {len(y_true)}")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("Confusion Matrix:\n", cm)

    # # Save predictions and scores
    np.save('test_labels.npy', y_true)
    np.save('test_pred.npy', y_pred)
    np.save('test_scores.npy', y_scores)
    print("Saved predictions to ./data/test_pred.npy and scores to ./data/test_scores.npy")