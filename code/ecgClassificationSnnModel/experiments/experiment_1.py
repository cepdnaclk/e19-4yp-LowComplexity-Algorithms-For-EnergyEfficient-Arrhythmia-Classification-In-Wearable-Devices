"""
Experiment 1: Baseline SNN training with default hyperparameters.
"""

import torch
from models.snn_model import SNNBinaryClassifier
from training.train import train, evaluate
from training.config import Config
import numpy as np

# Load preprocessed data
X_train = np.load('data/train_data.npy')
y_train = np.load('data/train_labels.npy')
X_val = np.load('data/test_data.npy')
y_val = np.load('data/test_labels.npy')

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

# Model configuration
input_size = X_train.shape[1]
hidden_size = 64
output_size = 1  # For binary classification

model = SNNBinaryClassifier(input_size, hidden_size, output_size, time_steps=50)
config = Config(
    lr=0.001,
    batch_size=32,
    epochs=20
)

# Train and evaluate
train(model, X_train, y_train, config)
evaluate(model, X_val, y_val)
