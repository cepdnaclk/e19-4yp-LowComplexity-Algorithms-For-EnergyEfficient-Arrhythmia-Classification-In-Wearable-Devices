"""
evaluate.py
-----------
Evaluate the trained model on a labeled test set and print metrics.
"""

import torch
import numpy as np
from models.snn_model import SNNBinaryClassifier
from training.utils import accuracy_score, confusion_matrix

def load_model(model_path, input_size, hidden_size, output_size, time_steps=100):
    model = SNNBinaryClassifier(input_size, hidden_size, output_size, time_steps)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def evaluate(model, X, y):
    with torch.no_grad():
        outputs = model(X)
        preds = (outputs > 0.5).float()
        acc = accuracy_score(y.numpy(), preds.numpy())
        cm = confusion_matrix(y.numpy(), preds.numpy())
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    # Example usage
    X_test = np.load('data/test_data.npy')
    y_test = np.load('data/test_labels.npy')
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Adjust these according to your model
    input_size = X_test.shape[1]
    hidden_size = 64
    output_size = 1
    time_steps = 10

    model = load_model('saved_models/best_model.pth', input_size, hidden_size, output_size, time_steps)
    evaluate(model, X_test, y_test)
