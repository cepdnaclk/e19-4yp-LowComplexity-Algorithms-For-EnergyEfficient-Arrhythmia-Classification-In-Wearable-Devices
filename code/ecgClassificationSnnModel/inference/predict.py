"""
predict.py
----------
Load a trained model and make predictions on new data.
"""

import torch
import numpy as np
from models.snn_model import SNNBinaryClassifier

def load_model(model_path, input_size, hidden_size, output_size, time_steps=100):
    model = SNNBinaryClassifier(input_size, hidden_size, output_size, time_steps)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict(model, X):
    with torch.no_grad():
        outputs = model(X)
        preds = (outputs > 0.5).float()
    return preds

if __name__ == "__main__":
    # Example usage
    X_new = np.load('data/test_data.npy')
    X_new = torch.tensor(X_new, dtype=torch.float32)

    # Adjust these according to your model
    input_size = X_new.shape[1]
    hidden_size = 64
    output_size = 1
    time_steps = 50

    model = load_model('saved_models/best_model.pth', input_size, hidden_size, output_size, time_steps)
    predictions = predict(model, X_new)

    print("Predictions:", predictions.numpy().flatten())
