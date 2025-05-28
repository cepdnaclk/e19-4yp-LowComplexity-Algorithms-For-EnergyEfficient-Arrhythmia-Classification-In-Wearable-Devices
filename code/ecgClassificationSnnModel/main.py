import torch
import numpy as np
import os
from models.snn_model import SNNConvClassifier
from training.train import train, evaluate
from training.config import Config

def main():
    # Load data
    X_train = np.load('testScripts/data/train_data.npy')
    y_train = np.load('testScripts/data/train_labels.npy')
    X_test = np.load('testScripts/data/test_data.npy')
    y_test = np.load('testScripts/data/test_labels.npy')

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # Reshape for Conv1d: (batch, channels=1, length)
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    num_inputs = X_train.shape[2]
    num_hidden = 128
    num_outputs = 1  # For binary classification with BCEWithLogitsLoss
    num_steps = 10   # For faster training

    model = SNNConvClassifier(num_inputs=num_inputs, num_hidden=num_hidden, num_outputs=num_outputs, num_steps=num_steps)
    config = Config(lr=0.001, batch_size=32, epochs=10)  # Fewer epochs for speed

    print("Starting training...")
    accuracy_history = train(model, X_train, y_train, config)

    print("Evaluating model on test data...")
    evaluate(model, X_test, y_test)

    # Create directories if they don't exist
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)

    # Save the trained model
    torch.save(model.state_dict(), 'saved_models/best_model.pth')
    print("Model saved to saved_models/best_model.pth")

    # Save accuracy history for plotting
    np.save('visualizations/accuracy_history.npy', np.array(accuracy_history))
    print("Training accuracy history saved to visualizations/accuracy_history.npy")

if __name__ == "__main__":
    main()
