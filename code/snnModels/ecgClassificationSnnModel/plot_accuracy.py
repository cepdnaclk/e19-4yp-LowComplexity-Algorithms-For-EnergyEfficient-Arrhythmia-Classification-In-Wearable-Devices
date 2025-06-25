import numpy as np
import matplotlib.pyplot as plt
import os

def plot_accuracy_curve(acc_history_path='accuracy_history.npy'):
    if not os.path.exists(acc_history_path):
        print(f"Accuracy history file not found at {acc_history_path}")
        return

    accuracy_history = np.load(acc_history_path)
    epochs = np.arange(1, len(accuracy_history) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, accuracy_history, marker='o', linestyle='-', color='b')
    plt.title('Training Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.xticks(epochs)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_accuracy_curve()
