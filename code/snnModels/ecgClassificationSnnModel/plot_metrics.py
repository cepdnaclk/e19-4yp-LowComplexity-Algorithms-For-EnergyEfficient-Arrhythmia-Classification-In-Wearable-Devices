import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve

def plot_classification_metrics(y_true, y_pred, y_scores=None, class_names=['Normal', 'Abnormal']):
    """
    Plot accuracy, confusion matrix, and optionally ROC and Precision-Recall curves.

    Args:
        y_true (np.array or list): True binary labels (0/1).
        y_pred (np.array or list): Predicted binary labels (0/1).
        y_scores (np.array or list, optional): Predicted probabilities or scores for positive class.
        class_names (list): Names of the classes for confusion matrix display.
    """
    # Compute accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(f"Number of test samples: {len(y_true)}")

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

    # If prediction scores provided, plot ROC and Precision-Recall curves
    if y_scores is not None:
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        plt.figure(figsize=(6, 6))
        plt.plot(recall, precision, color='purple', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    y_true = np.load('test_labels.npy')  # If you saved labels
    y_pred = np.load('test_pred.npy')      # If you saved predictions
    y_scores = np.load('test_scores.npy')  # If you saved scores/probabilities

    plot_classification_metrics(y_true, y_pred, y_scores)
