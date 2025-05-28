"""
utils.py
--------
Utility functions for training and evaluation.
"""

import numpy as np
from sklearn.metrics import accuracy_score as sk_accuracy_score, confusion_matrix as sk_confusion_matrix

def accuracy_score(y_true, y_pred):
    """
    Compute accuracy score.
    Args:
        y_true (np.array): True binary labels.
        y_pred (np.array): Predicted binary labels.
    Returns:
        float: Accuracy score.
    """
    return sk_accuracy_score(y_true, y_pred)

def confusion_matrix(y_true, y_pred):
    """
    Compute confusion matrix.
    Args:
        y_true (np.array): True binary labels.
        y_pred (np.array): Predicted binary labels.
    Returns:
        np.array: Confusion matrix.
    """
    return sk_confusion_matrix(y_true, y_pred)
