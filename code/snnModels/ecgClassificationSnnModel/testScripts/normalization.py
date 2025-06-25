# normalize.py

import numpy as np

def zscore_normalize(X):
    """
    Apply z-score normalization to each sample in the dataset.

    Args:
        X (np.array): 2D array of shape (num_samples, segment_length)

    Returns:
        np.array: Normalized array with zero mean and unit variance per sample
    """
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-8  # avoid division by zero
    X_norm = (X - mean) / std
    return X_norm

def minmax_normalize(X):
    """
    Apply min-max normalization to each sample in the dataset.

    Args:
        X (np.array): 2D array of shape (num_samples, segment_length)

    Returns:
        np.array: Normalized array scaled to [0, 1] per sample
    """
    min_val = X.min(axis=1, keepdims=True)
    max_val = X.max(axis=1, keepdims=True)
    X_norm = (X - min_val) / (max_val - min_val + 1e-8)
    return X_norm
