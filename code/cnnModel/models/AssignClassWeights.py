import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def get_class_weights(y_train):
    """
    Compute class weights dictionary for imbalanced data.
    Args:
        y_train: 1D numpy array of training labels
    Returns:
        class_weights: dict mapping class indices to weights
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
    print(f"Computed class weights: {class_weights}")
    return class_weights

