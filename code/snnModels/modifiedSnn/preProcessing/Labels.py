import numpy as np

# Binary AAMI classes
AAMI_classes = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  # Normal
    'A': 1, 'a': 1, 'J': 1, 'S': 1,         # Abnormal (Supraventricular)
    'V': 1, 'E': 1,                         # Abnormal (Ventricular)
    'F': 1,                                 # Abnormal (Fusion)
    'P': 1, '/': 1, 'f': 1, 'u': 1          # Abnormal (Paced/Unknown)
}

def create_labels(rpeaks, ann):
    """
    Create binary labels for ECG beats based on annotations.
    Args:
        rpeaks: Array of R-peak sample indices
        ann: Annotation object with 'sample' and 'symbol' attributes
    Returns:
        labels: Array of binary labels (0: Normal, 1: Abnormal)
    """
    labels = []
    for rpeak in rpeaks:
        idx = np.where(ann.sample == rpeak)[0]
        if len(idx) > 0:
            symbol = ann.symbol[idx[0]]
            labels.append(AAMI_classes.get(symbol, 1))  # Default to abnormal if symbol not in AAMI_classes
        else:
            labels.append(1)  # Default to abnormal if no annotation
    return np.array(labels)