# This code is to assign labels for ECG classes

# AAMI_classes = {
#     0: ['N', 'L', 'R', 'e', 'j'],  -> Normal beats
#     1: ['A', 'a', 'J', 'S'],       -> SVEB - Supraventricular ectopic beats
#     2: ['V', 'E'],                 -> VEB - Ventricular ectopic beats
#     3: ['F'],                      -> Fusion beats
#     4: ['P', '/', 'f', 'u']        -> Unknown / unclassified beats
# }

import numpy as np

AAMI_classes = {
    0: ['N', 'L', 'R', 'e', 'j'],      
    1: ['A', 'a', 'J', 'S'],          
    2: ['V', 'E'],                    
    3: ['F'],                         
    4: ['P', '/', 'f', 'u']           
}

def get_class_from_symbol(symbol):
    """Map beat symbol to class index based on AAMI classes."""
    for class_idx, symbols in AAMI_classes.items():
        if symbol in symbols:
            return class_idx
    # If symbol not found in any class, consider it normal (class 0)
    return 0

def create_labels(rpeaks, annotation):
    """
    Create multi-class labels for detected R-peaks based on annotation symbols.
    
    Args:
        rpeaks (np.array): Detected R-peak sample indices
        annotation (wfdb.Annotation): Annotation object with .sample and .symbol arrays
    
    Returns:
        np.array: Array of class labels for each R-peak
    """
    labels = []
    beat_symbols = annotation.symbol
    annotation_samples = annotation.sample
    
    for peak in rpeaks:
        idx = np.argmin(np.abs(annotation_samples - peak))
        symbol = beat_symbols[idx]
        class_idx = get_class_from_symbol(symbol)
        labels.append(class_idx)
    
    return np.array(labels)

