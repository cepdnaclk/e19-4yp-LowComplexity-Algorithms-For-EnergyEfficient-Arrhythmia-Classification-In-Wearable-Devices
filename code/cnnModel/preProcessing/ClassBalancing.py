# Contains the code for class balancing
# Uses Synthetic Minority Over-sampling Technique(SMOTE) algorithm

from imblearn.over_sampling import SMOTE, RandomOverSampler
import numpy as np
from collections import Counter

def balance_classes(X, y):
    y = np.ravel(y)
    class_counts = Counter(y)
    min_class_size = min(class_counts.values())
    
    # print(f"Class distribution before balancing: {class_counts}")
    
    if min_class_size < 2:
        # print("Some classes have fewer than 2 samples. Using RandomOverSampler instead of SMOTE.")
        ros = RandomOverSampler(random_state=42)
        return ros.fit_resample(X, y)
    
    k_neighbors = min(5, min_class_size - 1)
    # print(f"Using SMOTE with k_neighbors={k_neighbors}")
    smote = SMOTE(random_state=42, sampling_strategy='not majority', k_neighbors=k_neighbors)
    return smote.fit_resample(X, y)

