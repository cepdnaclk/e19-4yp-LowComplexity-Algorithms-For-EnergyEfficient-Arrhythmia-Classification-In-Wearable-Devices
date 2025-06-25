from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import numpy as np

def balance_classes(X, y, method='smote'):
    """Handle class imbalance using different techniques."""
    if method == 'smote':
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
    elif method == 'undersample':
        rus = RandomUnderSampler(random_state=42)
        X_res, y_res = rus.fit_resample(X, y)
    else:
        return X, y  # No balancing
    
    return X_res, y_res
