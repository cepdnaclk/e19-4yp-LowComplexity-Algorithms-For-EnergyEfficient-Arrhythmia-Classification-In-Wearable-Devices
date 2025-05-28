# Contains the code for normalization

import numpy as np

def normalize_beats(beats):
    """Normalize each beat independently"""
    return np.array([(beat - np.mean(beat))/np.std(beat) for beat in beats])
