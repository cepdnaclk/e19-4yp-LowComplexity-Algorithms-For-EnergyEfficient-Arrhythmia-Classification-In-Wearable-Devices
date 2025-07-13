import numpy as np

# Contains the code for normalization -> Beat-wise z-score normalization
# This method standardizes each individual beat by centering its values around zero and scaling them to have unit variance.
# For each beat (a vector of ECG samples), the mean value of that beat is subtracted from every sample, centering the beat's 
# data around zero. Then, the result is divided by the standard deviation of that beat, scaling the beat’s amplitude so that 
# it has a standard deviation of one.

# ECG signals vary in amplitude due to patient differences, electrode placement, and noise. Normalization removes these variations
# so the model focuses on waveform shape rather than absolute voltage.

def normalize_beats_beat_wise_z_score(beats):
    """Normalize each beat independently"""
    return np.array([(beat - np.mean(beat))/np.std(beat) for beat in beats])



# Min-Max Normalization (Scaling to )
# Scales each beat independently so that its minimum value maps to 0 and maximum to 1.
def normalize_beats_min_max(beats):
    normalized = []
    for beat in beats:
        min_val = np.min(beat)
        max_val = np.max(beat)
        norm_beat = (beat - min_val) / (max_val - min_val + 1e-8)  # Add epsilon to avoid div by zero
        normalized.append(norm_beat)
    return np.array(normalized)


# Median and Interquartile Range (Robust Scaling)
# Centers the beat using the median and scales by the interquartile range (IQR), which is robust to outliers.
def normalize_beats_median_interquartile(beats):
    normalized = []
    for beat in beats:
        median = np.median(beat)
        q75, q25 = np.percentile(beat, [75 ,25])
        iqr = q75 - q25 + 1e-8
        norm_beat = (beat - median) / iqr
        normalized.append(norm_beat)
    return np.array(normalized)


#  Global Z-Score Normalization
# Compute mean and std over the entire training dataset and normalize all beats using these global statistics.
def normalize_beats_global_z_score(beats):
    all_samples = beats.flatten()
    
    global_mean = np.mean(all_samples)
    global_std = np.std(all_samples)
    
    # Normalize using global statistics
    normalized_beats = (beats - global_mean) / (global_std + 1e-8)
    
    print(f"Global mean: {global_mean}")
    print(f"Global std: {global_std}")
    
    return normalized_beats


# L2 Normalization (Unit Norm)
# Scales each beat so that its L2 norm (Euclidean length) is 1.
def normalize_beats_l2_normalization(beats):
    normalized = []
    for beat in beats:
        norm = np.linalg.norm(beat) + 1e-8
        norm_beat = beat / norm
        normalized.append(norm_beat)
    return np.array(normalized)


# Max Absolute Scaling
# Scales each beat by dividing by its maximum absolute value, so the values lie in [-1, 1].
def normalize_beats(beats):
    normalized = []
    for beat in beats:
        max_abs = np.max(np.abs(beat)) + 1e-8
        norm_beat = beat / max_abs
        normalized.append(norm_beat)
    return np.array(normalized)

