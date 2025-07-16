# DeltaModulation.py
import numpy as np

def delta_modulation(beats, threshold=0.01, decay=0.9):
    spikes = np.zeros_like(beats, dtype=float)
    prev_spike = np.zeros(beats.shape[1])
    for i in range(beats.shape[0]):
        for j in range(beats.shape[1]):
            diff = beats[i, j] - prev_spike[j]
            if diff > threshold:
                spikes[i, j] = 1.0
            elif diff < -threshold:
                spikes[i, j] = -1.0
            prev_spike[j] = decay * prev_spike[j] + (1 - decay) * beats[i, j]
    return spikes