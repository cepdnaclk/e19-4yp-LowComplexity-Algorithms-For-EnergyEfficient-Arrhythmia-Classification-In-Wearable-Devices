# Delta Modulation Function -> generete spikes
# Delta Modulation (DM) is a signal encoding technique that converts 
# an analog signal into a digital pulse stream by encoding the difference 
# between successive samples rather than the absolute sample values. 

import numpy as np 
import torch 
from snntorch import spikegen

def delta_modulation(beats, threshold=0.01):
    return np.array([spikegen.delta(torch.tensor(beat), threshold=threshold, off_spike=True).numpy() for beat in beats])

# off_spike=True means spikes are generated for both positive and negative changes (i.e., when the signal goes up or down).