import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from preProcessing.Segment import extract_heartbeats
from preProcessing.Load import load_ecg
import matplotlib.pyplot as plt 

record_id = 101
data_dir = './data/mitdb'

signal, rpeaks, fs, ann = load_ecg(record_id, data_dir)
print("Number of events/R peaks found by annotations : ", len(rpeaks))

beats, valid_rpeaks = extract_heartbeats(signal,fs,ann.sample)
print("Number of events/R peaks found by Pan-Tompkins algorithm : ", len(valid_rpeaks))
print("Number of segments taken from the record : ", len(beats))
print("Accuracy of the algorithm : ", len(valid_rpeaks)/len(rpeaks))

# Plot ECG with detected R-peaks
plt.figure(figsize=(15, 4))
plt.plot(signal, label='Synthetic ECG')
plt.scatter(rpeaks, signal[rpeaks], color='red', label='Detected R-peaks')
plt.title('ECG Signal with Detected R-peaks (Pan-Tompkins)')
plt.xlabel('Sample Number')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

# Plot a few extracted beats (QRS complexes)
plt.figure(figsize=(12, 6))
for i in range(min(5, len(beats))):
    plt.plot(beats[i], label=f'Beat {i+1}')
plt.title('Extracted Fixed-Length Heartbeats Centered at R-peaks')
plt.xlabel('Sample Number (per beat)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()