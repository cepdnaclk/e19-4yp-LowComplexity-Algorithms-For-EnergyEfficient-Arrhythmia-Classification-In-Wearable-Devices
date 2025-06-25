import matplotlib.pyplot as plt
import numpy as np
import torch
from preProcessing.Load import load_ecg
from preProcessing.Denoise import bandpass_filter, notch_filter, remove_baseline
from preProcessing.Normalization import normalize_beats
from preProcessing.Segment import extract_heartbeats
from snntorch import spikegen

def delta_modulation(beats, threshold=0.1):
    return np.array([
        spikegen.delta(torch.tensor(beat), threshold=threshold, off_spike=True).numpy()
        for beat in beats
    ])

def plot_beat_and_spikes(beat, spikes, fs, title_prefix=""):
    time_axis = np.arange(len(beat)) / fs
    plt.figure(figsize=(12, 5))
    
    plt.subplot(2,1,1)
    plt.plot(time_axis, beat)
    plt.title(f"{title_prefix} Normalized ECG Beat")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    plt.subplot(2,1,2)
    plt.plot(time_axis, spikes)
    plt.title(f"{title_prefix} Delta Modulation Spike Train")
    plt.xlabel("Time (s)")
    plt.ylabel("Spike")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data_dir = 'data/mitdb'
    record_id = '101'
    
    # Load raw ECG signal and R-peaks
    signal, rpeaks, fs, ann = load_ecg(record_id, data_dir)
    
    # Preprocess signal (denoising)
    signal = bandpass_filter(signal, fs)
    signal = notch_filter(signal, fs)
    signal = remove_baseline(signal, fs)
    
    # Extract beats around R-peaks (e.g., 250 samples centered on R-peak)
    # beats = []
    # half_len = 125
    # for rpeak in rpeaks[:5]:  # take first 5 beats for example
    #     start = max(rpeak - half_len, 0)
    #     end = min(rpeak + half_len, len(signal))
    #     beat = signal[start:end]
    #     if len(beat) == 2*half_len:
    #         beats.append(beat)
    # beats = np.array(beats)
    beats, valid_rpeaks = extract_heartbeats(signal, fs, ann.sample)
    
    # Normalize beats
    beats_norm = normalize_beats(beats)
    
    # Delta modulation to generate spikes
    beats_spikes = delta_modulation(beats_norm, threshold=0.1)
    
    # Plot one example beat and its spike train
    plot_beat_and_spikes(beats_norm[0], beats_spikes[0], fs, title_prefix="Example")
