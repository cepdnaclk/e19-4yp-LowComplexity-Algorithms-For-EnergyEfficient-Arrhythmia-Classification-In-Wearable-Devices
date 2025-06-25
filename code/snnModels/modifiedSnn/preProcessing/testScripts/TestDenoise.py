import matplotlib.pyplot as plt
import numpy as np
import neurokit2 as nk
from preProcessing.Denoise import bandpass_filter, notch_filter, remove_baseline
from preProcessing.Load import load_ecg

def plot_qrs_complex(signal, rpeak, title="QRS Complex"):
    """Plot 250-sample QRS complex centered on R-peak"""
    start = max(rpeak - 125, 0)
    end = min(rpeak + 125, len(signal))
    qrs = signal[start:end]
    
    plt.figure(figsize=(10, 4))
    plt.plot(qrs, label=title)
    plt.xlabel("Samples (250 samples total)")
    plt.ylabel("Amplitude")
    plt.title(f"{title}\nR-peak at sample 125")
    plt.axvline(125, color='r', linestyle='--', label='R-peak')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    
    data_dir = 'data/mitdb'
    record_id = '101'
    
    # Load raw ECG
    signal, rpeaks, fs, ann = load_ecg(record_id, data_dir)
    
    # Detect R-peaks with NeuroKit2
    try:
        ecg_signals, info = nk.ecg_process(signal, sampling_rate=fs)
        rpeaks_nk = info['ECG_R_Peaks']
    except Exception as e:
        print(f"NeuroKit2 processing failed: {e}")
        rpeaks_nk = rpeaks  # fallback to original annotations

    # Use first detected R-peak
    if len(rpeaks_nk) > 0:
        first_rpeak = rpeaks_nk[0]
    else:
        first_rpeak = rpeaks[0]  # fallback if no peaks detected

    # Plot raw QRS complex
    plot_qrs_complex(signal, first_rpeak, "Raw QRS Complex (No Denoising)")

    # Apply filters sequentially and plot
    processed_signals = []
    
    # Bandpass filter
    signal_bp = bandpass_filter(signal, fs)
    plot_qrs_complex(signal_bp, first_rpeak, "After Bandpass Filter")
    processed_signals.append(signal_bp)
    
    # Notch filter
    signal_notch = notch_filter(signal_bp, fs)
    plot_qrs_complex(signal_notch, first_rpeak, "After Notch Filter")
    processed_signals.append(signal_notch)
    
    # Baseline removal
    signal_clean = remove_baseline(signal_notch, fs)
    plot_qrs_complex(signal_clean, first_rpeak, "After Baseline Removal")
    processed_signals.append(signal_clean)
