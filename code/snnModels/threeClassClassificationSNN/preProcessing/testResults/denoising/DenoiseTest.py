import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from Denoise import bandpass_filter, notch_filter, remove_baseline

record_name = '100'  # Change to other records as needed
record = wfdb.rdrecord(f'data/mitdb/{record_name}')
fs = record.fs
signal = record.p_signal[:, 0]  # First channel

# Load annotations to find QRS complexes
anno = wfdb.rdann(f'data/mitdb/{record_name}', 'atr')

# Handle cases where anno.symbol might be a scalar or an array
if isinstance(anno.symbol, str):
    # If anno.symbol is a scalar (string), check if it is 'N'
    if anno.symbol == 'N':
        n_indices = np.array([0])  # Single annotation index
    else:
        n_indices = np.array([])  # No 'N' annotations
elif isinstance(anno.symbol, (list, np.ndarray)):
    # If anno.symbol is a list or array, use np.where as usual
    is_N = np.array(anno.symbol) == 'N'
    n_indices = np.where(is_N)[0]
else:
    print("Unexpected type for anno.symbol")
    n_indices = np.array([])

if len(n_indices) == 0:
    print("No normal beats found")
else:
    # Handle anno.sample similarly, as it might also be a scalar
    if isinstance(anno.sample, (int, np.integer)):
        first_n = anno.sample  # Single annotation sample
    elif isinstance(anno.sample, (list, np.ndarray)):
        first_n = anno.sample[n_indices[0]]  # Use the index from n_indices
    else:
        print("Unexpected type for anno.sample")
        first_n = None

    # If first_n is None, you cannot proceed with segment extraction
    if first_n is None:
        print("Cannot extract segment due to invalid anno.sample")
    else:
        # Select a 2-second segment around the first normal beat
        segment_length = int(2 * fs)
        start = max(0, first_n - fs)
        end = start + segment_length

        # Extract original segment
        original_segment = signal[start:end]

        # Apply denoising steps
        s1 = bandpass_filter(signal, fs)
        s1_segment = s1[start:end]

        s2 = notch_filter(s1, fs)
        s2_segment = s2[start:end]

        s3 = remove_baseline(s2, fs)
        s3_segment = s3[start:end]

        # Plot time domain signals
        plt.figure(figsize=(12, 8))
        plt.subplot(4, 1, 1)
        plt.plot(original_segment)
        plt.title('Original Signal')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude (mV)')

        plt.subplot(4, 1, 2)
        plt.plot(s1_segment)
        plt.title('After Bandpass Filter')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude (mV)')

        plt.subplot(4, 1, 3)
        plt.plot(s2_segment)
        plt.title('After Notch Filter')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude (mV)')

        plt.subplot(4, 1, 4)
        plt.plot(s3_segment)
        plt.title('After Baseline Removal')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude (mV)')

        plt.tight_layout()
        plt.show()

        # Plot Power Spectral Density (PSD)
        f, Pxx_orig = welch(original_segment, fs=fs, nperseg=1024)
        f, Pxx_s1 = welch(s1_segment, fs=fs, nperseg=1024)
        f, Pxx_s2 = welch(s2_segment, fs=fs, nperseg=1024)
        f, Pxx_s3 = welch(s3_segment, fs=fs, nperseg=1024)

        plt.figure(figsize=(12, 8))
        plt.subplot(4, 1, 1)
        plt.semilogy(f, Pxx_orig)
        plt.title('Original PSD')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')

        plt.subplot(4, 1, 2)
        plt.semilogy(f, Pxx_s1)
        plt.title('After Bandpass PSD')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')

        plt.subplot(4, 1, 3)
        plt.semilogy(f, Pxx_s2)
        plt.title('After Notch PSD')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')

        plt.subplot(4, 1, 4)
        plt.semilogy(f, Pxx_s3)
        plt.title('After Baseline Removal PSD')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')

        plt.tight_layout()
        plt.show()

        # Check QRS amplitude preservation
        qrs_window = slice(int(fs - 0.1 * fs), int(fs + 0.1 * fs))  # ~0.2s around QRS
        max_orig = np.max(np.abs(original_segment[qrs_window]))
        max_s1 = np.max(np.abs(s1_segment[qrs_window]))
        max_s2 = np.max(np.abs(s2_segment[qrs_window]))
        max_s3 = np.max(np.abs(s3_segment[qrs_window]))

        print("Max absolute amplitude in QRS window:")
        print(f"  Original: {max_orig:.3f} mV")
        print(f"  After Bandpass: {max_s1:.3f} mV")
        print(f"  After Notch: {max_s2:.3f} mV")
        print(f"  After Baseline Removal: {max_s3:.3f} mV")