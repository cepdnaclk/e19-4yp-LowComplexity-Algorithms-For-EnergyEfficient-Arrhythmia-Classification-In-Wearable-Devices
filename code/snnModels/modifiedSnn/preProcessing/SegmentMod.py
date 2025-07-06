import neurokit2 as nk
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os

def extract_heartbeats(signal, fs, annotation_rpeaks=None, annotation=None, before=0.25, after=0.4, fixed_length=250, plot_dir='ecgPlots'):
    """
    Extract segments including previous, current, and next heartbeats centered at R-peaks using a sliding window
    for all valid beats in the record.

    Args:
        signal: ECG signal
        fs: Sampling frequency (Hz)
        annotation_rpeaks: Optional pre-annotated R-peaks
        annotation: Annotation object with .sample and .symbol arrays for labeling
        before: Seconds before R-peak for a single beat (default 0.25)
        after: Seconds after R-peak for a single beat (default 0.4)
        fixed_length: Target samples per beat (default 250)
        plot_dir: Directory to save segment plots (default 'plots/ecgPlots')

    Returns:
        segments: Array of fixed-length segments, each including previous, current, and next beats (n_segments, fixed_length*3)
        valid_rpeaks: Array of used R-peak positions (middle beats)
    """
    # Clean signal and detect R-peaks using neurokit2
    cleaned = nk.ecg_clean(signal, sampling_rate=fs)
    rpeaks = annotation_rpeaks if annotation_rpeaks is not None else \
             nk.ecg_findpeaks(cleaned, sampling_rate=fs)['ECG_R_Peaks']

    # Ensure plot directory exists
    print(f"Creating directory: {plot_dir}")
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Directory {plot_dir} is ready or already exists")

    segments = []
    valid_rpeaks = []

    # Single beat window size in samples
    samples_before = int(before * fs)
    samples_after = int(after * fs)
    single_beat_length = samples_before + samples_after
    total_length = fixed_length * 3  # Three beats per segment (750 samples)

    for i, peak in enumerate(rpeaks):
        # Skip first R-peak if no previous beat, but include last if next beat is available
        if i == 0 and len(rpeaks) > 1:
            continue
        if i == len(rpeaks) - 1 and len(rpeaks) > 1:
            break

        # Calculate midpoints for sliding window
        prev_mid = (rpeaks[i-1] + peak) // 2 if i > 0 else peak - single_beat_length
        next_mid = (peak + rpeaks[i+1]) // 2 if i < len(rpeaks) - 1 else peak + single_beat_length

        # Define segment boundaries
        start = prev_mid - single_beat_length if i > 0 else peak - 2 * single_beat_length
        end = next_mid + single_beat_length if i < len(rpeaks) - 1 else peak + 2 * single_beat_length

        # Handle edge cases
        if start < 0:
            start = 0
        if end > len(signal):
            end = len(signal)

        # Extract segment
        segment = signal[start:end]

        # Pad or trim to fixed length (750 samples)
        if len(segment) < total_length:
            segment = np.pad(segment, (0, total_length - len(segment)), mode='constant')
        elif len(segment) > total_length:
            segment = segment[:total_length]

        segments.append(segment)
        valid_rpeaks.append(peak)

        # Plot specific segments (first, 1001st, and every 1000th thereafter)
        if annotation is not None:
            segment_idx = len(segments) - 1
            if segment_idx == 0 or segment_idx == 1000 or (segment_idx - 1000) % 1000 == 0:
                print(f"Preparing to plot segment {segment_idx}")
                idx = np.argmin(np.abs(annotation.sample - peak))
                symbol = annotation.symbol[idx]
                print(f"Plotting segment {segment_idx} with annotation {symbol}")
                plot_segment(signal, start, end, peak, rpeaks[i-1] if i > 0 else None, rpeaks[i+1] if i < len(rpeaks) - 1 else None, fs, segment_idx, plot_dir, symbol)

    print(f"Total segments created: {len(segments)}")
    return np.array(segments), np.array(valid_rpeaks)

def plot_segment(signal, start, end, current_rpeak, prev_rpeak, next_rpeak, fs, segment_idx, plot_dir, annotation_symbol):
    """
    Plot a single ECG segment with marked R-peaks and annotation.

    Args:
        signal: Full ECG signal
        start: Start index of the segment
        end: End index of the segment
        current_rpeak: Current R-peak index
        prev_rpeak: Previous R-peak index (or None)
        next_rpeak: Next R-peak index (or None)
        fs: Sampling frequency (Hz)
        segment_idx: Index of the segment for naming
        plot_dir: Directory to save the plot
        annotation_symbol: Annotation symbol for the current beat
    """
    plt.figure(figsize=(10, 4))
    time = np.arange(start, end) / fs
    segment = signal[start:end]
    
    # Plot segment
    plt.plot(time, segment, label='ECG Segment')
    
    # Mark R-peaks
    if prev_rpeak is not None and prev_rpeak >= start and prev_rpeak < end:
        plt.axvline(x=(prev_rpeak / fs), color='g', linestyle='--', label='Previous R-peak')
    plt.axvline(x=(current_rpeak / fs), color='r', linestyle='--', label='Current R-peak')
    if next_rpeak is not None and next_rpeak >= start and next_rpeak < end:
        plt.axvline(x=(next_rpeak / fs), color='b', linestyle='--', label='Next R-peak')

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Segment {segment_idx}: Previous, Current, and Next Beats (Annotation: {annotation_symbol})')
    plt.legend()
    plt.grid(True)
    filepath = os.path.join(plot_dir, f'segment_{segment_idx}.png')
    print(f"Saving plot to: {filepath}")
    plt.savefig(filepath)
    plt.close()
    print(f"Plot saved to: {filepath}")

def pan_tompkins_rpeak_detection(signal, fs):
    """
    Pan-Tompkins algorithm to detect R-peaks in ECG signal.
    Args:
        signal: preprocessed ECG signal (1D numpy array)
        fs: sampling frequency in Hz
    Returns:
        rpeaks: numpy array of detected R-peak sample indices
    """
    # 1. Derivative filter (5-point derivative)
    derivative_kernel = np.array([1, 2, 0, -2, -1]) * (1/(8/fs))
    differentiated = np.convolve(signal, derivative_kernel, mode='same')

    # 2. Squaring
    squared = differentiated ** 2

    # 3. Moving window integration (150 ms window)
    window_size = int(0.15 * fs)
    integrated = np.convolve(squared, np.ones(window_size)/window_size, mode='same')

    # 4. Peak detection with minimum distance of 200 ms (refractory period)
    min_distance = int(0.2 * fs)
    threshold = 0.5 * np.max(integrated)  
    peaks, _ = find_peaks(integrated, distance=min_distance, height=threshold)

    # 5. Refine R-peak locations: find max in original signal ±50 ms around detected peaks
    rpeaks = []
    search_radius = int(0.05 * fs)
    for peak in peaks:
        start = max(peak - search_radius, 0)
        end = min(peak + search_radius, len(signal))
        local_max = np.argmax(signal[start:end]) + start
        rpeaks.append(local_max)

    return np.array(rpeaks)