import neurokit2 as nk
import numpy as np
from scipy.signal import find_peaks

def extract_heartbeats(signal, fs, annotation_rpeaks=None, before=0.25, after=0.4, fixed_length=300):
    """
    Extract fixed-length heartbeats centered at R-peaks
    
    Args:
        signal: ECG signal
        fs: Sampling frequency (Hz)
        annotation_rpeaks: Optional pre-annotated R-peaks
        before: Seconds before R-peak (default 0.25)
        after: Seconds after R-peak (default 0.4)
        fixed_length: Target samples per beat (default 250)
        
    Returns:
        beats: Array of fixed-length beats (n_beats, fixed_length)
        valid_rpeaks: Array of used R-peak positions
    """
    # Clean signal and detect R-peaks using neurokit2 library
    cleaned = nk.ecg_clean(signal, sampling_rate=fs)
    rpeaks = annotation_rpeaks if annotation_rpeaks is not None else \
             nk.ecg_findpeaks(cleaned, sampling_rate=fs)['ECG_R_Peaks']
    
    beats = []
    valid_rpeaks = []
    
    for peak in rpeaks:
        # Calculate fixed window around R-peak
        center = int(peak)
        start = center - fixed_length//2
        end = center + fixed_length//2
        
        # Handle edge cases
        if start < 0:
            start, end = 0, fixed_length
        elif end > len(signal):
            start, end = len(signal)-fixed_length, len(signal)
        
        beat = signal[start:end]
        
        # Ensure exact length (in case of sampling rate rounding)
        if len(beat) < fixed_length:
            beat = np.pad(beat, (0, fixed_length - len(beat)))
        elif len(beat) > fixed_length:
            beat = beat[:fixed_length]
        
        beats.append(beat)
        valid_rpeaks.append(peak)

        # Plot segment
        plot_segment(signal, start, end, peak, rpeaks[i-1] if i > 0 else None, rpeaks[i+1] if i < len(rpeaks) - 1 else None, fs, i, plot_dir)

    return np.array(segments), np.array(valid_rpeaks)

def plot_segment(signal, start, end, current_rpeak, prev_rpeak, next_rpeak, fs, segment_idx, plot_dir):
    """
    Plot a single ECG segment with marked R-peaks.

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
    plt.title(f'Segment {segment_idx}: Previous, Current, and Next Beats')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f'segment_{segment_idx}.png'))
    plt.close()

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

def main():
    """
    Main function to test the plot_segment function
    """
    print("Testing plot_segment function...")
    
    # Generate synthetic ECG signal
    fs = 500  # 500 Hz sampling rate
    duration = 10  # 10 seconds
    signal = nk.ecg_simulate(duration=duration, sampling_rate=fs, noise=0.1)
    
    # Clean signal and detect R-peaks
    cleaned = nk.ecg_clean(signal, sampling_rate=fs)
    rpeaks = nk.ecg_findpeaks(cleaned, sampling_rate=fs)['ECG_R_Peaks']
    
    print(f"Generated {len(signal)} samples of ECG signal")
    print(f"Detected {len(rpeaks)} R-peaks")
    
    # Create test plots directory
    plot_dir = 'test_plots'
    os.makedirs(plot_dir, exist_ok=True)
    
    # Test plot_segment function with different segments
    test_cases = [
        {"idx": 2, "description": "Normal segment with 3 beats"},
        {"idx": 5, "description": "Middle segment"},
        {"idx": len(rpeaks) - 3, "description": "Near end segment"}
    ]
    
    for case in test_cases:
        idx = case["idx"]
        if idx < len(rpeaks) - 1 and idx > 0:
            current_rpeak = rpeaks[idx]
            prev_rpeak = rpeaks[idx - 1] if idx > 0 else None
            next_rpeak = rpeaks[idx + 1] if idx < len(rpeaks) - 1 else None
            
            # Define segment boundaries (±0.5 seconds around current R-peak)
            samples_before = int(0.5 * fs)
            samples_after = int(0.5 * fs)
            
            start = max(current_rpeak - samples_before, 0)
            end = min(current_rpeak + samples_after, len(signal))
            
            print(f"\nTesting {case['description']}:")
            print(f"  Current R-peak at sample {current_rpeak} ({current_rpeak/fs:.2f}s)")
            print(f"  Previous R-peak at sample {prev_rpeak} ({prev_rpeak/fs:.2f}s)" if prev_rpeak else "  No previous R-peak")
            print(f"  Next R-peak at sample {next_rpeak} ({next_rpeak/fs:.2f}s)" if next_rpeak else "  No next R-peak")
            print(f"  Segment from {start} to {end} (samples)")
            
            # Test the plot_segment function
            plot_segment(
                signal=signal,
                start=start,
                end=end,
                current_rpeak=current_rpeak,
                prev_rpeak=prev_rpeak,
                next_rpeak=next_rpeak,
                fs=fs,
                segment_idx=idx,
                plot_dir=plot_dir
            )
            
            print(f"  Plot saved as: {plot_dir}/segment_{idx}.png")
    
    # Test with Pan-Tompkins algorithm
    print("\n" + "="*50)
    print("Testing with Pan-Tompkins R-peak detection:")
    
    pt_rpeaks = pan_tompkins_rpeak_detection(signal, fs)
    print(f"Pan-Tompkins detected {len(pt_rpeaks)} R-peaks")
    
    if len(pt_rpeaks) > 2:
        idx = 1  # Use second R-peak
        current_rpeak = pt_rpeaks[idx]
        prev_rpeak = pt_rpeaks[idx - 1] if idx > 0 else None
        next_rpeak = pt_rpeaks[idx + 1] if idx < len(pt_rpeaks) - 1 else None
        
        start = max(current_rpeak - samples_before, 0)
        end = min(current_rpeak + samples_after, len(signal))
        
        plot_segment(
            signal=signal,
            start=start,
            end=end,
            current_rpeak=current_rpeak,
            prev_rpeak=prev_rpeak,
            next_rpeak=next_rpeak,
            fs=fs,
            segment_idx=999,  # Special index for Pan-Tompkins test
            plot_dir=plot_dir
        )
        
        print(f"Pan-Tompkins plot saved as: {plot_dir}/segment_999.png")
    
    # Create a summary plot comparing both methods
    plt.figure(figsize=(15, 8))
    
    # Plot full signal with both R-peak detection methods
    time = np.arange(len(signal)) / fs
    plt.plot(time, signal, 'b-', alpha=0.7, label='ECG Signal')
    
    # Plot neurokit2 R-peaks
    plt.scatter(rpeaks/fs, signal[rpeaks], color='red', s=60, marker='o', 
               label=f'Neurokit2 R-peaks ({len(rpeaks)})', alpha=0.8)
    
    # Plot Pan-Tompkins R-peaks
    plt.scatter(pt_rpeaks/fs, signal[pt_rpeaks], color='green', s=80, marker='x', 
               label=f'Pan-Tompkins R-peaks ({len(pt_rpeaks)})', alpha=0.8)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('R-peak Detection Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'rpeak_comparison.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nSummary comparison plot saved as: {plot_dir}/rpeak_comparison.png")
    print(f"\nAll test plots saved in '{plot_dir}' directory")
    print("Testing completed successfully!")

if __name__ == "__main__":
    main()