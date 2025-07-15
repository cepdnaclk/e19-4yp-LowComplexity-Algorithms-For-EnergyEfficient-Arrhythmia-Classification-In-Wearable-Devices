import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
from SegmentMod import extract_heartbeats, pan_tompkins_rpeak_detection, plot_segment
import os

def test_pan_tompkins_detection():
    """Test Pan-Tompkins R-peak detection algorithm"""
    print("Testing Pan-Tompkins R-peak detection...")
    
    # Generate synthetic ECG signal
    fs = 500  # 500 Hz sampling rate
    duration = 10  # 10 seconds
    signal = nk.ecg_simulate(duration=duration, sampling_rate=fs, noise=0.1)
    
    # Get reference R-peaks using neurokit2
    _, info = nk.ecg_process(signal, sampling_rate=fs)
    reference_rpeaks = info['ECG_R_Peaks']
    
    # Test our Pan-Tompkins implementation
    detected_rpeaks = pan_tompkins_rpeak_detection(signal, fs)
    
    # Plot comparison
    plt.figure(figsize=(15, 8))
    
    # Plot 1: Full signal with detected peaks
    plt.subplot(2, 1, 1)
    time = np.arange(len(signal)) / fs
    plt.plot(time, signal, label='ECG Signal', alpha=0.7)
    plt.scatter(reference_rpeaks/fs, signal[reference_rpeaks], 
               color='red', marker='o', s=50, label='Reference R-peaks (neurokit2)')
    plt.scatter(detected_rpeaks/fs, signal[detected_rpeaks], 
               color='blue', marker='x', s=80, label='Detected R-peaks (Pan-Tompkins)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('R-peak Detection Comparison')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Zoomed view of first 3 seconds
    plt.subplot(2, 1, 2)
    zoom_end = int(3 * fs)
    plt.plot(time[:zoom_end], signal[:zoom_end], label='ECG Signal', alpha=0.7)
    
    # Filter peaks for zoom window
    ref_zoom = reference_rpeaks[reference_rpeaks < zoom_end]
    det_zoom = detected_rpeaks[detected_rpeaks < zoom_end]
    
    plt.scatter(ref_zoom/fs, signal[ref_zoom], 
               color='red', marker='o', s=50, label='Reference R-peaks')
    plt.scatter(det_zoom/fs, signal[det_zoom], 
               color='blue', marker='x', s=80, label='Detected R-peaks')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('R-peak Detection - Zoomed View (First 3 seconds)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('rpeak_detection_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Calculate performance metrics
    tolerance = int(0.05 * fs)  # 50ms tolerance
    true_positives = 0
    
    for detected in detected_rpeaks:
        if np.any(np.abs(reference_rpeaks - detected) <= tolerance):
            true_positives += 1
    
    false_positives = len(detected_rpeaks) - true_positives
    false_negatives = len(reference_rpeaks) - true_positives
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Pan-Tompkins Performance:")
    print(f"  Reference R-peaks: {len(reference_rpeaks)}")
    print(f"  Detected R-peaks: {len(detected_rpeaks)}")
    print(f"  True Positives: {true_positives}")
    print(f"  False Positives: {false_positives}")
    print(f"  False Negatives: {false_negatives}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1-Score: {f1_score:.3f}")
    
    return signal, detected_rpeaks, reference_rpeaks

def test_extract_heartbeats():
    """Test heartbeat extraction and plotting function"""
    print("\nTesting heartbeat extraction and plotting...")
    
    # Generate synthetic ECG signal
    fs = 500  # 500 Hz sampling rate
    duration = 15  # 15 seconds to get more beats
    signal = nk.ecg_simulate(duration=duration, sampling_rate=fs, noise=0.1)
    
    # Clean up plots directory
    plot_dir = 'test_plots'
    if os.path.exists(plot_dir):
        for file in os.listdir(plot_dir):
            os.remove(os.path.join(plot_dir, file))
    
    # Extract heartbeat segments
    segments, valid_rpeaks = extract_heartbeats(
        signal=signal, 
        fs=fs, 
        annotation_rpeaks=None,  # Let it auto-detect
        before=0.25, 
        after=0.4, 
        fixed_length=300,
        plot_dir=plot_dir
    )
    
    print(f"Extracted {len(segments)} heartbeat segments")
    print(f"Each segment has {segments.shape[1]} samples")
    print(f"Valid R-peaks: {len(valid_rpeaks)}")
    print(f"Segment plots saved in '{plot_dir}' directory")
    
    # Plot summary of extracted segments
    plt.figure(figsize=(12, 8))
    
    # Plot first few segments
    max_segments_to_plot = min(6, len(segments))
    for i in range(max_segments_to_plot):
        plt.subplot(2, 3, i+1)
        plt.plot(segments[i])
        plt.title(f'Segment {i+1}')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('extracted_segments_summary.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return segments, valid_rpeaks

def test_plot_function():
    """Test the individual plot function"""
    print("\nTesting individual plot function...")
    
    # Generate test data
    fs = 500
    duration = 5
    signal = nk.ecg_simulate(duration=duration, sampling_rate=fs, noise=0.1)
    
    # Detect R-peaks
    _, info = nk.ecg_process(signal, sampling_rate=fs)
    rpeaks = info['ECG_R_Peaks']
    
    # Test plot function with a specific segment
    if len(rpeaks) >= 3:
        current_idx = 1  # Use second R-peak
        current_rpeak = rpeaks[current_idx]
        prev_rpeak = rpeaks[current_idx - 1]
        next_rpeak = rpeaks[current_idx + 1]
        
        # Define segment boundaries
        samples_before = int(0.5 * fs)  # 500ms before
        samples_after = int(0.5 * fs)   # 500ms after
        
        start = max(current_rpeak - samples_before, 0)
        end = min(current_rpeak + samples_after, len(signal))
        
        # Test the plot function
        plot_segment(
            signal=signal,
            start=start,
            end=end,
            current_rpeak=current_rpeak,
            prev_rpeak=prev_rpeak,
            next_rpeak=next_rpeak,
            fs=fs,
            segment_idx=999,  # Test segment
            plot_dir='test_plots'
        )
        
        print("Individual plot function test completed. Check 'test_plots/segment_999.png'")
    else:
        print("Not enough R-peaks detected for individual plot test")

def main():
    """Main test function"""
    print("=" * 60)
    print("TESTING SEGMENTMOD.PY FUNCTIONS")
    print("=" * 60)
    
    # Test 1: Pan-Tompkins R-peak detection
    signal, detected_rpeaks, reference_rpeaks = test_pan_tompkins_detection()
    
    # Test 2: Extract heartbeats function
    segments, valid_rpeaks = test_extract_heartbeats()
    
    # Test 3: Individual plot function
    test_plot_function()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
    print("Generated files:")
    print("- rpeak_detection_test.png: R-peak detection comparison")
    print("- extracted_segments_summary.png: Summary of extracted segments")
    print("- test_plots/: Directory with individual segment plots")
    print("Check these files to verify the functions work correctly.")

if __name__ == "__main__":
    main()
