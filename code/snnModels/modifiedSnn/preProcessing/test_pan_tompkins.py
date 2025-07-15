import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
from scipy.signal import find_peaks
from SegmentMod import pan_tompkins_rpeak_detection

def visualize_pan_tompkins_steps(signal, fs):
    """
    Visualize each step of the Pan-Tompkins algorithm
    """
    print("Visualizing Pan-Tompkins algorithm steps...")
    
    # Step 1: Derivative filter
    derivative_kernel = np.array([1, 2, 0, -2, -1]) * (1/(8/fs))
    differentiated = np.convolve(signal, derivative_kernel, mode='same')
    
    # Step 2: Squaring
    squared = differentiated ** 2
    
    # Step 3: Moving window integration
    window_size = int(0.15 * fs)
    integrated = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
    
    # Step 4: Peak detection
    min_distance = int(0.2 * fs)
    threshold = 0.5 * np.max(integrated)
    peaks, _ = find_peaks(integrated, distance=min_distance, height=threshold)
    
    # Step 5: Refine R-peak locations
    rpeaks = []
    search_radius = int(0.05 * fs)
    for peak in peaks:
        start = max(peak - search_radius, 0)
        end = min(peak + search_radius, len(signal))
        local_max = np.argmax(signal[start:end]) + start
        rpeaks.append(local_max)
    
    # Create visualization
    fig, axes = plt.subplots(6, 1, figsize=(15, 12))
    time = np.arange(len(signal)) / fs
    
    # Plot 1: Original signal
    axes[0].plot(time, signal, 'b-', linewidth=1)
    axes[0].set_title('Step 1: Original ECG Signal')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True)
    
    # Plot 2: After derivative filter
    axes[1].plot(time, differentiated, 'g-', linewidth=1)
    axes[1].set_title('Step 2: After Derivative Filter')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True)
    
    # Plot 3: After squaring
    axes[2].plot(time, squared, 'r-', linewidth=1)
    axes[2].set_title('Step 3: After Squaring')
    axes[2].set_ylabel('Amplitude')
    axes[2].grid(True)
    
    # Plot 4: After integration
    axes[3].plot(time, integrated, 'm-', linewidth=1)
    axes[3].axhline(y=threshold, color='k', linestyle='--', label=f'Threshold: {threshold:.2f}')
    axes[3].scatter(peaks/fs, integrated[peaks], color='red', s=50, zorder=5, label='Detected Peaks')
    axes[3].set_title('Step 4: After Integration with Peak Detection')
    axes[3].set_ylabel('Amplitude')
    axes[3].legend()
    axes[3].grid(True)
    
    # Plot 5: Final R-peaks on original signal
    axes[4].plot(time, signal, 'b-', linewidth=1, alpha=0.7)
    axes[4].scatter(np.array(rpeaks)/fs, signal[rpeaks], color='red', s=80, zorder=5, 
                   marker='x', label='Final R-peaks')
    axes[4].set_title('Step 5: Final R-peaks on Original Signal')
    axes[4].set_ylabel('Amplitude')
    axes[4].legend()
    axes[4].grid(True)
    
    # Plot 6: Comparison with neurokit2
    _, info = nk.ecg_process(signal, sampling_rate=fs)
    reference_rpeaks = info['ECG_R_Peaks']
    
    axes[5].plot(time, signal, 'b-', linewidth=1, alpha=0.5, label='ECG Signal')
    axes[5].scatter(reference_rpeaks/fs, signal[reference_rpeaks], 
                   color='green', s=60, marker='o', label='Reference (neurokit2)')
    axes[5].scatter(np.array(rpeaks)/fs, signal[rpeaks], 
                   color='red', s=80, marker='x', label='Pan-Tompkins')
    axes[5].set_title('Step 6: Comparison with Reference')
    axes[5].set_xlabel('Time (s)')
    axes[5].set_ylabel('Amplitude')
    axes[5].legend()
    axes[5].grid(True)
    
    plt.tight_layout()
    plt.savefig('pan_tompkins_steps.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return rpeaks, reference_rpeaks

def simple_test():
    """Simple test of the Pan-Tompkins algorithm"""
    print("Running simple Pan-Tompkins test...")
    
    # Generate clean ECG signal
    fs = 500
    duration = 10
    signal = nk.ecg_simulate(duration=duration, sampling_rate=fs, noise=0.05)
    
    # Test the algorithm
    detected_rpeaks = pan_tompkins_rpeak_detection(signal, fs)
    
    # Get reference
    _, info = nk.ecg_process(signal, sampling_rate=fs)
    reference_rpeaks = info['ECG_R_Peaks']
    
    print(f"Reference R-peaks: {len(reference_rpeaks)}")
    print(f"Detected R-peaks: {len(detected_rpeaks)}")
    
    # Simple visualization
    plt.figure(figsize=(12, 6))
    time = np.arange(len(signal)) / fs
    plt.plot(time, signal, 'b-', alpha=0.7, label='ECG Signal')
    plt.scatter(reference_rpeaks/fs, signal[reference_rpeaks], 
               color='green', s=60, marker='o', label='Reference')
    plt.scatter(detected_rpeaks/fs, signal[detected_rpeaks], 
               color='red', s=80, marker='x', label='Detected')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Pan-Tompkins R-peak Detection Results')
    plt.legend()
    plt.grid(True)
    plt.savefig('simple_test_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return detected_rpeaks, reference_rpeaks

if __name__ == "__main__":
    print("=" * 50)
    print("PAN-TOMPKINS ALGORITHM TESTING")
    print("=" * 50)
    
    # Run simple test
    detected, reference = simple_test()
    
    print("\n" + "=" * 50)
    print("STEP-BY-STEP VISUALIZATION")
    print("=" * 50)
    
    # Run detailed visualization
    fs = 500
    duration = 5  # Shorter duration for clearer visualization
    signal = nk.ecg_simulate(duration=duration, sampling_rate=fs, noise=0.05)
    
    visualize_pan_tompkins_steps(signal, fs)
    
    print("\nTesting completed!")
    print("Generated files:")
    print("- simple_test_result.png: Simple test results")
    print("- pan_tompkins_steps.png: Step-by-step algorithm visualization")
