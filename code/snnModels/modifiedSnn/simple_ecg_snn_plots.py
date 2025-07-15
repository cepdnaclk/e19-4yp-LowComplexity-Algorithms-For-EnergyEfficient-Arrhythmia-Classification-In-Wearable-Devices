import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import torch
import os

def generate_snn_ecg_plots():
    """
    Generate ECG plots specifically for SNN model visualization
    """
    print("Generating ECG plots for SNN model...")
    
    # Create plots directory
    os.makedirs('snn_ecg_plots', exist_ok=True)
    
    # Parameters
    fs = 500  # Sampling frequency
    duration = 8  # Duration in seconds
    
    # Generate ECG signal
    ecg_raw = nk.ecg_simulate(duration=duration, sampling_rate=fs, noise=0.1)
    
    # Normalize for SNN (0-1 range)
    ecg_normalized = (ecg_raw - np.min(ecg_raw)) / (np.max(ecg_raw) - np.min(ecg_raw))
    
    # Time axis
    time = np.arange(len(ecg_raw)) / fs
    
    # Create comprehensive plot
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # Plot 1: Raw ECG Signal
    axes[0, 0].plot(time, ecg_raw, 'b-', linewidth=2)
    axes[0, 0].set_title('Raw ECG Signal', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Amplitude (mV)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(0, duration)
    
    # Plot 2: Normalized ECG for SNN Input
    axes[0, 1].plot(time, ecg_normalized, 'r-', linewidth=2)
    axes[0, 1].set_title('Normalized ECG for SNN Input', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Normalized Amplitude')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(0, duration)
    axes[0, 1].axhline(y=0.5, color='k', linestyle='--', alpha=0.7, label='Spike Threshold')
    axes[0, 1].legend()
    
    # Plot 3: Spike Conversion for SNN
    threshold = 0.5
    spikes = (ecg_normalized > threshold).astype(int)
    axes[1, 0].plot(time, ecg_normalized, 'g-', alpha=0.6, linewidth=1, label='ECG Signal')
    axes[1, 0].plot(time, spikes, 'r-', linewidth=2, label='Spike Train')
    axes[1, 0].set_title('ECG to Spike Conversion', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    axes[1, 0].set_xlim(0, duration)
    
    # Plot 4: Spike Raster for SNN
    spike_times = np.where(spikes == 1)[0] / fs
    axes[1, 1].eventplot([spike_times], lineoffsets=1, linelengths=0.8, 
                        colors=['red'], linewidths=3)
    axes[1, 1].set_title('Spike Raster for SNN', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Neuron')
    axes[1, 1].set_ylim(0.5, 1.5)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(0, duration)
    
    # Plot 5: ECG Segments for SNN Training
    segment_length = 1.0  # 1 second segments
    segment_samples = int(segment_length * fs)
    num_segments = len(ecg_normalized) // segment_samples
    
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    for i in range(min(5, num_segments)):
        start_idx = i * segment_samples
        end_idx = start_idx + segment_samples
        segment = ecg_normalized[start_idx:end_idx]
        segment_time = np.arange(len(segment)) / fs
        
        axes[2, 0].plot(segment_time, segment + i*0.2, 
                       color=colors[i % len(colors)], linewidth=2, 
                       label=f'Segment {i+1}')
    
    axes[2, 0].set_title('ECG Segments for SNN Training', fontsize=14, fontweight='bold')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Normalized Amplitude (offset)')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend()
    
    # Plot 6: Frequency Analysis for SNN
    freqs = np.fft.fftfreq(len(ecg_raw), 1/fs)
    fft_vals = np.abs(np.fft.fft(ecg_raw))
    
    # Plot only positive frequencies up to 50 Hz
    pos_freqs = freqs[:len(freqs)//2]
    pos_fft = fft_vals[:len(fft_vals)//2]
    freq_mask = pos_freqs <= 50
    
    axes[2, 1].plot(pos_freqs[freq_mask], pos_fft[freq_mask], 'purple', linewidth=2)
    axes[2, 1].set_title('ECG Frequency Spectrum for SNN', fontsize=14, fontweight='bold')
    axes[2, 1].set_xlabel('Frequency (Hz)')
    axes[2, 1].set_ylabel('Magnitude')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('snn_ecg_plots/comprehensive_ecg_snn.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return ecg_raw, ecg_normalized, spikes

def create_snn_input_format(ecg_signal, fs=500, window_size=1.0):
    """
    Create input format suitable for SNN model
    """
    # Normalize signal
    normalized = (ecg_signal - np.min(ecg_signal)) / (np.max(ecg_signal) - np.min(ecg_signal))
    
    # Create windowed data
    window_samples = int(window_size * fs)
    num_windows = len(normalized) // window_samples
    
    # Reshape into batches
    windowed_data = []
    for i in range(num_windows):
        start = i * window_samples
        end = start + window_samples
        window = normalized[start:end]
        windowed_data.append(window)
    
    # Convert to tensor
    tensor_data = torch.tensor(windowed_data, dtype=torch.float32)
    
    # Plot tensor visualization
    plt.figure(figsize=(12, 8))
    
    # Plot first 4 windows
    for i in range(min(4, tensor_data.shape[0])):
        plt.subplot(2, 2, i+1)
        time_axis = np.arange(tensor_data.shape[1]) / fs
        plt.plot(time_axis, tensor_data[i].numpy(), 'b-', linewidth=2)
        plt.title(f'SNN Input Window {i+1}', fontweight='bold')
        plt.ylabel('Normalized Amplitude')
        plt.xlabel('Time (s)')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('snn_ecg_plots/snn_input_format.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return tensor_data

def plot_ecg_arrhythmia_types_for_snn():
    """
    Generate different types of ECG patterns for SNN arrhythmia classification
    """
    print("Generating different ECG patterns for SNN arrhythmia classification...")
    
    fs = 500
    duration = 3
    
    # Create different ECG patterns
    patterns = {
        'Normal': {'noise': 0.05, 'heart_rate': 70},
        'Tachycardia': {'noise': 0.05, 'heart_rate': 120},
        'Bradycardia': {'noise': 0.05, 'heart_rate': 45},
        'Noisy': {'noise': 0.3, 'heart_rate': 70}
    }
    
    plt.figure(figsize=(15, 10))
    
    for i, (pattern_name, params) in enumerate(patterns.items()):
        # Generate ECG with specific characteristics
        ecg = nk.ecg_simulate(duration=duration, sampling_rate=fs, 
                             noise=params['noise'], heart_rate=params['heart_rate'])
        
        # Normalize for SNN
        ecg_norm = (ecg - np.min(ecg)) / (np.max(ecg) - np.min(ecg))
        
        time = np.arange(len(ecg)) / fs
        
        # Plot original
        plt.subplot(4, 2, i*2 + 1)
        plt.plot(time, ecg, 'b-', linewidth=2)
        plt.title(f'{pattern_name} ECG - Raw', fontweight='bold')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        # Plot normalized for SNN
        plt.subplot(4, 2, i*2 + 2)
        plt.plot(time, ecg_norm, 'r-', linewidth=2)
        plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.7, label='SNN Threshold')
        plt.title(f'{pattern_name} ECG - SNN Input', fontweight='bold')
        plt.ylabel('Normalized Amplitude')
        plt.grid(True, alpha=0.3)
        
        if i == 0:
            plt.legend()
        if i == 3:
            plt.xlabel('Time (s)')
    
    plt.tight_layout()
    plt.savefig('snn_ecg_plots/ecg_arrhythmia_types_snn.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function to generate all ECG plots for SNN model
    """
    print("=" * 50)
    print("ECG PLOTS FOR SNN MODEL")
    print("=" * 50)
    
    # Generate comprehensive ECG plots
    ecg_raw, ecg_normalized, spikes = generate_snn_ecg_plots()
    
    # Create SNN input format
    tensor_data = create_snn_input_format(ecg_raw)
    
    # Generate arrhythmia types
    plot_ecg_arrhythmia_types_for_snn()
    
    print("\n" + "=" * 50)
    print("PLOTS GENERATED SUCCESSFULLY!")
    print("=" * 50)
    print("Generated files in 'snn_ecg_plots/' directory:")
    print("- comprehensive_ecg_snn.png: Complete ECG analysis for SNN")
    print("- snn_input_format.png: Tensor format visualization")
    print("- ecg_arrhythmia_types_snn.png: Different ECG patterns for classification")
    
    print(f"\nSNN Input Tensor Shape: {tensor_data.shape}")
    print(f"Number of samples: {tensor_data.shape[0]}")
    print(f"Time steps per sample: {tensor_data.shape[1]}")
    print(f"Ready for SNN model input!")

if __name__ == "__main__":
    main()
