import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import torch
import os
from scipy import signal as scipy_signal

def generate_ecg_for_snn_visualization(duration=10, fs=500, noise_level=0.1):
    """
    Generate ECG signals suitable for SNN model visualization
    
    Args:
        duration: Duration of signal in seconds
        fs: Sampling frequency
        noise_level: Noise level (0-1)
    
    Returns:
        ecg_signal: Raw ECG signal
        normalized_signal: Normalized signal for SNN input
        spike_times: Spike timing data for SNN
    """
    # Generate synthetic ECG signal
    ecg_signal = nk.ecg_simulate(duration=duration, sampling_rate=fs, noise=noise_level)
    
    # Normalize signal for SNN input (0-1 range)
    normalized_signal = (ecg_signal - np.min(ecg_signal)) / (np.max(ecg_signal) - np.min(ecg_signal))
    
    # Convert to spike times for SNN (threshold crossing method)
    threshold = 0.5
    spike_times = []
    for i in range(1, len(normalized_signal)):
        if normalized_signal[i-1] < threshold and normalized_signal[i] >= threshold:
            spike_times.append(i)
    
    return ecg_signal, normalized_signal, np.array(spike_times)

def plot_ecg_for_snn_model(ecg_signal, normalized_signal, spike_times, fs, save_dir='snn_plots'):
    """
    Create comprehensive ECG plots for SNN model analysis
    """
    os.makedirs(save_dir, exist_ok=True)
    
    time = np.arange(len(ecg_signal)) / fs
    
    # Plot 1: Original ECG Signal
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Raw ECG
    plt.subplot(4, 1, 1)
    plt.plot(time, ecg_signal, 'b-', linewidth=1.5)
    plt.title('Raw ECG Signal for SNN Model', fontsize=14, fontweight='bold')
    plt.ylabel('Amplitude (mV)')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Normalized ECG (SNN Input)
    plt.subplot(4, 1, 2)
    plt.plot(time, normalized_signal, 'r-', linewidth=1.5)
    plt.title('Normalized ECG Signal (SNN Input)', fontsize=14, fontweight='bold')
    plt.ylabel('Normalized Amplitude')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Spike Times
    plt.subplot(4, 1, 3)
    plt.plot(time, normalized_signal, 'g-', alpha=0.5, linewidth=1)
    plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.7, label='Spike Threshold')
    plt.scatter(spike_times/fs, np.ones(len(spike_times))*0.5, 
               color='red', s=50, marker='|', label='Spike Times')
    plt.title('Spike Times for SNN Processing', fontsize=14, fontweight='bold')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Spike Raster Plot
    plt.subplot(4, 1, 4)
    plt.eventplot([spike_times/fs], lineoffsets=1, linelengths=0.8, 
                  colors=['red'], linewidths=2)
    plt.title('Spike Raster Plot for SNN', fontsize=14, fontweight='bold')
    plt.ylabel('Neuron')
    plt.xlabel('Time (s)')
    plt.ylim(0.5, 1.5)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ecg_snn_overview.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return os.path.join(save_dir, 'ecg_snn_overview.png')

def plot_ecg_segments_for_snn(duration=10, fs=500, segment_length=1.0, save_dir='snn_plots'):
    """
    Create individual ECG segments for SNN model training/testing
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate longer ECG signal
    ecg_signal = nk.ecg_simulate(duration=duration, sampling_rate=fs, noise=0.1)
    normalized_signal = (ecg_signal - np.min(ecg_signal)) / (np.max(ecg_signal) - np.min(ecg_signal))
    
    # Create segments
    samples_per_segment = int(segment_length * fs)
    num_segments = len(ecg_signal) // samples_per_segment
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for i in range(min(9, num_segments)):
        start_idx = i * samples_per_segment
        end_idx = start_idx + samples_per_segment
        
        segment = normalized_signal[start_idx:end_idx]
        time_segment = np.arange(len(segment)) / fs
        
        # Plot segment
        axes[i].plot(time_segment, segment, 'b-', linewidth=1.5)
        axes[i].set_title(f'ECG Segment {i+1} for SNN', fontweight='bold')
        axes[i].set_ylabel('Normalized Amplitude')
        axes[i].grid(True, alpha=0.3)
        
        # Add spike threshold line
        axes[i].axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Spike Threshold')
        
        if i >= 6:  # Add x-label for bottom row
            axes[i].set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ecg_segments_snn.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return os.path.join(save_dir, 'ecg_segments_snn.png')

def plot_different_ecg_types_for_snn(save_dir='snn_plots'):
    """
    Generate different types of ECG signals for SNN model
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fs = 500
    duration = 5
    
    # Different ECG types
    ecg_types = [
        {'noise': 0.01, 'title': 'Clean ECG', 'color': 'blue'},
        {'noise': 0.1, 'title': 'Normal Noise ECG', 'color': 'green'},
        {'noise': 0.3, 'title': 'Noisy ECG', 'color': 'red'},
        {'noise': 0.05, 'title': 'Low Noise ECG', 'color': 'purple'}
    ]
    
    plt.figure(figsize=(15, 12))
    
    for i, ecg_type in enumerate(ecg_types):
        # Generate ECG
        ecg_signal = nk.ecg_simulate(duration=duration, sampling_rate=fs, noise=ecg_type['noise'])
        normalized_signal = (ecg_signal - np.min(ecg_signal)) / (np.max(ecg_signal) - np.min(ecg_signal))
        
        time = np.arange(len(ecg_signal)) / fs
        
        # Plot original
        plt.subplot(4, 2, i*2 + 1)
        plt.plot(time, ecg_signal, color=ecg_type['color'], linewidth=1.5)
        plt.title(f'{ecg_type["title"]} - Original', fontweight='bold')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        # Plot normalized for SNN
        plt.subplot(4, 2, i*2 + 2)
        plt.plot(time, normalized_signal, color=ecg_type['color'], linewidth=1.5)
        plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.7, label='SNN Threshold')
        plt.title(f'{ecg_type["title"]} - SNN Input', fontweight='bold')
        plt.ylabel('Normalized Amplitude')
        plt.grid(True, alpha=0.3)
        
        if i == 3:  # Add x-label for bottom row
            plt.xlabel('Time (s)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ecg_types_snn.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return os.path.join(save_dir, 'ecg_types_snn.png')

def create_snn_input_tensor(ecg_signal, fs, time_window=1.0):
    """
    Create tensor format suitable for SNN model input
    """
    # Normalize signal
    normalized_signal = (ecg_signal - np.min(ecg_signal)) / (np.max(ecg_signal) - np.min(ecg_signal))
    
    # Create time windows
    window_samples = int(time_window * fs)
    num_windows = len(normalized_signal) // window_samples
    
    # Reshape into tensor format (batch_size, time_steps, features)
    tensor_data = []
    for i in range(num_windows):
        start_idx = i * window_samples
        end_idx = start_idx + window_samples
        window_data = normalized_signal[start_idx:end_idx]
        tensor_data.append(window_data)
    
    # Convert to PyTorch tensor
    tensor = torch.tensor(tensor_data, dtype=torch.float32)
    
    return tensor

def plot_snn_tensor_visualization(tensor_data, fs, save_dir='snn_plots'):
    """
    Visualize the tensor data format for SNN model
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 8))
    
    # Plot first few tensor slices
    for i in range(min(4, tensor_data.shape[0])):
        plt.subplot(2, 2, i+1)
        time_axis = np.arange(tensor_data.shape[1]) / fs
        plt.plot(time_axis, tensor_data[i].numpy(), 'b-', linewidth=1.5)
        plt.title(f'SNN Input Tensor - Batch {i+1}', fontweight='bold')
        plt.ylabel('Normalized Amplitude')
        plt.xlabel('Time (s)')
        plt.grid(True, alpha=0.3)
        
        # Add spike threshold
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Spike Threshold')
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'snn_tensor_visualization.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return os.path.join(save_dir, 'snn_tensor_visualization.png')

def main():
    """
    Main function to generate all ECG plots for SNN model
    """
    print("=" * 60)
    print("GENERATING ECG PLOTS FOR SNN MODEL")
    print("=" * 60)
    
    # Parameters
    fs = 500
    duration = 10
    save_dir = 'snn_ecg_plots'
    
    # 1. Generate basic ECG data
    print("\n1. Generating ECG signals for SNN...")
    ecg_signal, normalized_signal, spike_times = generate_ecg_for_snn_visualization(duration, fs)
    
    # 2. Create comprehensive overview plot
    print("2. Creating SNN overview plot...")
    overview_plot = plot_ecg_for_snn_model(ecg_signal, normalized_signal, spike_times, fs, save_dir)
    
    # 3. Create segment plots
    print("3. Creating ECG segments for SNN...")
    segments_plot = plot_ecg_segments_for_snn(duration, fs, segment_length=1.0, save_dir=save_dir)
    
    # 4. Create different ECG types
    print("4. Creating different ECG types for SNN...")
    types_plot = plot_different_ecg_types_for_snn(save_dir)
    
    # 5. Create tensor visualization
    print("5. Creating SNN tensor visualization...")
    tensor_data = create_snn_input_tensor(ecg_signal, fs, time_window=1.0)
    tensor_plot = plot_snn_tensor_visualization(tensor_data, fs, save_dir)
    
    print("\n" + "=" * 60)
    print("ECG PLOTS FOR SNN MODEL GENERATED!")
    print("=" * 60)
    print(f"All plots saved in: {save_dir}/")
    print("\nGenerated files:")
    print(f"- ecg_snn_overview.png: Complete ECG to SNN processing overview")
    print(f"- ecg_segments_snn.png: Individual ECG segments for SNN training")
    print(f"- ecg_types_snn.png: Different noise levels for SNN robustness testing")
    print(f"- snn_tensor_visualization.png: Tensor format visualization for SNN input")
    
    print(f"\nTensor shape for SNN model: {tensor_data.shape}")
    print(f"  - Batch size: {tensor_data.shape[0]}")
    print(f"  - Time steps: {tensor_data.shape[1]}")
    print(f"  - Features: 1 (single ECG channel)")
    
    return tensor_data

if __name__ == "__main__":
    tensor_data = main()
