import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import torch
import os

def generate_pqrst_based_spikes():
    """
    Generate spikes based on detected P, Q, R, S, T waves in ECG signal
    """
    print("Generating spikes based on P, Q, R, S, T wave detection...")
    
    # Create plots directory
    os.makedirs('pqrst_spike_plots', exist_ok=True)
    
    # Parameters
    fs = 500  # Sampling frequency
    duration = 8  # Duration in seconds
    
    # Generate ECG signal
    ecg_raw = nk.ecg_simulate(duration=duration, sampling_rate=fs, noise=0.1)
    
    # Clean signal and detect all ECG fiducial points
    cleaned_ecg = nk.ecg_clean(ecg_raw, sampling_rate=fs)
    signals, info = nk.ecg_process(cleaned_ecg, sampling_rate=fs)
    
    # Extract all fiducial points
    rpeaks = info['ECG_R_Peaks']
    
    # Detect P, Q, S, T waves using neurokit2
    try:
        # Detect P waves
        p_waves = nk.ecg_findpeaks(cleaned_ecg, sampling_rate=fs, method='neurokit')
        
        # Use ECG delineation to find P, Q, S, T waves
        waves = nk.ecg_delineate(cleaned_ecg, rpeaks, sampling_rate=fs, method='dwt')
        
        # Extract wave positions
        p_onsets = waves.get('ECG_P_Onsets', [])
        p_peaks = waves.get('ECG_P_Peaks', [])
        q_peaks = waves.get('ECG_Q_Peaks', [])
        s_peaks = waves.get('ECG_S_Peaks', [])
        t_peaks = waves.get('ECG_T_Peaks', [])
        t_offsets = waves.get('ECG_T_Offsets', [])
        
        # Remove NaN values
        p_onsets = [p for p in p_onsets if not np.isnan(p)]
        p_peaks = [p for p in p_peaks if not np.isnan(p)]
        q_peaks = [q for q in q_peaks if not np.isnan(q)]
        s_peaks = [s for s in s_peaks if not np.isnan(s)]
        t_peaks = [t for t in t_peaks if not np.isnan(t)]
        t_offsets = [t for t in t_offsets if not np.isnan(t)]
        
    except Exception as e:
        print(f"Advanced wave detection failed: {e}")
        print("Using simplified wave detection...")
        # Fallback to simplified detection
        p_onsets, p_peaks, q_peaks, s_peaks, t_peaks, t_offsets = detect_pqrst_waves_simple(cleaned_ecg, rpeaks, fs)
    
    print(f"Detected fiducial points:")
    print(f"  R-peaks: {len(rpeaks)}")
    print(f"  P-waves: {len(p_peaks)}")
    print(f"  Q-waves: {len(q_peaks)}")
    print(f"  S-waves: {len(s_peaks)}")
    print(f"  T-waves: {len(t_peaks)}")
    
    # Normalize ECG for SNN
    ecg_normalized = (ecg_raw - np.min(ecg_raw)) / (np.max(ecg_raw) - np.min(ecg_raw))
    
    # Generate spikes for all PQRST waves
    spike_train = generate_spikes_for_pqrst(ecg_normalized, rpeaks, p_peaks, q_peaks, s_peaks, t_peaks, fs)
    
    # Time axis
    time = np.arange(len(ecg_raw)) / fs
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(5, 1, figsize=(18, 15))
    
    # Plot 1: Original ECG with all PQRST waves
    axes[0].plot(time, ecg_raw, 'b-', linewidth=1.5, label='ECG Signal')
    
    # Plot all detected waves
    if len(p_peaks) > 0:
        axes[0].scatter(np.array(p_peaks)/fs, ecg_raw[p_peaks], color='green', s=80, 
                       marker='o', zorder=5, label=f'P-waves ({len(p_peaks)})')
    if len(q_peaks) > 0:
        axes[0].scatter(np.array(q_peaks)/fs, ecg_raw[q_peaks], color='orange', s=80, 
                       marker='v', zorder=5, label=f'Q-waves ({len(q_peaks)})')
    axes[0].scatter(rpeaks/fs, ecg_raw[rpeaks], color='red', s=100, 
                   marker='x', zorder=5, label=f'R-peaks ({len(rpeaks)})')
    if len(s_peaks) > 0:
        axes[0].scatter(np.array(s_peaks)/fs, ecg_raw[s_peaks], color='purple', s=80, 
                       marker='^', zorder=5, label=f'S-waves ({len(s_peaks)})')
    if len(t_peaks) > 0:
        axes[0].scatter(np.array(t_peaks)/fs, ecg_raw[t_peaks], color='brown', s=80, 
                       marker='s', zorder=5, label=f'T-waves ({len(t_peaks)})')
    
    axes[0].set_title('ECG Signal with All PQRST Waves Detected', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Amplitude (mV)')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Normalized ECG with PQRST waves
    axes[1].plot(time, ecg_normalized, 'b-', linewidth=1.5, label='Normalized ECG')
    
    # Plot all normalized waves
    if len(p_peaks) > 0:
        axes[1].scatter(np.array(p_peaks)/fs, ecg_normalized[p_peaks], color='green', s=80, 
                       marker='o', zorder=5, label='P-waves')
    if len(q_peaks) > 0:
        axes[1].scatter(np.array(q_peaks)/fs, ecg_normalized[q_peaks], color='orange', s=80, 
                       marker='v', zorder=5, label='Q-waves')
    axes[1].scatter(rpeaks/fs, ecg_normalized[rpeaks], color='red', s=100, 
                   marker='x', zorder=5, label='R-peaks')
    if len(s_peaks) > 0:
        axes[1].scatter(np.array(s_peaks)/fs, ecg_normalized[s_peaks], color='purple', s=80, 
                       marker='^', zorder=5, label='S-waves')
    if len(t_peaks) > 0:
        axes[1].scatter(np.array(t_peaks)/fs, ecg_normalized[t_peaks], color='brown', s=80, 
                       marker='s', zorder=5, label='T-waves')
    
    axes[1].axhline(y=0.5, color='k', linestyle='--', alpha=0.7, label='Spike Threshold')
    axes[1].set_title('Normalized ECG with PQRST Waves for SNN Input', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Normalized Amplitude')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Combined spike train
    axes[2].plot(time, ecg_normalized, 'b-', alpha=0.4, linewidth=1, label='ECG Signal')
    axes[2].plot(time, spike_train['Combined'], 'r-', linewidth=2, label='Combined Spike Train')
    axes[2].set_title('Combined Spike Train for All PQRST Waves', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Spike Amplitude')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Individual spike trains
    colors = ['green', 'orange', 'red', 'purple', 'brown']
    wave_types = ['P', 'Q', 'R', 'S', 'T']
    
    for i, (wave_type, color) in enumerate(zip(wave_types, colors)):
        offset = i * 0.2
        axes[3].plot(time, spike_train[wave_type] + offset, color=color, linewidth=1.5, 
                    label=f'{wave_type}-wave spikes')
    
    axes[3].set_title('Individual Spike Trains for Each Wave Type', fontsize=14, fontweight='bold')
    axes[3].set_ylabel('Spike Amplitude (offset)')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    # Plot 5: Spike raster for all waves
    spike_raster_data = []
    spike_labels = []
    
    for i, wave_type in enumerate(wave_types):
        spike_times_wave = np.where(spike_train[wave_type] > 0.5)[0] / fs
        if len(spike_times_wave) > 0:
            spike_raster_data.append(spike_times_wave)
            spike_labels.append(f'{wave_type}-wave')
    
    if spike_raster_data:
        axes[4].eventplot(spike_raster_data, lineoffsets=range(1, len(spike_raster_data)+1), 
                         linelengths=0.8, colors=colors[:len(spike_raster_data)], linewidths=3)
        axes[4].set_yticks(range(1, len(spike_raster_data)+1))
        axes[4].set_yticklabels(spike_labels)
    
    axes[4].set_title('Spike Raster Plot for All PQRST Waves', fontsize=14, fontweight='bold')
    axes[4].set_ylabel('Wave Type')
    axes[4].set_xlabel('Time (s)')
    axes[4].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pqrst_spike_plots/pqrst_spike_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return ecg_raw, ecg_normalized, rpeaks, p_peaks, q_peaks, s_peaks, t_peaks, spike_train

def generate_spikes_around_rpeaks(ecg_normalized, rpeaks, fs, spike_window=0.1):
    """
    Generate spikes in time windows around detected R-peaks
    
    Args:
        ecg_normalized: Normalized ECG signal (0-1)
        rpeaks: R-peak locations in samples
        fs: Sampling frequency
        spike_window: Time window around R-peak to generate spikes (seconds)
    
    Returns:
        spike_train: Binary spike train
    """
    spike_train = np.zeros_like(ecg_normalized)
    window_samples = int(spike_window * fs)
    
    for rpeak in rpeaks:
        # Define window around R-peak
        start = max(0, rpeak - window_samples//2)
        end = min(len(ecg_normalized), rpeak + window_samples//2)
        
        # Generate spikes where signal crosses threshold in this window
        window_signal = ecg_normalized[start:end]
        threshold = 0.5
        
        # Find threshold crossings
        crossings = np.where(np.diff(np.signbit(window_signal - threshold)))[0]
        
        # Generate spikes at crossings
        for crossing in crossings:
            spike_idx = start + crossing
            if spike_idx < len(spike_train):
                spike_train[spike_idx] = 1.0
    
    return spike_train

def analyze_pqrst_intervals(rpeaks, p_peaks, q_peaks, s_peaks, t_peaks, fs):
    """
    Analyze intervals and timing of all PQRST waves
    """
    if len(rpeaks) < 2:
        return None
    
    plt.figure(figsize=(15, 12))
    
    # Calculate intervals
    rr_intervals = np.diff(rpeaks) / fs
    heart_rates = 60 / rr_intervals
    
    # Plot 1: PQRST timing for first few beats
    plt.subplot(3, 3, 1)
    max_beats = min(3, len(rpeaks))
    colors = ['green', 'orange', 'red', 'purple', 'brown']
    wave_names = ['P', 'Q', 'R', 'S', 'T']
    
    for beat in range(max_beats):
        r_time = rpeaks[beat] / fs
        plt.axvline(x=r_time, color='red', linestyle='-', alpha=0.8, linewidth=2)
        
        # Plot relative wave timings
        if beat < len(p_peaks):
            plt.axvline(x=p_peaks[beat]/fs, color='green', linestyle='--', alpha=0.7)
        if beat < len(q_peaks):
            plt.axvline(x=q_peaks[beat]/fs, color='orange', linestyle='--', alpha=0.7)
        if beat < len(s_peaks):
            plt.axvline(x=s_peaks[beat]/fs, color='purple', linestyle='--', alpha=0.7)
        if beat < len(t_peaks):
            plt.axvline(x=t_peaks[beat]/fs, color='brown', linestyle='--', alpha=0.7)
    
    plt.title('PQRST Timing (First 3 Beats)', fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Wave Type')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: RR intervals
    plt.subplot(3, 3, 2)
    plt.plot(range(len(rr_intervals)), rr_intervals, 'bo-', linewidth=2, markersize=8)
    plt.title('RR Intervals', fontweight='bold')
    plt.ylabel('RR Interval (s)')
    plt.xlabel('Beat Number')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Heart Rate
    plt.subplot(3, 3, 3)
    plt.plot(range(len(heart_rates)), heart_rates, 'ro-', linewidth=2, markersize=8)
    plt.title('Heart Rate', fontweight='bold')
    plt.ylabel('Heart Rate (BPM)')
    plt.xlabel('Beat Number')
    plt.grid(True, alpha=0.3)
    
    # Calculate wave intervals relative to R-peaks
    pr_intervals = []
    qr_intervals = []
    rs_intervals = []
    rt_intervals = []
    
    for i, rpeak in enumerate(rpeaks):
        if i < len(p_peaks) and not np.isnan(p_peaks[i]):
            pr_intervals.append((rpeak - p_peaks[i]) / fs)
        if i < len(q_peaks) and not np.isnan(q_peaks[i]):
            qr_intervals.append((rpeak - q_peaks[i]) / fs)
        if i < len(s_peaks) and not np.isnan(s_peaks[i]):
            rs_intervals.append((s_peaks[i] - rpeak) / fs)
        if i < len(t_peaks) and not np.isnan(t_peaks[i]):
            rt_intervals.append((t_peaks[i] - rpeak) / fs)
    
    # Plot wave intervals
    intervals_data = [pr_intervals, qr_intervals, rs_intervals, rt_intervals]
    interval_names = ['PR', 'QR', 'RS', 'RT']
    interval_colors = ['green', 'orange', 'purple', 'brown']
    
    for i, (intervals, name, color) in enumerate(zip(intervals_data, interval_names, interval_colors)):
        if intervals:
            plt.subplot(3, 3, 4 + i)
            plt.plot(range(len(intervals)), intervals, 'o-', color=color, linewidth=2, markersize=8)
            plt.title(f'{name} Intervals', fontweight='bold')
            plt.ylabel('Interval (s)')
            plt.xlabel('Beat Number')
            plt.grid(True, alpha=0.3)
    
    # Plot 8: Wave count statistics
    plt.subplot(3, 3, 8)
    wave_counts = [len(p_peaks), len(q_peaks), len(rpeaks), len(s_peaks), len(t_peaks)]
    plt.bar(wave_names, wave_counts, color=colors)
    plt.title('Detected Wave Counts', fontweight='bold')
    plt.ylabel('Number of Waves')
    plt.xlabel('Wave Type')
    plt.grid(True, alpha=0.3)
    
    # Plot 9: Statistics summary
    plt.subplot(3, 3, 9)
    stats_text = f"""
    Total Beats: {len(rpeaks)}
    P-waves: {len(p_peaks)}
    Q-waves: {len(q_peaks)}
    R-waves: {len(rpeaks)}
    S-waves: {len(s_peaks)}
    T-waves: {len(t_peaks)}
    
    Mean HR: {np.mean(heart_rates):.1f} BPM
    Mean RR: {np.mean(rr_intervals):.3f} s
    """
    
    if pr_intervals:
        stats_text += f"\nMean PR: {np.mean(pr_intervals):.3f} s"
    if qr_intervals:
        stats_text += f"\nMean QR: {np.mean(qr_intervals):.3f} s"
    if rs_intervals:
        stats_text += f"\nMean RS: {np.mean(rs_intervals):.3f} s"
    if rt_intervals:
        stats_text += f"\nMean RT: {np.mean(rt_intervals):.3f} s"
    
    plt.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    plt.axis('off')
    plt.title('PQRST Statistics', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('pqrst_spike_plots/pqrst_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return rr_intervals, heart_rates

def create_pqrst_based_snn_input(ecg_signal, spike_trains, fs, beat_window=0.8):
    """
    Create SNN input format based on PQRST spike trains
    
    Args:
        ecg_signal: Normalized ECG signal
        spike_trains: Dictionary of spike trains for each wave type
        fs: Sampling frequency
        beat_window: Time window for each input sample (seconds)
    
    Returns:
        multi_channel_tensor: Tensor with separate channels for each wave type
        combined_tensor: Tensor with combined spike train
    """
    window_samples = int(beat_window * fs)
    num_windows = len(ecg_signal) // window_samples
    
    # Create multi-channel input (one channel per wave type)
    wave_types = ['P', 'Q', 'R', 'S', 'T']
    multi_channel_data = []
    combined_data = []
    
    for i in range(num_windows):
        start_idx = i * window_samples
        end_idx = start_idx + window_samples
        
        # Multi-channel window (5 channels for P,Q,R,S,T)
        window_channels = []
        for wave_type in wave_types:
            window_spike = spike_trains[wave_type][start_idx:end_idx]
            if len(window_spike) < window_samples:
                window_spike = np.pad(window_spike, (0, window_samples - len(window_spike)), mode='constant')
            window_channels.append(window_spike)
        
        multi_channel_data.append(window_channels)
        
        # Combined window
        combined_window = spike_trains['Combined'][start_idx:end_idx]
        if len(combined_window) < window_samples:
            combined_window = np.pad(combined_window, (0, window_samples - len(combined_window)), mode='constant')
        combined_data.append(combined_window)
    
    # Convert to tensors
    multi_channel_tensor = torch.tensor(multi_channel_data, dtype=torch.float32)
    combined_tensor = torch.tensor(combined_data, dtype=torch.float32)
    
    # Visualize multi-channel SNN input
    plt.figure(figsize=(18, 12))
    
    # Plot first window with all channels
    if len(multi_channel_data) > 0:
        window_idx = 0
        time_axis = np.arange(window_samples) / fs
        
        for i, wave_type in enumerate(wave_types):
            plt.subplot(3, 2, i+1)
            plt.plot(time_axis, multi_channel_data[window_idx][i], 
                    color=['green', 'orange', 'red', 'purple', 'brown'][i], linewidth=2)
            plt.title(f'{wave_type}-wave Channel (Window 1)', fontweight='bold')
            plt.ylabel('Spike Amplitude')
            plt.xlabel('Time (s)')
            plt.grid(True, alpha=0.3)
        
        # Plot combined channel
        plt.subplot(3, 2, 6)
        plt.plot(time_axis, combined_data[window_idx], 'k-', linewidth=2)
        plt.title('Combined PQRST Channel (Window 1)', fontweight='bold')
        plt.ylabel('Spike Amplitude')
        plt.xlabel('Time (s)')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pqrst_spike_plots/pqrst_snn_input.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return multi_channel_tensor, combined_tensor

def detect_pqrst_waves_simple(ecg_signal, rpeaks, fs):
    """
    Simplified detection of P, Q, S, T waves relative to R-peaks
    """
    p_onsets = []
    p_peaks = []
    q_peaks = []
    s_peaks = []
    t_peaks = []
    t_offsets = []
    
    # Typical timing relative to R-peak (in samples)
    p_offset = int(-0.15 * fs)  # P wave ~150ms before R
    q_offset = int(-0.04 * fs)  # Q wave ~40ms before R
    s_offset = int(0.04 * fs)   # S wave ~40ms after R
    t_offset = int(0.15 * fs)   # T wave ~150ms after R
    
    for rpeak in rpeaks:
        # P wave detection
        p_search_start = max(0, rpeak + int(-0.2 * fs))
        p_search_end = min(len(ecg_signal), rpeak + int(-0.08 * fs))
        if p_search_end > p_search_start:
            p_window = ecg_signal[p_search_start:p_search_end]
            p_local_peak = np.argmax(p_window) + p_search_start
            p_peaks.append(p_local_peak)
            p_onsets.append(max(0, p_local_peak - int(0.04 * fs)))
        
        # Q wave detection (minimum before R)
        q_search_start = max(0, rpeak + int(-0.08 * fs))
        q_search_end = min(len(ecg_signal), rpeak)
        if q_search_end > q_search_start:
            q_window = ecg_signal[q_search_start:q_search_end]
            q_local_min = np.argmin(q_window) + q_search_start
            q_peaks.append(q_local_min)
        
        # S wave detection (minimum after R)
        s_search_start = max(0, rpeak)
        s_search_end = min(len(ecg_signal), rpeak + int(0.08 * fs))
        if s_search_end > s_search_start:
            s_window = ecg_signal[s_search_start:s_search_end]
            s_local_min = np.argmin(s_window) + s_search_start
            s_peaks.append(s_local_min)
        
        # T wave detection
        t_search_start = max(0, rpeak + int(0.08 * fs))
        t_search_end = min(len(ecg_signal), rpeak + int(0.25 * fs))
        if t_search_end > t_search_start:
            t_window = ecg_signal[t_search_start:t_search_end]
            t_local_peak = np.argmax(t_window) + t_search_start
            t_peaks.append(t_local_peak)
            t_offsets.append(min(len(ecg_signal)-1, t_local_peak + int(0.08 * fs)))
    
    return p_onsets, p_peaks, q_peaks, s_peaks, t_peaks, t_offsets

def generate_spikes_for_pqrst(ecg_normalized, rpeaks, p_peaks, q_peaks, s_peaks, t_peaks, fs, spike_window=0.02):
    """
    Generate spikes for all P, Q, R, S, T waves
    
    Args:
        ecg_normalized: Normalized ECG signal (0-1)
        rpeaks, p_peaks, q_peaks, s_peaks, t_peaks: Wave locations
        fs: Sampling frequency
        spike_window: Time window around each wave to generate spikes (seconds)
    
    Returns:
        spike_trains: Dictionary with spike trains for each wave type
    """
    spike_trains = {
        'P': np.zeros_like(ecg_normalized),
        'Q': np.zeros_like(ecg_normalized),
        'R': np.zeros_like(ecg_normalized),
        'S': np.zeros_like(ecg_normalized),
        'T': np.zeros_like(ecg_normalized),
        'Combined': np.zeros_like(ecg_normalized)
    }
    
    window_samples = int(spike_window * fs)
    
    # Generate spikes for each wave type
    wave_data = {
        'P': p_peaks,
        'Q': q_peaks,
        'R': rpeaks,
        'S': s_peaks,
        'T': t_peaks
    }
    
    for wave_type, wave_peaks in wave_data.items():
        for peak in wave_peaks:
            if 0 <= peak < len(ecg_normalized):
                # Generate spike at exact peak location
                spike_trains[wave_type][peak] = 1.0
                spike_trains['Combined'][peak] = 1.0
                
                # Optional: Generate spikes in window around peak
                start = max(0, peak - window_samples//2)
                end = min(len(ecg_normalized), peak + window_samples//2)
                
                # Add some spikes around the peak based on signal amplitude
                window_signal = ecg_normalized[start:end]
                threshold = np.mean(window_signal)
                
                for i in range(len(window_signal)):
                    if window_signal[i] > threshold:
                        spike_idx = start + i
                        if spike_idx < len(spike_trains[wave_type]):
                            spike_trains[wave_type][spike_idx] = min(1.0, spike_trains[wave_type][spike_idx] + 0.3)
                            spike_trains['Combined'][spike_idx] = min(1.0, spike_trains['Combined'][spike_idx] + 0.3)
    
    return spike_trains

def main():
    """
    Main function to analyze PQRST waves and generate spikes
    """
    print("=" * 70)
    print("PQRST WAVE BASED SPIKE GENERATION FOR SNN")
    print("=" * 70)
    
    # Generate PQRST based spikes
    results = generate_pqrst_based_spikes()
    ecg_raw, ecg_normalized, rpeaks, p_peaks, q_peaks, s_peaks, t_peaks, spike_trains = results
    
    # Analyze PQRST intervals
    print("\nAnalyzing PQRST wave intervals...")
    rr_intervals, heart_rates = analyze_pqrst_intervals(rpeaks, p_peaks, q_peaks, s_peaks, t_peaks, fs=500)
    
    # Create PQRST based SNN input
    print("\nCreating PQRST based SNN input format...")
    multi_channel_tensor, combined_tensor = create_pqrst_based_snn_input(ecg_normalized, spike_trains, fs=500)
    
    # Summary statistics
    wave_spike_counts = {}
    total_spikes = 0
    
    for wave_type in ['P', 'Q', 'R', 'S', 'T', 'Combined']:
        spike_count = np.sum(spike_trains[wave_type] > 0.5)
        wave_spike_counts[wave_type] = spike_count
        if wave_type != 'Combined':
            total_spikes += spike_count
    
    print("\n" + "=" * 70)
    print("PQRST ANALYSIS RESULTS")
    print("=" * 70)
    print(f"Detected waves:")
    print(f"  P-waves: {len(p_peaks)}")
    print(f"  Q-waves: {len(q_peaks)}")
    print(f"  R-waves: {len(rpeaks)}")
    print(f"  S-waves: {len(s_peaks)}")
    print(f"  T-waves: {len(t_peaks)}")
    
    print(f"\nGenerated spikes:")
    for wave_type in ['P', 'Q', 'R', 'S', 'T']:
        print(f"  {wave_type}-wave spikes: {wave_spike_counts[wave_type]}")
    print(f"  Combined spikes: {wave_spike_counts['Combined']}")
    print(f"  Total individual spikes: {total_spikes}")
    
    print(f"\nSNN Input Tensor Shapes:")
    print(f"  Multi-channel tensor: {multi_channel_tensor.shape}")
    print(f"    - Batch size: {multi_channel_tensor.shape[0]}")
    print(f"    - Channels (P,Q,R,S,T): {multi_channel_tensor.shape[1]}")
    print(f"    - Time steps: {multi_channel_tensor.shape[2]}")
    print(f"  Combined tensor: {combined_tensor.shape}")
    print(f"    - Batch size: {combined_tensor.shape[0]}")
    print(f"    - Time steps: {combined_tensor.shape[1]}")
    
    if rr_intervals is not None:
        print(f"\nCardiac Metrics:")
        print(f"  Mean RR interval: {np.mean(rr_intervals):.3f} seconds")
        print(f"  Mean heart rate: {np.mean(heart_rates):.1f} BPM")
        print(f"  Heart rate range: {np.min(heart_rates):.1f} - {np.max(heart_rates):.1f} BPM")
    
    print("\nGenerated files in 'pqrst_spike_plots/' directory:")
    print("- pqrst_spike_analysis.png: Complete PQRST wave and spike analysis")
    print("- pqrst_analysis.png: PQRST wave intervals and timing analysis")
    print("- pqrst_snn_input.png: Multi-channel SNN input format")
    
    print(f"\nReady for SNN model with {len(['P', 'Q', 'R', 'S', 'T'])} input channels!")
    
    return rpeaks, p_peaks, q_peaks, s_peaks, t_peaks, spike_trains, multi_channel_tensor, combined_tensor

if __name__ == "__main__":
    results = main()
    rpeaks, p_peaks, q_peaks, s_peaks, t_peaks, spike_trains, multi_channel_tensor, combined_tensor = results
