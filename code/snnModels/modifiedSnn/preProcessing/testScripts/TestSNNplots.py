import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import spikegen, surrogate
import matplotlib.pyplot as plt
import wfdb
from scipy.signal import resample
import neurokit2 as nk
import os

# Module 1: Data Preprocessing
class ECGDataProcessor:
    def __init__(self, record_name='100', sampling_rate=360, beat_length=250, pn_dir='mitdb'):
        self.record_name = record_name
        self.sampling_rate = sampling_rate
        self.beat_length = beat_length
        self.pn_dir = pn_dir

    def load_ecg_beat(self, sample_idx=0):
        """Load and preprocess a single ECG beat."""
        try:
            if os.path.exists(os.path.join(self.pn_dir, f"{self.record_name}.hea")):
                record = wfdb.rdrecord(self.record_name, pn_dir=self.pn_dir)
            else:
                record = wfdb.rdrecord(self.record_name, pn_dir='mitdb', physical=True)
            ecg_signal = record.p_signal[:, 0]  # Use lead MLII
            ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=self.sampling_rate)
            rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=self.sampling_rate)[1]['ECG_R_Peaks']
            if len(rpeaks) <= sample_idx:
                raise ValueError(f"Sample index {sample_idx} exceeds available R-peaks ({len(rpeaks)})")
            beat_idx = rpeaks[sample_idx]
            beat = ecg_cleaned[beat_idx-100:beat_idx+150]  # Extract 250 samples around R-peak
            if len(beat) < self.beat_length:
                beat = np.pad(beat, (0, self.beat_length - len(beat)), mode='constant')
            beat = resample(beat, self.beat_length)
            beat = beat / np.max(np.abs(beat)) if np.max(np.abs(beat)) != 0 else beat
            return beat
        except Exception as e:
            print(f"Failed to load ECG beat for record {self.record_name}: {str(e)}")
            return None

    def delta_modulation(self, beat, threshold=0.005):
        """Convert ECG beat to spike train using delta modulation."""
        spikes = spikegen.delta(torch.tensor(beat), threshold=threshold, off_spike=True).numpy()
        print(f"Record {self.record_name} - Input spikes generated: {np.sum(spikes)} spikes")
        return spikes

# Module 2: SNN Model
class SNN(nn.Module):
    def __init__(self, num_inputs=250, num_hidden=128, num_outputs=5, num_steps=50, beta=0.9, threshold=0.5):
        super().__init__()
        self.num_steps = num_steps
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        """Forward pass returning spikes and membrane potentials."""
        batch_size = x.size(0)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk1_rec, spk2_rec, mem1_rec, mem2_rec = [], [], [], []
        for step in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk1_rec.append(spk1)
            spk2_rec.append(spk2)
            mem1_rec.append(mem1)
            mem2_rec.append(mem2)
        spk1_rec = torch.stack(spk1_rec, dim=0)
        spk2_rec = torch.stack(spk2_rec, dim=0)
        mem1_rec = torch.stack(mem1_rec, dim=0)
        mem2_rec = torch.stack(mem2_rec, dim=0)
        print(f"Record {self.record_name} - Hidden layer spikes: {spk1_rec.sum().item()} spikes")
        print(f"Record {self.record_name} - Output layer spikes: {spk2_rec.sum().item()} spikes")
        print(f"Record {self.record_name} - Hidden layer membrane potential range: {mem1_rec.min().item():.4f} to {mem1_rec.max().item():.4f}")
        print(f"Record {self.record_name} - Output layer membrane potential range: {mem2_rec.min().item():.4f} to {mem2_rec.max().item():.4f}")
        return spk2_rec, mem1_rec, mem2_rec

# Module 3: Plotting
class SNNPlotter:
    def __init__(self):
        self.figsize = (12, 10)

    def plot_signals(self, ecg_beat, input_spikes, mem1, mem2, spk2, title_prefix="", output_file=None):
        """Plot raw ECG, input spikes, membrane potentials, and output spikes."""
        plt.figure(figsize=self.figsize)

        # Raw ECG Signal
        plt.subplot(4, 1, 1)
        plt.plot(ecg_beat, label='Raw ECG Signal')
        plt.title(f'{title_prefix}Raw ECG Signal')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude (Normalized)')
        plt.grid(True)
        plt.legend()

        # Input Spikes
        plt.subplot(4, 1, 2)
        plt.eventplot(np.where(input_spikes == 1)[0], lineoffsets=1, colors='black')
        plt.title(f'{title_prefix}Input Spikes (Delta Modulation)')
        plt.xlabel('Sample')
        plt.ylabel('Spike')
        plt.grid(True)

        # Membrane Potential (Hidden Layer, First 5 Neurons)
        plt.subplot(4, 1, 3)
        for i in range(min(5, mem1.shape[2])):
            plt.plot(mem1[:, 0, i], label=f'Neuron {i+1}')
        plt.title(f'{title_prefix}Membrane Potential (Hidden Layer)')
        plt.xlabel('Time Step')
        plt.ylabel('Membrane Potential')
        plt.grid(True)
        plt.legend()

        # Output Spikes (All 5 Classes)
        plt.subplot(4, 1, 4)
        for i in range(spk2.shape[2]):
            spike_times = np.where(spk2[:, 0, i] == 1)[0]
            plt.eventplot(spike_times, lineoffsets=i+1, colors=f'C{i}', label=f'Class {i}')
        plt.title(f'{title_prefix}Output Spikes')
        plt.xlabel('Time Step')
        plt.ylabel('Class')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        if output_file:
            # Ensure unique filename by appending a counter if file exists
            base, ext = os.path.splitext(output_file)
            counter = 1
            unique_output_file = output_file
            while os.path.exists(unique_output_file):
                unique_output_file = f"{base}_{counter}{ext}"
                counter += 1
            try:
                plt.savefig(unique_output_file)
                print(f"Plot saved to {unique_output_file}")
            except Exception as e:
                print(f"Failed to save plot to {unique_output_file}: {str(e)}")
            plt.close()
        else:
            plt.show()

# Main Pipeline
def main():
    # List of MIT-BIH record numbers
    record_numbers = [
        '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
        '112', '113', '114', '115', '116', '117', '118', '119', '121', '122',
        '123', '124', '200', '201', '202', '203', '205', '207', '208', '209',
        '210', '212', '213', '214', '215', '217', '219', '220', '221', '222',
        '223', '228', '230', '231', '232', '234'
    ]

    # Initialize modules
    snn_model = SNN(num_inputs=250, num_hidden=128, num_outputs=5, num_steps=50, beta=0.9, threshold=0.5)
    plotter = SNNPlotter()

    # Create output directory for plots
    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)

    # Loop over all records
    for record_name in record_numbers:
        print(f"\nProcessing record {record_name}...")
        data_processor = ECGDataProcessor(record_name=record_name, beat_length=250, pn_dir='mitdb')
        snn_model.record_name = record_name  # For diagnostic printing

        # Load and preprocess ECG beat
        ecg_beat = data_processor.load_ecg_beat(sample_idx=0)
        if ecg_beat is None:
            continue  # Skip if loading fails

        # Apply delta modulation
        input_spikes = data_processor.delta_modulation(ecg_beat, threshold=0.005)

        # Prepare input for SNN
        input_tensor = torch.tensor([input_spikes], dtype=torch.float32)

        # Run through SNN
        spk2, mem1, mem2 = snn_model(input_tensor)

        # Convert to numpy for plotting
        spk2 = spk2.detach().numpy()
        mem1 = mem1.detach().numpy()
        mem2 = mem2.detach().numpy()

        # Save plot
        output_file = os.path.join(output_dir, f'ecg_snn_analysis_record_{record_name}.png')
        plotter.plot_signals(
            ecg_beat, input_spikes, mem1, mem2, spk2,
            title_prefix=f"Record {record_name} - ",
            output_file=output_file
        )

if __name__ == "__main__":
    main()