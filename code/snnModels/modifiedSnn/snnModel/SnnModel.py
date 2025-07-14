import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import torch
import warnings

class SNN(nn.Module):
    def __init__(self, num_inputs=750, num_hidden=[256, 512, 512, 256, 128], num_outputs=2, num_steps=10, beta=0.9, dropout_rate=0.1):
        super().__init__()
        self.num_steps = num_steps
        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        # Layer definitions
        layers = []
        prev_size = num_inputs // num_steps if num_inputs % num_steps == 0 else num_inputs
        self.layer_sizes = [prev_size] + num_hidden
        
        # Input to first hidden
        layers.extend([
            nn.Linear(prev_size, num_hidden[0]),
            nn.BatchNorm1d(num_hidden[0]),
            snn.Leaky(beta=beta, spike_grad=spike_grad),
            nn.Dropout(dropout_rate)
        ])
        prev_size = num_hidden[0]
        
        # Hidden layers with residual connections
        for i in range(len(num_hidden) - 1):
            layers.extend([
                nn.Linear(prev_size, num_hidden[i + 1]),
                nn.BatchNorm1d(num_hidden[i + 1]),
                snn.Leaky(beta=beta, spike_grad=spike_grad),
                nn.Dropout(dropout_rate)
            ])
            if prev_size == num_hidden[i + 1]:
                layers.append(ResidualAdd())
            prev_size = num_hidden[i + 1]
        
        # Output layer
        layers.extend([
            nn.Linear(prev_size, num_outputs),
            snn.Leaky(beta=beta, spike_grad=spike_grad)
        ])
        
        self.network = nn.Sequential(*layers)
        self.residual_applied = [i for i in range(4, len(layers) - 2, 4) if i > 0 and self.layer_sizes[i // 4] == self.layer_sizes[(i - 4) // 4]]

    def forward(self, x):
        batch_size = x.size(0)
        if x.dim() == 2:
            if x.size(1) == 300:
                x = x.view(batch_size, self.num_steps, 30)  # Reshape (batch_size, 300) to (batch_size, 10, 30)
                #print(f"Reshaped input from (batch_size, 300) to (batch_size, {self.num_steps}, 30)")
            elif x.size(1) == 750:
                x = x.view(batch_size, self.num_steps, 75)  # Original handling for 750
                #print(f"Reshaped input from (batch_size, 750) to (batch_size, {self.num_steps}, 75)")
            elif x.size(1) == 250:
                x = x.unsqueeze(1).repeat(1, self.num_steps, 1) // (self.num_steps / 3.33)  # Original handling for 250
                warnings.warn(f"Input shape (batch_size, 250) detected. Adjusted to (batch_size, {self.num_steps}, 75) for compatibility.")
            else:
                raise ValueError(f"Unsupported input size {x.size(1)}. Expected 250, 300, or 750.")
        elif x.dim() == 3:
            if x.size(1) == self.num_steps and x.size(2) in [30, 75]:
                print(f"Input already shaped as (batch_size, {self.num_steps}, {x.size(2)})")
            else:
                raise ValueError(f"Unsupported input shape {x.shape}. Expected (batch_size, 10, 30) or (batch_size, 10, 75).")

        mems = [layer.init_leaky() for layer in self.network if isinstance(layer, snn.Leaky)]
        spk_rec = []

        for step in range(self.num_steps):
            x_step = x[:, step, :]
            spk = x_step
            mem_idx = 0
            for i, layer in enumerate(self.network):
                if isinstance(layer, nn.Linear):
                    spk = layer(spk)
                elif isinstance(layer, nn.BatchNorm1d):
                    spk = layer(spk)
                elif isinstance(layer, snn.Leaky):
                    spk, mems[mem_idx] = layer(spk, mems[mem_idx])
                    mem_idx += 1
                elif isinstance(layer, nn.Dropout):
                    spk = layer(spk)
                elif isinstance(layer, ResidualAdd) and i in self.residual_applied:
                    prev_spk = self.network[i - 4](spk)
                    spk = spk + prev_spk
            spk_rec.append(spk)

        return torch.stack(spk_rec, dim=0).sum(dim=0)

class ResidualAdd(nn.Module):
    def forward(self, x):
        return x