
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import snntorch.utils as utils

class CSNN(nn.Module):
    def __init__(self, num_inputs=250, num_outputs=5, num_steps=50, beta=0.5, device='cpu'):
        super().__init__()
        self.num_steps = num_steps
        self.device = device
        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        self.net = nn.Sequential(
            nn.Conv1d(1, 12, kernel_size=5),  # 12 filters, kernel size 5
            nn.MaxPool1d(2),                  # Max pooling with kernel size 2
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Conv1d(12, 64, kernel_size=5), # 64 filters, kernel size 5
            nn.MaxPool1d(2),                  # Max pooling with kernel size 2
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Flatten(),
            nn.Linear(64 * 59, num_outputs),  # Adjusted for input size 250
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
        ).to(device)
    
    def forward(self, x):
        spk_rec = []
        mem_rec = []
        utils.reset(self.net)  # Reset hidden states
        
        for step in range(self.num_steps):
            spk, mem = self.net(x)
            spk_rec.append(spk)
            mem_rec.append(mem)
        
        return torch.stack(spk_rec), torch.stack(mem_rec)
