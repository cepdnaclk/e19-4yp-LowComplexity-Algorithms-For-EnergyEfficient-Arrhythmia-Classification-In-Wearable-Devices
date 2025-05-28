import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

class SNNConvClassifier(nn.Module):
    def __init__(self, num_inputs=200, num_hidden=128, num_outputs=1, num_steps=3, beta=0.9):
        super().__init__()
        self.num_steps = num_steps
        spike_grad = surrogate.fast_sigmoid(slope=25)

        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.fc1 = nn.Linear(32 * (num_inputs // 4), num_hidden)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        batch_size = x.size(0)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk_rec = []
        for _ in range(self.num_steps):
            cur = self.conv1(x)
            spk1, mem1 = self.lif1(self.pool1(cur), mem1)
            cur = self.conv2(spk1)
            spk2, mem2 = self.lif2(self.pool2(cur), mem2)
            cur = spk2.view(batch_size, -1)
            cur = self.fc1(cur)
            spk3, mem3 = self.lif3(cur, mem3)
            out = self.fc2(spk3)
            spk_rec.append(out)

        return torch.stack(spk_rec, dim=0).sum(dim=0)