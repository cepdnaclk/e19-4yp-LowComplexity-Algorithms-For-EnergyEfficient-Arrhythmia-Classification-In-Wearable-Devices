# SNN Model for 5-Class Classification

import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import torch 

class SNN(nn.Module):
    def __init__(self, num_inputs=250, num_hidden=128, num_outputs=5, num_steps=25, beta=0.9):
        super().__init__()
        self.num_steps = num_steps
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        batch_size = x.size(0)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec = []
        for step in range(self.num_steps):
            cur1 = self.fc1(x)  
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
        return torch.stack(spk2_rec, dim=0).sum(dim=0)
    
# if __name__ == "__main__":
#     from torchsummary import summary
#     import torch

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = SNN(num_inputs=250, num_outputs=5).to(device)
#     summary(model, input_size=(250,), device=str(device))

    

