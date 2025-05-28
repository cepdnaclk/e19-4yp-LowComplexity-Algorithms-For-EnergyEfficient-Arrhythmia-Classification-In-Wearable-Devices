import torch
import torch.nn as nn

class LIFNeuronLayer(nn.Module):
    def __init__(self, size, tau=2.0, v_threshold=1.0, v_reset=0.0):
        super(LIFNeuronLayer, self).__init__()
        self.size = size
        self.tau = tau
        self.v_threshold = v_threshold
        self.v_reset = v_reset

    def forward(self, input, mem):
        # Leaky Integrate-and-Fire neuron update
        mem = mem + (input - mem) / self.tau
        spike = (mem >= self.v_threshold).float()
        mem = torch.where(spike.bool(), torch.full_like(mem, self.v_reset), mem)
        return spike, mem
