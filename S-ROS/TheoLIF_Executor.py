"""
TheoLIF_Executor.py

BIOLOGICAL GROUNDING
Discrete-time LIF recurrent executor that provides noise resilience and long state-holding periods as proven in Sun et al. (2025) DTSRNN for FSM emulation. Weights are frozen after crystallization from the plastic core.

Primary grounding paper:
Sun Z et al. (2025). "General and stable emulation of finite state machines with spiking neural networks." Neuromorphic Computing and Engineering, 5(1):014016. DOI: 10.1088/2634-4386/adc32d
"""

import torch
import torch.nn as nn

class TheoLIF_Executor(nn.Module):
    def __init__(self, hidden_dim: int = 128, beta: float = 0.7, num_steps: int = 25):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.num_steps = num_steps
        # Recurrent weights (frozen after crystallization)
        self.rec_weight = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.input_weight = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.threshold = 1.0

    @torch.no_grad()
    def forward(self, initial_state: torch.Tensor) -> torch.Tensor:
        # initial_state is one-hot retrieved engram (128-dim)
        v = torch.zeros_like(initial_state)  # membrane potential
        spikes = torch.zeros_like(initial_state)
        for _ in range(self.num_steps):
            v = self.beta * v + (1 - self.beta) * torch.matmul(spikes, self.rec_weight) + torch.matmul(initial_state, self.input_weight)
            spikes = (v >= self.threshold).float()
            v = v * (1 - spikes)  # reset
        return spikes  # final spike train for action decoding
