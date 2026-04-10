"""
Theo_S-CAM_t.py

BIOLOGICAL GROUNDING
This file models spiking content-addressable memory as the rigid engram store of the deterministic core. It holds crystallized, immutable sensor-to-action mappings exactly as neocortical engrams do after systems consolidation. Now uses strict one-hot encoding per Sun et al. (2025) for FSM stability.

Primary grounding papers:
Hopfield JJ (1982). "Neural networks and physical systems with emergent collective computational abilities." PNAS, 79(8):2554-2558. DOI: 10.1073/pnas.79.8.2554
Sun Z et al. (2025). "General and stable emulation of finite state machines with spiking neural networks." Neuromorphic Computing and Engineering, 5(1):014016. DOI: 10.1088/2634-4386/adc32d
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict

class TheoSCAM(nn.Module):
    def __init__(self, spike_dim: int = 128, max_engrams: int = 4096):
        super().__init__()
        self.spike_dim = spike_dim
        self.max_engrams = max_engrams
        self.register_buffer("sensor_keys", torch.zeros(max_engrams, spike_dim))
        self.register_buffer("action_values", torch.zeros(max_engrams, spike_dim))
        self.register_buffer("is_active", torch.zeros(max_engrams, dtype=torch.bool))
        self.register_buffer("usage_counts", torch.zeros(max_engrams, dtype=torch.long))

    def _validate_and_align(self, tensor: torch.Tensor, name: str = "tensor") -> torch.Tensor:
        if tensor.shape[-1] != self.spike_dim:
            raise ValueError(f"{name} shape mismatch. Expected {self.spike_dim}, got {tensor.shape[-1]}")
        if tensor.device != self.sensor_keys.device:
            tensor = tensor.to(self.sensor_keys.device)
        if torch.all(tensor == 0):
            raise ValueError(f"Zero tensor passed to SCAM for {name} before crystallization.")
        # Enforce one-hot sparsity (Sun et al. 2025)
        if not torch.all((tensor.sum(dim=-1) == 1) | (tensor.sum(dim=-1) == 0)):
            raise ValueError("Sensor keys must be strictly one-hot encoded for FSM stability.")
        return tensor

    def external_retrieve(self, sensor_spikes: torch.Tensor) -> Tuple[torch.Tensor, float, int]:
        sensor_spikes = self._validate_and_align(sensor_spikes, "sensor_spikes")
        if not self.is_active.any():
            return torch.zeros_like(sensor_spikes), 0.0, -1
        similarity = torch.matmul(sensor_spikes.float(), self.sensor_keys.T)
        similarity = similarity.masked_fill(~self.is_active, float("-inf"))
        best_idx = torch.argmax(similarity, dim=-1).item()
        confidence = similarity[0, best_idx].item()
        if confidence > 0.95:
            self.usage_counts[best_idx] += 1
        return self.action_values[best_idx], confidence, best_idx

    @torch.no_grad()
    def crystallize_skill(self, sensor_pattern: torch.Tensor, action_pattern: torch.Tensor):
        sensor_pattern = self._validate_and_align(sensor_pattern, "sensor")
        action_pattern = self._validate_and_align(action_pattern, "action")
        if not self.is_active.all():
            idx = torch.where(~self.is_active)[0][0].item()
        else:
            idx = torch.argmin(self.usage_counts).item()
            if self.usage_counts[idx] <= 1:
                self.is_active[idx] = False
                self.usage_counts[idx] = 0
                idx = torch.where(~self.is_active)[0][0].item()
        self.sensor_keys[idx] = sensor_pattern.float()
        self.action_values[idx] = action_pattern.float()
        self.is_active[idx] = True
        self.usage_counts[idx] = 1

    def inspect_engrams(self) -> Dict[str, any]:
        active_count = self.is_active.sum().item()
        if active_count == 0:
            return {"status": "EMPTY", "active_count": 0}
        return {
            "status": "NOMINAL",
            "active_count": active_count,
            "max_capacity": self.max_engrams,
            "utilization": active_count / self.max_engrams,
            "stale_count": (self.usage_counts[self.is_active] == 1).sum().item()
        }
