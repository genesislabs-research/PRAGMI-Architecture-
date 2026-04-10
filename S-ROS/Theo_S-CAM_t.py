"""
Theo_S-CAM_t.py
Spiking Content-Addressable Memory (64K Procedural RAM Edition)

BIOLOGICAL & COMPUTATIONAL GROUNDING:
Upgrades the rigid engram store of the deterministic core into a massive 
64-Kilobyte (65,536 slot) procedural memory space. It hybridizes associative 
memory (content-addressable lookup) with sequential execution (program counter), 
allowing S-ROS to store complex, nested procedural logic natively.

Uses strict one-hot encoding per Sun et al. (2025) for FSM stability, mapping 
direct associative jumps (GOTO/interrupts) and sequential stepping.

Primary grounding papers:
Hopfield JJ (1982). "Neural networks and physical systems with emergent collective computational abilities." PNAS. DOI: 10.1073/pnas.79.8.2554
Sun Z et al. (2025). "General and stable emulation of finite state machines with spiking neural networks." Neuromorphic Computing and Engineering. DOI: 10.1088/2634-4386/adc32d
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict

class TheoSCAM(nn.Module):
    """
    64K Deterministic Program Memory.
    Total Capacity: 65,536 instructions mapped to 128-dim spike vectors.
    """
    def __init__(self, spike_dim: int = 128, max_engrams: int = 65536):
        super().__init__()
        self.spike_dim = spike_dim
        self.max_engrams = max_engrams
        
        # 64K Memory Block
        self.register_buffer("sensor_keys", torch.zeros(max_engrams, spike_dim))
        self.register_buffer("action_values", torch.zeros(max_engrams, spike_dim))
        
        # State Tracking
        self.register_buffer("is_active", torch.zeros(max_engrams, dtype=torch.bool))
        self.register_buffer("usage_counts", torch.zeros(max_engrams, dtype=torch.long))
        
        # Sequential Execution Pointer
        self.register_buffer("program_counter", torch.tensor(0, dtype=torch.long))

    def _validate_and_align(self, incoming_spikes: torch.Tensor, label: str) -> torch.Tensor:
        if incoming_spikes.shape[-1] != self.spike_dim:
            raise ValueError(f"[Theo 64K RAM] {label} shape mismatch. Expected {self.spike_dim}, got {incoming_spikes.shape[-1]}")
        if incoming_spikes.device != self.sensor_keys.device:
            incoming_spikes = incoming_spikes.to(self.sensor_keys.device)
        return incoming_spikes

    def retrieve_engram(self, sensor_spikes: torch.Tensor) -> Tuple[torch.Tensor, float, int]:
        """
        Associative lookup for context shifts, interrupts, or targeted jumps.
        Finds the exact memory address matching the sensor state and updates the PC.
        """
        sensor_spikes = self._validate_and_align(sensor_spikes, "sensor")
        
        if not self.is_active.any():
            return torch.zeros_like(sensor_spikes), 0.0, -1

        similarity = torch.matmul(sensor_spikes.float(), self.sensor_keys.T)
        similarity.masked_fill_(~self.is_active, float("-inf"))
        
        best_match_idx = torch.argmax(similarity, dim=-1)
        confidence = similarity[0, best_match_idx].item()
        
        if confidence > 0.95:
            self.usage_counts[best_match_idx] += 1
            # Snap the Program Counter to the new execution address
            self.program_counter.copy_(best_match_idx)
            
        return self.action_values[best_match_idx], confidence, best_match_idx.item()

    def step_execution(self) -> torch.Tensor:
        """
        Sequential execution. Steps to the next instruction in the 64K RAM block,
        returning the action spike for the next cycle.
        """
        next_address = (self.program_counter + 1) % self.max_engrams
        if self.is_active[next_address]:
            self.program_counter.copy_(next_address)
            return self.action_values[next_address]
        return torch.zeros(self.spike_dim, device=self.action_values.device)

    @torch.no_grad()
    def crystallize_skill(self, sensor_pattern: torch.Tensor, action_pattern: torch.Tensor, address: int = None):
        """
        Direct memory writing. Can act as a sequential append or a targeted POKE.
        """
        sensor_pattern = self._validate_and_align(sensor_pattern, "sensor")
        action_pattern = self._validate_and_align(action_pattern, "action")
        
        if address is None:
            # Find first empty slot, or least used if full
            if not self.is_active.all():
                idx = torch.where(~self.is_active)[0][0].item()
            else:
                idx = torch.argmin(self.usage_counts).item()
        else:
            if address >= self.max_engrams or address < 0:
                raise IndexError(f"Address {address} out of bounds for 64K RAM.")
            idx = address

        self.sensor_keys[idx] = torch.sign(sensor_pattern).float()
        self.action_values[idx] = torch.sign(action_pattern).float()
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
            "program_counter": self.program_counter.item()
        }

