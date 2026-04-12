"""
timmy/memory.py
Working Memory Cortex with Multi-Head Temporal Attention Readout

BIOLOGICAL GROUNDING:
This file implements Timmy's short-term working memory, modeled on the
sustained firing activity observed in prefrontal cortex (PFC) during
delay periods. PFC neurons can maintain task-relevant information across
seconds via persistent activity, likely supported by recurrent excitation
and intrinsic membrane properties with slow time constants.

The MemoryCortex uses AssociativeLIF neurons with tau_mem=0.99 (effectively
a 100ms time constant), much slower than the default tau_mem=0.85 in the
processing layers. These slow neurons act as a leaky buffer: they accumulate
spike-driven input across the processing window and retain it, providing a
within-context memory that downstream layers can read from via multi-head
temporal attention.

This is NOT the kernel's episodic memory. The MemoryCortex is a fast,
within-context scratchpad that Timmy uses during a single forward pass.
The kernel's CA3 stores episodes that persist across context windows.

Key grounding papers:
1. Fuster JM (1973). "Unit activity in prefrontal cortex during delayed-
   response performance: neuronal correlates of transient memory." Journal
   of Neurophysiology, 36(1):61-78. DOI: 10.1152/jn.1973.36.1.61

2. Wang XJ (2001). "Synaptic reverberation underlying mnemonic persistent
   activity." Trends in Neurosciences, 24(8):455-463.
   DOI: 10.1016/S0166-2236(00)01868-3

3. Goldman-Rakic PS (1995). "Cellular basis of working memory." Neuron,
   14(3):477-485. DOI: 10.1016/0896-6273(95)90304-6
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Tuple

from timmy_neuron import AssociativeLIF, NeuronConfig


class MemoryCortex(nn.Module):
    """
    Short-term working memory with slow-decaying LIF neurons and
    multi-head temporal attention readout.

    BIOLOGICAL STRUCTURE: Prefrontal cortex delay-period neurons with
    slow membrane time constants enabling persistent activity.

    BIOLOGICAL FUNCTION: Maintains task-relevant information across the
    processing window via sustained firing. The slow tau_mem (0.99) means
    the membrane loses only 1% of its charge per timestep, allowing
    information from early timesteps to influence late timestep processing.

    Reference: Fuster JM (1973). DOI: 10.1152/jn.1973.36.1.61
    Reference: Wang XJ (2001). DOI: 10.1016/S0166-2236(00)01868-3

    COMPUTATIONAL IMPLEMENTATION:
        1. Project input from d_model to memory_size via to_memory.
        2. Run through slow LIF (tau_mem=0.99, persistent=True).
        3. Spike-gated write: a separate LIF determines whether the
           memory content should be "let through" or suppressed.
        4. Multi-head temporal attention: learned query vectors attend
           across all T timesteps of the membrane trace, producing a
           temporally-weighted readout.
        5. Project back to d_model and mix with the residual stream.
    """

    def __init__(self, cfg, neuron_cfg: NeuronConfig = None):
        """
        Args:
            cfg: model-level config providing d_model, memory_size,
                memory_tau_mem, memory_gate_threshold, memory_n_read_heads.
            neuron_cfg: neuron config for the gate LIF. If None, default used.
        """
        super().__init__()
        if neuron_cfg is None:
            neuron_cfg = NeuronConfig()
        self.cfg = cfg
        D = cfg.d_model
        M = cfg.memory_size

        # Input/output projections between d_model and memory space.
        self.to_memory = nn.Linear(D, M, bias=False)
        self.from_memory = nn.Linear(M, D, bias=False)

        # Slow-decaying LIF for persistent memory trace (tau_mem=0.99).
        # Persistent=True: state carries between forward() calls.
        # Reference: Wang XJ (2001). DOI: 10.1016/S0166-2236(00)01868-3
        self.memory_lif = AssociativeLIF(
            M, neuron_cfg, persistent=True, tau_mem_override=cfg.memory_tau_mem
        )

        # Gate LIF + projection: determines what gets through to readout.
        # Non-persistent: gate resets each call (gating is per-window, not
        # persistent across windows).
        self.gate_lif = AssociativeLIF(M, neuron_cfg)
        self.gate_proj = nn.Linear(D, M, bias=False)
        self.gate_threshold = nn.Parameter(torch.tensor(cfg.memory_gate_threshold))

        # Multi-head temporal attention readout.
        # Learned query vectors attend across T timesteps of the memory
        # trace, producing a temporally-weighted readout rather than a
        # naive mean. This allows the model to "focus" on specific moments
        # in the processing window.
        # Reference: Goldman-Rakic PS (1995). DOI: 10.1016/0896-6273(95)90304-6
        H = cfg.memory_n_read_heads
        hd = M // H
        self.n_read_heads = H
        self.read_query = nn.Parameter(torch.randn(H, hd) * 0.02)
        self.read_key_proj = nn.Linear(M, M, bias=False)
        self.read_scale = 1.0 / math.sqrt(hd)

        # Output normalization and residual mixing.
        self.mem_norm = nn.LayerNorm(D)
        self.memory_mix = nn.Parameter(torch.tensor(0.1))

    def reset_state(self):
        """Reset the persistent memory LIF state."""
        self.memory_lif.reset_state()

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        """
        Process input through the working memory cortex.

        Args:
            x: (T, B, S, D) spiking tensor from previous block.

        Returns:
            x_enhanced: (T, B, S, D) input enhanced with memory readout.
            stats: dict with memory activity diagnostics.
        """
        T, B, S, D = x.shape
        M = self.cfg.memory_size
        N = B * S
        H, hd = self.n_read_heads, M // self.n_read_heads
        xf = x.reshape(T, N, D)

        # Write to memory: project to memory space, run through slow LIF.
        mi = self.to_memory(xf.reshape(T * N, D)).reshape(T, N, M)
        ms, mv = self.memory_lif(mi)  # ms: spikes, mv: membrane trace

        # Gate: separate LIF determines what passes through.
        gi = self.gate_proj(xf.reshape(T * N, D)).reshape(T, N, M)
        gs, _ = self.gate_lif(gi)
        gate_sig = gs.mean(dim=0)  # (N, M) mean gate activity
        gate_mask = torch.sigmoid((gate_sig - self.gate_threshold) * 10.0)

        # Multi-head temporal attention readout.
        # Keys come from the membrane trace (not spikes) because the membrane
        # retains graded information that spikes binarize away.
        mvh = mv.reshape(T, N, H, hd)
        mk = self.read_key_proj(mv.reshape(T * N, M)).reshape(T, N, H, hd)
        q = self.read_query.unsqueeze(0).unsqueeze(0)  # (1, 1, H, hd)
        attn_s = (q * mk).sum(-1) * self.read_scale     # (T, N, H)
        attn_w = F.softmax(attn_s.float(), dim=0).to(mv.dtype)

        # Weighted readout across timesteps, then apply gate.
        mem_read = (mvh * attn_w.unsqueeze(-1)).sum(0).reshape(N, M)
        mem_read = mem_read * gate_mask

        # Project back to d_model, normalize, mix with residual.
        mem_out = self.mem_norm(self.from_memory(mem_read).float()).to(x.dtype)
        mix = torch.sigmoid(self.memory_mix)
        x_enhanced = x + mix * mem_out.reshape(1, B, S, D).expand_as(x)

        stats = {
            "memory_spike_rate": ms.mean().item(),
            "gate_activity": gate_sig.mean().item(),
            "memory_mix": mix.item(),
            "memory_attn_entropy": -(
                attn_w.float() * (attn_w.float() + 1e-8).log()
            ).sum(0).mean().item(),
        }
        return x_enhanced, stats
