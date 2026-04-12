"""
timmy/blocks.py
Processing Blocks: FFN, Output Clamping, Zone Blocks, and Spike Regulation

BIOLOGICAL GROUNDING:
This file assembles the processing blocks that stack into Timmy's three-zone
architecture (sensory, association, executive). Each block performs one round
of synaptic resonance (attention) followed by a feed-forward transformation,
with residual connections, layer normalization, and output clamping.

The three zones model the hierarchical organization of neocortex:
    Sensory zones  (FFN, bidirectional propagation):
        Low-level feature extraction, analogous to primary sensory cortex.
    Association zones (MoE, divergent routing):
        Cross-modal integration, analogous to association cortex.
    Executive zones (FFN, force-nonneg output):
        Decision and action selection, analogous to prefrontal cortex.
        Non-negative output prevents inhibitory signals from propagating
        upstream through the executive pathway.

Key grounding papers:
1. Felleman DJ, Van Essen DC (1991). "Distributed hierarchical processing
   in the primate cerebral cortex." Cerebral Cortex, 1(1):1-47.
   DOI: 10.1093/cercor/1.1.1

2. He K, Zhang X, Ren S, Sun J (2016). "Deep residual learning for image
   recognition." CVPR 2016, pp. 770-778. DOI: 10.1109/CVPR.2016.90
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from typing import Dict, List, Tuple

from timmy_neuron import AssociativeLIF, NeuronConfig
from timmy_attention import SpikingSynapticResonance
from timmy_experts import SpikeDrivenMoE


# =========================================================================
# Spiking Feed-Forward Network
# =========================================================================

class SpikingFeedForward(nn.Module):
    """
    Standard up/down feed-forward network with LIF nonlinearity on both
    projections. Replaces the GELU/ReLU activation in a transformer FFN
    with spiking dynamics.

    NOT a distinct biological structure. Corresponds to the local
    excitatory-excitatory synaptic processing within a cortical column.
    """

    def __init__(self, d_model: int, d_ff: int, neuron_cfg: NeuronConfig = None):
        """
        Args:
            d_model: input/output dimensionality.
            d_ff: hidden dimensionality (expansion factor).
            neuron_cfg: neuron config for the LIF populations.
        """
        super().__init__()
        if neuron_cfg is None:
            neuron_cfg = NeuronConfig()
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        self.lif_hidden = AssociativeLIF(d_ff, neuron_cfg)
        self.lif_output = AssociativeLIF(d_model, neuron_cfg)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (T, B, S, D) spiking tensor.
        Returns:
            (T, B, S, D) processed tensor.
        """
        T, B, S, D = x.shape
        h = self.up(x.reshape(T * B * S, D)).reshape(T, B * S, -1)
        h, _ = self.lif_hidden(h)
        h = self.down(h.reshape(T * B * S, -1)).reshape(T, B * S, D)
        h, _ = self.lif_output(h)
        return h.reshape(T, B, S, D)


# =========================================================================
# Leaky Clamp (Output Normalization)
# =========================================================================

class LeakyClamp(nn.Module):
    """
    Output normalization with learnable floor and leaky negative slope.

    For executive zone blocks, force_nonneg=True replaces this with a
    simple ReLU, preventing negative spike signals from propagating through
    the executive pathway (FIX M). This ensures the executive zone can
    only excite downstream targets, not inhibit them.

    NOT a biological structure. Engineering output normalization.
    """

    def __init__(self, d: int, floor_init: float = -0.1, leak_init: float = 0.1,
                 force_nonneg: bool = False):
        """
        Args:
            d: feature dimension (d_model).
            floor_init: minimum output value for negative inputs.
            leak_init: slope for negative inputs (before sigmoid).
            force_nonneg: if True, clamp to ReLU (executive zone).
        """
        super().__init__()
        self.force_nonneg = force_nonneg
        if force_nonneg:
            floor_init = 0.0
        self.floor = nn.Parameter(torch.full((d,), floor_init))
        self.leak_raw = nn.Parameter(
            torch.full((d,), math.log(leak_init / (1 - leak_init + 1e-6)))
        )

    @property
    def leak(self) -> Tensor:
        """Leak slope in (0, 1), via sigmoid of learnable logit."""
        return torch.sigmoid(self.leak_raw)

    def forward(self, x: Tensor) -> Tensor:
        """Clamp output: ReLU for executive, leaky clamp otherwise."""
        if self.force_nonneg:
            return F.relu(x)
        return torch.where(x >= 0, x, (self.leak * x).clamp(min=self.floor))


# =========================================================================
# Timmy Block (Processing Unit)
# =========================================================================

class TimmyBlock(nn.Module):
    """
    One processing block in Timmy's zone architecture.

    Structure:
        LayerNorm -> SpikingSynapticResonance -> residual (gamma_attn)
        LayerNorm -> FFN or MoE -> residual (gamma_ffn)
        LeakyClamp output

    BIOLOGICAL STRUCTURE: One laminar processing stage in neocortex. Each
    block corresponds to one cortical area in the processing hierarchy.

    BIOLOGICAL FUNCTION: Integrates input via attention (synaptic resonance),
    transforms it via feed-forward processing (local columnar computation),
    and applies output normalization. The zone parameter determines whether
    the block uses MoE routing (association) or plain FFN (sensory/executive)
    and whether the output is force-nonneg (executive only).

    Reference: Felleman DJ, Van Essen DC (1991). "Distributed hierarchical
    processing in the primate cerebral cortex." Cerebral Cortex, 1(1):1-47.
    DOI: 10.1093/cercor/1.1.1

    Residual scaling (gamma_attn, gamma_ffn) initialized to 0.1/n_layers,
    following the principle that residual branches should start small to
    prevent deep networks from diverging during early training.
    Reference: He K et al. (2016). "Deep residual learning." CVPR 2016.
    DOI: 10.1109/CVPR.2016.90

    Gradient checkpointing (FIX H): when enabled, the block's inner
    computation is wrapped in torch.utils.checkpoint to trade compute for
    VRAM. The stats dict is empty when checkpointing is active because
    the inner function must return a single tensor.
    """

    def __init__(self, cfg, layer_idx: int = 0, use_moe: bool = False,
                 zone: str = "sensory", neuron_cfg: NeuronConfig = None):
        """
        Args:
            cfg: model-level config providing d_model, n_layers_total,
                clamp_floor, gradient_checkpointing.
            layer_idx: index of this block in the global stack.
            use_moe: if True, use SpikeDrivenMoE instead of SpikingFeedForward.
            zone: "sensory", "association", or "executive".
            neuron_cfg: neuron config. If None, default used.
        """
        super().__init__()
        if neuron_cfg is None:
            neuron_cfg = NeuronConfig()
        D = cfg.d_model
        self.use_moe = use_moe
        self.zone = zone
        self.layer_idx = layer_idx
        self.use_checkpoint = cfg.gradient_checkpointing

        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)
        self.resonance = SpikingSynapticResonance(cfg, neuron_cfg)

        if use_moe:
            self.moe = SpikeDrivenMoE(cfg, neuron_cfg)
        else:
            self.ffn = SpikingFeedForward(D, cfg.d_ff, neuron_cfg)

        # Residual scaling: small initial contribution from each branch.
        sc = 0.1 / max(cfg.n_layers_total, 1)
        self.gamma_attn = nn.Parameter(torch.full((D,), sc))
        self.gamma_ffn = nn.Parameter(torch.full((D,), sc))

        # Output clamp. Executive zone forces non-negative (FIX M).
        self.clamp = LeakyClamp(
            D, floor_init=cfg.clamp_floor,
            force_nonneg=(zone == "executive"),
        )

    @staticmethod
    def _safe_layernorm(norm: nn.LayerNorm, x: Tensor) -> Tensor:
        """LayerNorm in float32 for numerical stability, cast back to input dtype."""
        od = x.dtype
        return F.layer_norm(
            x.float(),
            norm.normalized_shape,
            norm.weight.float() if norm.weight is not None else None,
            norm.bias.float() if norm.bias is not None else None,
            norm.eps,
        ).to(od)

    def _forward_inner(self, x: Tensor) -> Tuple[Tensor, Dict]:
        """Core block computation (not checkpointed)."""
        stats = {}
        x = x + self.gamma_attn * self.resonance(self._safe_layernorm(self.norm1, x))
        xn = self._safe_layernorm(self.norm2, x)
        if self.use_moe:
            fo, ms = self.moe(xn)
            stats.update(ms)
        else:
            fo = self.ffn(xn)
        return self.clamp(x + self.gamma_ffn * fo), stats

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict]:
        """
        Args:
            x: (T, B, S, D) spiking tensor.
        Returns:
            output: (T, B, S, D) processed tensor.
            stats: dict (empty when gradient checkpointing is active).
        """
        if self.use_checkpoint and self.training:
            x = grad_checkpoint(
                lambda inp: self._forward_inner(inp)[0],
                x, use_reentrant=False,
            )
            return x, {}
        return self._forward_inner(x)


# =========================================================================
# Auxiliary Spike Rate Regulator
# =========================================================================

class AuxiliarySpikeRegulator(nn.Module):
    """
    Asymmetric spike rate loss that penalizes neurons firing too much or
    too little relative to the target rate.

    Under-firing receives 3x the penalty of over-firing (FIX L), because
    a silent network is catastrophically useless while a slightly overactive
    network still transmits information. An additional hard floor penalty
    at rate < 0.01 prevents complete neural death.

    NOT a biological structure. Training regularization loss.
    """

    def __init__(self, target_rate: float = 0.03, weight: float = 0.5):
        """
        Args:
            target_rate: desired mean firing rate across the population.
            weight: coefficient for the spike loss in the total objective.
        """
        super().__init__()
        self.target = target_rate
        self.weight = weight
        self.min_rate = 0.01

    def forward(self, spike_tensors: List[Tensor]) -> Tensor:
        """
        Compute the asymmetric spike rate loss across all layer spike tensors.

        Args:
            spike_tensors: list of (T, B, S, D) or (T, N, D) spike tensors
                from each block in the network.

        Returns:
            Scalar loss (weighted).
        """
        if not spike_tensors:
            return torch.tensor(0.0)
        loss = torch.tensor(0.0, device=spike_tensors[0].device, dtype=torch.float32)
        for s in spike_tensors:
            # FIX K: only count non-negative values as spikes.
            rate = s.float().clamp(min=0).mean()
            diff = self.target - rate
            if diff > 0:
                loss = loss + 3.0 * diff ** 2   # under-firing: 3x penalty
            else:
                loss = loss + diff ** 2          # over-firing: 1x penalty
            if rate < self.min_rate:
                loss = loss + 10.0 * (self.min_rate - rate) ** 2  # death penalty
        return self.weight * loss / len(spike_tensors)
