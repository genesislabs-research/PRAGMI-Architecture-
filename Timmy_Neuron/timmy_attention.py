"""
timmy/attention.py
Spike-Driven Synaptic Resonance with Rotary Position Encoding

BIOLOGICAL GROUNDING:
This file implements Timmy's attention mechanism, modeled as synaptic resonance
between spiking populations. In biological cortex, attention emerges from the
synchronized oscillatory coupling between neural populations: when two
populations resonate at the same frequency, information transfer between them
is enhanced. This is the communication-through-coherence hypothesis.

The mechanism here uses spiking query and key populations that filter the input
through LIF dynamics before computing attention scores. The temporal mixing
weights aggregate spike information across T=10 timesteps into a single
attention score per token pair, solving the problem of how to extract a
meaningful scalar similarity from two spike trains.

Top-k sparse attention limits each token to attending over at most K=64 other
tokens. This serves a dual purpose: computational efficiency (avoiding the
O(S^2) full attention matrix) and biological plausibility (cortical neurons
have limited connectivity, not all-to-all).

Key grounding papers:
1. Fries P (2005). "A mechanism for cognitive dynamics: neuronal communication
   through neuronal coherence." Trends in Cognitive Sciences, 9(10):474-480.
   DOI: 10.1016/j.tics.2005.08.011

2. Su J, Lu Y, Pan S, Murtadha A, Wen B, Liu Y (2024). "RoFormer: Enhanced
   transformer with rotary position embedding." Neurocomputing, 568:127063.
   DOI: 10.1016/j.neucom.2023.127063

3. Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez AN, Kaiser L,
   Polosukhin I (2017). "Attention is all you need." NeurIPS 2017.
   DOI: 10.48550/arXiv.1706.03762
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

from timmy_neuron import AssociativeLIF, NeuronConfig


# =========================================================================
# Rotary Position Embedding
# =========================================================================

class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for encoding sequence position into
    attention scores via rotation of query/key vectors in complex plane.

    NOT a biological structure. Engineering technique for injecting position
    information into attention without additive position embeddings.

    Reference: Su J et al. (2024). "RoFormer: Enhanced transformer with
    rotary position embedding." Neurocomputing, 568:127063.
    DOI: 10.1016/j.neucom.2023.127063
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        """
        Args:
            dim: head dimension (d_model // n_heads). RoPE rotates pairs of
                dimensions, so this must be even.
            max_seq_len: maximum sequence length for cached cos/sin tables.
            theta: frequency base for the inverse frequency schedule.
                Higher theta -> slower frequency decay -> better long-range
                position encoding. Default 10000.0 from Vaswani et al. (2017).
        """
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())

    def forward(self, x: Tensor, seq_len: int) -> Tuple[Tensor, Tensor]:
        """Return cached cos/sin tables truncated to seq_len."""
        return (
            self.cos_cached[:seq_len].to(x.dtype),
            self.sin_cached[:seq_len].to(x.dtype),
        )


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """
    Apply rotary position embedding to a tensor.

    Rotates consecutive pairs of dimensions by position-dependent angles.
    If x has more dimensions than 2*d (where d = cos.shape[-1]), the
    excess dimensions are passed through unchanged.

    Args:
        x: (..., D) tensor to rotate.
        cos: (S, d) cosine table from RotaryPositionEmbedding.
        sin: (S, d) sine table.

    Returns:
        (..., D) rotated tensor.
    """
    d = cos.shape[-1]
    x1, x2 = x[..., :d], x[..., d:2 * d]
    c = cos.unsqueeze(0).unsqueeze(0)
    s = sin.unsqueeze(0).unsqueeze(0)
    rot = torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
    if x.shape[-1] > 2 * d:
        return torch.cat([rot, x[..., 2 * d:]], dim=-1)
    return rot


# =========================================================================
# Spiking Synaptic Resonance (Spike-Driven Attention)
# =========================================================================

class SpikingSynapticResonance(nn.Module):
    """
    Spike-driven attention via synchronized oscillatory coupling between
    query and key populations.

    BIOLOGICAL STRUCTURE: Cortico-cortical communication mediated by
    oscillatory coherence. When two neural populations fire in synchrony
    (phase-locked), their mutual information transfer is enhanced. This
    selective coupling allows flexible routing of information between
    cortical areas.

    BIOLOGICAL FUNCTION: Computes a similarity score between token positions
    by passing queries and keys through LIF neuron populations, then
    aggregating their spike patterns across timesteps via learned temporal
    mixing weights. The resulting attention scores determine how much each
    position contributes to the representation of every other position.

    Reference: Fries P (2005). "A mechanism for cognitive dynamics: neuronal
    communication through neuronal coherence." Trends in Cognitive Sciences,
    9(10):474-480. DOI: 10.1016/j.tics.2005.08.011

    COMPUTATIONAL IMPLEMENTATION:
    The attention computation proceeds as:

        1. Project input spikes to Q, K, V via linear maps.
        2. Pass Q and K through dedicated LIF populations (spiking Q/K).
        3. Aggregate spiking Q/K across T timesteps via learned temporal
           mixing weights (FIX E: not naive mean or last-timestep).
        4. Apply RoPE to the temporally-mixed Q and K.
        5. Compute scaled dot-product attention with causal mask.
        6. Apply top-k sparsification (keep only K strongest connections).
        7. Softmax -> weighted sum of V -> output projection.

    The temporal mixing weights (FIX E) solve a fundamental problem: how to
    extract a meaningful scalar attention score from two multi-timestep spike
    trains. Naive approaches (mean across time, or using only the last
    timestep) discard temporal structure. Learned mixing weights allow the
    model to discover which timesteps carry the most information about
    inter-token relationships.

    Reference: Vaswani A et al. (2017). "Attention is all you need."
    NeurIPS 2017. DOI: 10.48550/arXiv.1706.03762
    """

    def __init__(self, cfg, neuron_cfg: NeuronConfig = None):
        """
        Args:
            cfg: model-level config providing d_model, n_heads, T_total,
                resonance_top_k, max_seq_len, rope_theta.
            neuron_cfg: neuron config for the Q/K LIF populations. If None,
                a default NeuronConfig is used.
        """
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.top_k = cfg.resonance_top_k
        D = cfg.d_model
        T_t = cfg.T_total

        if neuron_cfg is None:
            neuron_cfg = NeuronConfig()

        # Query, Key, Value projections (standard multi-head attention).
        self.W_q = nn.Linear(D, D, bias=False)
        self.W_k = nn.Linear(D, D, bias=False)
        self.W_v = nn.Linear(D, D, bias=False)
        self.W_o = nn.Linear(D, D, bias=False)

        # Dedicated LIF populations for Q and K. These convert the projected
        # continuous vectors into spike trains before temporal mixing. The
        # spike trains capture which dimensions are "active" at each timestep,
        # providing a rate-coded representation that the temporal mixer
        # aggregates into per-token attention features.
        self.lif_q = AssociativeLIF(D, neuron_cfg)
        self.lif_k = AssociativeLIF(D, neuron_cfg)

        # Learnable temperature for the resonance (attention) scores.
        # Initialized to 1/sqrt(d_head) following standard scaled dot-product.
        # Reference: Vaswani et al. (2017). DOI: 10.48550/arXiv.1706.03762
        self.resonance_temp = nn.Parameter(
            torch.tensor(1.0 / math.sqrt(self.d_head))
        )

        # Temporal mixing weights (FIX E). Per-timestep weights for Q and K
        # that are softmax-normalized before application. Initialized uniform
        # so all timesteps contribute equally at the start of training.
        # NOT a biological quantity. Engineering solution to the spike train
        # aggregation problem.
        self.temporal_mix_q = nn.Parameter(torch.ones(T_t) / T_t)
        self.temporal_mix_k = nn.Parameter(torch.ones(T_t) / T_t)

        # Rotary position embedding for the per-head dimension.
        self.rope = RotaryPositionEmbedding(
            self.d_head, cfg.max_seq_len, cfg.rope_theta
        )

    def forward(self, x_spikes: Tensor) -> Tensor:
        """
        Compute spike-driven attention over a spiking input tensor.

        Args:
            x_spikes: (T_total, B, S, D) spiking tensor from the previous
                block or the input LIF.

        Returns:
            (T_total, B, S, D) attended output, broadcast across timesteps.
        """
        T_t, B, S, D = x_spikes.shape
        H, Dh = self.n_heads, self.d_head

        # Project to Q, K, V.
        xf = x_spikes.reshape(T_t * B * S, D)
        qc = self.W_q(xf).reshape(T_t, B * S, D)
        kc = self.W_k(xf).reshape(T_t, B * S, D)
        vr = self.W_v(xf).reshape(T_t, B, S, D)

        # Pass Q and K through dedicated LIF populations.
        # The spike trains encode which Q/K dimensions are "active" per timestep.
        qs, _ = self.lif_q(qc)
        ks, _ = self.lif_k(kc)

        # Reshape for multi-head: (T, B, S, H, Dh).
        qs = qs.reshape(T_t, B, S, H, Dh)
        ks = ks.reshape(T_t, B, S, H, Dh)

        # Temporal mixing (FIX E): weighted sum across timesteps.
        # Softmax ensures the weights are a proper probability distribution
        # over timesteps.
        twq = F.softmax(self.temporal_mix_q, dim=0).reshape(T_t, 1, 1, 1, 1)
        twk = F.softmax(self.temporal_mix_k, dim=0).reshape(T_t, 1, 1, 1, 1)
        qm = (qs * twq).sum(0).permute(0, 2, 1, 3)  # (B, H, S, Dh)
        km = (ks * twk).sum(0).permute(0, 2, 1, 3)   # (B, H, S, Dh)

        # Apply rotary position embedding.
        cos, sin = self.rope(qm, S)
        qm = apply_rope(qm, cos, sin)
        km = apply_rope(km, cos, sin)

        # Scaled dot-product resonance scores.
        res = torch.matmul(qm, km.transpose(-2, -1)) * self.resonance_temp

        # Causal mask: prevent attention to future positions.
        cmask = torch.triu(
            torch.ones(S, S, device=x_spikes.device, dtype=torch.bool),
            diagonal=1,
        )
        res.masked_fill_(cmask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Top-k sparsification: each query attends to at most K keys.
        # NOT a biological quantity. Sparse connectivity approximation.
        K = min(self.top_k, S)
        if K < S:
            tv, ti = torch.topk(res, K, dim=-1)
            sr = torch.full_like(res, float("-inf"))
            sr.scatter_(-1, ti, tv)
            res = sr

        # Softmax attention weights.
        attn = F.softmax(res.float(), dim=-1).to(res.dtype)

        # Value aggregation: mean across timesteps, then standard attention.
        vm = vr.mean(dim=0).reshape(B, S, H, Dh).permute(0, 2, 1, 3)
        ctx = torch.matmul(attn, vm).permute(0, 2, 1, 3).reshape(B, S, D)

        # Output projection, broadcast across all timesteps.
        return self.W_o(ctx).unsqueeze(0).expand(T_t, -1, -1, -1)
