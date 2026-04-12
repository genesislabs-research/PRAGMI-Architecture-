"""
timmy/encoder.py
Multi-Scale Temporal Spike Encoder with External Embedding Injection

BIOLOGICAL GROUNDING:
This file implements the translation layer between discrete token space and
Timmy's spiking dynamics. In biological terms, this is the thalamocortical
relay: thalamic neurons receive compressed sensory input and convert it into
temporal patterns of firing that drive cortical populations. The thalamus does
not simply relay input unchanged; it imposes temporal structure through burst
and tonic firing modes that encode salience and timing information.

The encoder produces a multi-scale temporal current drive from each token
embedding. A fast basis (T=8 timesteps) captures fine temporal structure
within a single processing window, analogous to gamma-band oscillatory
modulation. A slow basis (T_slow=2 timesteps) captures sustained activation
that persists across the processing window, analogous to theta-band envelope
modulation. The concatenation of fast and slow bases produces a T_total=10
dimensional temporal code per token per neuron.

The external float embedding path (MEM 3) allows continuous vectors from
outside the token vocabulary to be injected into the encoder. This is the
interface through which the Cognitive Kernel returns reconstructed episodic
coordinates to Timmy, and through which external LLM embeddings (e.g. from
a frozen teacher model) or sensor streams can enter the spiking pipeline.
The injection uses a learnable gate initialized near-zero (sigmoid(-4)=0.018)
so the model starts token-only and gradually opens the gate during training.

Key grounding papers:
1. Sherman SM, Guillery RW (2002). "The role of the thalamus in the flow of
   information to the cortex." Philosophical Transactions of the Royal
   Society B, 357(1428):1695-1708. DOI: 10.1098/rstb.2002.1161

2. Buzsaki G (2006). "Rhythms of the Brain." Oxford University Press.
   DOI: 10.1093/acprof:oso/9780195301069.001.0001
   (Theta-gamma coupling, multi-scale temporal coding)

3. Llinás R, Jahnsen H (1982). "Electrophysiology of mammalian thalamic
   neurones in vitro." Nature, 297(5865):406-408. DOI: 10.1038/297406a0
   (Thalamic burst/tonic firing modes)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass
from typing import Optional


# =========================================================================
# Configuration
# =========================================================================

@dataclass
class EncoderConfig:
    """
    Configuration for the TemporalSpikeEncoder.
    Separated from the full TimmyConfig so the encoder can be tested standalone.
    """

    # Vocabulary and embedding dimensions.
    tokenizer_id: str = "meta-llama/Llama-3.2-1B"
    vocab_size: int = 128_256
    d_model: int = 496

    # Fast temporal basis: T timesteps of gamma-band modulation.
    # NOT a biological quantity. Engineering choice matching the spiking
    # window depth in AssociativeLIF.
    T: int = 8

    # Slow temporal basis: T_slow timesteps of theta-band envelope.
    # NOT a biological quantity. Adds sustained activation to complement
    # the fast transient. Together, T + T_slow = T_total = 10 timesteps.
    # Reference: Buzsaki G (2006). "Rhythms of the Brain." Oxford University
    # Press. DOI: 10.1093/acprof:oso/9780195301069.001.0001
    # (Theta-gamma coupling as multi-scale temporal code)
    T_slow: int = 2

    # External float embedding dimension. 0 = disabled.
    # When > 0, the encoder accepts optional float_embeds and blends them
    # with token embeddings via a learnable gate.
    # Set to d_model for same-space injection (e.g. PRAGMI episodic context).
    # Set to a different value for external LLM embeddings.
    float_embed_dim: int = 0

    @property
    def T_total(self) -> int:
        """Total temporal depth: fast + slow basis."""
        return self.T + self.T_slow


# =========================================================================
# Temporal Spike Encoder
# =========================================================================

class TemporalSpikeEncoder(nn.Module):
    """
    Multi-scale temporal encoder that converts token embeddings (and optional
    external continuous embeddings) into a temporal current drive for
    downstream spiking populations.

    BIOLOGICAL STRUCTURE: Thalamocortical relay neurons. The thalamus receives
    compressed multimodal input and converts it into temporal firing patterns
    that drive cortical populations via thalamocortical projections.

    BIOLOGICAL FUNCTION: The thalamus imposes temporal structure on sensory
    input through burst and tonic firing modes. Burst mode (short, high-
    frequency spike packets) signals novel or salient input. Tonic mode
    (regular, lower-frequency firing) signals sustained familiar input.

    Reference: Sherman SM, Guillery RW (2002). "The role of the thalamus in
    the flow of information to the cortex." Philosophical Transactions of
    the Royal Society B, 357(1428):1695-1708.
    DOI: 10.1098/rstb.2002.1161

    Reference: Llinás R, Jahnsen H (1982). "Electrophysiology of mammalian
    thalamic neurones in vitro." Nature, 297(5865):406-408.
    DOI: 10.1038/297406a0

    COMPUTATIONAL IMPLEMENTATION:
    The encoder produces current of shape (T_total, B*S, D) from tokens (B, S):

        1. Token embedding: token_ids -> embed -> temporal_proj -> x (B*S, D)
        2. Optional float blend: x = (1-gate)*x + gate*float_proj(float_embeds)
        3. Fast basis: sigmoid(fast_basis) * x * drive_scale   (T, B*S, D)
        4. Slow basis: sigmoid(slow_basis) * x * slow_scale    (T_slow, B*S, D)
        5. Concatenate: (T + T_slow, B*S, D)

    The sigmoid on each basis vector produces a per-timestep, per-neuron
    gating pattern in [0, 1] that modulates the shared embedding x. This is
    analogous to how thalamic relay neurons modulate cortical drive through
    temporally structured burst/tonic switching.

    EXTERNAL EMBEDDING INJECTION (MEM 3):
    When float_embed_dim > 0, the encoder accepts an optional float_embeds
    tensor that is projected to d_model space and blended with token
    embeddings via a learnable gate. The gate is initialized at sigmoid(-4)
    = 0.018 so the model starts near token-only and can open the gate
    gradually during training. This prevents the external signal from
    overwhelming the token signal before the model has learned to use it.

    Use cases for float_embeds:
        - Episodic context reconstructed by the Cognitive Kernel
        - External LLM embeddings from a frozen teacher model
        - Continuous sensor streams (audio features, visual features)
        - Robby's real-time experience vectors
    """

    def __init__(self, cfg: EncoderConfig):
        """
        Args:
            cfg: encoder configuration with vocab_size, d_model, T, T_slow,
                and float_embed_dim.
        """
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model

        # Token embedding table. Kaiming uniform initialization following
        # PyTorch nn.Embedding default (which uses the same init internally).
        self.embed = nn.Embedding(cfg.vocab_size, D)
        nn.init.kaiming_uniform_(self.embed.weight, a=math.sqrt(5))

        # Temporal projection: learned linear map applied to embeddings before
        # temporal basis expansion. Allows the model to learn a
        # representation optimized for temporal coding rather than using raw
        # embedding vectors directly.
        # NOT a biological quantity. Engineering projection.
        self.temporal_proj = nn.Linear(D, D, bias=False)

        # Drive scale for the fast basis. Controls the overall amplitude of
        # the current injected into the first LIF layer. Initialized to 25.0,
        # which was tuned during the 13K-step training run to produce firing
        # rates in the target range (0.03-0.05) at the input LIF.
        # NOT a biological quantity. Training artifact.
        self.drive_scale = nn.Parameter(torch.tensor(25.0))

        # Fast temporal basis: (T, D) learnable gating pattern.
        # Each row is a per-neuron modulation weight for one timestep.
        # After sigmoid, values in [0, 1] gate the projected embedding.
        # T=8 timesteps: analogous to gamma-band (30-100 Hz) modulation
        # within one theta cycle.
        # Reference: Buzsaki G (2006). DOI: 10.1093/acprof:oso/9780195301069.001.0001
        self.fast_basis = nn.Parameter(torch.randn(cfg.T, D) * 0.02)

        # Slow temporal basis: (T_slow, D) learnable gating pattern.
        # T_slow=2 timesteps: analogous to theta-band (4-8 Hz) envelope
        # that provides sustained drive across the processing window.
        # Reference: Buzsaki G (2006). DOI: 10.1093/acprof:oso/9780195301069.001.0001
        self.slow_basis = nn.Parameter(torch.randn(cfg.T_slow, D) * 0.02)

        # Drive scale for the slow basis. Lower than fast_scale because the
        # slow basis provides background sustained activation, not phasic drive.
        # NOT a biological quantity. Training artifact.
        self.slow_scale = nn.Parameter(torch.tensor(8.0))

        # External float embedding path (MEM 3).
        if cfg.float_embed_dim > 0:
            # Projection from external embedding space to d_model.
            self.float_proj = nn.Linear(cfg.float_embed_dim, D, bias=False)

            # Learnable gate in logit space. sigmoid(-4) = 0.018, so the
            # model starts with ~98% token signal and ~2% external signal.
            # The gate opens gradually during training as the model learns
            # to use the external input.
            # NOT a biological quantity. Training artifact for stable
            # curriculum learning (token-first, then blended).
            self.float_gate_raw = nn.Parameter(torch.tensor(-4.0))

    @property
    def float_gate(self) -> Tensor:
        """Current gate value for external embeddings, in [0, 1]."""
        return torch.sigmoid(self.float_gate_raw)

    def forward(
        self,
        token_ids: Tensor,
        float_embeds: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Encode tokens (and optional external embeddings) into a multi-scale
        temporal current drive.

        INTERFACE BOUNDARY:
            SENDING:   Token vocabulary (discrete) + optional continuous vectors
            RECEIVING: Input LIF population (first spiking layer in Timmy)
            CONNECTION: Thalamocortical projection (thalamus -> cortical layer 4)

        Reference: Sherman SM, Guillery RW (2002).
        DOI: 10.1098/rstb.2002.1161

        Args:
            token_ids: (B, S) integer token indices.
            float_embeds: optional (B, S, float_embed_dim) external continuous
                embeddings. Requires cfg.float_embed_dim > 0.

        Returns:
            (T_total, B*S, D) temporal current drive. First T timesteps are the
            fast (gamma) basis; last T_slow timesteps are the slow (theta)
            basis. Fed directly into the input AssociativeLIF population.
        """
        B, S = token_ids.shape
        D = self.cfg.d_model

        # Token embedding -> temporal projection.
        x = self.temporal_proj(self.embed(token_ids)).reshape(B * S, D)

        # External float embedding injection (MEM 3).
        # Gate blends token and external signals: x = (1-g)*x_token + g*x_external.
        if float_embeds is not None and self.cfg.float_embed_dim > 0:
            fe = self.float_proj(float_embeds.reshape(B * S, self.cfg.float_embed_dim))
            gate = self.float_gate
            x = (1.0 - gate) * x + gate * fe

        # Fast basis: per-timestep sigmoid gating * shared embedding * drive scale.
        # Shape: (T, 1, D) * (1, B*S, D) * scalar -> (T, B*S, D)
        fast = torch.sigmoid(self.fast_basis).unsqueeze(1) * x.unsqueeze(0) * self.drive_scale

        # Slow basis: same structure, lower scale, fewer timesteps.
        slow = torch.sigmoid(self.slow_basis).unsqueeze(1) * x.unsqueeze(0) * self.slow_scale

        # Concatenate: fast gamma modulation followed by slow theta envelope.
        return torch.cat([fast, slow], dim=0)  # (T_total, B*S, D)

    def get_diagnostics(self) -> dict:
        """Encoder health report."""
        diag = {
            "drive_scale": self.drive_scale.item(),
            "slow_scale": self.slow_scale.item(),
            "fast_basis_mean": torch.sigmoid(self.fast_basis).mean().item(),
            "slow_basis_mean": torch.sigmoid(self.slow_basis).mean().item(),
        }
        if self.cfg.float_embed_dim > 0:
            diag["float_gate"] = self.float_gate.item()
        return diag
