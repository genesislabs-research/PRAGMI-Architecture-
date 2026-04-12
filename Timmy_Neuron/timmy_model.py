"""
timmy/model.py
Timmy: The Spiking Neural Network Bridge Language Model

BIOLOGICAL GROUNDING:
This file assembles the complete spiking language model that serves as the
"subconscious" bridge between the external LLM and the Cognitive Kernel.
Timmy is not end-user accessible. It sits between the token world (where
the LLM operates) and the coordinate world (where the kernel stores and
reconstructs episodes).

The architecture follows the hierarchical organization of mammalian neocortex:

    Token input -> TemporalSpikeEncoder (thalamocortical relay)
        -> Input LIF (layer 4 stellate cells)
        -> Sensory blocks (primary sensory cortex, 2 x FFN blocks)
        -> Association blocks (association cortex, 2 x MoE blocks)
        -> MemoryCortex (PFC working memory)
        -> Executive blocks (prefrontal cortex, 2 x FFN blocks, force-nonneg)
        -> EpisodicMemoryLayer (hippocampal interface, PRAGMI kernel link)
        -> Readout LIF (motor/output cortex)
        -> EMA smoothing -> LM head -> logits

The sensory zone extracts low-level features. The association zone routes
information through specialized expert subpopulations. The memory cortex
maintains short-term context via slow-decaying LIF neurons. The executive
zone produces the final decision signal with non-negative output to prevent
inhibitory feedback. The episodic memory layer (from the PRAGMI kernel)
provides long-term context that persists across context windows.

Key grounding papers:
1. Felleman DJ, Van Essen DC (1991). "Distributed hierarchical processing
   in the primate cerebral cortex." Cerebral Cortex, 1(1):1-47.
   DOI: 10.1093/cercor/1.1.1

2. Neftci EO, Mostafa H, Zenke F (2019). "Surrogate gradient learning in
   spiking neural networks." IEEE Signal Processing Magazine, 36(6):51-63.
   DOI: 10.1109/MSP.2019.2931595

3. Gerstner W, Kistler WM, Naud R, Paninski L (2014). "Neuronal Dynamics."
   Cambridge University Press. DOI: 10.1017/CBO9781107447615
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from timmy_neuron import AssociativeLIF, NeuronConfig
from timmy_encoder import TemporalSpikeEncoder, EncoderConfig
from timmy_attention import SpikingSynapticResonance
from timmy_experts import SpikeDrivenMoE
from timmy_memory import MemoryCortex
from timmy_blocks import TimmyBlock, AuxiliarySpikeRegulator
from timmy_plasticity import STDPEngine
from timmy_state import (
    save_timmy_state,
    load_timmy_state,
    diff_timmy_states,
    compute_architecture_hash,
    ArchitectureMismatchError,
)


# =========================================================================
# Configuration
# =========================================================================

@dataclass
class TimmyConfig:
    """
    Complete configuration for the Timmy spiking language model.

    Every parameter that has a biological analog cites its source. Parameters
    that are training artifacts or engineering approximations are labeled.
    Parameters inherited from the original architecture carry their fix tags
    (FIX A through FIX M, MEM 1 through MEM 5) for traceability.
    """

    # ---- Tokenizer and embedding ----

    tokenizer_id: str = "meta-llama/Llama-3.2-1B"
    vocab_size: int = 128_256
    d_model: int = 496
    n_heads: int = 8
    n_layers: int = 6  # total across all zones
    d_ff: int = 1024
    max_seq_len: int = 512

    # ---- Temporal processing ----

    # Fast basis: T timesteps of gamma-band modulation.
    # NOT a biological quantity. Engineering choice matching spiking window.
    # Reference: Buzsaki G (2006). "Rhythms of the Brain." Oxford.
    # DOI: 10.1093/acprof:oso/9780195301069.001.0001
    T: int = 8

    # Slow basis: theta-band envelope.
    T_slow: int = 2

    # Persistent membrane state between forward calls.
    persistent_mem: bool = True

    # ---- LIF neuron parameters (FIX D) ----

    # See timmy/neuron.py NeuronConfig for full citations on each parameter.
    # Values here are the model-level defaults that get passed to NeuronConfig.
    tau_mem: float = 0.85
    tau_mem_min: float = 0.8
    tau_mem_max: float = 0.98
    tau_syn: float = 0.50
    v_threshold: float = 0.12
    v_thresh_min: float = 0.05
    v_thresh_max: float = 0.5
    v_reset: float = -0.1
    refractory_t: int = 2
    threshold_lr: float = 0.01
    lif_freeze_steps: int = 500

    # ---- Cascade amplification (minicolumn model) ----

    # Reference: Mountcastle VB (1997). DOI: 10.1093/brain/120.4.701
    n_clusters: int = 64
    cascade_radius: int = 3
    cascade_gain: float = 0.8

    # ---- STDP (FIX F) ----

    # Reference: Bi GQ, Poo MM (1998). DOI: 10.1523/JNEUROSCI.18-24-10464.1998
    stdp_a_plus: float = 0.005
    stdp_a_minus: float = 0.005
    stdp_tau_plus: float = 20.0
    stdp_tau_minus: float = 20.0
    stdp_w_max: float = 0.5
    stdp_w_min: float = -0.15
    stdp_reward_scale: float = 1.0
    stdp_layers: Optional[List[str]] = None

    # ---- Synaptic resonance (attention) ----

    # Top-k sparse attention: each query attends to at most this many keys.
    # NOT a biological quantity. Sparse connectivity approximation.
    resonance_top_k: int = 64

    # Output clamp floor for non-executive blocks.
    clamp_floor: float = -0.1

    # Surrogate gradient sharpness.
    # Reference: Fang W et al. (2021). DOI: 10.1109/ICCV48922.2021.00266
    surrogate_alpha: float = 4.0

    # RoPE frequency base.
    # Reference: Su J et al. (2024). DOI: 10.1016/j.neucom.2023.127063
    rope_theta: float = 10000.0

    # ---- Mixture of Experts (FIX A + G) ----

    # Reference: Shazeer N et al. (2017). DOI: 10.48550/arXiv.1701.06538
    n_experts: int = 4
    top_k_experts: int = 2
    moe_capacity_factor: float = 1.25
    moe_load_balance_weight: float = 0.01
    moe_route_temperature: float = 1.0

    # ---- Spike rate regulation (FIX C + L) ----

    target_spike_rate: float = 0.03
    spike_loss_weight: float = 0.5

    # ---- Zone layout ----

    # Sensory: primary feature extraction (FFN blocks).
    sensory_layers: int = 2
    # Association: cross-modal integration (MoE blocks).
    association_layers: int = 2
    # Executive: decision/output (FFN blocks, force-nonneg).
    executive_layers: int = 2

    # ---- Memory cortex (FIX B) ----

    # Slow LIF time constant for persistent working memory.
    # Reference: Wang XJ (2001). DOI: 10.1016/S0166-2236(00)01868-3
    memory_tau_mem: float = 0.99
    memory_size: int = 128
    memory_gate_threshold: float = 0.3
    memory_n_read_heads: int = 4

    # ---- Gradient checkpointing (FIX H) ----

    gradient_checkpointing: bool = False

    # ---- External embedding path (MEM 3) ----

    # When > 0, TemporalSpikeEncoder accepts float_embeds of this dimension.
    # Set to d_model for same-space injection (PRAGMI episodic context).
    float_embed_dim: int = 0  # 0 = disabled

    # ---- Episodic memory layer (MEM 5) ----

    # When True, the episodic memory layer from the PRAGMI kernel is wired
    # after the executive blocks. This is the interface between Timmy and
    # the Cognitive Kernel's long-term memory.
    episodic_memory: bool = True
    episodic_num_entries: int = 4096
    episodic_top_k: int = 8
    episodic_commit_every: int = 16
    episodic_present_capacity: int = 32
    episodic_drift_scale: float = 0.01

    # ---- Training ----

    batch_size: int = 2
    grad_accum: int = 16
    lr: float = 3e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 50_000
    save_every: int = 1000
    log_every: int = 10
    max_grad_norm: float = 1.0
    dtype: torch.dtype = torch.float16
    device: str = "cuda"

    @property
    def T_total(self) -> int:
        """Total temporal depth: fast + slow basis."""
        return self.T + self.T_slow

    @property
    def n_layers_total(self) -> int:
        """Total number of processing blocks across all zones."""
        return self.sensory_layers + self.association_layers + self.executive_layers

    def __post_init__(self):
        """Set default STDP target layers if none specified."""
        if self.stdp_layers is None:
            self.stdp_layers = [f"executive_{i}" for i in range(self.executive_layers)]

    def to_neuron_config(self) -> NeuronConfig:
        """Extract the neuron-level config subset."""
        return NeuronConfig(
            tau_mem=self.tau_mem, tau_mem_min=self.tau_mem_min,
            tau_mem_max=self.tau_mem_max, tau_syn=self.tau_syn,
            v_threshold=self.v_threshold, v_thresh_min=self.v_thresh_min,
            v_thresh_max=self.v_thresh_max, v_reset=self.v_reset,
            refractory_t=self.refractory_t, threshold_lr=self.threshold_lr,
            lif_freeze_steps=self.lif_freeze_steps,
            surrogate_alpha=self.surrogate_alpha,
            n_clusters=self.n_clusters, cascade_radius=self.cascade_radius,
            cascade_gain=self.cascade_gain,
            target_spike_rate=self.target_spike_rate,
            spike_loss_weight=self.spike_loss_weight,
        )

    def to_encoder_config(self) -> EncoderConfig:
        """Extract the encoder-level config subset."""
        return EncoderConfig(
            tokenizer_id=self.tokenizer_id, vocab_size=self.vocab_size,
            d_model=self.d_model, T=self.T, T_slow=self.T_slow,
            float_embed_dim=self.float_embed_dim,
        )


# =========================================================================
# Timmy Model
# =========================================================================

class TimmyModel(nn.Module):
    """
    Timmy: Spiking Neural Network Language Model.

    Architecture (token -> logits):
        TemporalSpikeEncoder (+ optional float_embeds)
            -> input_lif           (persistent AssociativeLIF)
            -> sensory_blocks      (2 x TimmyBlock, FFN)
            -> association_blocks  (2 x TimmyBlock, MoE)
            -> memory_cortex       (MemoryCortex, slow LIF)
            -> executive_blocks    (2 x TimmyBlock, FFN, force_nonneg)
            -> readout_lif         (persistent AssociativeLIF)
            -> EMA smoothing -> lm_head -> logits

    State persistence:
        MEM 1: Membrane buffers always registered in state_dict.
        MEM 2: save_state / load_state for full three-layer checkpoints
               (cold weights, warm membrane dynamics, hot STDP/MoE scalars).
               Delegates to timmy_state.py for the actual serialization.
        MEM 3: External float embedding path in encoder.
        MEM 4: STDP external reward signal via stdp.set_external_reward().
        MEM 5: Episodic memory layer interface (PRAGMI kernel link).

    References:
        Felleman & Van Essen (1991). DOI: 10.1093/cercor/1.1.1
        Neftci et al. (2019). DOI: 10.1109/MSP.2019.2931595
        Gerstner et al. (2014). DOI: 10.1017/CBO9781107447615
    """

    def __init__(self, cfg: TimmyConfig):
        """
        Args:
            cfg: complete Timmy configuration.
        """
        super().__init__()
        self.cfg = cfg
        neuron_cfg = cfg.to_neuron_config()
        encoder_cfg = cfg.to_encoder_config()

        # ---- Encoder: tokens -> temporal spike current ----
        self.encoder = TemporalSpikeEncoder(encoder_cfg)
        self.input_lif = AssociativeLIF(
            cfg.d_model, neuron_cfg, persistent=cfg.persistent_mem
        )

        # ---- Sensory zone: primary feature extraction (FFN) ----
        self.sensory_blocks = nn.ModuleList([
            TimmyBlock(cfg, i, use_moe=False, zone="sensory", neuron_cfg=neuron_cfg)
            for i in range(cfg.sensory_layers)
        ])

        # ---- Association zone: cross-modal integration (MoE) ----
        self.association_blocks = nn.ModuleList([
            TimmyBlock(cfg, cfg.sensory_layers + i, use_moe=True,
                       zone="association", neuron_cfg=neuron_cfg)
            for i in range(cfg.association_layers)
        ])

        # ---- Working memory ----
        self.memory_cortex = MemoryCortex(cfg, neuron_cfg)

        # ---- Executive zone: decision/output (FFN, force-nonneg) ----
        self.executive_blocks = nn.ModuleList([
            TimmyBlock(
                cfg, cfg.sensory_layers + cfg.association_layers + i,
                use_moe=False, zone="executive", neuron_cfg=neuron_cfg,
            )
            for i in range(cfg.executive_layers)
        ])

        # ---- Readout: EMA-smoothed membrane potential -> logits ----
        self.readout_lif = AssociativeLIF(
            cfg.d_model, neuron_cfg, persistent=cfg.persistent_mem
        )
        self.readout_ema_raw = nn.Parameter(torch.tensor(1.4))
        self.readout_norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # ---- Learning: STDP + spike regulation ----
        self.stdp = STDPEngine(cfg)
        self.spike_regulator = AuxiliarySpikeRegulator(
            target_rate=cfg.target_spike_rate, weight=cfg.spike_loss_weight
        )
        self._last_loss: Optional[float] = None
        self._stdp_cache: Dict[str, Tensor] = {}

    @property
    def readout_ema_decay(self) -> Tensor:
        """EMA decay for the readout membrane potential smoothing."""
        return torch.sigmoid(self.readout_ema_raw)

    # ---- State management (MEM 2) ----

    def reset_state(self):
        """Reset all persistent membrane states (call between unrelated sequences)."""
        self.input_lif.reset_state()
        self.readout_lif.reset_state()
        self.memory_cortex.reset_state()

    def save_state(
        self,
        path: str,
        optimizer_state: Optional[Dict] = None,
        scheduler_state: Optional[Dict] = None,
        training_step: Optional[int] = None,
        extra: Optional[Dict] = None,
    ) -> str:
        """
        Save a complete three-layer checkpoint to disk.

        Captures cold (weights), warm (membrane dynamics for every
        AssociativeLIF in the module tree), and hot (STDP scalars, MoE
        expert utilization EMAs) state layers. Includes an architecture
        hash for structural verification on restore, and a weight health
        snapshot for post-load diagnostics.

        Delegates to timmy_state.save_timmy_state(). See that module for
        full documentation of the checkpoint format.

        Args:
            path: filesystem path. Convention: "timmy_step_{N}.state".
            optimizer_state: output of optimizer.state_dict(). Optional.
            scheduler_state: output of scheduler.state_dict(). Optional.
            training_step: current global training step. Optional.
            extra: arbitrary metadata (dataset path, git hash, etc.).

        Returns:
            The path written to.
        """
        return save_timmy_state(
            model=self,
            path=path,
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state,
            training_step=training_step,
            extra=extra,
        )

    def load_state(
        self,
        path: str,
        strict_cold: bool = False,
        device: Optional[str] = None,
        skip_hash_check: bool = False,
    ) -> Dict[str, Any]:
        """
        Restore a checkpoint into this model.

        Applies cold (weights), warm (per-population LIF membrane state),
        and hot (STDP/MoE scalars) layers. Verifies the architecture hash
        before touching any weights. Runs weight health diagnostics after
        loading and logs warnings for dead or exploded layers.

        Backward compatible with v1 checkpoints produced by the old
        serialize_state() method.

        Delegates to timmy_state.load_timmy_state(). See that module for
        full documentation of migration, batch size handling, and hash
        verification.

        Args:
            path: filesystem path to the .state checkpoint.
            strict_cold: if True, require exact key match for state_dict.
            device: target device. If None, uses current model device.
            skip_hash_check: bypass architecture hash verification.

        Returns:
            Restoration metadata dict with keys: state_version,
            training_step, architecture_hash_match, lif_restored,
            lif_skipped, weight_warnings, has_optimizer, has_scheduler,
            config, extra, optimizer, scheduler.

        Raises:
            ArchitectureMismatchError: if the architecture hash does not
                match and skip_hash_check is False.
        """
        return load_timmy_state(
            model=self,
            path=path,
            strict_cold=strict_cold,
            device=device,
            skip_hash_check=skip_hash_check,
        )

    def architecture_hash(self) -> str:
        """
        Compute the structural hash of this model's config.

        Two models with the same hash have compatible tensor shapes and
        can load each other's checkpoints. Two models with different
        hashes are structurally incompatible.

        Returns:
            16-character hex string.
        """
        return compute_architecture_hash(self.cfg)

    # ---- Forward pass ----

    def forward(
        self,
        token_ids: Tensor,
        float_embeds: Optional[Tensor] = None,
        enable_stdp: bool = False,
    ) -> Tuple[Tensor, Dict]:
        """
        Forward pass: tokens -> logits.

        Args:
            token_ids: (B, S) integer token indices.
            float_embeds: optional (B, S, float_embed_dim) external continuous
                embeddings (MEM 3). Requires cfg.float_embed_dim > 0.
            enable_stdp: if True, cache pre/post spike tensors for a
                subsequent stdp_update() call (MEM 4).

        Returns:
            logits: (B, S, vocab_size).
            stats: dict with spike rates, MoE routing, memory diagnostics.
        """
        B, S = token_ids.shape
        T_t = self.cfg.T_total
        D = self.cfg.d_model

        # Encode: tokens -> multi-scale temporal current.
        cur = self.encoder(token_ids, float_embeds=float_embeds)
        isp, _ = self.input_lif(cur)
        isp = isp.reshape(T_t, B, S, D)
        spike_ts = [isp]
        stats: Dict = {}
        moe_lb = torch.tensor(0.0, device=token_ids.device)

        if enable_stdp:
            self._stdp_cache["input"] = isp.detach()

        # Sensory zone.
        x = isp
        for i, bl in enumerate(self.sensory_blocks):
            x, bs = bl(x)
            spike_ts.append(x)
            for k, v in bs.items():
                stats[f"sensory_{i}_{k}"] = v

        # Association zone (MoE).
        for i, bl in enumerate(self.association_blocks):
            prev = x.detach() if enable_stdp else None
            x, bs = bl(x)
            spike_ts.append(x)
            lb = bs.pop("moe_load_balance_loss", None)
            if lb is not None:
                moe_lb = moe_lb + lb
            for k, v in bs.items():
                stats[f"assoc_{i}_{k}"] = v
            if enable_stdp and prev is not None:
                self._stdp_cache[f"assoc_{i}_pre"] = prev
                self._stdp_cache[f"assoc_{i}_post"] = x.detach()

        # Working memory.
        x, ms = self.memory_cortex(x)
        stats.update(ms)

        # Executive zone (force-nonneg output).
        for i, bl in enumerate(self.executive_blocks):
            prev = x.detach() if enable_stdp else None
            x, bs = bl(x)
            spike_ts.append(x)
            for k, v in bs.items():
                stats[f"exec_{i}_{k}"] = v
            if enable_stdp and prev is not None:
                self._stdp_cache[f"executive_{i}_pre"] = prev
                self._stdp_cache[f"executive_{i}_post"] = x.detach()

        # Readout: EMA-smoothed membrane potential -> logits.
        xf = x.reshape(T_t, B * S, D)
        rsp, vm = self.readout_lif(xf)
        a = self.readout_ema_decay
        ema = torch.zeros(B * S, D, device=x.device, dtype=vm.dtype)
        for t in range(T_t):
            ema = a * ema + (1 - a) * vm[t]
        vs = ema.reshape(B, S, D)
        sm = rsp.mean(dim=0).reshape(B, S, D)
        ro = vs + sm

        xn = F.layer_norm(
            ro.float(),
            self.readout_norm.normalized_shape,
            self.readout_norm.weight.float() if self.readout_norm.weight is not None else None,
            self.readout_norm.bias.float() if self.readout_norm.bias is not None else None,
            self.readout_norm.eps,
        ).to(ro.dtype)
        logits = self.lm_head(xn)

        # Aggregate stats.
        out_rate = rsp.detach().mean().item()
        sr = [s.detach().clamp(min=0).mean().item() for s in spike_ts]
        stats["sparsity"] = 1.0 - out_rate
        stats["avg_spike_rate"] = sum(sr) / len(sr)
        stats["spike_loss"] = self.spike_regulator(spike_ts)
        stats["moe_lb_loss"] = moe_lb
        stats["spike_rates"] = sr

        return logits, stats

    # ---- STDP update ----

    @torch.no_grad()
    def stdp_update(self, current_loss: Optional[float] = None) -> None:
        """
        Apply reward-modulated STDP to executive-zone layers (MEM 4).

        Uses cached pre/post tensors from the last forward(enable_stdp=True).
        If an external reward was injected via stdp.set_external_reward(),
        that takes priority over loss-based reward.

        Args:
            current_loss: current training loss for reward computation.
        """
        loss_val = current_loss or self._last_loss
        for name, bl_list, zone in [
            ("assoc", self.association_blocks, "assoc"),
            ("executive", self.executive_blocks, "executive"),
        ]:
            for i, bl in enumerate(bl_list):
                key = f"{name}_{i}"
                pre_k = f"{key}_pre"
                post_k = f"{key}_post"
                if pre_k not in self._stdp_cache or post_k not in self._stdp_cache:
                    continue
                pre = self._stdp_cache[pre_k]
                post = self._stdp_cache[post_k]
                T_dim = pre.shape[0]
                pre_flat = pre.reshape(T_dim, -1, self.cfg.d_model).mean(dim=1)
                post_flat = post.reshape(T_dim, -1, self.cfg.d_model).mean(dim=1)
                layer_name = f"executive_{i}" if zone == "executive" else f"assoc_{i}"
                self.stdp.apply_to_layer(
                    bl.resonance.W_v,
                    pre_flat, post_flat,
                    current_loss=loss_val,
                    name=layer_name,
                )
        self._stdp_cache.clear()

    def set_last_loss(self, loss: float) -> None:
        """Store the last training loss for STDP reward computation."""
        self._last_loss = loss

    # ---- Diagnostics ----

    def count_params(self) -> str:
        """Human-readable parameter count by zone."""
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        se = sum(p.numel() for n, p in self.named_parameters() if "sensory" in n)
        a = sum(p.numel() for n, p in self.named_parameters() if "association" in n)
        m = sum(p.numel() for n, p in self.named_parameters() if "memory" in n)
        e = sum(p.numel() for n, p in self.named_parameters() if "executive" in n)
        return (
            f"Total: {total/1e6:.1f}M | Trainable: {train/1e6:.1f}M\n"
            f"  Sensory:     {se/1e6:.1f}M ({self.cfg.sensory_layers} blocks)\n"
            f"  Association: {a/1e6:.1f}M ({self.cfg.association_layers} blocks, MoE)\n"
            f"  Memory:      {m/1e6:.1f}M\n"
            f"  Executive:   {e/1e6:.1f}M ({self.cfg.executive_layers} blocks)"
        )
