"""
timmy/neuron.py
The Spiking Neuron: Surrogate Gradient and Associative LIF with Cascade Amplification

BIOLOGICAL GROUNDING:
This file implements the fundamental spiking unit used throughout Timmy, the
spiking neural network language model that bridges the external LLM and the
Cognitive Kernel. Every neuron in Timmy's sensory, association, and executive
zones is an instance of AssociativeLIF.

The neuron model extends the textbook Leaky Integrate-and-Fire (LIF) with
three mechanisms:

    1. Synaptic current filtering. Input current passes through a first-order
       exponential filter before reaching the membrane, modeling the kinetics
       of postsynaptic receptor channels (AMPA: 1-5ms, NMDA: 50-150ms). The
       filter smooths discrete input spikes into a continuous current that the
       membrane can integrate, preventing physically unrealistic instantaneous
       voltage jumps from single input events.

    2. Absolute refractory period. After firing, the membrane is clamped to
       a hyperpolarized reset voltage for a fixed number of timesteps, modeling
       the inactivation of voltage-gated sodium channels that makes the cell
       physically unable to produce another action potential regardless of
       input strength. At T=8 with refractory_t=2, this limits maximum firing
       rate to T/(refractory_t+1) = 2.67 spikes per processing window.

    3. Cascade amplification. When neurons in one cortical minicolumn fire,
       lateral excitatory connections boost the synaptic current of neurons in
       neighboring minicolumns. This produces correlated population bursting
       that carries more information per timestep than independent firing.
       The connectivity wraps circularly (toroidal topology) with strength
       decaying linearly with distance.

The spike function is a hard Heaviside step in the forward pass (binary 0/1
output). Backpropagation uses an arctangent surrogate gradient because the
true Heaviside has zero derivative everywhere except at the threshold where
it is undefined. The surrogate is a standard technique in modern SNN training
and is not specific to this architecture.

Key grounding papers:
1. Neftci EO, Mostafa H, Zenke F (2019). "Surrogate gradient learning in
   spiking neural networks: Bringing the power of gradient-based optimization
   to spiking neural networks." IEEE Signal Processing Magazine, 36(6):51-63.
   DOI: 10.1109/MSP.2019.2931595

2. Gerstner W, Kistler WM, Naud R, Paninski L (2014). "Neuronal Dynamics:
   From single neurons to networks and models of cognition." Cambridge
   University Press. DOI: 10.1017/CBO9781107447615

3. Mountcastle VB (1997). "The columnar organization of the neocortex."
   Brain, 120(4):701-722. DOI: 10.1093/brain/120.4.701

4. Hodgkin AL, Huxley AF (1952). "A quantitative description of membrane
   current and its application to conduction and excitation in nerve."
   Journal of Physiology, 117(4):500-544.
   DOI: 10.1113/jphysiol.1952.sp004764

5. Fang W, Yu Z, Chen Y, Masquelier T, Huang T, Tian Y (2021).
   "Incorporating learnable membrane time constants to enhance learning of
   spiking neural networks." In Proceedings of the IEEE/CVF International
   Conference on Computer Vision, pp. 2661-2671.
   DOI: 10.1109/ICCV48922.2021.00266
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


# =========================================================================
# Configuration
# =========================================================================

@dataclass
class NeuronConfig:
    """
    Configuration for the Associative LIF neuron.
    Parameters with biological analogs cite their source. Parameters that
    are training artifacts or engineering approximations are labeled NOT
    biological.
    """

    # ---- Membrane dynamics ----

    # Discrete-time membrane decay factor. In continuous time the membrane
    # time constant of cortical pyramidal cells is 10-30ms; in discrete time
    # with a 1ms step, beta = exp(-dt/tau) gives 0.90-0.97. Our value of
    # 0.85 corresponds to a faster effective time constant (~6ms), tuned
    # empirically for stable training at T=8.
    # NOT a biological quantity. Training artifact.
    # Reference: Gerstner et al. (2014). DOI: 10.1017/CBO9781107447615, Ch. 1
    tau_mem: float = 0.85

    # Stability clamps on the learnable membrane decay. tau_mem_min prevents
    # the neuron from becoming so leaky it cannot integrate across two
    # timesteps. tau_mem_max prevents it from becoming a perfect integrator
    # (no leak) which causes unbounded membrane growth.
    # NOT biological. Stability guards.
    tau_mem_min: float = 0.8
    tau_mem_max: float = 0.98

    # Synaptic current decay factor. Models first-order postsynaptic current
    # dynamics. 0.50 corresponds to fast AMPA-like kinetics (~2ms at 1ms step).
    # Biological range: AMPA 1-5ms, NMDA 50-150ms.
    # NOT a biological quantity. Training artifact biased toward fast dynamics.
    # Reference: Gerstner et al. (2014). DOI: 10.1017/CBO9781107447615, Ch. 3
    tau_syn: float = 0.50

    # Initial spike threshold in arbitrary units (learnable, per-neuron).
    # Biological action potential threshold is ~-55mV to -40mV relative to
    # resting potential -70mV, but this model operates in unitless space.
    # Only the ratio to input scale matters, not the absolute value.
    # Reference: Gerstner et al. (2014). DOI: 10.1017/CBO9781107447615, Ch. 1
    v_threshold: float = 0.12

    # Clamps on learnable threshold.
    # v_thresh_min prevents fire-on-everything. v_thresh_max prevents fire-on-nothing.
    # NOT biological. Stability guards.
    v_thresh_min: float = 0.05
    v_thresh_max: float = 0.5

    # Membrane potential during refractory period. Negative value models
    # afterhyperpolarization (AHP) from potassium channel activation
    # following an action potential.
    # Reference: Gerstner et al. (2014). DOI: 10.1017/CBO9781107447615, Ch. 1
    v_reset: float = -0.1

    # Absolute refractory period in timesteps. Biological absolute refractory
    # period for cortical pyramidal cells is approximately 1-2ms.
    # Reference: Hodgkin AL, Huxley AF (1952). "A quantitative description
    # of membrane current and its application to conduction and excitation
    # in nerve." Journal of Physiology, 117(4):500-544.
    # DOI: 10.1113/jphysiol.1952.sp004764
    refractory_t: int = 2

    # Learning rate for the per-neuron threshold adaptation.
    # NOT biological. Training hyperparameter.
    threshold_lr: float = 0.01

    # Number of training steps before LIF parameters (tau, threshold) are
    # unfrozen. During early training the spike dynamics are unstable; holding
    # the biophysical parameters fixed while the projection weights warm up
    # prevents compounding instability.
    # NOT biological. Training artifact for early stability.
    lif_freeze_steps: int = 500

    # ---- Surrogate gradient ----

    # Sharpness of the arctangent surrogate gradient. Higher values produce
    # a sharper approximation to the Heaviside (better spike fidelity,
    # smaller gradients far from threshold). Lower values give smoother
    # gradients (worse fidelity, easier optimization).
    # NOT a biological quantity. Training artifact only. Biology has no
    # backpropagation. Value from Fang et al. (2021).
    # Reference: Fang W et al. (2021). "Incorporating learnable membrane
    # time constants to enhance learning of spiking neural networks."
    # ICCV 2021, pp. 2661-2671. DOI: 10.1109/ICCV48922.2021.00266
    # Framework: Neftci et al. (2019). DOI: 10.1109/MSP.2019.2931595
    surrogate_alpha: float = 4.0

    # ---- Cascade amplification (minicolumn model) ----

    # Number of neuron clusters (minicolumn analogs). Neurons are assigned
    # round-robin: neuron i belongs to cluster (i mod n_clusters). Each
    # cluster models a cortical minicolumn, a vertical group of ~80-100
    # neurons sharing similar tuning properties with strong lateral excitation.
    # Reference: Mountcastle VB (1997). "The columnar organization of the
    # neocortex." Brain, 120(4):701-722. DOI: 10.1093/brain/120.4.701
    n_clusters: int = 64

    # Lateral excitation radius in cluster index space. A spike in cluster i
    # excites clusters (i-r) through (i+r) with linearly decaying strength.
    # NOT a biological quantity. Real lateral excitation depends on axonal
    # arbor size (~200-500um) which does not map to an integer index.
    cascade_radius: int = 3

    # Initial coupling strength for cascade amplification. Learnable
    # per-cluster during training.
    # NOT a biological quantity. Engineering approximation.
    cascade_gain: float = 0.8

    # ---- Homeostatic monitoring ----

    # Target firing rate for the exponential moving average tracker. Used by
    # AuxiliarySpikeRegulator (in blocks.py) to penalize neurons that fire
    # too much or too little.
    # NOT biological. Training regularization target.
    target_spike_rate: float = 0.03

    # Weight of the spike rate loss in the total training objective.
    # NOT biological. Training hyperparameter.
    spike_loss_weight: float = 0.5


# =========================================================================
# Surrogate Gradient
# =========================================================================

class ATanSurrogate(torch.autograd.Function):
    """
    Arctangent surrogate gradient for spiking neuron training.

    Forward: hard Heaviside step (binary 0/1 spikes).
    Backward: smooth arctan derivative centered at threshold.

        ds/dv = alpha / (2*pi * (1 + (alpha * (v - threshold))^2))

    This is the derivative of (1/pi)*arctan(alpha*x) + 0.5, a smooth
    sigmoid-like approximation to the Heaviside. The gradient with respect
    to the threshold is the negative of the gradient with respect to the
    membrane potential (increasing threshold has the opposite effect of
    increasing potential on the firing decision).

    NOT a biological structure. There is no analog to backpropagation in
    biological neural circuits.

    Reference: Neftci EO, Mostafa H, Zenke F (2019). "Surrogate gradient
    learning in spiking neural networks." IEEE Signal Processing Magazine,
    36(6):51-63. DOI: 10.1109/MSP.2019.2931595
    """

    alpha = 4.0

    @staticmethod
    def forward(ctx, v_mem: Tensor, v_threshold: Tensor) -> Tensor:
        """Emit binary spike where membrane potential exceeds threshold."""
        ctx.save_for_backward(v_mem, v_threshold)
        return (v_mem >= v_threshold).to(v_mem.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Surrogate gradient: d/d_threshold = -d/d_v."""
        v_mem, v_threshold = ctx.saved_tensors
        x = (v_mem.float() - v_threshold.float())
        surrogate = ATanSurrogate.alpha / (
            2.0 * math.pi * (1.0 + (ATanSurrogate.alpha * x) ** 2)
        )
        grad_v = (grad_output.float() * surrogate).to(v_mem.dtype)
        return grad_v, -grad_v


def spike_fn(v_mem: Tensor, v_threshold: Tensor, alpha: float = 4.0) -> Tensor:
    """Convenience wrapper: set alpha and call ATanSurrogate."""
    ATanSurrogate.alpha = alpha
    return ATanSurrogate.apply(v_mem, v_threshold)


# =========================================================================
# Associative LIF Neuron
# =========================================================================

class AssociativeLIF(nn.Module):
    """
    Population of Leaky Integrate-and-Fire neurons with synaptic current
    filtering, absolute refractory period, and minicolumn cascade amplification.

    BIOLOGICAL STRUCTURE: Cortical pyramidal cell population with first-order
    synaptic dynamics and lateral excitatory connections organized into
    minicolumns (Mountcastle, 1997).

    BIOLOGICAL FUNCTION: Subthreshold membrane integration, threshold-based
    action potential generation, post-spike afterhyperpolarization with
    absolute refractory silence, and lateral excitation within minicolumns
    that produces correlated population bursting.

    Per-timestep dynamics:
        i_syn = tau_syn * i_syn + input                    (synaptic filtering)
        v_mem = tau_mem * v_mem + (1 - tau_mem) * i_syn    (leaky integration)
        spike = Heaviside(v_mem - threshold)               (spike generation)
        i_syn += cascade_amplify(spike)                    (lateral excitation)
        v_mem = v_mem - spike * threshold                  (soft reset)

    State persistence (MEM 1): when persistent=True, membrane potential and
    synaptic current carry between forward() calls via registered buffers.
    When persistent=False, buffers are (0,0) sentinels that keep state_dict()
    keys consistent for checkpoint loading without contributing to the
    forward pass.

    Per-neuron heterogeneity: the spike threshold is a learnable parameter
    with independent values per neuron. Over training, different neurons
    develop different firing sensitivities, producing the diverse response
    profiles observed in biological cortical populations.
    Reference: Graves AR et al. (2012). "Hippocampal pyramidal neurons
    comprise two distinct cell types." Neuron, 76(6):1126-1137.
    DOI: 10.1016/j.neuron.2012.10.036

    References:
        Gerstner et al. (2014). DOI: 10.1017/CBO9781107447615
        Mountcastle (1997). DOI: 10.1093/brain/120.4.701
        Neftci et al. (2019). DOI: 10.1109/MSP.2019.2931595
    """

    def __init__(
        self,
        n_neurons: int,
        cfg: NeuronConfig,
        persistent: bool = False,
        tau_mem_override: Optional[float] = None,
    ):
        """
        Args:
            n_neurons: population size (typically d_model=496 per layer).
            cfg: neuron configuration with all biophysical and training params.
            persistent: if True, membrane/synaptic state carries between calls.
                Used for layers that must maintain temporal continuity across
                sequential input chunks.
            tau_mem_override: override cfg.tau_mem for this population. Used by
                MemoryCortex for slow-decaying memory neurons with longer
                integration windows (tau_mem=0.99 vs default 0.85).
        """
        super().__init__()
        self.cfg = cfg
        self.n_neurons = n_neurons
        self.persistent = persistent

        # Per-neuron learnable spike threshold.
        # Initialized uniformly; develops heterogeneity during training.
        self.v_threshold_raw = nn.Parameter(
            torch.full((n_neurons,), cfg.v_threshold)
        )

        # Membrane decay in logit space: sigmoid(raw) clamped to [min, max].
        # Logit parameterization guarantees the decay factor stays in (0, 1)
        # before clamping, preventing NaN from negative or >1 values.
        # Reference: Fang et al. (2021). DOI: 10.1109/ICCV48922.2021.00266
        tau_mem = tau_mem_override if tau_mem_override is not None else cfg.tau_mem
        self.beta_mem_raw = nn.Parameter(
            torch.tensor(math.log(tau_mem / (1.0 - tau_mem + 1e-6)))
        )

        # Synaptic current decay in logit space.
        self.beta_syn_raw = nn.Parameter(
            torch.tensor(math.log(cfg.tau_syn / (1.0 - cfg.tau_syn + 1e-6)))
        )

        # Cascade amplification: minicolumn lateral excitation.
        # Round-robin cluster assignment: neuron i -> cluster (i % n_clusters).
        nc = cfg.n_clusters
        self.register_buffer("cluster_ids", torch.arange(n_neurons) % nc)

        # Neighbor excitation weight matrix (nc x nc). Weight decays linearly
        # with cluster distance, wraps circularly (toroidal topology matching
        # cortical sheet models). Diagonal is zero (no self-excitation;
        # self-reinforcement comes from the recurrent connectivity at the
        # network level, not the single-neuron level).
        r = cfg.cascade_radius
        idx = torch.arange(nc)
        iw = torch.zeros(nc, nc)
        for offset in range(-r, r + 1):
            if offset != 0:
                iw[idx, (idx + offset) % nc] = 1.0 - abs(offset) / (r + 1)
        self.neighbor_weights = nn.Parameter(iw)

        # Per-cluster gain for cascade amplification (learnable).
        self.cluster_gain = nn.Parameter(torch.full((nc,), cfg.cascade_gain))

        # Persistent state buffers (MEM 1).
        # persistent=True:  real (B, n_neurons) state carried between calls.
        # persistent=False: (0, 0) sentinel; never read, but key present in
        # state_dict() so checkpoint loading with strict=True never fails.
        if persistent:
            self.register_buffer("_v_mem_state", torch.zeros(1, n_neurons))
            self.register_buffer("_i_syn_state", torch.zeros(1, n_neurons))
        else:
            self.register_buffer("_v_mem_state", torch.zeros(0, 0))
            self.register_buffer("_i_syn_state", torch.zeros(0, 0))

        # Firing rate exponential moving average (for spike rate regulator).
        self.register_buffer(
            "_firing_rate_ema",
            torch.full((n_neurons,), cfg.target_spike_rate)
        )
        self.register_buffer("_step_counter", torch.tensor(0, dtype=torch.long))

    # ---- Clamped property accessors ----

    @property
    def v_threshold(self) -> Tensor:
        """Per-neuron threshold, clamped to [v_thresh_min, v_thresh_max]."""
        return self.v_threshold_raw.clamp(self.cfg.v_thresh_min, self.cfg.v_thresh_max)

    @property
    def beta_mem(self) -> Tensor:
        """Membrane decay factor: sigmoid of learnable logit, clamped."""
        return torch.sigmoid(self.beta_mem_raw).clamp(
            self.cfg.tau_mem_min, self.cfg.tau_mem_max
        )

    @property
    def beta_syn(self) -> Tensor:
        """Synaptic current decay factor in (0, 1)."""
        return torch.sigmoid(self.beta_syn_raw)

    # ---- Cascade amplification ----

    def _cascade_amplify(self, spikes: Tensor) -> Tensor:
        """
        Lateral excitatory current from spiking neurons to minicolumn neighbors.

        Aggregate per-cluster spike rates, spread to neighbors via the
        sigmoid-gated weight matrix, scale by per-cluster gain, and
        redistribute to individual neurons.

        Reference: Mountcastle VB (1997). "The columnar organization of the
        neocortex." Brain, 120(4):701-722. DOI: 10.1093/brain/120.4.701

        Args:
            spikes: (batch, n_neurons) binary spike tensor.

        Returns:
            (batch, n_neurons) lateral excitatory current to inject into i_syn.
        """
        B, D = spikes.shape
        nc = self.cfg.n_clusters
        cid = self.cluster_ids.unsqueeze(0).expand(B, -1)

        # Per-cluster mean spike rate.
        cf = torch.zeros(B, nc, device=spikes.device, dtype=spikes.dtype)
        cf.scatter_add_(1, cid, spikes)
        cf = cf / max(D // nc, 1)

        # Spread to neighbors. Sigmoid gates prevent negative excitation.
        W = torch.sigmoid(self.neighbor_weights)
        ns = (W.to(cf.dtype) @ cf.T).T * self.cluster_gain.to(cf.dtype).unsqueeze(0)

        # Map back from cluster space to neuron space.
        return ns.gather(1, cid)

    # ---- State management ----

    def reset_state(self):
        """Reset persistent membrane and synaptic state to zero."""
        if self.persistent:
            self._v_mem_state.zero_()
            self._i_syn_state.zero_()

    def get_state(self) -> dict:
        """Serialize neuron state for .soul file."""
        return {
            "v_mem_state": self._v_mem_state.cpu().clone(),
            "i_syn_state": self._i_syn_state.cpu().clone(),
            "firing_rate_ema": self._firing_rate_ema.cpu().clone(),
            "step_counter": self._step_counter.item(),
            "v_threshold_raw": self.v_threshold_raw.data.cpu().clone(),
            "beta_mem_raw": self.beta_mem_raw.data.cpu().clone(),
            "beta_syn_raw": self.beta_syn_raw.data.cpu().clone(),
            "neighbor_weights": self.neighbor_weights.data.cpu().clone(),
            "cluster_gain": self.cluster_gain.data.cpu().clone(),
        }

    def restore_state(self, state: dict):
        """Restore from .soul checkpoint."""
        device = self._v_mem_state.device
        self._v_mem_state = state["v_mem_state"].to(device)
        self._i_syn_state = state["i_syn_state"].to(device)
        self._firing_rate_ema = state["firing_rate_ema"].to(device)
        self._step_counter.fill_(state["step_counter"])
        self.v_threshold_raw.data = state["v_threshold_raw"].to(device)
        self.beta_mem_raw.data = state["beta_mem_raw"].to(device)
        self.beta_syn_raw.data = state["beta_syn_raw"].to(device)
        self.neighbor_weights.data = state["neighbor_weights"].to(device)
        self.cluster_gain.data = state["cluster_gain"].to(device)

    # ---- Forward pass ----

    def forward(self, i_input: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Run LIF dynamics for T timesteps.

        Args:
            i_input: (T, batch, n_neurons) input current per timestep from
                the upstream synaptic projection.

        Returns:
            spikes:  (T, batch, n_neurons) binary spike output.
            v_trace: (T, batch, n_neurons) membrane potential trace (used by
                STDP engine for eligibility traces and by diagnostics for
                monitoring neuron health).
        """
        T, B, D = i_input.shape
        device, dtype = i_input.device, i_input.dtype
        bm = self.beta_mem
        bs = self.beta_syn
        thresh = self.v_threshold
        alpha = self.cfg.surrogate_alpha
        ref_t = self.cfg.refractory_t

        # Initialize or restore persistent state.
        if self.persistent and self._v_mem_state.shape[0] == B:
            v_mem = self._v_mem_state.clone()
            i_syn = self._i_syn_state.clone()
        else:
            v_mem = torch.zeros(B, D, device=device, dtype=dtype)
            i_syn = torch.zeros(B, D, device=device, dtype=dtype)
            if self.persistent:
                self._v_mem_state = torch.zeros(B, D, device=device, dtype=dtype)
                self._i_syn_state = torch.zeros(B, D, device=device, dtype=dtype)

        # Refractory counter (integer, not differentiable).
        refrac = torch.zeros(B, D, device=device, dtype=torch.int32)
        refractory_val = torch.full_like(v_mem, self.cfg.v_reset)
        spikes_out = []
        v_trace = []

        for t in range(T):
            # Synaptic current: exponential decay + new input.
            # Reference: Gerstner et al. (2014), Ch. 3.
            # DOI: 10.1017/CBO9781107447615
            i_syn = bs * i_syn + i_input[t]

            # Membrane integration, clamped to reset during refractory period.
            # Reference: Gerstner et al. (2014), Ch. 1.
            # DOI: 10.1017/CBO9781107447615
            rmask = refrac > 0
            v_new = bm * v_mem + (1.0 - bm) * i_syn
            v_mem = torch.where(rmask, refractory_val, v_new)

            # Spike generation with surrogate gradient in backward pass.
            # Reference: Neftci et al. (2019). DOI: 10.1109/MSP.2019.2931595
            s = spike_fn(v_mem, thresh, alpha)

            # Cascade amplification: lateral minicolumn excitation.
            # Reference: Mountcastle (1997). DOI: 10.1093/brain/120.4.701
            if s.sum() > 0:
                i_syn = i_syn + self._cascade_amplify(s)

            # Soft reset: subtract threshold from spiking neurons. Preserves
            # the "residual" above threshold, which carries information about
            # input strength that a hard reset to v_reset would discard.
            v_mem = v_mem - s * thresh.detach()

            # Refractory counter: absolute refractory period.
            # Reference: Hodgkin & Huxley (1952).
            # DOI: 10.1113/jphysiol.1952.sp004764
            refrac = torch.where(
                s.bool(),
                torch.full_like(refrac, ref_t),
                (refrac - 1).clamp(min=0),
            )

            spikes_out.append(s)
            v_trace.append(v_mem)

        # Persist state for next call (detach to avoid graph retention).
        if self.persistent:
            self._v_mem_state = v_mem.detach()
            self._i_syn_state = i_syn.detach()

        ss = torch.stack(spikes_out)

        # Update firing rate tracker (no gradient, homeostatic monitoring only).
        with torch.no_grad():
            self._firing_rate_ema.lerp_(ss.mean(dim=(0, 1)), 0.01)
            self._step_counter += 1

        return ss, torch.stack(v_trace)

    # ---- Diagnostics ----

    def get_diagnostics(self) -> Dict[str, object]:
        """Population health report."""
        return {
            "mean_firing_rate": self._firing_rate_ema.mean().item(),
            "firing_rate_std": self._firing_rate_ema.std().item(),
            "threshold_mean": self.v_threshold.mean().item(),
            "threshold_std": self.v_threshold.std().item(),
            "beta_mem": self.beta_mem.item(),
            "beta_syn": self.beta_syn.item(),
            "cascade_gain_mean": self.cluster_gain.mean().item(),
            "steps": self._step_counter.item(),
            "n_neurons": self.n_neurons,
            "persistent": self.persistent,
        }
