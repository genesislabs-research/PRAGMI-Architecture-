"""
timmy/plasticity.py
Reward-Modulated Spike-Timing-Dependent Plasticity (STDP)

BIOLOGICAL GROUNDING:
This file implements the three-factor learning rule used for online synaptic
modification in Timmy's executive zone. STDP is a biologically observed
phenomenon where the relative timing of pre- and post-synaptic spikes
determines the direction and magnitude of synaptic weight change: pre-before-
post strengthens the synapse (LTP), post-before-pre weakens it (LTD).

The three-factor extension adds a modulatory reward signal that gates the
STDP update. Without reward modulation, STDP is unsupervised and can be
unstable. The reward signal converts it into a reinforcement learning rule:
synapses that contributed to rewarded outcomes are strengthened, and synapses
that contributed to punished outcomes are weakened.

The STDP engine is only applied to designated layers (executive zone by
default). This isolation (FIX F) prevents STDP from destabilizing the
sensory and association zones, which are trained solely by backpropagation.

The external reward signal path (MEM 4) allows the Cognitive Kernel, a human
evaluator, or an environment signal to directly inject reward values that
bypass the loss-based reward computation. This is how the system transitions
from self-supervised training to externally-guided learning.

Key grounding papers:
1. Bi GQ, Poo MM (1998). "Synaptic modifications in cultured hippocampal
   neurons: dependence on spike timing, synaptic strength, and postsynaptic
   cell type." Journal of Neuroscience, 18(24):10464-10472.
   DOI: 10.1523/JNEUROSCI.18-24-10464.1998

2. Izhikevich EM (2007). "Solving the distal reward problem through linkage
   of STDP and dopamine signaling." Cerebral Cortex, 17(10):2443-2452.
   DOI: 10.1093/cercor/bhl152

3. Fremaux N, Gerstner W (2016). "Neuromodulated spike-timing-dependent
   plasticity, and theory of three-factor learning rules." Frontiers in
   Neural Circuits, 9:85. DOI: 10.3389/fncir.2015.00085
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Optional, Set


class STDPEngine:
    """
    Reward-modulated STDP with eligibility traces and external reward injection.

    BIOLOGICAL STRUCTURE: Synaptic plasticity at glutamatergic synapses,
    modulated by dopaminergic (reward) signaling from the VTA/SNc.

    BIOLOGICAL FUNCTION: Pre-before-post spike timing strengthens the synapse
    (LTP) with amplitude a_plus and time constant tau_plus. Post-before-pre
    weakens it (LTD) with amplitude a_minus and time constant tau_minus.
    The update is then scaled by a reward signal that determines whether
    the weight change is applied (positive reward) or reversed (negative
    reward).

    Reference: Bi GQ, Poo MM (1998). "Synaptic modifications in cultured
    hippocampal neurons." Journal of Neuroscience, 18(24):10464-10472.
    DOI: 10.1523/JNEUROSCI.18-24-10464.1998

    Reference: Izhikevich EM (2007). "Solving the distal reward problem."
    Cerebral Cortex, 17(10):2443-2452.
    DOI: 10.1093/cercor/bhl152

    ISOLATION (FIX F): STDP is only applied to layers in the allowed set
    (executive zone by default). Sensory and association zones are trained
    by backpropagation only. This prevents STDP from destabilizing the
    feature extraction layers while allowing the decision-making layers
    to benefit from reinforcement.

    EXTERNAL REWARD (MEM 4): set_external_reward(reward) injects a reward
    value in [-1, 1] directly, bypassing the loss-based EMA computation.
    The external reward is consumed on the next apply_to_layer call and
    then cleared (one-shot).

    WEIGHT BOUNDING: Updates are norm-clipped to max_update_norm=0.01 per
    apply call, and weights are clamped to [w_min, w_max] after each update.
    This prevents runaway weight growth that would otherwise occur because
    STDP is a positive feedback loop (strong synapses cause correlated
    firing, which strengthens them further).

    Reference: Fremaux N, Gerstner W (2016). "Neuromodulated STDP and
    three-factor learning rules." Frontiers in Neural Circuits, 9:85.
    DOI: 10.3389/fncir.2015.00085
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: model-level config providing stdp_a_plus, stdp_a_minus,
                stdp_tau_plus, stdp_tau_minus, stdp_w_max, stdp_w_min,
                stdp_reward_scale, stdp_layers.
        """
        self.cfg = cfg

        # STDP amplitude parameters.
        # a_plus: LTP magnitude (pre-before-post strengthening).
        # a_minus: LTD magnitude (post-before-pre weakening).
        # Symmetric defaults (a_plus = a_minus = 0.005) produce balanced
        # potentiation and depression.
        # Reference: Bi & Poo (1998). DOI: 10.1523/JNEUROSCI.18-24-10464.1998
        self.a_plus = cfg.stdp_a_plus
        self.a_minus = cfg.stdp_a_minus

        # Timing window parameters (in timesteps, not ms).
        # tau_plus: time constant for the pre-synaptic eligibility trace.
        # tau_minus: time constant for the post-synaptic eligibility trace.
        # Biological values are ~20ms for both. With T=8 timesteps these
        # control how far back in the spike train timing matters.
        # Reference: Bi & Poo (1998). DOI: 10.1523/JNEUROSCI.18-24-10464.1998
        self.tau_plus = cfg.stdp_tau_plus
        self.tau_minus = cfg.stdp_tau_minus

        # Weight bounds.
        self.w_max = cfg.stdp_w_max
        self.w_min = cfg.stdp_w_min

        # Reward scaling for the loss-based reward computation.
        self.reward_scale = cfg.stdp_reward_scale

        # Layer isolation (FIX F): only these layers receive STDP updates.
        self.allowed: Set[str] = set(cfg.stdp_layers or [])

        # Loss-based reward EMA baseline.
        self._loss_ema: float = 10.0
        self._ema_decay: float = 0.99

        # Maximum norm of a single STDP weight update (stability guard).
        self.max_update_norm: float = 0.01

        # External reward signal (MEM 4). One-shot: consumed on next apply.
        self._external_reward: Optional[float] = None

    def set_external_reward(self, reward: float) -> None:
        """
        Inject an external reward signal for the next STDP update.

        This bypasses the loss-based reward computation entirely. The value
        is consumed on the next apply_to_layer call and then cleared.

        Args:
            reward: float in [-1, 1].
                +1: maximum positive reinforcement (strengthen active synapses).
                -1: maximum negative reinforcement (weaken active synapses).
                 0: no modulation (raw STDP without reward gating).
        """
        self._external_reward = float(reward)

    def update_reward(self, current_loss: float) -> None:
        """Update the loss-based reward EMA baseline."""
        self._loss_ema = (
            self._ema_decay * self._loss_ema
            + (1 - self._ema_decay) * current_loss
        )

    def _compute_reward(self, current_loss: float) -> float:
        """
        Compute reward from loss improvement relative to EMA baseline.
        Returns a value in (0, 1): >0.5 means current loss is below baseline
        (positive reinforcement), <0.5 means above baseline (negative).
        """
        return float(torch.sigmoid(
            torch.tensor((self._loss_ema - current_loss) * self.reward_scale)
        ).item())

    def is_allowed(self, name: str) -> bool:
        """Check if a named layer is in the STDP-allowed set."""
        return name in self.allowed

    @torch.no_grad()
    def compute_stdp_update(self, pre: Tensor, post: Tensor) -> Tensor:
        """
        Compute the raw STDP weight update from pre/post spike trains.

        Iterates through timesteps, maintaining exponentially decaying
        eligibility traces for pre and post spikes. At each post-spike,
        LTP is applied proportional to the pre-trace. At each pre-spike,
        LTD is applied proportional to the post-trace.

        Reference: Bi GQ, Poo MM (1998).
        DOI: 10.1523/JNEUROSCI.18-24-10464.1998

        Args:
            pre:  (T, N) pre-synaptic spike train (population-averaged).
            post: (T, N) post-synaptic spike train (population-averaged).

        Returns:
            (post_dim, pre_dim) weight update matrix, norm-clipped.
        """
        T = pre.shape[0]
        # Eligibility traces decay exponentially.
        tp = torch.zeros_like(pre[0])     # pre-synaptic trace
        tpo = torch.zeros_like(post[0])   # post-synaptic trace
        dp = math.exp(-1.0 / self.tau_plus)
        dm = math.exp(-1.0 / self.tau_minus)
        dW = torch.zeros(post.shape[1], pre.shape[1],
                         device=pre.device, dtype=pre.dtype)

        for t in range(T):
            tp = tp * dp + pre[t]       # update pre-trace
            tpo = tpo * dm + post[t]    # update post-trace

            # LTP: post spike -> strengthen synapses from recently active pre.
            if post[t].any():
                dW += self.a_plus * torch.outer(post[t], tp)

            # LTD: pre spike -> weaken synapses to recently active post.
            if pre[t].any():
                dW -= self.a_minus * torch.outer(tpo, pre[t])

        # Norm clip for stability.
        n = dW.norm()
        if n > self.max_update_norm:
            dW = dW * (self.max_update_norm / n)

        return dW

    @torch.no_grad()
    def apply_to_layer(
        self,
        layer: nn.Linear,
        pre: Tensor,
        post: Tensor,
        current_loss: Optional[float] = None,
        name: str = "",
    ) -> None:
        """
        Apply reward-modulated STDP to a linear layer's weights.

        Reward priority:
            1. External reward (MEM 4) if set via set_external_reward().
            2. Loss-based EMA reward if current_loss is provided.
            3. Unmodulated raw STDP if neither is available.

        Reference: Izhikevich EM (2007). "Solving the distal reward problem."
        DOI: 10.1093/cercor/bhl152

        Args:
            layer: nn.Linear whose weight to update.
            pre: (T, B, N) or (T, N) pre-synaptic spike tensor.
            post: (T, B, N) or (T, N) post-synaptic spike tensor.
            current_loss: current training loss (for loss-based reward).
            name: layer name for the allowed-set check.
        """
        if name and not self.is_allowed(name):
            return

        # Average over batch if 3D.
        if pre.dim() == 3:
            pre = pre.mean(dim=1)
        if post.dim() == 3:
            post = post.mean(dim=1)

        dW = self.compute_stdp_update(pre, post)

        # Reward modulation: scale the raw STDP update by reward signal.
        if self._external_reward is not None:
            dW = dW * self._external_reward
            self._external_reward = None  # consume and clear
        elif current_loss is not None:
            r = self._compute_reward(current_loss)
            dW = dW * (2.0 * r - 1.0)  # map (0,1) -> (-1,1)
            self.update_reward(current_loss)

        # Apply update to the layer weight, respecting shape and bounds.
        o, i = layer.weight.shape
        dW = dW[:o, :i]
        layer.weight.data = (layer.weight.data + dW).clamp(self.w_min, self.w_max)
