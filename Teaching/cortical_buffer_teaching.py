"""
cortical_buffer_teaching.py

BIOLOGICAL GROUNDING
====================
This file models the persistent activity mechanism of the prefrontal cortex (PFC),
specifically the delay-period firing that sustains working memory representations
across seconds-long gaps in the absence of external input.

In the brain, pyramidal neurons in layers II/III and V of the PFC maintain elevated
firing rates during the delay period of working memory tasks through recurrent
synaptic reverberation: populations of excitatory neurons mutually excite each other
via AMPA and NMDA receptors, sustaining activity that would otherwise decay. This
persistent state biases subsequent sensory processing toward task-relevant
representations. It is the biological substrate of "keeping something in mind."

Within the PRAGMI architecture, each Timmy column implements a 32-dimensional
bottleneck buffer that compresses the column's post-forward-pass membrane state into
a compact representation, and injects that representation as an additive bias on the
initial membrane potential at the start of the next forward pass. The buffer persists
across wake cycles and is reconsolidated during sleep: replay passes update the buffer
through the bottleneck projection, so the post-sleep buffer reflects the
post-consolidation state rather than the pre-sleep working context.

Primary papers grounding this file:

Goldman-Rakic PS (1995). "Cellular basis of working memory." Neuron, 14(3), 477-485.
DOI: 10.1016/0896-6273(95)90304-6

Compte A, Brunel N, Goldman-Rakic PS, Wang XJ (2000). "Synaptic mechanisms and
network dynamics underlying spatial working memory in a cortical network model."
Cerebral Cortex, 10(11), 910-923. DOI: 10.1093/cercor/10.11.910

Wang XJ (2001). "Synaptic reverberation underlying mnemonic persistent activity."
Trends in Neurosciences, 24(8), 455-463. DOI: 10.1016/S0166-2236(00)01844-0
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class CorticalBufferConfig:
    """
    Configuration for the persistent cortical state buffer.

    Attributes:
        d_model: Internal representational width of the parent column.
            This value is NOT fixed at construction time in the dynamic
            expansion architecture. The buffer's projection layers must
            support width expansion via expand_for_new_d_model().
            NOT a biological quantity: engineering parameter determined
            by the column's current d_model after dynamic expansion.

        buffer_dim: Dimension of the compressed bottleneck buffer.
            [DECISION] Set to 32. Rationale: the associative cascade uses
            64 clusters. A 16-dim buffer preserves fewer than one value
            per cluster on average, insufficient to capture cluster-level
            activation state. A 32-dim buffer provides one value per two
            clusters, preserving coarse cluster topology. 64-dim would
            mirror the coordinate manifold but adds unnecessary cost for
            per-column runtime state.
            NOT a biological quantity: engineering approximation of the
            compression performed by corticothalamic feedback projections.
            Fuster JM (1973). DOI: 10.1152/jn.1973.36.1.61

        injection_scale: Scalar multiplier on the buffer projection before
            it is added to the initial membrane potential. Controls how
            strongly prior context biases the current forward pass.
            NOT a biological quantity: engineering hyperparameter.
            Initial value 0.1 chosen to keep the injected bias sub-threshold
            so that prior context primes but does not force firing.

        buffer_tau: EMA decay rate for buffer state updates. Controls how
            much of the previous buffer state is retained versus the new
            compressed membrane state on each update call.
            NOT a biological quantity: engineering approximation of the
            recurrent attractor timescale in PFC persistent activity. In the
            brain, reverberation timescale is determined by NMDA receptor
            kinetics and network connectivity. Here it is a tunable scalar.
            Wang XJ (2001). DOI: 10.1016/S0166-2236(00)01844-0
            Initial value 0.8 chosen so the buffer integrates approximately
            5 forward passes of history before old state decays below 1%.
    """
    d_model: int = 256
    buffer_dim: int = 32
    injection_scale: float = 0.1
    buffer_tau: float = 0.8


class CorticalBuffer(nn.Module):
    """
    Persistent cortical state buffer implementing PFC delay-period working memory.

    BIOLOGICAL STRUCTURE: Prefrontal cortex persistent activity.
    BIOLOGICAL FUNCTION: PFC neurons in layers II/III and V sustain elevated
    firing rates during delay periods through recurrent AMPA/NMDA synaptic
    reverberation. This activity biases subsequent sensory processing toward
    task-relevant representations, implementing working memory as a dynamical
    attractor state rather than a static store.

    Goldman-Rakic PS (1995). "Cellular basis of working memory." Neuron, 14(3),
    477-485. DOI: 10.1016/0896-6273(95)90304-6

    Compte A, et al. (2000). "Synaptic mechanisms and network dynamics underlying
    spatial working memory in a cortical network model." Cerebral Cortex, 10(11),
    910-923. DOI: 10.1093/cercor/10.11.910

    ANATOMICAL INTERFACE (write path):
        Sending structure: AssociativeLIF executive zone (post-forward-pass
            mean membrane potential across timesteps).
        Receiving structure: CorticalBuffer bottleneck projection.
        Connection: corticothalamic feedback, Layer 6 pyramidal axons projecting
            to thalamic relay nuclei and back to cortex.
        Sherman SM, Guillery RW (2002). DOI: 10.1098/rstb.2002.1161

    ANATOMICAL INTERFACE (read path):
        Sending structure: CorticalBuffer.
        Receiving structure: AssociativeLIF initial membrane potential (v_mem
            at t=0 of the next forward pass).
        Connection: recurrent collaterals within PFC layers II/III and V.
        Wang XJ (2001). DOI: 10.1016/S0166-2236(00)01844-0

    The buffer is a registered buffer (not a Parameter) because it is runtime
    state, not a learned weight. It is serialized in the HOT layer.
    The projection weights (compress and expand) are learned Parameters
    serialized in the COLD layer.
    """

    def __init__(self, cfg: CorticalBufferConfig) -> None:
        """
        Initialize the cortical buffer with bottleneck projections.

        Args:
            cfg: CorticalBufferConfig specifying d_model and buffer_dim.
        """
        super().__init__()
        self.cfg = cfg

        # Bottleneck compression: d_model -> buffer_dim.
        # Learned linear map approximating the compression performed by
        # corticothalamic Layer 6 feedback projections.
        # NOT a biological quantity: engineering approximation.
        # Fuster JM (1973). DOI: 10.1152/jn.1973.36.1.61
        self.compress = nn.Linear(cfg.d_model, cfg.buffer_dim, bias=False)

        # Bottleneck expansion: buffer_dim -> d_model.
        # Learned linear map approximating the thalamocortical relay back
        # to cortical layers IV and I.
        # NOT a biological quantity: engineering approximation.
        # Sherman SM, Guillery RW (2002). DOI: 10.1098/rstb.2002.1161
        self.expand = nn.Linear(cfg.buffer_dim, cfg.d_model, bias=False)

        # The persistent buffer itself. Registered as a buffer (not a Parameter)
        # because it is runtime state that evolves during inference, not a
        # weight that is updated by the optimizer.
        # Shape: (buffer_dim,). Scalar per compressed dimension.
        # Serialized in HOT layer, not COLD.
        # NOT a biological quantity: engineering approximation.
        self.register_buffer(
            "state",
            torch.zeros(cfg.buffer_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize projection weights.

        Compress is initialized with small random values so early forward
        passes see a near-zero bias before the buffer accumulates meaningful
        state. Expand is initialized with small random values for the same
        reason. Both use scaled normal initialization.

        NOT a biological quantity: engineering initialization choice.
        """
        nn.init.normal_(self.compress.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.expand.weight, mean=0.0, std=0.01)

    def get_injection(self) -> torch.Tensor:
        """
        Compute the additive bias to inject into the initial membrane potential.

        BIOLOGICAL FUNCTION: recurrent collateral currents from previously
        active PFC populations provide a tonic depolarizing input that
        biases the network toward previously active attractor states.
        Compte et al. (2000). DOI: 10.1093/cercor/10.11.910

        Returns:
            Tensor of shape (d_model,) representing the bias to add to
            v_mem at t=0. Scaled by injection_scale to keep the injected
            current sub-threshold, priming rather than forcing firing.
        """
        return self.cfg.injection_scale * self.expand(self.state)

    def update(self, v_mem: torch.Tensor) -> None:
        """
        Update the buffer from the membrane potential of the completed forward pass.

        ANATOMICAL INTERFACE:
            Sending structure: AssociativeLIF executive zone.
            Receiving structure: CorticalBuffer (this method).
            Connection: corticothalamic feedback, Layer 6 pyramidal axons.
            Sherman SM, Guillery RW (2002). DOI: 10.1098/rstb.2002.1161

        This update runs after each forward pass during wake phase and after
        each replay pass during sleep phase. During sleep, successive replay
        passes gradually reconsolidate the buffer toward the post-sleep
        attractor state. The final replay pass before wake-up defines the
        context for the first wake-phase forward pass.

        The update is an EMA rather than a hard overwrite:
            state <- tau * state + (1 - tau) * compress(mean(v_mem, batch))
        This means the buffer accumulates a weighted history of recent column
        states rather than being a single-step snapshot, matching the
        attractor dynamics of PFC persistent activity driven by recurrent
        NMDA reverberation. Wang XJ (2001). DOI: 10.1016/S0166-2236(00)01844-0

        The batch dimension is reduced by mean before compression. The buffer
        represents the column's aggregate working memory context across the
        current batch, not a per-sample state. Per-sample persistent state
        would require a fundamentally different design (episodic rather than
        semantic working memory) and is out of scope for this module.
        This is a deliberate architectural choice, NOT a simplification.

        Args:
            v_mem: Membrane potential tensor. Accepts either shape (d_model,)
                for single-sample inference or (B, d_model) for batched
                training. Batch dimension is averaged before compression.
                Must be detached from the computation graph before being
                passed here to prevent the buffer update from participating
                in backpropagation through time.
                NOT a biological quantity: the detachment is a training
                artifact required by BPTT.
                Neftci EO et al. (2019). DOI: 10.1109/MSP.2019.2931595
        """
        with torch.no_grad():
            v = v_mem.detach()
            if v.dim() == 2:
                v = v.mean(dim=0)
            compressed = self.compress(v)
            self.state.copy_(
                self.cfg.buffer_tau * self.state + (1.0 - self.cfg.buffer_tau) * compressed
            )

    def expand_for_new_d_model(self, new_d_model: int) -> None:
        """
        Expand the bottleneck projection layers after a d_model expansion event.

        BIOLOGICAL GROUNDING: cortical columns that widen through experience-
        dependent structural plasticity maintain their functional connectivity
        patterns while gaining new synaptic capacity.
        Holtmaat A, Svoboda K (2009). DOI: 10.1038/nrn2699

        FUNCTION-PRESERVING INITIALIZATION: new weights are set to near-zero
        with symmetry-breaking noise. The forward pass is unchanged immediately
        after expansion because new output weights (expand) contribute nothing
        at near-zero, and new input weights (compress) read nothing meaningful
        from the new d_model dimensions which are themselves near-zero in the
        expanded weight matrices throughout the column.
        Adapted from SSONN (Anonymous, under double-blind review).

        The buffer state itself (shape: buffer_dim) does not change. Only
        the projection weights change shape. This is intentional: the buffer
        provides a fixed-dimension summary of column state independent of
        internal representational capacity.

        NOT a biological quantity: function-preserving initialization is an
        engineering technique with no direct biological analog.

        Args:
            new_d_model: The new d_model after expansion. Must equal
                current d_model + expansion_increment (64).
        """
        old_d_model = self.cfg.d_model
        delta = new_d_model - old_d_model
        assert delta > 0, f"new_d_model {new_d_model} must exceed current {old_d_model}"

        # Expand compress: (buffer_dim, old_d_model) -> (buffer_dim, new_d_model)
        # New input columns read from the new d_model dimensions.
        # Initialized near-zero with symmetry-breaking noise.
        old_compress_weight = self.compress.weight.data  # (buffer_dim, old_d_model)
        new_cols = torch.zeros(
            self.cfg.buffer_dim, delta,
            device=old_compress_weight.device,
            dtype=old_compress_weight.dtype,
        )
        new_cols.uniform_(0.0, 1e-8)
        new_compress_weight = torch.cat([old_compress_weight, new_cols], dim=1)
        new_compress = nn.Linear(new_d_model, self.cfg.buffer_dim, bias=False)
        new_compress.weight = nn.Parameter(new_compress_weight)
        self.compress = new_compress

        # Expand expand: (old_d_model, buffer_dim) -> (new_d_model, buffer_dim)
        # New output rows emit into the new d_model dimensions.
        # Initialized near-zero with symmetry-breaking noise.
        old_expand_weight = self.expand.weight.data  # (old_d_model, buffer_dim)
        new_rows = torch.zeros(
            delta, self.cfg.buffer_dim,
            device=old_expand_weight.device,
            dtype=old_expand_weight.dtype,
        )
        new_rows.uniform_(0.0, 1e-8)
        new_expand_weight = torch.cat([old_expand_weight, new_rows], dim=0)
        new_expand = nn.Linear(self.cfg.buffer_dim, new_d_model, bias=False)
        new_expand.weight = nn.Parameter(new_expand_weight)
        self.expand = new_expand

        self.cfg = CorticalBufferConfig(
            d_model=new_d_model,
            buffer_dim=self.cfg.buffer_dim,
            injection_scale=self.cfg.injection_scale,
            buffer_tau=self.cfg.buffer_tau,
        )

    def reset(self) -> None:
        """
        Zero the buffer state.

        Called when a column is first initialized or when a specialist is
        cloned from Prime. The clone starts with no prior working memory
        context, matching the biological observation that newly differentiated
        cortical columns begin without specialized persistent activity patterns.
        Rakic P (1988). DOI: 10.1126/science.3291116
        """
        with torch.no_grad():
            self.state.zero_()

    def get_hot_state(self) -> dict:
        """
        Return the buffer state for HOT layer serialization.

        The buffer is runtime state, not a learned weight. It goes in the
        HOT layer of the PRAGMI serialization bridge, not the COLD layer.
        COLD contains weights (state_dict). HOT contains runtime scalars
        and buffers including STDP traces, MoE EMAs, and this buffer.

        Returns:
            Dict with 'state' key containing the buffer tensor.
        """
        return {"state": self.state.detach().cpu().clone()}

    def load_hot_state(self, hot: dict) -> None:
        """
        Restore the buffer state from a HOT layer checkpoint.

        Args:
            hot: Dict produced by get_hot_state().
        """
        with torch.no_grad():
            self.state.copy_(hot["state"].to(self.state.device))
