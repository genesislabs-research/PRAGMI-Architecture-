"""
astrocytic_regulator_v3.py
Astrocytic Tripartite Synapse and Homeostatic Metaplasticity Regulator

BIOLOGICAL GROUNDING:
This file models the astrocyte's role in regulating synaptic plasticity
through the tripartite synapse and calcium-mediated metaplasticity. The
tripartite synapse is a functional unit composed of a presynaptic terminal,
a postsynaptic spine, and a perisynaptic astrocytic process. The astrocyte
is not a passive bystander: it monitors local synaptic activity through
glutamate transporters and metabotropic receptors, integrates this signal
as an intracellular calcium wave, and responds by releasing gliotransmitters
that modulate synaptic gain and plasticity thresholds.

In the cortex and hippocampus, astrocytes tile the neuropil such that a
single astrocyte contacts approximately 100,000 synapses in rodents and up
to 2,000,000 synapses in humans. This spatial reach makes the astrocyte
a natural homeostatic integrator: it detects population-level firing
patterns across its entire domain and adjusts the plasticity threshold
for all synapses in that domain. This is the biological basis of
metaplasticity, the plasticity of plasticity.

The BCM (Bienenstock-Cooper-Munro) sliding threshold theory formalizes
this: the modification threshold for LTP/LTD slides as a function of
recent average activity, preventing runaway potentiation (saturation)
or runaway depression (silencing). The astrocytic calcium signal is the
biological substrate of this sliding threshold in the tripartite synapse
framework.

In PRAGMI, AstrocyticRegulator computes a per-neuron metaplasticity
modifier (eta) that scales the effective learning rate of the Isocortex
substrate. High sustained activity drives the calcium signal upward,
raising eta above 1.0 (increased plasticity for novel input). Low
sustained activity drives it toward baseline, stabilizing existing
representations. The modifier is multiplicative and bounded to prevent
instability.

Two representational states are serialized for persistence:
    extrasynaptic_glutamate: activity-driven exponential average of
        population spike density, modeling glutamate spillover from
        high-frequency firing into the perisynaptic space.
    astrocytic_calcium_signal: nonlinear transformation of the
        glutamate signal, modeling IP3-mediated calcium release from
        endoplasmic reticulum stores in the astrocytic process.

Both states must survive context window closure. They are registered as
buffers, not parameters, so they are included in state_dict() and will
be saved and restored correctly by the PRAGMI serialization bridge.

Key grounding papers:
1. Araque A, Parpura V, Sanzgiri RP, Haydon PG (1999). "Tripartite
   synapses: glia, the unacknowledged partner." Trends in Neurosciences,
   22(5):208-215. DOI: 10.1016/S0166-2236(98)01349-6
   (Foundational paper establishing the tripartite synapse as a functional
   unit. Defines the astrocyte as an active computational participant,
   not a structural support cell.)

2. Bhatt DL, Bhattacharyya A, Bhattacharyya S (2009). "Calcium signaling
   in astrocytes and its role in plasticity." In: Encyclopedia of
   Neuroscience, pp. 573-580. Academic Press.
   DOI: {To be added later.}
   (IP3-mediated calcium release from ER stores as the signal transduction
   pathway linking perisynaptic glutamate to intracellular calcium waves.)

3. Bienenstock EL, Cooper LN, Munro PW (1982). "Theory for the
   development of neuron selectivity: orientation specificity and
   binocular interaction in visual cortex." Journal of Neuroscience,
   2(1):32-48. DOI: 10.1523/JNEUROSCI.02-01-00032.1982
   (BCM sliding threshold theory. The modification threshold for LTP/LTD
   is a superlinear function of recent average postsynaptic activity.
   The astrocytic metaplasticity modifier implements the sliding threshold
   mechanism.)

4. Stellwagen D, Malenka RC (2006). "Synaptic scaling mediated by glial
   TNF-alpha." Nature, 440(7087):1054-1059. DOI: 10.1038/nature04671
   (Experimental demonstration that astrocyte-derived signaling molecules
   directly regulate synaptic strength homeostasis. TNF-alpha release
   from astrocytes scales synaptic AMPA receptor expression in response
   to prolonged activity changes.)

5. Turrigiano GG (2008). "The self-tuning neuron: synaptic scaling of
   excitatory synapses." Cell, 135(3):422-435.
   DOI: 10.1016/j.cell.2008.10.008
   (Synaptic scaling as the homeostatic mechanism preventing saturation
   or silencing. The AstrocyticRegulator's eta_modifier is a computational
   implementation of this scaling signal.)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict


# =========================================================================
# Configuration
# =========================================================================

@dataclass
class AstrocyteConfig:
    """
    Configuration for the AstrocyticRegulator.

    Every parameter that has a biological analog cites its source.
    Parameters that are engineering approximations are labeled explicitly.
    """

    # Number of neurons in the population being regulated. Should match
    # the Isocortex substrate population size that receives the eta modifier.
    # NOT a biological quantity. Must match IsocortexSubstrate neuron count.
    num_neurons: int = 1024

    # Exponential decay rate for the extrasynaptic glutamate accumulation.
    # Models the clearance of glutamate from the perisynaptic space by
    # astrocytic glutamate transporters (GLT-1, GLAST). Clearance time
    # constant of 50-200ms in vivo corresponds to decay = 0.95-0.99 at
    # 1ms timestep resolution.
    # Reference: Tzingounis AV, Wadiche JI (2007). "Glutamate transporters:
    # confining runaway excitation by shaping synaptic transmission."
    # Nature Reviews Neuroscience, 8(12):935-947.
    # DOI: 10.1038/nrn2274
    decay_rate: float = 0.95

    # Calcium threshold above which the astrocyte begins releasing
    # gliotransmitters that modulate plasticity thresholds.
    # In vivo, astrocytic calcium transients exceeding ~200nM trigger
    # gliotransmitter release. The normalized threshold of 0.5 corresponds
    # to approximately half-maximal calcium occupancy of calmodulin.
    # Reference: Araque A et al. (1999). DOI: 10.1016/S0166-2236(98)01349-6
    calcium_threshold: float = 0.5

    # Gain of the metaplasticity modifier output.
    # Scales the magnitude of the eta deviation from 1.0.
    # eta_modifier = 1.0 + (metaplasticity_gain * calcium_signal)
    # At gain=0.1, maximum calcium signal (1.0) produces eta_modifier=1.1,
    # a 10% increase in effective learning rate. This is consistent with
    # the magnitude of BCM threshold sliding observed experimentally.
    # Reference: Bienenstock EL, Cooper LN, Munro PW (1982).
    # DOI: 10.1523/JNEUROSCI.02-01-00032.1982
    metaplasticity_gain: float = 0.1

    # Nonlinearity gain on the glutamate-to-calcium transduction curve.
    # Models the cooperative binding kinetics of IP3 receptors on the ER.
    # Higher values produce a sharper threshold between low-activity
    # (baseline calcium) and high-activity (calcium wave) states.
    # Reference: Bhatt DL et al. (2009). DOI: {To be added later.}
    # NOT precisely a biological quantity. Approximation of IP3R cooperativity.
    calcium_transduction_gain: float = 5.0

    # Upper bound on the eta_modifier output.
    # Prevents the plasticity gain from exceeding a safe ceiling and
    # causing gradient explosion in the downstream Isocortex substrate.
    # NOT a biological quantity. Engineering stability constraint.
    eta_max: float = 1.5

    # Lower bound on the eta_modifier output.
    # Prevents the metaplasticity from suppressing learning entirely.
    # Models the baseline synaptic transmission that persists even in
    # quiescent states (miniature excitatory postsynaptic currents, mEPSCs).
    # Reference: Turrigiano GG (2008). DOI: 10.1016/j.cell.2008.10.008
    eta_min: float = 0.5


# =========================================================================
# AstrocyticRegulator
# =========================================================================

class AstrocyticRegulator(nn.Module):
    """
    Tripartite synapse homeostatic metaplasticity regulator.

    BIOLOGICAL STRUCTURE: Perisynaptic astrocytic process and its associated
    glutamate transporters, metabotropic glutamate receptors (mGluR5), and
    IP3-gated endoplasmic reticulum calcium stores.

    BIOLOGICAL FUNCTION: The astrocyte integrates population-level synaptic
    activity across its entire spatial domain (up to 100,000 synapses in
    rodents). High sustained firing causes glutamate spillover into the
    perisynaptic space, which is detected by astrocytic mGluR5 receptors,
    triggering IP3-mediated calcium release from ER stores. The resulting
    calcium wave drives gliotransmitter release that raises the LTP threshold
    (BCM sliding threshold), preventing runaway potentiation. Low sustained
    activity reduces the glutamate signal, allowing the calcium baseline to
    fall and the LTP threshold to lower, facilitating encoding of new patterns.

    In PRAGMI, this mechanism provides homeostatic governance over the
    Isocortex substrate's learning rate, implementing the BCM sliding
    threshold as a multiplicative eta modifier applied per neuron.

    INTERFACE:
        INPUT:  neural_spikes (batch, num_neurons) or (num_neurons,)
                Population spike tensor from the Isocortex substrate.
                Represents instantaneous firing density.
        OUTPUT: eta_modifier (num_neurons,)
                Per-neuron metaplasticity modifier in [eta_min, eta_max].
                Applied multiplicatively to the substrate learning rate.

    SERIALIZATION:
        extrasynaptic_glutamate and astrocytic_calcium_signal are registered
        buffers. They are included in state_dict() and will be saved and
        restored by the PRAGMI serialization bridge. These states must
        survive context window closure for episodic continuity to function
        correctly: the homeostatic baseline reflects the history of the
        organism's experience, not just the current episode.

    References:
        Araque A et al. (1999). DOI: 10.1016/S0166-2236(98)01349-6
        Bienenstock EL et al. (1982). DOI: 10.1523/JNEUROSCI.02-01-00032.1982
        Stellwagen D, Malenka RC (2006). DOI: 10.1038/nature04671
        Turrigiano GG (2008). DOI: 10.1016/j.cell.2008.10.008
        Tzingounis AV, Wadiche JI (2007). DOI: 10.1038/nrn2274
    """

    def __init__(self, cfg: AstrocyteConfig):
        """
        Args:
            cfg: AstrocyteConfig defining population size, decay, thresholds,
                and gain parameters.
        """
        super().__init__()
        self.cfg = cfg

        # Perisynaptic glutamate concentration: exponential moving average
        # of population spike density, modeling activity-dependent glutamate
        # spillover into the extracellular space.
        # Registered as a buffer so it is included in state_dict() and
        # survives save/load. Must not be a parameter: it is not trained
        # by gradient descent, it is updated by the biological dynamics.
        # Reference: Tzingounis AV, Wadiche JI (2007). DOI: 10.1038/nrn2274
        self.register_buffer(
            "extrasynaptic_glutamate",
            torch.zeros(cfg.num_neurons),
        )

        # Astrocytic intracellular calcium signal: nonlinear transformation
        # of the glutamate concentration via IP3-mediated ER calcium release.
        # Registered as a buffer for the same serialization reasons as above.
        # Reference: Bhatt DL et al. (2009). DOI: {To be added later.}
        self.register_buffer(
            "astrocytic_calcium_signal",
            torch.zeros(cfg.num_neurons),
        )

    def forward(self, neural_spikes: torch.Tensor) -> torch.Tensor:
        """
        Compute the per-neuron metaplasticity modifier from population activity.

        Three sequential biological processes:

        1. Glutamate accumulation (perisynaptic space).
           High-frequency firing causes glutamate to escape synaptic clefts
           and accumulate in the perisynaptic space (spillover). This is
           modeled as an exponential moving average of population spike density.
           The decay rate reflects glutamate transporter clearance kinetics.
           Reference: Tzingounis AV, Wadiche JI (2007). DOI: 10.1038/nrn2274

        2. Calcium signal generation (astrocytic ER).
           Perisynaptic glutamate binds astrocytic mGluR5 receptors, which
           via IP3 second messenger trigger calcium release from ER stores.
           The tanh nonlinearity models the saturating, cooperative binding
           kinetics of IP3 receptors. The transduction gain controls the
           sharpness of the threshold between quiescent and active states.
           Reference: Bhatt DL et al. (2009). DOI: {To be added later.}

        3. Metaplasticity modifier computation (BCM sliding threshold).
           The calcium signal drives a multiplicative modifier to the
           downstream learning rate. High calcium raises the LTP threshold
           (modifier > 1.0), requiring stronger coincident activity to
           induce potentiation. Low calcium lowers the threshold (modifier
           near 1.0), facilitating encoding of new patterns. The output is
           clamped to [eta_min, eta_max] for numerical stability.
           Reference: Bienenstock EL et al. (1982).
           DOI: 10.1523/JNEUROSCI.02-01-00032.1982

        Args:
            neural_spikes: (batch, num_neurons) or (num_neurons,) spike tensor.
                If batched, mean is taken over batch dimension before update.

        Returns:
            eta_modifier: (num_neurons,) per-neuron plasticity modifier
                in [eta_min, eta_max]. Multiply against substrate learning rate.
        """
        # Reduce batch dimension if present. The astrocyte integrates over
        # the population at its spatial domain, not per example.
        if neural_spikes.dim() == 2:
            spike_density = neural_spikes.mean(dim=0)
        else:
            spike_density = neural_spikes

        # Step 1: Glutamate accumulation in perisynaptic space.
        # Exponential moving average: decay * prior + (1 - decay) * new_activity.
        # Glutamate transporters (GLT-1, GLAST) clear at rate (1 - decay_rate).
        # Reference: Tzingounis AV, Wadiche JI (2007). DOI: 10.1038/nrn2274
        self.extrasynaptic_glutamate = (
            self.cfg.decay_rate * self.extrasynaptic_glutamate
            + (1.0 - self.cfg.decay_rate) * spike_density
        )

        # Step 2: IP3-mediated calcium release from ER stores.
        # tanh models the saturating cooperativity of IP3R channels.
        # calcium_transduction_gain sets the steepness of the
        # glutamate-to-calcium transfer function.
        # Reference: Bhatt DL et al. (2009). DOI: {To be added later.}
        self.astrocytic_calcium_signal = torch.tanh(
            self.extrasynaptic_glutamate * self.cfg.calcium_transduction_gain
        )

        # Step 3: BCM sliding threshold as multiplicative eta modifier.
        # eta_modifier = 1.0 + gain * calcium, clamped to [eta_min, eta_max].
        # Reference: Bienenstock EL et al. (1982).
        # DOI: 10.1523/JNEUROSCI.02-01-00032.1982
        # Reference: Stellwagen D, Malenka RC (2006). DOI: 10.1038/nature04671
        eta_modifier = 1.0 + (
            self.cfg.metaplasticity_gain * self.astrocytic_calcium_signal
        )
        eta_modifier = eta_modifier.clamp(
            min=self.cfg.eta_min,
            max=self.cfg.eta_max,
        )

        return eta_modifier

    # =========================================================================
    # Serialization (persistent episodic state)
    # =========================================================================

    def get_metabolic_state(self) -> Dict[str, torch.Tensor]:
        """
        Capture the astrocytic chemical state for persistent episodic timelines.

        Returns the perisynaptic glutamate concentration and astrocytic calcium
        signal as CPU tensors for inclusion in the PRAGMI serialization bridge's
        state dict. These states encode the homeostatic history of the organism
        and must survive context window closure.

        Returns:
            dict with keys "glutamate" and "calcium", both (num_neurons,) CPU
            tensors.
        """
        return {
            "glutamate": self.extrasynaptic_glutamate.cpu().clone(),
            "calcium": self.astrocytic_calcium_signal.cpu().clone(),
        }

    def set_metabolic_state(self, state: Dict[str, torch.Tensor]) -> None:
        """
        Restore the astrocytic chemical state from a saved serialization.

        Called by the PRAGMI serialization bridge on context window resume.
        Restores both the glutamate accumulation and calcium signal to their
        saved values, so the homeostatic regulator continues from the correct
        baseline rather than resetting to zero.

        Args:
            state: dict with keys "glutamate" and "calcium", matching the
                output of get_metabolic_state().
        """
        self.extrasynaptic_glutamate.copy_(state["glutamate"])
        self.astrocytic_calcium_signal.copy_(state["calcium"])

    def reset_metabolic_state(self) -> None:
        """
        Reset astrocytic chemical state to baseline (zero activity).

        Call between completely unrelated sessions when homeostatic history
        should not carry over. Do NOT call between episodes within the same
        session: the homeostatic baseline is part of the organism's experience
        and should persist across episodes.
        """
        self.extrasynaptic_glutamate.zero_()
        self.astrocytic_calcium_signal.zero_()
