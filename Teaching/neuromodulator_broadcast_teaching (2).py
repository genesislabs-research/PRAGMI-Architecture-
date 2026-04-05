"""
neuromodulator_broadcast_teaching.py

BIOLOGICAL GROUNDING
====================
This file models the diffuse neuromodulatory systems of the brain: the four
major neuromodulator pathways that broadcast global chemical signals across
large brain regions simultaneously, reconfiguring the operating mode of entire
networks in response to reward, uncertainty, arousal, and behavioral context.

Unlike fast point-to-point neurotransmitters (glutamate, GABA) that act on
millisecond timescales at specific synapses, neuromodulators diffuse widely
through volume transmission and act on seconds-to-minutes timescales.
They function as real-time hyperparameters: changing how plastic synapses are,
how strongly the system attends to new input versus prior knowledge, and
modulating exploration versus exploitation trade-offs across the entire array.

The four modeled neuromodulators and their primary functions:

Dopamine (DA): Reward prediction error. Encodes the difference between
expected and received reward. Strengthens synapses for actions and outcomes
that were better than expected (LTP) and weakens them when outcomes were worse
(LTD). Originates primarily from substantia nigra and ventral tegmental area.
Schultz W (2016). DOI: 10.1038/nrn.2015.26
Izhikevich EM (2007). DOI: 10.1093/cercor/bhl152

Norepinephrine (NE): Arousal and unexpected uncertainty. Signals when
something surprising or salient occurs, increasing gain on sensory processing
and tightening the thalamic gate. Originates from the locus coeruleus.
Aston-Jones G, Cohen JD (2005). DOI: 10.1146/annurev.neuro.28.061604.135709
Yu AJ, Dayan P (2005). DOI: 10.1016/j.neuron.2005.04.026

Acetylcholine (ACh): Expected uncertainty and encoding/consolidation gating.
Controls the balance between bottom-up sensory drive and top-down expectations.
High ACh during waking: hippocampus in write mode, fast encoding. Low ACh
during sleep: hippocampus in read mode, replay and consolidation.
Hasselmo ME (2006). DOI: 10.1016/j.conb.2006.09.002

Serotonin (5-HT): Behavioral inhibition and tolerance to aversive uncertainty.
Modulates when to persist versus withdraw in ambiguous situations. Affects
output temperature and exploration-exploitation balance.
Dayan P, Huys QJM (2009). DOI: 10.1146/annurev.neuro.051508.135507

In addition to the four neuromodulators, this module owns global_maturity:
a scalar in [0, 1] representing the system's developmental stage. Maturity
is computed from four signals (routing stability, loss smoothness, probe
response, ensemble agreement) and gates structural plasticity events including
width expansion, width pruning, and specialist column spawning. Biologically
grounded in critical period closure.
Hensch TK (2005). DOI: 10.1038/nrn1787
Huang et al. (2022). DOI: 10.1007/s12021-022-09576-5

Primary papers grounding this file:

Yu AJ, Dayan P (2005). "Uncertainty, neuromodulation, and attention." Neuron,
46(4), 681-692. DOI: 10.1016/j.neuron.2005.04.026

Özçete ÖD et al. (2024). "Mechanisms of neuromodulatory volume transmission."
Molecular Psychiatry. DOI: 10.1038/s41380-024-02608-3

Hasselmo ME (2006). "The role of acetylcholine in learning and memory."
Current Opinion in Neurobiology, 16(6), 710-715.
DOI: 10.1016/j.conb.2006.09.002
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class NeuromodulatorConfig:
    """
    Configuration for NeuromodulatorBroadcast and MaturityComputer.

    Attributes:
        da_decay: EMA decay for dopamine baseline.
            NOT a biological quantity: engineering approximation of DA
            clearance timescale via reuptake transporters.
            Özçete ÖD et al. (2024). DOI: 10.1038/s41380-024-02608-3
            Value 0.995 from living document.

        ach_decay: EMA decay for acetylcholine baseline.
            NOT a biological quantity: engineering approximation.
            Value 0.98 from living document.

        ne_decay: EMA decay for norepinephrine baseline.
            NOT a biological quantity: engineering approximation.
            Value 0.99 from living document.

        ht_decay: EMA decay for serotonin baseline.
            NOT a biological quantity: engineering approximation.
            Value 0.999 from living document.

        da_baseline_init: Initial dopamine level.
            NOT a biological quantity: engineering initialization.

        ach_baseline_init: Initial acetylcholine level.
            NOT a biological quantity: engineering initialization.
            Higher than DA/NE because ACh is tonically active during waking.
            Hasselmo ME (2006). DOI: 10.1016/j.conb.2006.09.002

        ne_baseline_init: Initial norepinephrine level.
            NOT a biological quantity: engineering initialization.

        ht_baseline_init: Initial serotonin level.
            NOT a biological quantity: engineering initialization.

        tau_mature_up: EMA rate for maturity increase. Slow, because maturity
            should be hard to earn: sustained stability is required.
            NOT a biological quantity: engineering approximation of the slow
            timescale of critical period closure.
            Hensch TK (2005). DOI: 10.1038/nrn1787
            Value 0.995 from living document.

        tau_mature_down: EMA rate for maturity decrease. Faster than tau_up,
            because destabilization should be registered quickly. Easy to lose
            maturity, hard to earn it back. Asymmetry mirrors the observation
            that critical periods close slowly but can reopen rapidly under
            significant perturbation.
            NOT a biological quantity: engineering approximation.
            Value 0.95 from living document.

        maturity_component_weight: Weight of each of the four maturity
            components. All equal at 0.25. Adjust empirically.
            NOT a biological quantity: engineering hyperparameter.

        entropy_baseline_window: Number of early sleep cycles over which to
            compute the routing entropy baseline standard deviation.
            NOT a biological quantity: engineering hyperparameter.

        loss_baseline_window: Number of early sleep cycles over which to
            compute the loss CV baseline.
            NOT a biological quantity: engineering hyperparameter.

        variance_ceiling: Maximum ensemble variance used to normalize the
            ensemble_agreement_component. Prevents a single high-variance
            event from dominating maturity.
            NOT a biological quantity: engineering hyperparameter.
            [NEEDS INPUT] Placeholder 0.1 pending empirical calibration.
    """
    da_decay: float = 0.995
    ach_decay: float = 0.98
    ne_decay: float = 0.99
    ht_decay: float = 0.999
    da_baseline_init: float = 0.5
    ach_baseline_init: float = 0.7
    ne_baseline_init: float = 0.4
    ht_baseline_init: float = 0.5
    tau_mature_up: float = 0.995
    tau_mature_down: float = 0.95
    maturity_component_weight: float = 0.25
    entropy_baseline_window: int = 10
    loss_baseline_window: int = 10
    variance_ceiling: float = 0.1


class NeuromodulatorBroadcast(nn.Module):
    """
    Global neuromodulator state machine and maturity computer for the TimmyArray.

    BIOLOGICAL STRUCTURE: Diffuse neuromodulatory systems: locus coeruleus
    (NE), ventral tegmental area and substantia nigra (DA), basal forebrain
    (ACh), raphe nuclei (5-HT).
    BIOLOGICAL FUNCTION: These nuclei broadcast chemical signals that
    reconfigure the operating mode of large brain regions simultaneously.
    They are not point-to-point communicators but global modulators that
    act like real-time hyperparameters across the entire cortical network.

    Özçete ÖD et al. (2024). DOI: 10.1038/s41380-024-02608-3
    Yu AJ, Dayan P (2005). DOI: 10.1016/j.neuron.2005.04.026

    This module owns:
    - Four neuromodulator EMA baselines (DA, NE, ACh, 5-HT).
    - global_maturity scalar.
    - MaturityComputer logic (compute_maturity method).
    - Baseline statistics for maturity component normalization.
    - Wake/sleep mode flag (controlled by ACh level).

    All persistent state is stored in registered buffers so it survives
    checkpointing and device movement. The forward pass runs under
    @torch.no_grad() to prevent gradient leaks from instantaneous
    modulator signals into EMA buffer updates.

    Consumers of global_maturity (all read via broadcast.global_maturity):
    1. Width expansion eligibility (column.maturity > 0.6 gate).
    2. Width shrink eligibility (same gate).
    3. Critical period decay targeting.
    4. newborn_lr_multiplier scheduling.
    5. CuriosityHead EpistemicSelector damping.
    """

    def __init__(self, cfg: NeuromodulatorConfig) -> None:
        """
        Initialize neuromodulator baselines, maturity state, and baseline
        statistics buffers.

        Args:
            cfg: NeuromodulatorConfig.
        """
        super().__init__()
        self.cfg = cfg

        # Neuromodulator EMA baselines. Registered as buffers: runtime state,
        # not learned weights. Serialized in HOT layer.
        self.register_buffer("da", torch.tensor(cfg.da_baseline_init))
        self.register_buffer("ach", torch.tensor(cfg.ach_baseline_init))
        self.register_buffer("ne", torch.tensor(cfg.ne_baseline_init))
        self.register_buffer("ht", torch.tensor(cfg.ht_baseline_init))

        # Global maturity scalar in [0, 1].
        # Updated once per sleep cycle by compute_maturity().
        # NOT a biological quantity in this parameterization: the 0-1 scale
        # and update rate are engineering choices.
        # Biological grounding: critical period closure.
        # Hensch TK (2005). DOI: 10.1038/nrn1787
        self.register_buffer("global_maturity", torch.tensor(0.0))

        # Baseline statistics for maturity component normalization.
        # Set during early sleep cycles, held fixed thereafter.
        # NOT biological quantities: engineering normalization.
        self.register_buffer("entropy_baseline_std", torch.tensor(-1.0))
        self.register_buffer("loss_baseline_cv", torch.tensor(-1.0))

        # Rolling history lists for maturity computation.
        # Stored as plain lists (not buffers) because they contain variable-
        # length Python lists, not fixed-shape tensors.
        # Serialized in HOT layer via get_hot_state()/load_hot_state().
        self._entropy_history: list[float] = []
        self._loss_history: list[float] = []

        # Sleep cycle counter for baseline window tracking.
        # NOT a biological quantity.
        self._sleep_cycles_elapsed: int = 0

    @torch.no_grad()
    def update_da(self, signal: float) -> None:
        """
        Update the dopamine EMA baseline.

        BIOLOGICAL FUNCTION: Dopamine encodes reward prediction error (RPE).
        Phasic DA release occurs when outcomes are better than expected (LTP
        signal) and is suppressed when outcomes are worse (LTD signal). The
        EMA here approximates the tonic DA baseline against which phasic
        signals are measured.

        Schultz W (2016). "Dopamine reward prediction-error signalling: a
        two-component response." Nature Reviews Neuroscience, 17(3), 183-195.
        DOI: 10.1038/nrn.2015.26

        Args:
            signal: Instantaneous DA signal in [0, 1]. Must be detached from
                any computation graph before passing here. The @torch.no_grad()
                decorator prevents gradient accumulation into the buffer even
                if the caller forgets to detach. Failure to detach causes a
                VRAM memory leak through retained computation graphs.
                NOT a biological quantity: the detachment requirement is a
                training artifact.
        """
        self.da.copy_(self.cfg.da_decay * self.da + (1.0 - self.cfg.da_decay) * signal)

    @torch.no_grad()
    def update_ach(self, signal: float) -> None:
        """
        Update the acetylcholine EMA baseline.

        BIOLOGICAL FUNCTION: ACh from basal forebrain controls the balance
        between bottom-up sensory encoding (high ACh = write mode) and
        top-down consolidation replay (low ACh = read mode). The wake/sleep
        cycle is gated by ACh: high during waking, low during sleep.

        Hasselmo ME (2006). DOI: 10.1016/j.conb.2006.09.002

        Args:
            signal: Instantaneous ACh signal in [0, 1].
        """
        self.ach.copy_(self.cfg.ach_decay * self.ach + (1.0 - self.cfg.ach_decay) * signal)

    @torch.no_grad()
    def update_ne(self, signal: float) -> None:
        """
        Update the norepinephrine EMA baseline.

        BIOLOGICAL FUNCTION: NE from locus coeruleus signals unexpected
        uncertainty and arousal. High NE increases gain on sensory processing,
        tightens the thalamic gate, and lowers expansion thresholds. Linked
        to surprise signals from the CuriosityHead's world model prediction
        error.

        Aston-Jones G, Cohen JD (2005). DOI: 10.1146/annurev.neuro.28.061604.135709

        Args:
            signal: Instantaneous NE signal in [0, 1].
        """
        self.ne.copy_(self.cfg.ne_decay * self.ne + (1.0 - self.cfg.ne_decay) * signal)

    @torch.no_grad()
    def update_ht(self, signal: float) -> None:
        """
        Update the serotonin (5-HT) EMA baseline.

        BIOLOGICAL FUNCTION: 5-HT from raphe nuclei modulates behavioral
        inhibition and tolerance to aversive uncertainty. In this architecture
        it modulates output logit temperature and the exploration-exploitation
        balance of the active data loader.

        Dayan P, Huys QJM (2009). DOI: 10.1146/annurev.neuro.051508.135507

        Args:
            signal: Instantaneous 5-HT signal in [0, 1].
        """
        self.ht.copy_(self.cfg.ht_decay * self.ht + (1.0 - self.cfg.ht_decay) * signal)

    def is_sleep_phase(self) -> bool:
        """
        Return True if the system is currently in sleep phase.

        BIOLOGICAL FUNCTION: The wake/sleep distinction is gated by
        acetylcholine level. High ACh = waking encoding state. Low ACh =
        consolidation state. The threshold of 0.5 is a midpoint approximation.
        NOT a biological quantity: the exact threshold value is an engineering
        choice.

        Hasselmo ME (2006). DOI: 10.1016/j.conb.2006.09.002

        Returns:
            True if ach baseline is below 0.5 (sleep/consolidation mode).
        """
        return self.ach.item() < 0.5

    def compute_maturity(
        self,
        routing_entropy: float,
        loss_value: float,
        probe_response: float,
        mean_recent_variance: float,
    ) -> float:
        """
        Compute and update global_maturity. Called once per sleep cycle.

        BIOLOGICAL STRUCTURE: Critical period closure mechanism in developing
        cortex. Circuits that have stabilized their basic organization become
        less plastic, resisting perturbation.
        BIOLOGICAL FUNCTION: Mature circuits are less susceptible to
        large-scale reorganization from novel input. This protects consolidated
        knowledge while allowing gradual refinement. Critical periods can
        reopen under significant perturbation (reopening of plasticity windows).

        Hensch TK (2005). DOI: 10.1038/nrn1787
        Huang et al. (2022). DOI: 10.1007/s12021-022-09576-5

        FOUR-COMPONENT MATURITY (equal weights, 0.25 each):

        1. routing_component: 1 - normalized(std(routing_entropy_history)).
           Measures whether column utilization patterns have stabilized.
           High std = routing still shifting = low routing maturity.

        2. loss_component: 1 - normalized(CV(loss_history)).
           CV = std/mean. Measures whether the loss landscape has smoothed.
           High CV = loss still lurching = low loss maturity.

        3. probe_component: 1 - probe_response.
           probe_response is the normalized perturbation sensitivity from
           the critical period probe. Mature systems absorb perturbation
           without large representational shifts.

        4. ensemble_agreement_component: 1 - clip(variance / ceiling, 0, 1).
           High world model ensemble variance = system is uncertain about
           large portions of the manifold = low ensemble maturity.
           Separate from probe_component: a structurally mature system can
           still have high world model uncertainty in a novel domain.

        ASYMMETRIC EMA:
        Maturity increases slowly (tau_up=0.995): hard to earn.
        Maturity decreases faster (tau_down=0.95): easy to lose.
        This asymmetry mirrors the biological observation that critical periods
        close slowly through sustained stability but can reopen rapidly under
        significant perturbation.
        NOT a biological quantity: the specific tau values are engineering
        approximations.

        Args:
            routing_entropy: Current routing entropy scalar from the array.
            loss_value: Current training loss scalar.
            probe_response: Normalized perturbation response from the critical
                period probe. 0.0 = fully stable, 1.0 = fully sensitive.
            mean_recent_variance: Mean ensemble variance from
                WorldModelEnsemble.mean_recent_variance(). High = uncertain.

        Returns:
            Updated maturity scalar in [0, 1]. Also stored in
            self.global_maturity as a registered buffer.
        """
        self._sleep_cycles_elapsed += 1
        self._entropy_history.append(routing_entropy)
        self._loss_history.append(loss_value)

        # Set baseline statistics during early cycles.
        # Baselines normalize the maturity components so that early-training
        # volatility defines the reference point, not an arbitrary constant.
        # NOT a biological quantity: engineering normalization choice.
        if (self._sleep_cycles_elapsed == self.cfg.entropy_baseline_window
                and self.entropy_baseline_std.item() < 0.0):
            std_val = float(torch.tensor(self._entropy_history).std())
            self.entropy_baseline_std.copy_(
                torch.tensor(max(std_val, 1e-6))
            )

        if (self._sleep_cycles_elapsed == self.cfg.loss_baseline_window
                and self.loss_baseline_cv.item() < 0.0):
            t = torch.tensor(self._loss_history)
            cv_val = float(t.std() / (t.mean() + 1e-9))
            self.loss_baseline_cv.copy_(
                torch.tensor(max(cv_val, 1e-6))
            )

        # After baselines are set, cap history lists to the last 20 entries.
        # The maturity computation only ever reads the last 20 values
        # (routing_history[-20:] and loss_history[-20:]). Retaining the full
        # history beyond that is a memory leak in a system that runs for
        # thousands of sleep cycles. Before baselines are set, the full
        # history is retained so baseline computation has all early data.
        # NOT a biological quantity: memory management engineering choice.
        _HISTORY_CAP = 20
        if self.entropy_baseline_std.item() > 0.0 and len(self._entropy_history) > _HISTORY_CAP:
            self._entropy_history = self._entropy_history[-_HISTORY_CAP:]
        if self.loss_baseline_cv.item() > 0.0 and len(self._loss_history) > _HISTORY_CAP:
            self._loss_history = self._loss_history[-_HISTORY_CAP:]

        # Component 1: routing entropy stability.
        if (self.entropy_baseline_std.item() > 0.0
                and len(self._entropy_history) >= 2):
            recent_std = float(
                torch.tensor(self._entropy_history[-20:]).std()
            )
            routing_component = float(torch.clamp(
                torch.tensor(
                    1.0 - recent_std / self.entropy_baseline_std.item()
                ), 0.0, 1.0
            ))
        else:
            routing_component = 0.0

        # Component 2: loss smoothness via coefficient of variation.
        if (self.loss_baseline_cv.item() > 0.0
                and len(self._loss_history) >= 2):
            recent = torch.tensor(self._loss_history[-20:])
            cv = float(recent.std() / (recent.mean() + 1e-9))
            loss_component = float(torch.clamp(
                torch.tensor(1.0 - cv / self.loss_baseline_cv.item()),
                0.0, 1.0
            ))
        else:
            loss_component = 0.0

        # Component 3: probe response (perturbation sensitivity).
        # probe_response is already normalized in [0, 1] by the caller.
        # 0 = fully stable (mature), 1 = fully sensitive (immature).
        probe_component = float(torch.clamp(
            torch.tensor(1.0 - probe_response), 0.0, 1.0
        ))

        # Component 4: ensemble agreement.
        # High variance = uncertain world model = immature.
        # Normalized by variance_ceiling to prevent single events dominating.
        # [NEEDS INPUT] variance_ceiling=0.1 is a placeholder.
        ensemble_agreement_component = float(torch.clamp(
            torch.tensor(
                1.0 - mean_recent_variance / max(self.cfg.variance_ceiling, 1e-9)
            ), 0.0, 1.0
        ))

        w = self.cfg.maturity_component_weight  # 0.25
        new_maturity = (
            w * routing_component
            + w * loss_component
            + w * probe_component
            + w * ensemble_agreement_component
        )

        # Asymmetric EMA update.
        current = self.global_maturity.item()
        if new_maturity > current:
            updated = self.cfg.tau_mature_up * current + (1.0 - self.cfg.tau_mature_up) * new_maturity
        else:
            updated = self.cfg.tau_mature_down * current + (1.0 - self.cfg.tau_mature_down) * new_maturity

        updated = float(torch.clamp(torch.tensor(updated), 0.0, 1.0))
        with torch.no_grad():
            self.global_maturity.copy_(torch.tensor(updated))

        return updated

    def get_hot_state(self) -> dict:
        """
        Return runtime state for HOT layer serialization.

        Neuromodulator baselines, global_maturity, and baseline statistics
        are registered buffers and are already captured by state_dict().
        This method captures the variable-length history lists which cannot
        be stored as fixed-shape tensors.

        Returns:
            Dict with history lists, sleep cycle counter, and baseline flags.
        """
        return {
            "entropy_history": list(self._entropy_history),
            "loss_history": list(self._loss_history),
            "sleep_cycles_elapsed": self._sleep_cycles_elapsed,
        }

    def load_hot_state(self, hot: dict) -> None:
        """
        Restore runtime state from a HOT layer checkpoint.

        Args:
            hot: Dict produced by get_hot_state().
        """
        self._entropy_history = list(hot.get("entropy_history", []))
        self._loss_history = list(hot.get("loss_history", []))
        self._sleep_cycles_elapsed = hot.get("sleep_cycles_elapsed", 0)
