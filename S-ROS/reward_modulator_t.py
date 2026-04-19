"""
reward_modulator_t.py
Teaching implementation of the dopamine reward prediction error (RPE) module
for S-ROS / Tiny Tim STDP modulation.

BIOLOGICAL GROUNDING
====================
This file models the mesolimbic dopamine system, specifically the ventral
tegmental area (VTA) and its projections to the nucleus accumbens and
prefrontal cortex. VTA dopaminergic neurons do not simply signal reward;
they signal the *error* between predicted and received reward. When an
outcome is better than expected, these neurons fire above baseline (positive
RPE). When an outcome is worse than expected, they pause below baseline
(negative RPE). When an outcome perfectly matches prediction, firing is
unchanged. This RPE signal acts as a global teaching signal that is broadcast
diffusely and modulates the magnitude of concurrent synaptic weight changes
at recently active synapses, implementing what is now called reward-modulated
spike-timing-dependent plasticity (R-STDP).

In this implementation, the RewardModulator module stands in for VTA.
It receives scalar reward signals from the game environment (via Theo's
spike bus output), computes d (the RPE scalar), and the result is used
by the caller to gate Tiny Tim's STDP delta_w before weight commit.
CrystallizationManager (Neo) watches the *post-modulation* weight deltas,
which is correct: crystallization should track the modulated learning
trajectory, not raw STDP.

PRIMARY CITATIONS:
    Schultz W, Dayan P, Montague PR (1997). "A neural substrate of
    prediction and reward." Science, 275(5306), 1593-1599.
    DOI: 10.1126/science.275.5306.1593
    [Primary source for the RPE framework and VTA dopamine encoding.]

    Rescorla RA, Wagner AR (1972). "A theory of Pavlovian conditioning:
    Variations in the effectiveness of reinforcement and nonreinforcement."
    In Black AH, Prokasy WF (Eds.), Classical Conditioning II: Current
    Research and Theory. Appleton-Century-Crofts, 64-99.
    DOI: {To be added later.}
    [Primary source for the prediction-error learning rule this module
    instantiates computationally.]

    Frey U, Morris RGM (1997). "Synaptic tagging and long-term
    potentiation." Nature, 385(6616), 533-536. DOI: 10.1038/385533a0
    [Primary source for the eligibility trace / synaptic tag mechanism.]

SECONDARY / REVIEW:
    Schultz W (2016). "Dopamine reward prediction-error signalling: A
    two-component response." Nature Reviews Neuroscience, 17(3), 183-195.
    DOI: 10.1038/nrn.2015.26
    [Comprehensive review; useful for orientation but 1997 paper is lead
    citation for this module per project standard.]

ENGINEERING NOTES:
    - reward_baseline and eligibility_trace are nn.Module registered buffers
      (state_dict-serializable, optimizer-excluded). They are NOT nn.Parameters.
    - Multiplicative gating (1 + d) is an engineering approximation of the
      neuromodulatory effect, not a directly measured biological quantity.
    - All ablation flags are engineering controls, not biological quantities.
    - This is a teaching file (_t.py). No stripped _p.py variant is created here.
"""

import dataclasses
import math
import torch
import torch.nn as nn
from typing import List


# =============================================================================
# Part 2: RewardModulatorConfig
# =============================================================================

@dataclasses.dataclass
class RewardModulatorConfig:
    """
    Hyperparameter and ablation-flag container for RewardModulator.

    Every field is either a biological quantity (approximated) or an
    engineering parameter; each is labeled explicitly below. Ablation flags
    independently disable individual mechanisms so that each component of
    the DA-RPE loop can be tested in isolation.
    """

    # ------------------------------------------------------------------
    # Biological / approximated-biological quantities
    # ------------------------------------------------------------------

    baseline_decay: float = 0.95
    """
    Exponential moving average (EMA) decay for the running expected-reward
    baseline (V_total analog).

    Biological structure: Dopaminergic neurons in the ventral tegmental area
    (VTA) maintain a running expectation of reward that is subtracted from
    the actual reward to produce the RPE signal. The baseline represents the
    brain's learned prediction of reward magnitude, and only deviations from
    that prediction drive strong dopamine release. This is sometimes called
    metaplasticity: the reference point shifts so that habituation to repeated
    rewards reduces the RPE toward zero.

    NOT a directly measured biological timescale. EMA with this decay is an
    engineering approximation of the VTA's reward-expectation tracking dynamics.
    The value 0.95 is an engineering starting point, not empirically derived.

    Schultz W, Dayan P, Montague PR (1997). "A neural substrate of prediction
    and reward." Science, 275(5306), 1593-1599.
    DOI: 10.1126/science.275.5306.1593
    """

    eligibility_trace_decay: float = 0.9
    """
    Per-step exponential decay factor for the synaptic eligibility trace.

    Biological structure: Synaptic eligibility traces (also called "synaptic
    tags") are transient molecular marks left at recently active synapses.
    When a synapse participates in pre-before-post spiking, it is transiently
    "tagged." If a neuromodulatory signal (dopamine) arrives while the tag is
    still present, that synapse is selected for long-term potentiation. The
    tag decays in the absence of a reinforcing signal, which is why this
    mechanism bridges the temporal gap between action and delayed reward.

    NOT a directly measured biological timescale. 0.9 is an engineering
    starting point pending calibration from training-run trajectories.

    Frey U, Morris RGM (1997). "Synaptic tagging and long-term potentiation."
    Nature, 385(6616), 533-536. DOI: 10.1038/385533a0
    """

    da_clip_value: float = 1.0
    """
    Symmetric clip bound on the raw RPE scalar before saturation nonlinearity.

    Biological structure: VTA dopaminergic neurons have a maximum firing rate
    ceiling (~80 Hz physiological max) and a floor at zero Hz (neurons cannot
    fire at negative rates, though a pause below tonic baseline encodes
    negative RPE). The clip approximates this physiological bounded range.

    NOT a biological quantity in this exact parameterization. The value 1.0
    is an engineering hyperparameter defining the input range to tanh saturation.

    Schultz W, Dayan P, Montague PR (1997). DOI: 10.1126/science.275.5306.1593
    """

    da_saturation_sharpness: float = 3.0
    """
    Slope parameter of the soft saturation nonlinearity: tanh(sharpness * d).

    Applied after clipping to model the compressive nonlinearity of VTA
    dopamine encoding. Higher values approach a hard clip; lower values
    produce a smoother response curve. Biological motivation: DA release does
    not scale linearly with RPE at high magnitudes.

    NOT a biological quantity. Engineering hyperparameter. Value 3.0 is
    an engineering starting point, not empirically derived.
    """

    # ------------------------------------------------------------------
    # Ablation flags (engineering controls, not biological quantities)
    # ------------------------------------------------------------------

    use_reward_modulation: bool = True
    """
    Master ablation flag.

    When False, d is returned as 0.0 from compute_rpe and delta_w is returned
    unchanged from apply. Tiny Tim runs pure unmodulated STDP. This is the
    baseline ablation that isolates the total contribution of the DA-RPE loop
    to learning speed and crystallization trajectory. All other flags are
    irrelevant when this is False.

    Engineering control. Not a biological quantity.
    """

    use_prediction: bool = True
    """
    When True, RPE = actual - predicted (standard prediction error).
    When False, the forward-model prediction is bypassed and RPE = actual
    (raw reward signal with no expectation subtraction). Tests whether the
    prediction component of RPE contributes beyond simple reward gating.
    When False, compute_rpe ignores the predicted argument and treats
    expected reward as zero before baseline subtraction.

    Engineering control. Not a biological quantity.
    """

    use_eligibility_trace: bool = True
    """
    When True, the eligibility trace accumulates recent spike products and
    gates modulation: only synapses with active traces receive full d weighting.
    When False, the eligibility trace is bypassed and d is applied uniformly
    to all elements of delta_w as delta_w * (1 + d). Tests whether temporal
    credit assignment across the action-to-outcome delay is necessary.

    Engineering control. Not a biological quantity.
    """

    use_da_saturation: bool = True
    """
    When True, d is clipped to [-da_clip_value, da_clip_value] and passed
    through tanh(da_saturation_sharpness * d). When False, d is used raw
    with no clipping or nonlinearity. Tests whether saturation matters for
    training stability; with False, large reward signals can produce d >> 1.

    Engineering control. Not a biological quantity.
    """

    use_baseline_subtraction: bool = True
    """
    When True, the running EMA baseline is subtracted from raw RPE, so that
    repeated identical rewards gradually produce smaller d as expectation rises.
    When False, the baseline buffer is held at zero and never updated; every
    non-zero reward produces a positive d regardless of prior expectation.
    This is the purest ablation of the prediction-error structure: tests
    whether subtracting expectation matters beyond reward presence/absence.

    Engineering control. Not a biological quantity.
    """


# =============================================================================
# Part 3: RewardModulator class definition, __init__, and compute_rpe
# =============================================================================

class RewardModulator(nn.Module):
    """
    Dopamine reward prediction error (RPE) module for reward-modulated STDP.

    BIOLOGICAL STRUCTURE:
        The mesolimbic dopamine system, specifically the projection from ventral
        tegmental area (VTA) to nucleus accumbens and prefrontal cortex.

    BIOLOGICAL FUNCTION:
        VTA dopaminergic neurons compute a scalar signal encoding the difference
        between received reward and predicted reward. When outcomes exceed
        prediction, these neurons fire above their tonic baseline, broadcasting
        a positive RPE. When outcomes fall short, they pause below baseline,
        broadcasting a negative RPE. This signal is diffuse: it reaches many
        synaptic targets simultaneously. At each recently-active synapse (one
        carrying a synaptic tag / eligibility trace), the dopamine signal
        selectively determines whether the tag is converted to a lasting
        weight change. This implements reward-modulated spike-timing-dependent
        plasticity (R-STDP).

    Schultz W, Dayan P, Montague PR (1997). "A neural substrate of prediction
    and reward." Science, 275(5306), 1593-1599.
    DOI: 10.1126/science.275.5306.1593

    ANATOMICAL INTERFACE (inputs):
        Sending structure: Game environment feedback (score delta, win/lose
            event) decoded from Theo's spike bus output.
        Receiving structure: RewardModulator (this module), standing in for VTA.
        Connection: No direct anatomical analog; this is an engineering
            interface. Reward signal passed as a Python float each game tick.

    ANATOMICAL INTERFACE (outputs):
        Sending structure: RewardModulator (VTA analog).
        Receiving structure: Tiny Tim STDP update function (caller's
            responsibility; see call-site wiring spec, issued separately).
        Connection: Mesolimbic dopamine projection. The scalar d modulates
            weight update magnitude at recently active (tagged) synapses.
        Schultz W et al. (1997). DOI: 10.1126/science.275.5306.1593

    REGISTERED BUFFERS (state_dict-serializable, NOT optimizer parameters):
        reward_baseline: scalar float tensor, running EMA of received rewards.
            Biological analog: VTA reward expectation.
        eligibility_trace: tensor matching the shape of delta_w. Initialized as
            scalar zero; lazily reshaped on first apply() call.
            Biological structure: synaptic eligibility trace / synaptic tag.
            Frey U, Morris RGM (1997). DOI: 10.1038/385533a0
    """

    def __init__(self, cfg: RewardModulatorConfig) -> None:
        """
        Construct a RewardModulator from a RewardModulatorConfig.

        Args:
            cfg: Fully specified configuration dataclass. No defaults are
                 applied here; all defaults live in RewardModulatorConfig.
        """
        super().__init__()
        self.cfg = cfg

        # Biological analog: VTA running reward expectation.
        # Registered as buffer: persisted in state_dict, excluded from optimizer.
        # NOT nn.Parameter.
        self.register_buffer("reward_baseline", torch.tensor(0.0))

        # Biological structure: synaptic eligibility trace / synaptic tag.
        # Initialized as scalar zero; reshaped lazily on first apply() call
        # once the shape of delta_w is known.
        # NOT nn.Parameter.
        # Frey U, Morris RGM (1997). DOI: 10.1038/385533a0
        self.register_buffer("eligibility_trace", torch.tensor(0.0))

    def compute_rpe(self, predicted: float, actual: float) -> float:
        """
        Compute the dopamine reward prediction error (RPE) scalar d.

        BIOLOGICAL STRUCTURE:
            Ventral tegmental area (VTA) dopaminergic neurons.

        BIOLOGICAL FUNCTION:
            VTA dopaminergic neurons compare received reward against their
            running prediction and emit a signed scalar error signal. Firing
            above tonic baseline encodes positive RPE (outcome better than
            predicted). Pausing below tonic baseline encodes negative RPE
            (outcome worse than predicted). Tonic firing with no change
            encodes RPE = 0 (perfect prediction). This is the central
            computation of temporal difference learning as it manifests in
            the brain's reward circuitry.

        Schultz W, Dayan P, Montague PR (1997). "A neural substrate of
        prediction and reward." Science, 275(5306), 1593-1599.
        DOI: 10.1126/science.275.5306.1593

        Rescorla RA, Wagner AR (1972). "A theory of Pavlovian conditioning:
        Variations in the effectiveness of reinforcement and nonreinforcement."
        In Black AH, Prokasy WF (Eds.), Classical Conditioning II: Current
        Research and Theory. Appleton-Century-Crofts, 64-99.
        DOI: {To be added later.}

        Args:
            predicted: The outcome predicted by Tiny Tim's forward model
                       for the current game state. Ignored when
                       cfg.use_prediction is False.
            actual:    The actual reward received from the game environment
                       this tick.

        Returns:
            d: Python float. The modulated RPE scalar. Returned as 0.0
               when use_reward_modulation is False.
        """
        # Step 1: Master ablation gate.
        if not self.cfg.use_reward_modulation:
            return 0.0

        # Step 2: Raw RPE computation.
        # Biological analog: VTA neurons compute actual - predicted.
        # When use_prediction is False, the forward model is bypassed;
        # raw_rpe = actual (reward presence/absence only).
        if self.cfg.use_prediction:
            raw_rpe: float = actual - predicted
        else:
            raw_rpe = actual

        # Step 3: Baseline subtraction.
        # Biological analog: VTA reward expectation is subtracted so that
        # habituation to repeated identical rewards reduces d toward zero.
        # When use_baseline_subtraction is False, the baseline is held at
        # zero and never updated, so every non-zero reward produces positive d.
        if self.cfg.use_baseline_subtraction:
            self.reward_baseline = (
                self.cfg.baseline_decay * self.reward_baseline
                + (1.0 - self.cfg.baseline_decay) * actual
            )
            raw_rpe = raw_rpe - self.reward_baseline.item()
        # If False: leave raw_rpe unchanged; do not update reward_baseline.

        # Step 4: Saturation nonlinearity.
        # Biological analog: VTA firing rate is bounded; tanh approximates
        # the compressive nonlinearity of dopamine encoding at high magnitudes.
        # When use_da_saturation is False, d is used raw (can exceed 1.0).
        # NOT a biological quantity in this exact form; engineering approximation.
        if self.cfg.use_da_saturation:
            clipped = max(
                -self.cfg.da_clip_value,
                min(self.cfg.da_clip_value, raw_rpe)
            )
            d: float = math.tanh(self.cfg.da_saturation_sharpness * clipped)
        else:
            d = raw_rpe

        return d


# =============================================================================
# Part 4: RewardModulator.apply, reset_trace, reset_baseline
# =============================================================================

    def apply(self, delta_w: torch.Tensor, d: float) -> torch.Tensor:
        """
        Gate the STDP weight update tensor by the dopamine RPE scalar.

        BIOLOGICAL STRUCTURE:
            Mesolimbic dopamine projection from VTA to striatum (nucleus
            accumbens) and prefrontal cortex.

        BIOLOGICAL FUNCTION:
            The dopamine RPE signal gates plasticity at recently active
            synapses via the synaptic tagging mechanism. A synapse that
            recently participated in coincident pre-post spiking carries
            a transient eligibility trace (synaptic tag). When the dopamine
            signal arrives, only tagged synapses receive the full modulatory
            effect; the trace weight determines how much of the current DA
            signal each synapse absorbs. This implements temporal credit
            assignment: the trace bridges the delay between the action that
            caused the outcome and the delayed reward signal.

        Update equation (trace active):
            delta_w_final = delta_w * (1 + d * eligibility_trace)

        Update equation (trace disabled, use_eligibility_trace=False):
            delta_w_final = delta_w * (1 + d)

        Frey U, Morris RGM (1997). "Synaptic tagging and long-term
        potentiation." Nature, 385(6616), 533-536. DOI: 10.1038/385533a0

        Schultz W, Dayan P, Montague PR (1997). "A neural substrate of
        prediction and reward." Science, 275(5306), 1593-1599.
        DOI: 10.1126/science.275.5306.1593

        NOT a biological quantity in this exact form: multiplicative gating
        (1 + d) is an engineering approximation of the neuromodulatory effect
        on synaptic weight change magnitude.
        PRAGMI Complete Reference (April 18, 2026), Equation 17.

        Args:
            delta_w: Weight update tensor produced by Tiny Tim's STDP rule,
                     pre-commit. Shape must be consistent across calls once
                     the trace has been initialized.
            d:       RPE scalar produced by compute_rpe. Pass 0.0 to apply
                     a no-op modulation (delta_w unchanged).

        Returns:
            Modulated weight update tensor, same shape as delta_w.
            Returns delta_w unchanged when use_reward_modulation is False.
        """
        # Step 1: Master ablation gate.
        if not self.cfg.use_reward_modulation:
            return delta_w

        # Step 2: Lazy trace initialization.
        # On the first call, eligibility_trace is a scalar zero tensor.
        # Re-register with the correct shape once delta_w shape is known.
        # On subsequent calls with the same shape, this branch is skipped.
        if self.eligibility_trace.shape != delta_w.shape:
            self.register_buffer(
                "eligibility_trace",
                torch.zeros_like(delta_w)
            )

        # Step 3: Apply eligibility trace or bypass it.
        if self.cfg.use_eligibility_trace:
            # Biological analog: synaptic tag accumulation.
            # Trace accumulates recent |delta_w| products and decays,
            # bridging the action-to-reward temporal gap.
            # Frey U, Morris RGM (1997). DOI: 10.1038/385533a0
            # NOT a biological quantity in this parameterization; engineering
            # approximation of tag accumulation dynamics.
            self.eligibility_trace = (
                self.cfg.eligibility_trace_decay * self.eligibility_trace
                + (1.0 - self.cfg.eligibility_trace_decay) * delta_w.abs()
            )
            return delta_w * (1.0 + d * self.eligibility_trace)
        else:
            # Trace disabled: uniform d applied to all elements.
            # Do NOT update eligibility_trace buffer.
            return delta_w * (1.0 + d)

    def reset_trace(self) -> None:
        """
        Zero the eligibility trace buffer in place.

        Biological analog: Synaptic tags decay between unrelated behavioral
        episodes. A tag that was set during one rally or game episode should
        not influence plasticity in the next unrelated episode; the molecular
        mark dissipates if not reinforced by a DA signal during the relevant
        window. Calling this method between game episodes (between Pong rallies
        or full game resets) models that inter-episode tag decay.

        Must be called between game episodes. Failure to call reset_trace
        between episodes allows eligibility from one episode to leak into
        the next, which corrupts temporal credit assignment and does not
        correspond to the biological mechanism.

        Frey U, Morris RGM (1997). "Synaptic tagging and long-term
        potentiation." Nature, 385(6616), 533-536. DOI: 10.1038/385533a0
        """
        self.eligibility_trace.zero_()

    def reset_baseline(self) -> None:
        """
        Zero the reward_baseline buffer in place.

        Intended for ablation study resets between fully independent training
        runs, NOT for within-run use.

        WARNING: Calling this mid-run corrupts the running reward expectation
        that the baseline_subtraction mechanism depends on. The EMA baseline
        reflects the history of rewards seen during the run; zeroing it mid-run
        invalidates the use_baseline_subtraction=True ablation condition, because
        the baseline will transiently behave as if the agent has never seen any
        reward, producing artificially large d values until the EMA re-converges.
        Only call this between fully independent training runs.
        """
        self.reward_baseline.zero_()


# =============================================================================
# Part 5: to_dict and from_dict
# =============================================================================

    def to_dict(self) -> dict:
        """
        Serialize the full RewardModulator state to a JSON-compatible dict.

        Captures: configuration, reward_baseline scalar, and eligibility_trace
        tensor (shape and flattened values). All values are plain Python types
        (float, int, list, bool) to support JSON serialization without custom
        encoders.

        The eligibility_trace is stored as a flat list of floats. If the trace
        has never been initialized beyond the scalar-zero default (i.e., apply()
        has never been called), eligibility_trace_shape is [] and
        eligibility_trace is [].

        Returns:
            dict with keys:
                cfg (dict): dataclasses.asdict representation of self.cfg.
                reward_baseline (float): current scalar value of baseline buffer.
                eligibility_trace_shape (list[int]): shape of the trace tensor.
                    Empty list if trace is still the uninitialized scalar zero.
                eligibility_trace (list[float]): CPU-flattened trace values.
                    Empty list if trace is still the uninitialized scalar zero.
        """
        trace = self.eligibility_trace
        # Treat scalar (0-dim) tensor or shape [1] as uninitialized.
        if trace.dim() == 0 or list(trace.shape) == [1]:
            trace_shape: List[int] = []
            trace_values: List[float] = []
        else:
            trace_shape = list(trace.shape)
            trace_values = trace.detach().cpu().flatten().tolist()

        return {
            "cfg": dataclasses.asdict(self.cfg),
            "reward_baseline": self.reward_baseline.item(),
            "eligibility_trace_shape": trace_shape,
            "eligibility_trace": trace_values,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RewardModulator":
        """
        Reconstruct a RewardModulator from a dict produced by to_dict.

        Restores cfg, reward_baseline buffer, and eligibility_trace buffer to
        the values captured at serialization time. This supports full round-trip
        state recovery as required by the paper's checkpoint integrity claim:
        everything Tiny Tim and its associated modules learn must be persistently
        saveable and verifiable via load-and-test.

        Args:
            d: dict as produced by to_dict. Keys: cfg, reward_baseline,
               eligibility_trace_shape, eligibility_trace.

        Returns:
            RewardModulator with all buffers restored to serialized values.
        """
        cfg = RewardModulatorConfig(**d["cfg"])
        mod = cls(cfg)

        # Restore reward baseline.
        mod.reward_baseline.fill_(d["reward_baseline"])

        # Restore eligibility trace.
        shape = d["eligibility_trace_shape"]
        values = d["eligibility_trace"]
        if not shape or not values:
            # Never initialized beyond scalar zero default.
            mod.register_buffer("eligibility_trace", torch.zeros(1))
        else:
            tensor = torch.tensor(values, dtype=torch.float32).reshape(shape)
            mod.register_buffer("eligibility_trace", tensor)

        return mod


# =============================================================================
# Part 6: Self-tests
# =============================================================================

def _run_self_tests() -> None:
    """
    Six self-tests verifying the ablation flags and serialization round-trip.

    Each test fails for exactly the reason the corresponding claim would be
    false. All tests print PASS or FAIL with a specific failure reason.
    Called from the __main__ block.

    No unittest dependency. Plain assertions with descriptive print output.
    """

    all_passed = True

    # ------------------------------------------------------------------
    # Test 1: use_reward_modulation=False returns zero d and unmodified delta_w
    # ------------------------------------------------------------------
    try:
        cfg = RewardModulatorConfig(use_reward_modulation=False)
        mod = RewardModulator(cfg)
        d = mod.compute_rpe(predicted=0.5, actual=1.0)
        assert d == 0.0, f"compute_rpe returned {d}, expected 0.0"
        delta_w = torch.ones(4)
        result = mod.apply(delta_w, d=0.5)
        assert torch.allclose(result, delta_w), (
            f"apply returned {result}, expected {delta_w}"
        )
        print("PASS  Test 1: use_reward_modulation=False disables modulation.")
    except AssertionError as e:
        print(f"FAIL  Test 1: use_reward_modulation=False did not disable modulation. {e}")
        all_passed = False

    # ------------------------------------------------------------------
    # Test 2: use_prediction=False ignores the predicted argument
    # ------------------------------------------------------------------
    try:
        cfg = RewardModulatorConfig(
            use_prediction=False,
            use_baseline_subtraction=False,
            use_da_saturation=False,
            use_reward_modulation=True,
        )
        mod1 = RewardModulator(cfg)
        d1 = mod1.compute_rpe(predicted=999.0, actual=0.7)
        mod2 = RewardModulator(cfg)
        d2 = mod2.compute_rpe(predicted=0.0, actual=0.7)
        assert abs(d1 - d2) < 1e-6, (
            f"d1={d1}, d2={d2} differ by {abs(d1-d2):.2e}; predicted arg was used."
        )
        print("PASS  Test 2: use_prediction=False ignores the predicted argument.")
    except AssertionError as e:
        print(f"FAIL  Test 2: use_prediction=False still used the predicted argument. {e}")
        all_passed = False

    # ------------------------------------------------------------------
    # Test 3: use_eligibility_trace=False applies d directly, no accumulation
    # ------------------------------------------------------------------
    try:
        cfg = RewardModulatorConfig(
            use_eligibility_trace=False,
            use_reward_modulation=True,
            use_da_saturation=False,
            use_baseline_subtraction=False,
            use_prediction=False,
        )
        mod = RewardModulator(cfg)
        delta_w = torch.ones(4)
        result1 = mod.apply(delta_w, d=1.0)
        assert torch.allclose(result1, delta_w * 2.0), (
            f"First call: result1={result1}, expected {delta_w * 2.0}"
        )
        result2 = mod.apply(delta_w, d=1.0)
        assert torch.allclose(result2, result1), (
            f"Second call differed: result2={result2}, result1={result1}; trace accumulation occurred."
        )
        print("PASS  Test 3: use_eligibility_trace=False applies d directly, no accumulation.")
    except AssertionError as e:
        print(
            f"FAIL  Test 3: use_eligibility_trace=False did not produce delta_w * (1+d) "
            f"or produced different results on repeated calls. {e}"
        )
        all_passed = False

    # ------------------------------------------------------------------
    # Test 4: use_da_saturation=False allows d > 1.0
    # ------------------------------------------------------------------
    try:
        cfg = RewardModulatorConfig(
            use_da_saturation=False,
            use_prediction=False,
            use_baseline_subtraction=False,
            use_reward_modulation=True,
        )
        mod = RewardModulator(cfg)
        d = mod.compute_rpe(predicted=0.0, actual=5.0)
        assert d == 5.0, f"Expected d=5.0, got d={d}"
        print("PASS  Test 4: use_da_saturation=False allows d > 1.0.")
    except AssertionError as e:
        print(f"FAIL  Test 4: use_da_saturation=False still clipped or saturated d. {e}")
        all_passed = False

    # ------------------------------------------------------------------
    # Test 5: use_baseline_subtraction=True reduces d over repeated equal rewards
    # ------------------------------------------------------------------
    try:
        cfg = RewardModulatorConfig(
            use_baseline_subtraction=True,
            use_prediction=False,
            use_da_saturation=False,
            use_reward_modulation=True,
        )
        mod = RewardModulator(cfg)
        d_values = [mod.compute_rpe(predicted=0.0, actual=1.0) for _ in range(10)]
        assert d_values[-1] < d_values[0], (
            f"d did not decrease: d[0]={d_values[0]:.4f}, d[-1]={d_values[-1]:.4f}"
        )
        assert d_values[-1] < 0.1, (
            f"Baseline did not converge sufficiently: d[-1]={d_values[-1]:.4f}"
        )
        print(
            f"PASS  Test 5: use_baseline_subtraction=True reduced d from "
            f"{d_values[0]:.4f} to {d_values[-1]:.4f} over 10 identical rewards."
        )
    except AssertionError as e:
        print(
            f"FAIL  Test 5: use_baseline_subtraction=True did not reduce d over "
            f"repeated identical rewards. {e}"
        )
        all_passed = False

    # ------------------------------------------------------------------
    # Test 6: to_dict / from_dict round-trip
    # ------------------------------------------------------------------
    try:
        cfg = RewardModulatorConfig()
        mod = RewardModulator(cfg)
        # Advance trace by applying a delta_w.
        mod.apply(torch.ones(8), d=0.5)
        # Advance baseline by computing an RPE.
        mod.compute_rpe(predicted=0.2, actual=0.8)
        snapshot = mod.to_dict()
        mod2 = RewardModulator.from_dict(snapshot)
        assert abs(mod2.reward_baseline.item() - mod.reward_baseline.item()) < 1e-6, (
            f"reward_baseline mismatch: mod={mod.reward_baseline.item():.6f}, "
            f"mod2={mod2.reward_baseline.item():.6f}"
        )
        assert torch.allclose(mod2.eligibility_trace, mod.eligibility_trace), (
            f"eligibility_trace mismatch:\n  mod={mod.eligibility_trace}\n  mod2={mod2.eligibility_trace}"
        )
        print("PASS  Test 6: to_dict / from_dict round-trip restored RewardModulator state.")
    except AssertionError as e:
        print(f"FAIL  Test 6: Round-trip serialization did not restore RewardModulator state. {e}")
        all_passed = False

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    if all_passed:
        print("All 6 self-tests passed.")
    else:
        print("One or more self-tests FAILED. See output above.")


# =============================================================================
# Part 7: __main__ entry point
# =============================================================================

if __name__ == "__main__":
    _run_self_tests()
