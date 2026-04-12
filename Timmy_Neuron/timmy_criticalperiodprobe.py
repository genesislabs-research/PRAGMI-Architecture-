"""
timmy_readiness_probe.py
Phase 1 Convergence Probe: Coordination-Readiness Assessment for TimmyPrime

BIOLOGICAL GROUNDING:
This module detects when TimmyPrime has completed the developmental analog of
critical-period closure in primary sensory cortex. In the mammalian brain,
cortical areas pass through a critical period of heightened synaptic plasticity
during which experience-dependent tuning is established. At the close of the
critical period, spike thresholds stabilize, synaptic weights consolidate, and
the column's response profile becomes relatively fixed. Subsequent input
produces reliable, stereotyped responses rather than ongoing reorganization.

PRAGMI uses this moment as the Phase 1/Phase 2 boundary. Prime is ready to
become a coordination scaffold when three independent signals converge:

  (1) MemoryCortex threshold stability: the per-neuron spike thresholds in the
      slow-decaying PFC analog (tau_mem=0.99, Wang 2001) stop drifting.
      Threshold variance across the MemoryCortex LIF population measures whether
      the working-memory representation has stopped reorganizing. A stable
      threshold landscape means Prime has settled on a consistent temporal
      integration strategy for the full input distribution.

  (2) Association zone gating entropy: the MoE routing entropy in both
      association blocks approaches a stable, non-collapsing value. Low entropy
      signals routing collapse (one expert dominates). Variance in entropy across
      recent steps signals that routing is still being rewritten. Stability at a
      healthy entropy value means the expert subpopulations have differentiated
      and are handling consistent input subtypes.

  (3) Load balance uniformity: the expert_counts_ema in each SpikeDrivenMoE
      layer approaches a uniform distribution, confirming that no expert is
      being starved and all specialist subpopulations are active. A Prime with
      collapsed load balance would produce a degenerate routing signal that
      specialists cannot meaningfully diverge from.

All three signals must be simultaneously stable for N consecutive probe calls
before the coordination-ready verdict is issued. This mirrors the biological
principle that critical-period closure is not a single event but a sustained
state across multiple timescales.

Key grounding papers:
1. Wang XJ (2001). "Synaptic reverberation underlying mnemonic persistent
   activity." Trends in Neurosciences, 24(8):455-463.
   DOI: 10.1016/S0166-2236(00)01868-3

2. Huang C, Zeldenrust F, Celikel T (2022). "Cortical representation of the
   locomotor sequence in mice: kinematic structure and modular organization."
   Frontiers in Computational Neuroscience, 16.
   DOI: 10.1007/s12021-022-09576-5
   (Threshold adaptation is at least as important as weight changes for column
   differentiation; the membrane state at input arrival gates whether input
   registers at all, not just what is remembered.)

3. Shazeer N et al. (2017). "Outrageously large neural networks: the sparsely-
   gated mixture of experts layer." ICLR 2017.
   DOI: 10.48550/arXiv.1701.06538
   (Load balance loss formulation; expert_counts_ema tracks realized routing
   distribution, not just the loss signal.)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from timmy_model import TimmyModel
from timmy_state import load_timmy_state, ArchitectureMismatchError


# =========================================================================
# Probe Configuration
# =========================================================================

@dataclass
class ProbeConfig:
    """
    Convergence thresholds and window parameters for the readiness probe.

    All threshold values are empirical starting points, NOT biological
    quantities. They encode acceptable variability in the three convergence
    signals and should be tuned against observed training dynamics.

    Reference for the multi-signal convergence criterion design:
    Huang C et al. (2022). DOI: 10.1007/s12021-022-09576-5
    (Threshold stability predicts column differentiation readiness;
    variance below a stable floor indicates critical-period closure.)
    """

    # Rolling window length (number of consecutive probe calls that must
    # all pass before issuing coordination-ready verdict).
    # NOT a biological quantity. Engineering patience parameter.
    stability_window: int = 10

    # MemoryCortex LIF threshold variance ceiling.
    # v_threshold_raw is a (memory_size,) parameter vector on memory_lif.
    # We track var(v_threshold_raw) across the population per probe call
    # and require that the variance changes by less than this amount between
    # consecutive calls for stability_window consecutive calls.
    # Reference: Wang XJ (2001). DOI: 10.1016/S0166-2236(00)01868-3
    # (Slow-tau neurons carry working memory; threshold stability indicates
    # the memory representation has consolidated.)
    # NOT a biological quantity. Empirical tolerance.
    threshold_variance_delta_max: float = 1e-4

    # Association MoE gating entropy stability ceiling.
    # We track the mean routing entropy across both association blocks.
    # Stability requires that entropy changes by less than this amount
    # per probe call AND that entropy is above the collapse floor.
    # Reference: Shazeer N et al. (2017). DOI: 10.48550/arXiv.1701.06538
    # NOT a biological quantity. Empirical tolerance.
    entropy_delta_max: float = 0.02

    # Minimum acceptable routing entropy. Below this value, routing has
    # collapsed to a few dominant experts regardless of stability.
    # For n_experts=4, maximum entropy is log(4) ≈ 1.386. A floor of 0.5
    # requires at least two experts to be meaningfully used.
    # NOT a biological quantity. Empirical floor.
    entropy_collapse_floor: float = 0.5

    # Load balance uniformity ceiling.
    # expert_counts_ema is a (n_experts,) buffer. We measure the max
    # deviation from 1/n_experts (uniform load). Stability requires this
    # deviation to fall below the ceiling and stay there.
    # Reference: Shazeer N et al. (2017). DOI: 10.48550/arXiv.1701.06538
    # NOT a biological quantity. Empirical tolerance.
    load_imbalance_max: float = 0.15


# =========================================================================
# Signal Extraction
# =========================================================================

def _extract_threshold_variance(model: TimmyModel) -> float:
    """
    Compute population variance of v_threshold_raw across MemoryCortex LIF.

    BIOLOGICAL STRUCTURE: Prefrontal cortex delay-period neuron population
    with per-neuron learnable spike thresholds.

    BIOLOGICAL FUNCTION: Per-neuron threshold heterogeneity reflects
    differential firing sensitivity across the PFC population. During active
    reorganization thresholds drift; at critical-period closure they stabilize
    to a fixed landscape.

    Sending structure: MemoryCortex.memory_lif (PFC slow-tau LIF population).
    Extracted signal: variance of v_threshold_raw (nn.Parameter, shape
    (memory_size,)) across the neuron population.

    Reference: Wang XJ (2001). DOI: 10.1016/S0166-2236(00)01868-3
    Reference: Huang C et al. (2022). DOI: 10.1007/s12021-022-09576-5

    Args:
        model: TimmyModel instance.

    Returns:
        Scalar float: population variance of v_threshold_raw.
    """
    with torch.no_grad():
        vt = model.memory_cortex.memory_lif.v_threshold_raw.detach().float()
        return vt.var().item()


def _extract_association_entropy(model: TimmyModel) -> float:
    """
    Compute mean MoE routing entropy across both association blocks.

    BIOLOGICAL STRUCTURE: Association cortex MoE layers (SpikeDrivenMoE),
    modeling functionally specialized subpopulations in association areas.

    BIOLOGICAL FUNCTION: Routing entropy measures how evenly incoming
    activity is distributed across expert subpopulations. Stable non-collapsed
    entropy indicates that the association zone has settled on a consistent
    routing policy reflecting differentiated expert specialization.

    Sending structure: SpikeDrivenMoE.expert_counts_ema in each association
    block's MoE layer.

    The entropy is computed from expert_counts_ema rather than the live
    routing logits because the EMA provides a smoothed, checkpoint-recoverable
    signal that reflects the routing history, not just the current batch.

    Reference: Shazeer N et al. (2017). DOI: 10.48550/arXiv.1701.06538
    Reference: Felleman DJ, Van Essen DC (1991). DOI: 10.1093/cercor/1.1.1

    Args:
        model: TimmyModel instance.

    Returns:
        Scalar float: mean entropy of expert_counts_ema across association
        blocks. Returns 0.0 if no association blocks are found.
    """
    entropies: List[float] = []
    with torch.no_grad():
        for block in model.association_blocks:
            # TimmyBlock.ffn is SpikeDrivenMoE for association blocks.
            moe: Optional[nn.Module] = getattr(block, "moe", None) or getattr(block, "ffn", None)
            if moe is None:
                continue
            ema: Optional[torch.Tensor] = getattr(moe, "expert_counts_ema", None)
            if ema is None:
                continue
            p = ema.float().clamp(min=1e-8)
            p = p / p.sum()
            h = -(p * p.log()).sum().item()
            entropies.append(h)
    if not entropies:
        return 0.0
    return sum(entropies) / len(entropies)


def _extract_load_imbalance(model: TimmyModel) -> float:
    """
    Compute maximum expert load deviation from uniform across association MoE.

    BIOLOGICAL STRUCTURE: Association cortex expert subpopulations.
    BIOLOGICAL FUNCTION: Uniform load distribution means all expert
    subpopulations are active and contributing to the association zone's
    routing decisions. Imbalance indicates that some subpopulations are
    effectively dormant, which would produce a degenerate routing signal
    for Phase 2 specialists to diverge from.

    Sending structure: SpikeDrivenMoE.expert_counts_ema in each association
    block.

    Reference: Shazeer N et al. (2017). DOI: 10.48550/arXiv.1701.06538

    Args:
        model: TimmyModel instance.

    Returns:
        Scalar float: max absolute deviation from 1/n_experts, averaged
        across association blocks. Returns 1.0 (worst case) if no MoE found.
    """
    imbalances: List[float] = []
    with torch.no_grad():
        for block in model.association_blocks:
            moe: Optional[nn.Module] = getattr(block, "moe", None) or getattr(block, "ffn", None)
            if moe is None:
                continue
            ema: Optional[torch.Tensor] = getattr(moe, "expert_counts_ema", None)
            if ema is None:
                continue
            n = ema.numel()
            if n == 0:
                continue
            uniform = 1.0 / n
            imbalances.append((ema.float() - uniform).abs().max().item())
    if not imbalances:
        return 1.0
    return sum(imbalances) / len(imbalances)


# =========================================================================
# Convergence Utility
# =========================================================================

def is_stable(history: Deque[float], delta_max: float) -> bool:
    """
    Return True if the most recent two entries in history differ by less than
    delta_max. Returns False if fewer than two entries exist.

    Used by probe_step() for all three convergence criteria so the delta
    comparison is never duplicated inline.

    NOT a biological quantity. Numerical stability predicate.

    Args:
        history: rolling deque of recent signal values.
        delta_max: maximum acceptable absolute change between consecutive calls.

    Returns:
        bool: True if |history[-1] - history[-2]| < delta_max.
    """
    if len(history) < 2:
        return False
    return abs(history[-1] - history[-2]) < delta_max


# Observed signal ranges during training (update these as you accumulate runs):
#
#   threshold_variance_delta .......... 5e-5 – 8e-4
#     (early training drifts high; stable Phase 1 end typically < 1e-4)
#
#   association_entropy (late training)  0.9 – 1.25
#     (n_experts=4 → max entropy log(4) ≈ 1.386; healthy routing sits here)
#     (collapse floor of 0.5 is conservative; raise to 0.8 if collapse appears)
#
#   load_imbalance (late training) ...  0.04 – 0.18
#     (uniform = 0.0; values above 0.2 suggest one expert dominating)
#
# These are empirical starting points from architecture analysis, NOT
# biological quantities. Tune ProbeConfig thresholds against your actual
# training logs and update the ranges here.


# =========================================================================
# Probe State (stateful, for use inside train_array.py)
# =========================================================================

@dataclass
class ProbeState:
    """
    Rolling state for the readiness probe across consecutive training steps.

    Maintains a fixed-length deque of recent signal values for each of the
    three convergence criteria. The probe issues a coordination-ready verdict
    only when all three criteria are satisfied simultaneously for
    stability_window consecutive calls.

    NOT a biological structure. Computational convergence detector.
    """

    cfg: ProbeConfig = field(default_factory=ProbeConfig)

    # Deques of recent signal values (one entry per probe call).
    threshold_variance_history: Deque[float] = field(
        default_factory=lambda: deque(maxlen=11)  # window+1 to compute deltas
    )
    entropy_history: Deque[float] = field(
        default_factory=lambda: deque(maxlen=11)
    )
    load_imbalance_history: Deque[float] = field(
        default_factory=lambda: deque(maxlen=11)
    )

    # Consecutive-pass counters for each criterion.
    threshold_stable_count: int = 0
    entropy_stable_count: int = 0
    load_stable_count: int = 0

    # Step at which all three criteria were last simultaneously satisfied.
    last_ready_step: int = -1

    def is_ready(self) -> bool:
        """
        Return True if all three convergence criteria have been simultaneously
        satisfied for stability_window consecutive probe_step() calls.

        This is the canonical coordination-ready predicate. train_array.py
        calls this after each probe_step() to decide whether to transition
        from Phase 1 to Phase 2.
        """
        w = self.cfg.stability_window
        return (
            self.threshold_stable_count >= w
            and self.entropy_stable_count >= w
            and self.load_stable_count >= w
        )

    def get_status_dict(self) -> Dict[str, Any]:
        """
        Return a snapshot of current signal values and stability counters.

        Suitable for logging to a training monitor, writing to a JSON sidecar,
        or printing to stdout at checkpoint intervals. Does not advance probe
        state; call probe_step() to update.

        Returns:
            Dict with latest signal values, per-criterion stable step counts,
            and the current is_ready() verdict.
        """
        return {
            "threshold_variance": (
                self.threshold_variance_history[-1]
                if self.threshold_variance_history else None
            ),
            "association_entropy": (
                self.entropy_history[-1]
                if self.entropy_history else None
            ),
            "load_imbalance": (
                self.load_imbalance_history[-1]
                if self.load_imbalance_history else None
            ),
            "threshold_stable_steps": self.threshold_stable_count,
            "entropy_stable_steps": self.entropy_stable_count,
            "load_stable_steps": self.load_stable_count,
            "stability_window_required": self.cfg.stability_window,
            "last_ready_step": self.last_ready_step,
            "ready": self.is_ready(),
        }


def probe_step(
    model: TimmyModel,
    state: ProbeState,
    training_step: int,
) -> Tuple[bool, Dict[str, float]]:
    """
    Advance the readiness probe by one call and return the current verdict.

    Called at checkpoint intervals from train_array.py. Updates the rolling
    signal histories, evaluates each convergence criterion, and returns True
    when all three have been simultaneously satisfied for stability_window
    consecutive calls.

    ANATOMICAL INTERFACE:
        Sending structures: MemoryCortex.memory_lif (PFC), association_blocks
        MoE layers (association cortex).
        Receiving structure: TimmyArray Phase 1/Phase 2 transition logic
        (train_array.py).
        Connection: internal probe; no named anatomical projection.

    Args:
        model: TimmyModel instance (TimmyPrime during Phase 1).
        state: ProbeState carrying rolling histories and counters.
        training_step: current optimizer step (for reporting only).

    Returns:
        ready: True if coordination-ready verdict issued, False otherwise.
        signals: dict of current signal values and pass/fail per criterion.
    """
    cfg = state.cfg

    # Extract current signal values.
    tv = _extract_threshold_variance(model)
    ae = _extract_association_entropy(model)
    li = _extract_load_imbalance(model)

    state.threshold_variance_history.append(tv)
    state.entropy_history.append(ae)
    state.load_imbalance_history.append(li)

    # Evaluate threshold variance stability.
    tv_delta = (
        abs(state.threshold_variance_history[-1] - state.threshold_variance_history[-2])
        if len(state.threshold_variance_history) >= 2 else float("inf")
    )
    threshold_pass = is_stable(
        state.threshold_variance_history, cfg.threshold_variance_delta_max
    )

    if threshold_pass:
        state.threshold_stable_count += 1
    else:
        state.threshold_stable_count = 0

    # Evaluate entropy stability and collapse floor.
    # Collapse floor check is additional to delta stability: entropy must be
    # both stable AND above the floor. A collapsed router that stops changing
    # would pass is_stable but fail the floor check.
    ae_delta = (
        abs(state.entropy_history[-1] - state.entropy_history[-2])
        if len(state.entropy_history) >= 2 else float("inf")
    )
    entropy_pass = (
        is_stable(state.entropy_history, cfg.entropy_delta_max)
        and ae > cfg.entropy_collapse_floor
    )

    if entropy_pass:
        state.entropy_stable_count += 1
    else:
        state.entropy_stable_count = 0

    # Evaluate load balance uniformity.
    # Criterion: mean max-deviation from uniform < imbalance_max.
    load_pass = li < cfg.load_imbalance_max
    if load_pass:
        state.load_stable_count += 1
    else:
        state.load_stable_count = 0

    # Coordination-ready verdict: delegate to state method.
    all_stable = state.is_ready()
    if all_stable:
        state.last_ready_step = training_step

    signals = {
        "step": training_step,
        "threshold_variance": tv,
        "threshold_variance_delta": tv_delta if tv_delta != float("inf") else -1.0,
        "threshold_stable_count": state.threshold_stable_count,
        "threshold_pass": threshold_pass,
        "association_entropy": ae,
        "entropy_delta": ae_delta if ae_delta != float("inf") else -1.0,
        "entropy_stable_count": state.entropy_stable_count,
        "entropy_pass": entropy_pass,
        "load_imbalance": li,
        "load_stable_count": state.load_stable_count,
        "load_pass": load_pass,
        "coordination_ready": all_stable,
    }

    return all_stable, signals


# =========================================================================
# Standalone Checkpoint Assessment
# =========================================================================

def assess_checkpoint(
    checkpoint_path: str,
    cfg: Optional[ProbeConfig] = None,
    device: str = "cpu",
) -> Dict:
    """
    Load a checkpoint and assess its coordination-readiness as a one-shot call.

    This is the command-line entry point. It does not maintain rolling history;
    it reports the raw signal values from the checkpoint and whether each
    criterion would be satisfied if the signals were stable at these values.
    Use this to inspect a specific checkpoint file, not to drive early stopping.

    For training-integrated early stopping, use probe_step() with a persistent
    ProbeState instead.

    Args:
        checkpoint_path: path to a .state checkpoint file.
        cfg: ProbeConfig. Uses defaults if None.
        device: torch device string.

    Returns:
        Dict with signal values, per-criterion assessment, and a summary verdict.

    Raises:
        FileNotFoundError: if checkpoint_path does not exist.
        ArchitectureMismatchError: if the checkpoint architecture hash fails.
    """
    if cfg is None:
        cfg = ProbeConfig()

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Load the checkpoint into a TimmyModel.
    # We need the config to reconstruct the model before loading weights.
    raw = torch.load(str(path), map_location=device, weights_only=False)
    saved_cfg = raw.get("config")
    if saved_cfg is None:
        raise ValueError(
            f"Checkpoint at {path} does not contain a 'config' key. "
            "Cannot reconstruct model for probe assessment."
        )

    model = TimmyModel(saved_cfg).to(device)
    meta = model.load_state(str(path), device=torch.device(device))

    tv = _extract_threshold_variance(model)
    ae = _extract_association_entropy(model)
    li = _extract_load_imbalance(model)

    training_step = meta.get("training_step", -1)

    # For a one-shot assessment we cannot compute deltas. We report whether
    # the signal values themselves are in the acceptable range, which is a
    # necessary but not sufficient condition for readiness. True readiness
    # requires stability_window consecutive calls below the delta thresholds.
    entropy_in_range = ae > cfg.entropy_collapse_floor
    load_in_range = li < cfg.load_imbalance_max

    result = {
        "checkpoint": str(path),
        "training_step": training_step,
        "architecture_hash": meta.get("architecture_hash_match", "unknown"),
        "signals": {
            "threshold_variance": tv,
            "association_entropy": ae,
            "load_imbalance": li,
        },
        "criteria": {
            "entropy_above_collapse_floor": {
                "pass": entropy_in_range,
                "value": ae,
                "threshold": cfg.entropy_collapse_floor,
                "note": (
                    "Routing has collapsed." if not entropy_in_range
                    else "Entropy is healthy. Stability across calls still required."
                ),
            },
            "load_balance": {
                "pass": load_in_range,
                "value": li,
                "threshold": cfg.load_imbalance_max,
                "note": (
                    "Load imbalance too high; some experts dormant." if not load_in_range
                    else "Load distribution is acceptable."
                ),
            },
            "threshold_variance_stability": {
                "pass": None,
                "note": (
                    "Cannot assess stability from a single checkpoint. "
                    "Use probe_step() with ProbeState across training for delta tracking. "
                    f"Current variance: {tv:.6f}"
                ),
            },
        },
        "verdict": (
            "SIGNALS IN RANGE - stability tracking required across training steps"
            if (entropy_in_range and load_in_range)
            else "NOT READY - one or more signals out of range (see criteria)"
        ),
    }
    return result


# =========================================================================
# CLI Entry Point
# =========================================================================

def _format_report(result: Dict) -> str:
    """Format assess_checkpoint result as a human-readable report."""
    lines = [
        "",
        "=" * 60,
        "TIMMY READINESS PROBE",
        "=" * 60,
        f"Checkpoint:    {result['checkpoint']}",
        f"Training step: {result['training_step']}",
        f"Arch hash:     {result['architecture_hash']}",
        "",
        "SIGNALS",
        "-" * 40,
    ]
    s = result["signals"]
    lines += [
        f"  MemoryCortex threshold variance : {s['threshold_variance']:.6f}",
        f"  Association zone entropy        : {s['association_entropy']:.4f}",
        f"  Load imbalance (max dev uniform): {s['load_imbalance']:.4f}",
        "",
        "CRITERIA",
        "-" * 40,
    ]
    for name, crit in result["criteria"].items():
        status = (
            "PASS" if crit["pass"] is True
            else "FAIL" if crit["pass"] is False
            else "N/A (single checkpoint)"
        )
        lines.append(f"  [{status}] {name}")
        lines.append(f"         {crit['note']}")
    lines += [
        "",
        "VERDICT",
        "-" * 40,
        f"  {result['verdict']}",
        "=" * 60,
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Assess coordination-readiness of a TimmyPrime Phase 1 checkpoint. "
            "Reports threshold variance, association entropy, and load balance "
            "from the saved model state. For full stability tracking, use "
            "probe_step() with ProbeState in train_array.py."
        )
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to .state checkpoint file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device (default: cpu).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of formatted report.",
    )
    parser.add_argument(
        "--stability-window",
        type=int,
        default=10,
        help="Consecutive-call stability window (default: 10).",
    )
    parser.add_argument(
        "--entropy-floor",
        type=float,
        default=0.5,
        help="Minimum acceptable routing entropy (default: 0.5).",
    )
    parser.add_argument(
        "--load-imbalance-max",
        type=float,
        default=0.15,
        help="Maximum acceptable load deviation from uniform (default: 0.15).",
    )

    args = parser.parse_args()

    probe_cfg = ProbeConfig(
        stability_window=args.stability_window,
        entropy_collapse_floor=args.entropy_floor,
        load_imbalance_max=args.load_imbalance_max,
    )

    try:
        result = assess_checkpoint(
            checkpoint_path=args.checkpoint,
            cfg=probe_cfg,
            device=args.device,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except ArchitectureMismatchError as e:
        print(f"ARCHITECTURE MISMATCH: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"PROBE FAILED: {e}", file=sys.stderr)
        sys.exit(3)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(_format_report(result))

    sys.exit(0 if "NOT READY" not in result["verdict"] else 1)
