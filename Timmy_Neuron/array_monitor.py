"""
array_monitor.py
Phase 2 Online Diagnostic Monitor for the TimmyArray Cortical Column Ensemble

BIOLOGICAL GROUNDING:
This module provides the diagnostic instrumentation for Phase 2 divergence
training. The four signals it tracks correspond directly to measurable
properties of biological cortical column ensembles undergoing experience-
dependent specialization.

ROUTER ENTROPY measures how evenly the ColumnRouter distributes routing load
across specialist columns. In the biological analog, this corresponds to
balanced activation across the neocortical ensemble: no single area should
dominate processing to the exclusion of others. Routing collapse (entropy
approaching zero) indicates that the coordination mechanism has failed and
all input is being sent to one specialist, which is equivalent to lesioning
the rest of the ensemble.

SPECIALIST ACTIVATION FREQUENCY tracks how often each specialist column is
selected by the router over a rolling window. In the biological literature,
Pérez-Ortega et al. (2021) showed that ensemble stability is carried by a
connectivity core: neurons (and here, columns) that are persistently activated
across varied inputs develop the highest internal connectivity density and
become the stable core of the ensemble. Columns that are never selected
cannot develop that stability.

SUBSPACE EFFECTIVE RANK measures the dimensionality of the representation
space each column's association zone is using. Roy and Vetterli (2007) defined
effective rank as exp(H(singular values / sum)), where H is the Shannon entropy
of the normalized singular value spectrum. A column using full-rank
representations is distributing its representational capacity across many
directions. A column collapsing to low rank is developing a degenerate
representation. Divergence between columns on this metric is evidence of
genuine specialization: specialists are carving out different subspaces.

COSINE DISTANCE between Prime and specialist association zone outputs measures
how much each specialist has diverged from the broadband coordination scaffold.
At Phase 2 start (after clone_prime_to_specialists()), all columns are
identical and cosine distance is zero. Increasing distance is the primary
behavioral signature of column differentiation. See JZ et al. (2018) showed
that coordinated neuronal ensembles are defined by higher-order correlation
structure: the monitoring of cosine distance operationalizes this by asking
whether the specialist's representational geometry has diverged from Prime's.

Key grounding papers:
1. Pérez-Ortega J, Alejandre-García T, Yuste R (2021). "Long-term stability
   of cortical ensembles." eLife, 10:e64449.
   DOI: 10.7554/eLife.64449

2. See JZ, Atencio CA, Sohal VS, Schreiner CE (2018). "Coordinated neuronal
   ensembles in primary auditory cortical columns." eLife, 7:e35587.
   DOI: 10.7554/eLife.35587

3. Roy O, Vetterli M (2007). "The effective rank: a measure of effective
   dimensionality." Proceedings of the 15th European Signal Processing
   Conference (EUSIPCO), 606-610.
   DOI: {To be added later.}

4. Shazeer N et al. (2017). "Outrageously large neural networks: the sparsely-
   gated mixture of experts layer." ICLR 2017.
   DOI: 10.48550/arXiv.1701.06538
   (Load balance formulation; specialist_load_ema tracks realized routing.)
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

# TimmyArray is imported lazily inside functions to avoid circular imports
# when array_monitor is imported early in training scripts.


# =========================================================================
# Monitor Configuration
# =========================================================================

@dataclass
class MonitorConfig:
    """
    Configuration for the Phase 2 array monitor.

    All thresholds are empirical starting points, NOT biological quantities.
    Update the observed-ranges block below as you accumulate training runs.

    Observed signal ranges during Phase 2 training (update these!):
      router_entropy .................. 0.8 – 1.4  (n=5 specialists, max log(5)≈1.61)
      specialist_activation_freq ...... 0.1 – 0.4  per specialist (uniform = 0.2)
      subspace_effective_rank ......... 8 – 48     (d_model=496, n_clusters=64)
      cosine_distance_from_prime ...... 0.0 – 0.8  (0.0 at clone; >0.3 = healthy divergence)

    These are empirical starting points from architecture analysis.
    NOT biological quantities. Tune against your actual training logs.
    """

    # Rolling window for activation frequency tracking (steps).
    # NOT a biological quantity. Smoothing window.
    activation_window: int = 200

    # Log to file every this many steps. Set 0 to disable file logging.
    # NOT a biological quantity. I/O cadence.
    log_every: int = 100

    # Alert threshold: router entropy below this triggers a collapse warning.
    # For n=5 specialists max entropy is log(5)≈1.609. Floor of 0.5 means
    # at least two specialists must be meaningfully used.
    # NOT a biological quantity. Empirical alert threshold.
    entropy_collapse_floor: float = 0.5

    # Alert threshold: cosine distance above this is considered healthy
    # divergence between a specialist and Prime. Below this after
    # divergence_expect_step steps is a warning.
    # NOT a biological quantity. Empirical alert threshold.
    divergence_healthy_floor: float = 0.15

    # Step at which healthy divergence is expected. Before this step,
    # low cosine distance is not alarming. After this step, a specialist
    # still at near-zero distance may be failing to differentiate.
    # NOT a biological quantity. Empirical patience parameter.
    divergence_expect_step: int = 2000

    # Output directory for log files. None disables file logging.
    log_dir: Optional[str] = None


# =========================================================================
# Signal Extraction
# =========================================================================

def _extract_router_entropy(array) -> float:
    """
    Compute Shannon entropy of the ColumnRouter specialist_load_ema.

    BIOLOGICAL STRUCTURE: ColumnRouter coordinating specialist column
    selection, analogous to the thalamocortical gating that routes sensory
    input to appropriate cortical areas.

    BIOLOGICAL FUNCTION: Balanced routing across specialists is the
    computational analog of balanced cortical area recruitment. Entropy
    collapse means the coordination mechanism is degenerate.

    Sending structure: ColumnRouter.specialist_load_ema (n_specialists,)
    buffer tracking realized routing distribution via EMA.

    Reference: Shazeer N et al. (2017). DOI: 10.48550/arXiv.1701.06538

    Args:
        array: TimmyArray instance.

    Returns:
        Scalar float: Shannon entropy of specialist_load_ema.
        Returns 0.0 if router not found.
    """
    with torch.no_grad():
        ema = getattr(array.router, "specialist_load_ema", None)
        if ema is None:
            return 0.0
        p = ema.float().clamp(min=1e-8)
        p = p / p.sum()
        return -(p * p.log()).sum().item()


def _extract_activation_frequencies(array) -> Dict[str, float]:
    """
    Extract per-specialist selection frequency from specialist_selection_ema.

    BIOLOGICAL STRUCTURE: Neocortical column ensemble activation statistics.
    BIOLOGICAL FUNCTION: Columns that are persistently activated develop
    the stable connectivity core described by Pérez-Ortega et al. (2021).
    Columns that are never selected cannot develop that core.

    Sending structure: TimmyArray.specialist_selection_ema (n_specialists,)
    buffer updated each forward pass via lerp_ on routing_weights.

    Reference: Pérez-Ortega J et al. (2021). DOI: 10.7554/eLife.64449

    Args:
        array: TimmyArray instance.

    Returns:
        Dict mapping specialist name -> selection frequency in [0, 1].
        Returns empty dict if ema not found.
    """
    with torch.no_grad():
        ema = getattr(array, "specialist_selection_ema", None)
        if ema is None:
            return {}
        specialist_names = array.column_names[1:]
        return {
            name: ema[i].item()
            for i, name in enumerate(specialist_names)
        }


def _effective_rank(matrix: Tensor) -> float:
    """
    Compute the effective rank of a 2D matrix via singular value entropy.

    Effective rank = exp(H(sigma / sum(sigma))) where sigma are the singular
    values and H is Shannon entropy. A full-rank matrix has effective rank
    equal to min(m, n). A rank-1 matrix has effective rank 1.0.

    BIOLOGICAL FUNCTION: Measures how many independent representational
    dimensions a column's association zone is actively using. Divergence
    in effective rank between columns is a signature of specialization:
    each specialist carves out a different subspace geometry.

    Reference: Roy O, Vetterli M (2007). DOI: {To be added later.}
    NOT a biological quantity. Mathematical measure of representational
    dimensionality.

    Args:
        matrix: (N, D) 2D tensor.

    Returns:
        Scalar float: effective rank in [1.0, min(N, D)].
        Returns 1.0 on failure.
    """
    if matrix.ndim != 2 or min(matrix.shape) < 2:
        return 1.0
    try:
        with torch.no_grad():
            m = matrix.float()
            # Subsample rows if matrix is large to keep SVD tractable.
            if m.shape[0] > 512:
                idx = torch.randperm(m.shape[0], device=m.device)[:512]
                m = m[idx]
            sv = torch.linalg.svdvals(m)
            sv = sv.clamp(min=0.0)
            s = sv.sum()
            if s < 1e-10:
                return 1.0
            p = sv / s
            p = p.clamp(min=1e-10)
            h = -(p * p.log()).sum().item()
            return math.exp(h)
    except Exception:
        return 1.0


def _extract_association_outputs(array, token_ids: Tensor) -> Dict[str, Tensor]:
    """
    Run a single forward pass per column through the association zone only
    and return the flattened association output tensor per column.

    This is used to compute subspace effective rank and cosine distance.
    The pass is run under torch.no_grad() and does not update any state.

    ANATOMICAL INTERFACE:
        Sending structures: each column's association_blocks (association cortex).
        Receiving structure: array_monitor subspace and cosine distance metrics.
        Connection: internal diagnostic hook; no named anatomical projection.

    Args:
        array: TimmyArray instance.
        token_ids: (B, S) integer token indices. Should be a small batch
            (B=1 or B=2) to keep this fast.

    Returns:
        Dict mapping column name -> (N, D) flattened association output tensor,
        where N = B*S*T_total and D = d_model. Returns empty dict on failure.
    """
    outputs: Dict[str, Tensor] = {}
    with torch.no_grad():
        for name, col in zip(array.column_names, array.all_columns()):
            try:
                cfg = col.cfg
                T_t = cfg.T_total
                D = cfg.d_model
                cur = col.encoder(token_ids)
                isp, _ = col.input_lif(cur)
                x = isp.reshape(T_t, token_ids.shape[0], token_ids.shape[1], D)
                for bl in col.sensory_blocks:
                    x, _ = bl(x)
                for bl in col.association_blocks:
                    x, _ = bl(x)
                # Flatten to (N, D) for rank and cosine computations.
                outputs[name] = x.reshape(-1, D).detach()
            except Exception:
                pass
    return outputs


def _extract_subspace_ranks(assoc_outputs: Dict[str, Tensor]) -> Dict[str, float]:
    """
    Compute effective rank of each column's association zone output.

    Reference: Roy O, Vetterli M (2007). DOI: {To be added later.}

    Args:
        assoc_outputs: dict from _extract_association_outputs().

    Returns:
        Dict mapping column name -> effective rank scalar.
    """
    return {name: _effective_rank(out) for name, out in assoc_outputs.items()}


def _extract_cosine_distances(
    assoc_outputs: Dict[str, Tensor],
    prime_name: str = "prime",
) -> Dict[str, float]:
    """
    Compute mean cosine distance between Prime and each specialist's
    association zone output.

    Distance = 1 - cosine_similarity, averaged across the N token positions.
    At clone initialization all specialists are identical to Prime and
    distance is zero. Increasing distance signals genuine divergence.

    BIOLOGICAL FUNCTION: Measures how much each specialist's representational
    geometry has diverged from the broadband coordination scaffold. The
    ensemble becomes genuinely differentiated when specialist subspaces
    become non-overlapping with Prime's.

    Reference: See JZ et al. (2018). DOI: 10.7554/eLife.35587

    Args:
        assoc_outputs: dict from _extract_association_outputs().
        prime_name: key for Prime's output in assoc_outputs.

    Returns:
        Dict mapping specialist name -> mean cosine distance from Prime.
        Returns empty dict if Prime output not found.
    """
    prime_out = assoc_outputs.get(prime_name)
    if prime_out is None:
        return {}
    distances: Dict[str, float] = {}
    with torch.no_grad():
        for name, out in assoc_outputs.items():
            if name == prime_name:
                continue
            if out.shape != prime_out.shape:
                continue
            sim = F.cosine_similarity(
                prime_out.float(), out.float(), dim=-1
            )
            distances[name] = (1.0 - sim).mean().item()
    return distances


# =========================================================================
# Monitor State
# =========================================================================

@dataclass
class MonitorState:
    """
    Rolling state for the Phase 2 array monitor.

    Maintains per-step signal histories for all four diagnostic metrics.
    Call monitor_step() each time you want to record a measurement.

    NOT a biological structure. Computational diagnostic accumulator.
    """

    cfg: MonitorConfig = field(default_factory=MonitorConfig)

    # Per-step histories. Each entry is one monitor_step() call.
    router_entropy_history: List[float] = field(default_factory=list)
    activation_freq_history: List[Dict[str, float]] = field(default_factory=list)
    subspace_rank_history: List[Dict[str, float]] = field(default_factory=list)
    cosine_distance_history: List[Dict[str, float]] = field(default_factory=list)
    step_history: List[int] = field(default_factory=list)

    # Live alerts: list of (step, message) tuples.
    alerts: List[Tuple[int, str]] = field(default_factory=list)

    def get_latest(self) -> Dict[str, Any]:
        """
        Return the most recent snapshot across all four metrics.

        Returns empty dict if no steps have been recorded yet.
        """
        if not self.step_history:
            return {}
        return {
            "step": self.step_history[-1],
            "router_entropy": (
                self.router_entropy_history[-1]
                if self.router_entropy_history else None
            ),
            "activation_frequencies": (
                self.activation_freq_history[-1]
                if self.activation_freq_history else None
            ),
            "subspace_ranks": (
                self.subspace_rank_history[-1]
                if self.subspace_rank_history else None
            ),
            "cosine_distances": (
                self.cosine_distance_history[-1]
                if self.cosine_distance_history else None
            ),
            "alerts_since_last": [
                a for a in self.alerts
                if a[0] == self.step_history[-1]
            ],
        }

    def format_line(self) -> str:
        """
        Return a compact single-line summary of the latest snapshot for
        stdout logging during training.
        """
        d = self.get_latest()
        if not d:
            return "[monitor] no data yet"
        step = d["step"]
        entropy = d["router_entropy"]
        ranks = d["subspace_ranks"] or {}
        dists = d["cosine_distances"] or {}
        freqs = d["activation_frequencies"] or {}

        rank_str = " ".join(
            f"{n[:3]}={v:.1f}" for n, v in sorted(ranks.items())
        )
        dist_str = " ".join(
            f"{n[:3]}={v:.3f}" for n, v in sorted(dists.items())
        )
        freq_str = " ".join(
            f"{n[:3]}={v:.2f}" for n, v in sorted(freqs.items())
        )
        alert_flag = " [ALERT]" if d["alerts_since_last"] else ""
        return (
            f"[monitor step={step}] "
            f"entropy={entropy:.3f} | "
            f"rank: {rank_str} | "
            f"dist_from_prime: {dist_str} | "
            f"freq: {freq_str}"
            f"{alert_flag}"
        )


# =========================================================================
# Core Monitor Step
# =========================================================================

def monitor_step(
    array,
    state: MonitorState,
    training_step: int,
    probe_token_ids: Optional[Tensor] = None,
) -> Dict[str, Any]:
    """
    Advance the monitor by one call. Records all four diagnostic signals
    and evaluates alert conditions.

    Router entropy and activation frequency are extracted directly from
    array buffers (cheap, no forward pass). Subspace rank and cosine
    distance require a forward pass through each column's association zone
    (more expensive). If probe_token_ids is None, the rank and cosine
    distance metrics are skipped for that step.

    For training efficiency, call monitor_step() with probe_token_ids=None
    at every log_every step to capture entropy and frequency cheaply, and
    supply probe_token_ids only every N*log_every steps for the full
    subspace diagnostics.

    ANATOMICAL INTERFACE:
        Sending structures: ColumnRouter (routing statistics), TimmyArray
        specialist_selection_ema (activation frequency), each column's
        association_blocks (subspace geometry).
        Receiving structure: MonitorState histories and train_array.py logging.
        Connection: internal diagnostic hook.

    Args:
        array: TimmyArray instance (should be in eval mode for association
            pass; caller is responsible for mode management).
        state: MonitorState carrying rolling histories.
        training_step: current optimizer step.
        probe_token_ids: optional (B, S) token ids for association zone
            forward pass. If None, subspace rank and cosine distance are
            skipped this step.

    Returns:
        Dict matching MonitorState.get_latest() format.
    """
    cfg = state.cfg

    # Always-cheap signals: extracted from buffers, no forward pass.
    entropy = _extract_router_entropy(array)
    freqs = _extract_activation_frequencies(array)

    state.router_entropy_history.append(entropy)
    state.activation_freq_history.append(freqs)

    # Optional subspace signals: require association zone forward pass.
    if probe_token_ids is not None:
        assoc_outputs = _extract_association_outputs(array, probe_token_ids)
        ranks = _extract_subspace_ranks(assoc_outputs)
        dists = _extract_cosine_distances(assoc_outputs)
    else:
        ranks = {}
        dists = {}

    state.subspace_rank_history.append(ranks)
    state.cosine_distance_history.append(dists)
    state.step_history.append(training_step)

    # Alert evaluation.
    if entropy < cfg.entropy_collapse_floor:
        msg = (
            f"Router entropy {entropy:.3f} below collapse floor "
            f"{cfg.entropy_collapse_floor}. Routing may be collapsing."
        )
        state.alerts.append((training_step, msg))

    if training_step >= cfg.divergence_expect_step and dists:
        for name, dist in dists.items():
            if dist < cfg.divergence_healthy_floor:
                msg = (
                    f"Specialist '{name}' cosine distance from Prime "
                    f"{dist:.3f} below healthy floor "
                    f"{cfg.divergence_healthy_floor} at step {training_step}. "
                    "Column may not be diverging."
                )
                state.alerts.append((training_step, msg))

    snapshot = state.get_latest()

    # File logging.
    if cfg.log_dir and cfg.log_every > 0 and training_step % cfg.log_every == 0:
        _write_log_entry(cfg.log_dir, training_step, snapshot)

    return snapshot


# =========================================================================
# File Logging
# =========================================================================

def _write_log_entry(log_dir: str, step: int, snapshot: Dict) -> None:
    """
    Append a JSON log entry to array_monitor.jsonl in log_dir.

    Each line is one JSON object. The file can be streamed and parsed
    incrementally during training without loading the full history.

    NOT a biological structure. I/O utility.

    Args:
        log_dir: directory path. Created if it does not exist.
        step: training step for filename partitioning.
        snapshot: dict from MonitorState.get_latest().
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = Path(log_dir) / "array_monitor.jsonl"
    entry = {"step": step, "ts": time.time()}

    # Serialize tensor-free snapshot (alerts contain tuples, convert).
    def _safe(v):
        if isinstance(v, (int, float, str, bool, type(None))):
            return v
        if isinstance(v, dict):
            return {str(k): _safe(vv) for k, vv in v.items()}
        if isinstance(v, (list, tuple)):
            return [_safe(x) for x in v]
        return str(v)

    entry.update(_safe(snapshot))
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


# =========================================================================
# Summary Report
# =========================================================================

def summarize(state: MonitorState, last_n: int = 50) -> str:
    """
    Generate a human-readable summary of the last_n monitor steps.

    Suitable for printing at the end of a training run or at Phase 2
    checkpoint intervals to assess whether divergence is progressing.

    Args:
        state: MonitorState with accumulated history.
        last_n: number of most recent steps to summarize.

    Returns:
        Multi-line string summary.
    """
    if not state.step_history:
        return "MonitorState: no data recorded."

    steps = state.step_history[-last_n:]
    entropies = state.router_entropy_history[-last_n:]
    ranks_list = state.subspace_rank_history[-last_n:]
    dists_list = state.cosine_distance_history[-last_n:]

    lines = [
        "",
        "=" * 60,
        f"ARRAY MONITOR SUMMARY (last {len(steps)} steps)",
        f"Steps {steps[0]} - {steps[-1]}",
        "=" * 60,
        "",
        "ROUTER ENTROPY",
        "-" * 40,
        f"  Mean:  {sum(entropies)/len(entropies):.4f}",
        f"  Min:   {min(entropies):.4f}",
        f"  Max:   {max(entropies):.4f}",
        f"  Last:  {entropies[-1]:.4f}",
        f"  Collapse floor: {state.cfg.entropy_collapse_floor}",
        "",
        "SUBSPACE EFFECTIVE RANK (last step with data)",
        "-" * 40,
    ]

    last_ranks = next(
        (r for r in reversed(ranks_list) if r), {}
    )
    for name, rank in sorted(last_ranks.items()):
        lines.append(f"  {name:12s}: {rank:.2f}")

    lines += [
        "",
        "COSINE DISTANCE FROM PRIME (last step with data)",
        "-" * 40,
    ]
    last_dists = next(
        (d for d in reversed(dists_list) if d), {}
    )
    for name, dist in sorted(last_dists.items()):
        healthy = dist >= state.cfg.divergence_healthy_floor
        flag = "" if healthy else " [LOW]"
        lines.append(f"  {name:12s}: {dist:.4f}{flag}")

    lines += [
        "",
        f"ALERTS ({len(state.alerts)} total)",
        "-" * 40,
    ]
    recent_alerts = [a for a in state.alerts if a[0] >= steps[0]]
    if recent_alerts:
        for step, msg in recent_alerts[-10:]:
            lines.append(f"  [step={step}] {msg}")
    else:
        lines.append("  None.")

    lines += ["=" * 60, ""]
    return "\n".join(lines)
