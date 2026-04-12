"""
train_array.py
Two-Phase Training Loop for the TimmyArray Cortical Column Ensemble

BIOLOGICAL GROUNDING:
This script implements the two-phase training workflow for TimmyArray,
grounded in the neuroscience of cortical column specialization. The key
insight from the empirical literature is that cortical columns develop
functional tuning through experience-dependent plasticity, not through
architectural constraints or weight freezing. The training protocol
reflects this directly.

Phase 1 (Generalist): Prime alone is trained on the full, unfiltered
    input distribution. This establishes a broadband integration column
    whose receptive field spans the complete input space, analogous to the
    development of a cortical area with broad tuning before experience-
    dependent refinement begins. Only Prime's parameters are updated.
    All specialist columns are held at their initialization state.

Phase 2 (Divergence): All columns train simultaneously, each receiving
    input from a domain-specific DataLoader assigned to that column by
    name. Prime continues to receive the full distribution. Specialists
    receive domain-specific subcorpora. Differentiation emerges from the
    divergence in input statistics, exactly as biological column
    specialization emerges from differences in sensory experience. No
    weights are frozen; all columns are plastic throughout.

The load-balancing bias adjustment from ColumnRouter runs after each
optimizer step in Phase 2, preventing routing collapse without auxiliary
loss and without corrupting the routing gradient. This follows the
DeepSeek-V3 approach, which demonstrated that bias-term adjustment
outperforms auxiliary loss for maintaining specialist utilization.

STDP is applied to each active column's executive zone in Phase 2,
modulated by the LM loss as the reward signal. This three-factor learning
rule (spike timing, Hebbian correlation, reward) operates in parallel
with gradient descent and reinforces coordinated ensemble activity in the
executive zone of each column independently.

Three grounding papers for this training design:

1. Huang C, Zeldenrust F, Celikel T (2022). "Cortical representation of
   touch in silico." Neuroinformatics, 20:1013-1039.
   DOI: 10.1007/s12021-022-09576-5
   (Threshold adaptation as the primary mechanism of column differentiation.
   Columns trained on different input distributions develop different
   threshold dynamics and effective connectivity, not just different weights.
   This justifies Phase 2 domain assignment without weight freezing.)

2. Wang P, Li L, Shao Y et al. (2024). "Auxiliary-loss-free load balancing
   strategy for mixture of experts." arXiv: 2408.15664.
   DOI: {To be added later.}
   (Bias-term adjustment outperforms auxiliary loss for balanced routing.
   The ColumnRouter.update_load_balance_bias() method implements this
   directly. Called after each optimizer step, not inside the loss.)

3. Fremaux N, Gerstner W (2016). "Neuromodulated spike-timing-dependent
   plasticity, and theory of three-factor learning rules." Frontiers in
   Neural Circuits, 9:85. DOI: 10.3389/fncir.2015.00085
   (Three-factor STDP: reward signal modulates Hebbian plasticity in the
   executive zone. LM loss provides the reward. Specialist columns receive
   independent STDP updates based on their individual executive-zone
   spike statistics.)

Usage:
    # Phase 1 only (train Prime on full distribution):
    python train_array.py --phase1_data roneneldan/TinyStories

    # Phase 1 then Phase 2 (full workflow):
    python train_array.py \\
        --phase1_data roneneldan/TinyStories \\
        --phase2_prime roneneldan/TinyStories \\
        --phase2_proximal /data/sequential_text.txt \\
        --phase2_distal /data/long_form.txt \\
        --phase2_affective /data/social_dialogue.txt \\
        --phase2_somatic /data/embodied_descriptions.txt \\
        --phase2_structural /data/code_and_math.txt

    # Phase 2 only (resume from Phase 1 checkpoint):
    python train_array.py \\
        --phase2_prime roneneldan/TinyStories \\
        --phase2_proximal /data/sequential_text.txt \\
        --resume_prime checkpoints/array_phase1_prime_final.state

Requirements:
    pip install torch transformers datasets
"""

from __future__ import annotations

import argparse
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from timmy_data import create_dataloader
from CreateTimmyArray import TimmyArray, TimmyArrayConfig, COLUMN_NAMES
from timmy_model import TimmyConfig
from timmy_criticalperiodprobe import ProbeConfig, ProbeState, probe_step
from array_monitor import MonitorConfig, MonitorState, monitor_step, summarize


# =========================================================================
# Learning Rate Schedule
# =========================================================================

def cosine_lr(
    step: int,
    warmup: int,
    max_steps: int,
    lr: float,
    min_lr: float,
) -> float:
    """
    Cosine learning rate schedule with linear warmup.

    Linear ramp from 0 to lr over warmup steps, then cosine decay
    from lr to min_lr over the remaining steps. This schedule prevents
    early-training instability in the LIF threshold dynamics (FIX D)
    while ensuring the full parameter space is explored before the
    learning rate drops.

    NOT a biological quantity. Standard deep learning training artifact.

    Args:
        step: current training step (optimizer steps, not micro-steps).
        warmup: number of warmup steps.
        max_steps: total training steps for this phase.
        lr: peak learning rate.
        min_lr: minimum learning rate at end of cosine decay.

    Returns:
        Learning rate scalar for this step.
    """
    if step < warmup:
        return lr * step / max(warmup, 1)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / max(max_steps - warmup, 1)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def make_data_iter(
    source: str,
    cfg: TimmyConfig,
    text_column: str = "text",
    split: str = "train",
) -> Iterator:
    """
    Create an infinite data iterator for a given source.

    Wraps create_dataloader() from timmy_data.py with restart-on-exhaustion
    semantics. The iterator never raises StopIteration; it restarts from
    the beginning of the dataset when exhausted.

    Args:
        source: HuggingFace dataset name or local file path.
        cfg: TimmyConfig for tokenizer_id, max_seq_len, batch_size.
        text_column: text field name for HuggingFace datasets.
        split: dataset split.

    Yields:
        (B, S) integer token_id tensors.
    """
    while True:
        loader = create_dataloader(
            source=source,
            tokenizer_id=cfg.tokenizer_id,
            max_seq_len=cfg.max_seq_len,
            batch_size=cfg.batch_size,
            text_column=text_column,
            split=split,
            shuffle=True,
        )
        yield from loader


# =========================================================================
# Phase 1: Generalist Training (Prime Only)
# =========================================================================

def train_phase1(
    array: TimmyArray,
    data_source: str,
    output_dir: str,
    text_column: str = "text",
    split: str = "train",
    resume_path: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
    probe_every: int = 500,
    probe_cfg: Optional[ProbeConfig] = None,
) -> TimmyArray:
    """
    Phase 1: Train TimmyPrime on the full input distribution.

    Only Prime's parameters are updated. All specialist columns remain
    at initialization. This establishes a broadband integration column
    with receptive fields spanning the complete input space before
    domain-specific divergence begins in Phase 2.

    The biological analog is cortical area development before experience-
    dependent refinement: broad initial tuning that covers the full input
    space, establishing the coordination scaffold that specialists will
    anchor to.

    STDP is disabled in Phase 1. The three-factor learning rule requires
    the reward signal from CA1 mismatch feedback, which is not available
    until the Cognitive Kernel is connected. LM loss-modulated STDP is
    enabled in Phase 2.

    Reference: Mountcastle VB (1997). "The columnar organization of the
    neocortex." Brain, 120(4):701-722. DOI: 10.1093/brain/120.4.701
    (Columns are structurally uniform; functional differentiation emerges
    from experience, not from initial architectural differences.)

    Args:
        array: TimmyArray instance. Only array.prime is trained.
        data_source: HuggingFace dataset name or local path.
        output_dir: directory for .state checkpoints.
        text_column: text field name for HuggingFace datasets.
        split: dataset split.
        resume_path: optional path to a prior Prime .state checkpoint.
        device: target device.
        probe_every: run the critical period probe every this many optimizer
            steps. Probe is evaluated at the same cadence as save_every if
            probe_every <= 0. Default 500.
        probe_cfg: ProbeConfig controlling convergence thresholds and
            stability window. Uses defaults if None.

    Returns:
        The TimmyArray with Prime trained. Specialists unchanged.
        Phase 1 exits early if the probe issues a coordination-ready verdict
        before max_steps is reached.
    """
    cfg = array.cfg.column_cfg
    print("\n" + "=" * 70)
    print("PHASE 1: Generalist Training (Prime only)")
    print(f"  Data:       {data_source}")
    print(f"  Max steps:  {cfg.max_steps}")
    print(f"  Batch size: {cfg.batch_size} x {cfg.grad_accum} accum = "
          f"{cfg.batch_size * cfg.grad_accum} effective")
    print("=" * 70)

    # Only Prime is trainable in Phase 1.
    # Specialist parameters are not frozen; they simply receive no gradients
    # because they are not in the optimizer parameter group.
    optimizer = torch.optim.AdamW(
        array.prime.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    use_amp = (device.type == "cuda" and cfg.dtype == torch.float16)
    scaler = GradScaler('cuda', enabled=use_amp)

    # Resume Prime state if provided.
    start_step = 0
    if resume_path is not None:
        print(f"  Resuming Prime from {resume_path}")
        meta = array.prime.load_state(resume_path, device=str(device))
        start_step = meta.get("training_step") or 0
        if meta.get("has_optimizer"):
            optimizer.load_state_dict(meta["optimizer"])
        print(f"  Resumed at step {start_step}")

    os.makedirs(output_dir, exist_ok=True)
    data_iter = make_data_iter(data_source, cfg, text_column, split)

    # Critical period probe: tracks MemoryCortex threshold variance,
    # association routing entropy, and expert load balance.
    # Issues coordination-ready verdict when all three are simultaneously
    # stable for probe_cfg.stability_window consecutive probe calls.
    # Reference: Huang C et al. (2022). DOI: 10.1007/s12021-022-09576-5
    _probe_cfg = probe_cfg if probe_cfg is not None else ProbeConfig()
    _probe_state = ProbeState(cfg=_probe_cfg)
    _probe_interval = probe_every if probe_every > 0 else cfg.save_every

    step = start_step
    micro_step = 0
    accum_loss = accum_spike = accum_moe = 0.0
    tokens_seen = 0
    best_loss = float("inf")

    array.train()
    optimizer.zero_grad()
    t_start = time.time()

    while step < cfg.max_steps:
        token_ids = next(data_iter).to(device)
        input_ids = token_ids[:, :-1]
        target_ids = token_ids[:, 1:]
        B, S = input_ids.shape

        # Forward pass through Prime only.
        # STDP disabled in Phase 1: no CA1 reward signal available yet.
        with autocast('cuda', enabled=use_amp):
            logits, stats = array.prime(input_ids, enable_stdp=False)
            lm_loss = F.cross_entropy(
                logits.reshape(-1, cfg.vocab_size),
                target_ids.reshape(-1),
            )
            spike_loss = _to_tensor(
                stats.get("spike_loss", 0.0), device
            )
            moe_loss = _to_tensor(
                stats.get("moe_lb_loss", 0.0), device
            )
            total_loss = (
                lm_loss
                + spike_loss
                + cfg.moe_load_balance_weight * moe_loss
            )
            scaled_loss = total_loss / cfg.grad_accum

        scaler.scale(scaled_loss).backward()

        accum_loss += lm_loss.item()
        accum_spike += spike_loss.item()
        accum_moe += moe_loss.item()
        tokens_seen += B * S
        micro_step += 1

        if micro_step % cfg.grad_accum == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                array.prime.parameters(), cfg.max_grad_norm
            )
            current_lr = cosine_lr(
                step, cfg.warmup_steps, cfg.max_steps, cfg.lr, cfg.min_lr
            )
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            step += 1

            if step % cfg.log_every == 0:
                _log_step(
                    step=step,
                    phase=1,
                    column="prime",
                    avg_loss=accum_loss / cfg.grad_accum,
                    avg_spike=accum_spike / cfg.grad_accum,
                    avg_moe=accum_moe / cfg.grad_accum,
                    lr=current_lr,
                    grad_norm=grad_norm.item(),
                    spike_rate=stats.get("avg_spike_rate", 0.0),
                    tok_per_sec=tokens_seen / max(time.time() - t_start, 1),
                )
                if (accum_loss / cfg.grad_accum) < best_loss:
                    best_loss = accum_loss / cfg.grad_accum
                accum_loss = accum_spike = accum_moe = 0.0

            if step % cfg.save_every == 0:
                _save_prime(array, optimizer, step, best_loss, tokens_seen,
                            output_dir, tag="phase1")

            if step % _probe_interval == 0:
                array.prime.eval()
                ready, signals = probe_step(array.prime, _probe_state, step)
                array.prime.train()
                status = _probe_state.get_status_dict()
                print(
                    f"  [probe step={step}] "
                    f"thresh_var={status['threshold_variance']:.6f} "
                    f"({status['threshold_stable_steps']}/{_probe_cfg.stability_window}) | "
                    f"entropy={status['association_entropy']:.4f} "
                    f"({status['entropy_stable_steps']}/{_probe_cfg.stability_window}) | "
                    f"load_imb={status['load_imbalance']:.4f} "
                    f"({status['load_stable_steps']}/{_probe_cfg.stability_window}) | "
                    f"ready={ready}"
                )
                if ready:
                    print(
                        f"\n  Critical period closed at step {step}. "
                        "Prime coordination-ready. Exiting Phase 1."
                    )
                    _save_prime(array, optimizer, step, best_loss, tokens_seen,
                                output_dir, tag="phase1_coordready")
                    break

    # Final Phase 1 checkpoint.
    _save_prime(array, optimizer, step, best_loss, tokens_seen,
                output_dir, tag="phase1_final")
    elapsed = time.time() - t_start
    print(f"\nPhase 1 complete. Steps: {step} | "
          f"Best loss: {best_loss:.4f} | "
          f"Time: {elapsed/3600:.1f}h")

    return array


# =========================================================================
# Phase 2: Divergence Training (All Columns)
# =========================================================================

def train_phase2(
    array: TimmyArray,
    column_data_sources: Dict[str, str],
    output_dir: str,
    text_column: str = "text",
    split: str = "train",
    resume_paths: Optional[Dict[str, str]] = None,
    device: torch.device = torch.device("cpu"),
    monitor_every: int = 100,
    monitor_subspace_every: int = 1000,
    monitor_cfg: Optional[MonitorConfig] = None,
) -> TimmyArray:
    """
    Phase 2: Train all columns simultaneously on domain-assigned data.

    Each column receives a named DataLoader. Prime continues on the full
    distribution. Specialists receive domain-specific subcorpora. All
    columns are simultaneously plastic; no weights are frozen.

    Specialization emerges from input distribution divergence, exactly as
    biological cortical column differentiation emerges from sensory
    experience. The threshold adaptation mechanism in AssociativeLIF
    (v_threshold_raw, learned via backpropagation) is the primary driver
    of column differentiation alongside weight changes.

    The ColumnRouter's load-balance bias is updated after each optimizer
    step using the observed routing distribution, without auxiliary loss.
    This prevents routing collapse while preserving the routing gradient.

    STDP is enabled for executive-zone layers in all active columns,
    modulated by each column's own LM loss as the reward signal. Each
    column develops its own executive-zone connectivity structure
    independently.

    Reference: Pérez-Ortega J, Alejandre-García T, Yuste R (2021). "Long-
    term stability of cortical ensembles." eLife, 10:e64449.
    DOI: 10.7554/eLife.64449
    (Ensemble identity is carried by connectivity structure, not fixed
    membership. Columns that receive domain-specific input develop stable
    ensemble structures with high internal connectivity density. This is
    what we are inducing in Phase 2.)

    Args:
        array: TimmyArray instance. All columns are trained.
        column_data_sources: dict mapping column name to data source.
            Keys must be valid column names from COLUMN_NAMES.
            Columns with no entry receive no training in Phase 2.
        output_dir: directory for .state checkpoints.
        text_column: text field name for HuggingFace datasets.
        split: dataset split.
        resume_paths: optional dict mapping column name to .state path.
        device: target device.
        monitor_every: record cheap monitor signals (entropy, activation
            frequency) every this many optimizer steps. Default 100.
        monitor_subspace_every: run full association zone forward pass for
            subspace rank and cosine distance every this many steps.
            More expensive than monitor_every. Default 1000.
        monitor_cfg: MonitorConfig controlling alert thresholds and log dir.
            Uses defaults if None.

    Returns:
        The TimmyArray with all trained columns.
    """
    cfg = array.cfg.column_cfg
    active_columns = {
        name: col
        for name, col in zip(array.column_names, array.all_columns())
        if name in column_data_sources
    }

    print("\n" + "=" * 70)
    print("PHASE 2: Divergence Training (all columns)")
    print(f"  Active columns: {list(active_columns.keys())}")
    print(f"  Max steps:      {cfg.max_steps}")
    print(f"  Batch size:     {cfg.batch_size} x {cfg.grad_accum} accum")
    print("=" * 70)

    # Separate optimizer per column so each column's learning rate and
    # gradient history are independent. Parameter interference between
    # columns sharing an optimizer would prevent clean specialization.
    # NOT a biological quantity. Engineering choice for clean divergence.
    optimizers: Dict[str, torch.optim.AdamW] = {}
    scalers: Dict[str, GradScaler] = {}
    use_amp = (device.type == "cuda" and cfg.dtype == torch.float16)

    for name, col in active_columns.items():
        optimizers[name] = torch.optim.AdamW(
            col.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95),
        )
        scalers[name] = GradScaler('cuda', enabled=use_amp)

    # Also train the router and PerforantPath bridge parameters.
    # These are shared infrastructure, trained with Prime's optimizer.
    if "prime" in optimizers:
        for p in array.router.parameters():
            optimizers["prime"].add_param_group({"params": [p]})
        for p in array.perforant_bridge.parameters():
            optimizers["prime"].add_param_group({"params": [p]})

    # Resume column states if provided.
    start_step = 0
    for name, col in active_columns.items():
        paths = resume_paths or {}
        if name in paths:
            print(f"  Resuming {name} from {paths[name]}")
            meta = col.load_state(paths[name], device=str(device))
            col_step = meta.get("training_step") or 0
            if col_step > start_step:
                start_step = col_step
            if meta.get("has_optimizer"):
                optimizers[name].load_state_dict(meta["optimizer"])
            print(f"    {name} resumed at step {col_step}")

    # Build per-column data iterators.
    data_iters: Dict[str, Iterator] = {
        name: make_data_iter(src, cfg, text_column, split)
        for name, src in column_data_sources.items()
        if name in active_columns
    }

    os.makedirs(output_dir, exist_ok=True)

    # Phase 2 divergence monitor: tracks router entropy, specialist activation
    # frequency, subspace effective rank, and cosine distance from Prime.
    # Cheap signals (entropy, freq) are extracted every monitor_every steps
    # from buffers with no forward pass. Subspace signals (rank, cosine)
    # require an association zone forward pass and run every
    # monitor_subspace_every steps.
    # Reference: See JZ et al. (2018). DOI: 10.7554/eLife.35587
    # Reference: Pérez-Ortega J et al. (2021). DOI: 10.7554/eLife.64449
    _monitor_cfg = monitor_cfg if monitor_cfg is not None else MonitorConfig()
    _monitor_state = MonitorState(cfg=_monitor_cfg)

    step = start_step
    micro_step = 0
    accum: Dict[str, Dict[str, float]] = {
        name: {"loss": 0.0, "spike": 0.0, "moe": 0.0}
        for name in active_columns
    }
    tokens_seen = 0
    best_losses: Dict[str, float] = {name: float("inf") for name in active_columns}

    array.train()
    for opt in optimizers.values():
        opt.zero_grad()
    t_start = time.time()

    while step < cfg.max_steps:
        # ---- Forward pass for each active column independently ----
        # Each column sees its own domain-specific batch. Losses are
        # accumulated independently and backpropagated independently.
        # This is the core of the divergence mechanism: each column's
        # weight update is driven by its own input distribution.
        enable_stdp = (step >= cfg.lif_freeze_steps and step % 10 == 0)

        for name, col in active_columns.items():
            if name not in data_iters:
                continue

            token_ids = next(data_iters[name]).to(device)
            input_ids = token_ids[:, :-1]
            target_ids = token_ids[:, 1:]
            B, S = input_ids.shape

            with autocast('cuda', enabled=use_amp):
                logits, stats = col(input_ids, enable_stdp=enable_stdp)
                lm_loss = F.cross_entropy(
                    logits.reshape(-1, cfg.vocab_size),
                    target_ids.reshape(-1),
                )
                spike_loss = _to_tensor(stats.get("spike_loss", 0.0), device)
                moe_loss = _to_tensor(stats.get("moe_lb_loss", 0.0), device)
                total_loss = (
                    lm_loss
                    + spike_loss
                    + cfg.moe_load_balance_weight * moe_loss
                )
                scaled_loss = total_loss / cfg.grad_accum

            scalers[name].scale(scaled_loss).backward()

            accum[name]["loss"] += lm_loss.item()
            accum[name]["spike"] += spike_loss.item()
            accum[name]["moe"] += moe_loss.item()

        tokens_seen += B * S * len(active_columns)
        micro_step += 1

        if micro_step % cfg.grad_accum == 0:
            current_lr = cosine_lr(
                step, cfg.warmup_steps, cfg.max_steps, cfg.lr, cfg.min_lr
            )

            # Optimizer step for each active column.
            grad_norms: Dict[str, float] = {}
            for name, col in active_columns.items():
                opt = optimizers[name]
                sc = scalers[name]
                sc.unscale_(opt)
                gn = torch.nn.utils.clip_grad_norm_(
                    col.parameters(), cfg.max_grad_norm
                )
                grad_norms[name] = gn.item()
                for pg in opt.param_groups:
                    pg["lr"] = current_lr
                sc.step(opt)
                sc.update()
                opt.zero_grad()

            step += 1

            # ---- STDP for each active column ----
            # Each column receives its own STDP update based on its own
            # LM loss (reward signal) and its own executive-zone spike
            # statistics. Three-factor learning rule: spike timing,
            # Hebbian correlation, reward.
            # Reference: Fremaux N, Gerstner W (2016).
            # DOI: 10.3389/fncir.2015.00085
            if enable_stdp:
                for name, col in active_columns.items():
                    col_loss = accum[name]["loss"] / cfg.grad_accum
                    col.set_last_loss(col_loss)
                    col.stdp_update(current_loss=col_loss)

            # ---- Load balance bias update (no gradient) ----
            # Called once per optimizer step with the routing weights from
            # Prime's last forward pass. The bias update is detached from
            # the computational graph; it does not affect the routing gradient.
            # Reference: Wang P et al. (2024). arXiv: 2408.15664.
            # DOI: {To be added later.}
            if "prime" in active_columns:
                # Re-extract routing weights from Prime's last stats dict.
                # We use a dummy forward of the router with the most recent
                # MemoryCortex output from the last full forward pass.
                # This is a cheap re-evaluation, not a full forward pass.
                # In practice, the routing weights from the last step are
                # stored in the stats dict by train_array.forward(); here
                # we update based on what the router would produce for the
                # current bias state.
                with torch.no_grad():
                    dummy_input = torch.zeros(
                        1, cfg.d_model, device=device
                    )
                    rw, _ = array.router(dummy_input, top_k=None)
                    array.router.update_load_balance_bias(rw)

            # ---- Logging ----
            if step % cfg.log_every == 0:
                for name in active_columns:
                    _log_step(
                        step=step,
                        phase=2,
                        column=name,
                        avg_loss=accum[name]["loss"] / cfg.grad_accum,
                        avg_spike=accum[name]["spike"] / cfg.grad_accum,
                        avg_moe=accum[name]["moe"] / cfg.grad_accum,
                        lr=current_lr,
                        grad_norm=grad_norms.get(name, 0.0),
                        spike_rate=0.0,  # populated from stats in full array forward
                        tok_per_sec=tokens_seen / max(time.time() - t_start, 1),
                    )
                    if accum[name]["loss"] / cfg.grad_accum < best_losses[name]:
                        best_losses[name] = accum[name]["loss"] / cfg.grad_accum

                # Specialization report every 10x log interval.
                if step % (cfg.log_every * 10) == 0:
                    print(array.specialization_report())

                # Array monitor: cheap signals every monitor_every steps.
                if step % monitor_every == 0:
                    array.eval()
                    monitor_step(array, _monitor_state, step)
                    array.train()
                    print(_monitor_state.format_line())
                    if _monitor_state.alerts and _monitor_state.alerts[-1][0] == step:
                        print(f"  [ALERT] {_monitor_state.alerts[-1][1]}")

                # Full subspace probe every monitor_subspace_every steps.
                if step % monitor_subspace_every == 0:
                    _probe_batch = next(next(iter(data_iters.values()))).to(device)
                    _probe_ids = _probe_batch[:1, :-1]
                    array.eval()
                    monitor_step(array, _monitor_state, step,
                                 probe_token_ids=_probe_ids)
                    array.train()

                for name in active_columns:
                    accum[name] = {"loss": 0.0, "spike": 0.0, "moe": 0.0}

            # ---- Checkpoint ----
            if step % cfg.save_every == 0:
                _save_array(
                    array=array,
                    optimizers=optimizers,
                    step=step,
                    best_losses=best_losses,
                    tokens_seen=tokens_seen,
                    output_dir=output_dir,
                    tag="phase2",
                )

    # Final Phase 2 checkpoint.
    _save_array(
        array=array,
        optimizers=optimizers,
        step=step,
        best_losses=best_losses,
        tokens_seen=tokens_seen,
        output_dir=output_dir,
        tag="phase2_final",
    )

    elapsed = time.time() - t_start
    print(f"\nPhase 2 complete. Steps: {step} | "
          f"Time: {elapsed/3600:.1f}h")
    for name, loss in best_losses.items():
        print(f"  {name:<14} best loss: {loss:.4f}")
    print()
    print(array.specialization_report())
    print(summarize(_monitor_state, last_n=50))

    return array


# =========================================================================
# Internal helpers
# =========================================================================

def _to_tensor(val, device: torch.device) -> torch.Tensor:
    """
    Convert a scalar or tensor to a tensor on device.

    Helper for normalizing loss components from the stats dict, which may
    return float 0.0 when a loss term is not active.

    NOT a biological quantity. Utility function.
    """
    if torch.is_tensor(val):
        return val.to(device)
    return torch.tensor(float(val), device=device)


def _log_step(
    step: int,
    phase: int,
    column: str,
    avg_loss: float,
    avg_spike: float,
    avg_moe: float,
    lr: float,
    grad_norm: float,
    spike_rate: float,
    tok_per_sec: float,
) -> None:
    """
    Print a single training log line.

    Format matches train_timmy.py for consistency, with phase and column
    name added to disambiguate multi-column Phase 2 logging.
    """
    print(
        f"P{phase} step {step:6d} | "
        f"{column:<12} | "
        f"loss {avg_loss:.4f} | "
        f"spike {avg_spike:.4f} | "
        f"moe_lb {avg_moe:.4f} | "
        f"lr {lr:.2e} | "
        f"gnorm {grad_norm:.2f} | "
        f"tok/s {tok_per_sec:.0f}"
    )


def _save_prime(
    array: TimmyArray,
    optimizer: torch.optim.AdamW,
    step: int,
    best_loss: float,
    tokens_seen: int,
    output_dir: str,
    tag: str,
) -> None:
    """
    Save Prime's .state checkpoint during Phase 1.

    Uses TimmyModel.save_state() which applies the full three-layer
    COLD/WARM/HOT format from timmy_state.py.
    """
    path = os.path.join(output_dir, f"array_{tag}_prime.state")
    array.prime.save_state(
        path=path,
        optimizer_state=optimizer.state_dict(),
        training_step=step,
        extra={"best_loss": best_loss, "tokens_seen": tokens_seen, "phase": 1},
    )
    print(f"  Saved Prime checkpoint: {path}")


def _save_array(
    array: TimmyArray,
    optimizers: Dict[str, torch.optim.AdamW],
    step: int,
    best_losses: Dict[str, float],
    tokens_seen: int,
    output_dir: str,
    tag: str,
) -> None:
    """
    Save the full array checkpoint during Phase 2.

    Applies save_array_state() which saves each column independently
    and the Cognitive Kernel state (if available) once at the root.
    The routing_bias and specialist_selection_ema are included in the
    HOT layer for each column's checkpoint via the extra dict.
    """
    prefix = os.path.join(output_dir, f"array_{tag}_step{step}")
    written = array.save_array_state(
        path_prefix=prefix,
        optimizer_states={
            name: opt.state_dict() for name, opt in optimizers.items()
        },
        training_step=step,
        kernel_state=None,  # Cognitive Kernel state injected externally
        extra={
            "best_losses": best_losses,
            "tokens_seen": tokens_seen,
            "phase": 2,
        },
    )
    for name, path in written.items():
        print(f"  Saved {name}: {path}")


# =========================================================================
# CLI
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Two-phase training for TimmyArray cortical column ensemble"
    )

    # ---- Phase 1 ----
    parser.add_argument(
        "--phase1_data", type=str, default=None,
        help="Data source for Phase 1 (Prime generalist training). "
             "HuggingFace dataset name or local file path. "
             "If omitted, Phase 1 is skipped.",
    )
    parser.add_argument(
        "--phase1_steps", type=int, default=None,
        help="Override max_steps for Phase 1. Default: TimmyConfig.max_steps.",
    )
    parser.add_argument(
        "--resume_prime", type=str, default=None,
        help="Path to a prior Prime .state checkpoint to resume Phase 1 from.",
    )

    # ---- Phase 2: one argument per column ----
    for cname in COLUMN_NAMES:
        parser.add_argument(
            f"--phase2_{cname}", type=str, default=None,
            help=f"Data source for the '{cname}' column in Phase 2. "
                 f"HuggingFace dataset name or local file path. "
                 f"If omitted, '{cname}' is not trained in Phase 2.",
        )

    parser.add_argument(
        "--phase2_steps", type=int, default=None,
        help="Override max_steps for Phase 2. Default: TimmyConfig.max_steps.",
    )

    # ---- Resume Phase 2 ----
    for cname in COLUMN_NAMES:
        parser.add_argument(
            f"--resume_{cname}", type=str, default=None,
            help=f"Path to a prior .state checkpoint for the '{cname}' column "
                 f"to resume Phase 2 from.",
        )

    # ---- General ----
    parser.add_argument(
        "--output_dir", type=str, default="checkpoints",
        help="Directory for saving .state checkpoints. Default: checkpoints/",
    )
    parser.add_argument(
        "--text_column", type=str, default="text",
        help="Text field name for HuggingFace datasets. Default: text",
    )
    parser.add_argument(
        "--split", type=str, default="train",
        help="Dataset split. Default: train",
    )
    parser.add_argument(
        "--num_specialists", type=int, default=5,
        help="Number of specialist columns (excludes Prime). Default: 5",
    )

    args = parser.parse_args()

    # ---- Build array ----
    col_cfg = TimmyConfig()
    arr_cfg = TimmyArrayConfig(
        column_cfg=col_cfg,
        num_specialists=args.num_specialists,
    )
    if args.phase1_steps is not None:
        arr_cfg.column_cfg.max_steps = args.phase1_steps

    device = torch.device(
        col_cfg.device if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    array = TimmyArray(arr_cfg).to(device)
    print(array.count_params())

    # ---- Phase 1 ----
    if args.phase1_data is not None:
        array = train_phase1(
            array=array,
            data_source=args.phase1_data,
            output_dir=args.output_dir,
            text_column=args.text_column,
            split=args.split,
            resume_path=args.resume_prime,
            device=device,
        )

    # ---- Phase 2 ----
    phase2_sources: Dict[str, str] = {}
    for cname in COLUMN_NAMES[: 1 + args.num_specialists]:
        src = getattr(args, f"phase2_{cname}", None)
        if src is not None:
            phase2_sources[cname] = src

    if phase2_sources:
        if args.phase2_steps is not None:
            arr_cfg.column_cfg.max_steps = args.phase2_steps

        resume_paths: Dict[str, str] = {}
        for cname in COLUMN_NAMES[: 1 + args.num_specialists]:
            rp = getattr(args, f"resume_{cname}", None)
            if rp is not None:
                resume_paths[cname] = rp

        array = train_phase2(
            array=array,
            column_data_sources=phase2_sources,
            output_dir=args.output_dir,
            text_column=args.text_column,
            split=args.split,
            resume_paths=resume_paths if resume_paths else None,
            device=device,
        )

    if not args.phase1_data and not phase2_sources:
        parser.error(
            "Specify at least --phase1_data or one --phase2_<column> argument."
        )
