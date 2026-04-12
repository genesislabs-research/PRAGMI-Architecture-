"""
timmy/timmy_state.py
Full-Depth State Persistence for the Timmy Spiking Language Model

PURPOSE:
This module replaces the skeleton serialize_state / load_state on TimmyModel
with a three-layer checkpoint that captures every piece of recoverable state
in the system. The goal is a round-trip where you can save mid-training, shut
down, restart on different hardware, and resume with zero cold-start transient
in the membrane dynamics and zero drift in STDP reward baselines.

STATE LAYERS:
    COLD  -- Model weights, embedding tables, registered buffers.
            Handled by PyTorch's state_dict(). Always present.

    WARM  -- Persistent membrane potentials, synaptic currents, firing rate
            EMAs, refractory counters, and learned biophysical parameters
            (threshold, tau, cascade gain) for every AssociativeLIF instance
            in the model. This includes LIF populations buried inside
            SpikingSynapticResonance (lif_q, lif_k), SpikingExpertGroup
            (lif_hidden, lif_output), SpikingFeedForward (lif_hidden,
            lif_output), MemoryCortex (memory_lif, gate_lif), and the
            top-level input_lif and readout_lif.

    HOT   -- STDP engine scalars (loss EMA, external reward, allowed set),
            MoE expert_counts_ema buffers, readout_ema_raw, and the
            training step counter from each LIF population.

ARCHITECTURE HASH:
    Every checkpoint stores a hash computed from the structurally critical
    config fields (d_model, n_heads, d_ff, n_experts, sensory_layers,
    association_layers, executive_layers, memory_size, n_clusters,
    vocab_size). On restore, the hash is recomputed from the live model's
    config and compared. A mismatch triggers a hard error rather than a
    silent partial load, because loading weights from an incompatible
    architecture produces a model that looks initialized but has random
    values in every mismatched layer.

WEIGHT HEALTH DIAGNOSTICS:
    At save time, the mean and standard deviation of the executive zone
    projection weights are captured as metadata. At load time, these are
    compared against the restored weights to detect whether the model has
    drifted into a dead zone (weights too small to produce spikes) or an
    explosion zone (every neuron pinned at max firing rate). This is a
    warning, not a hard error, because the condition may be intentional
    during early training.

BATCH SIZE HANDLING:
    Persistent _v_mem_state and _i_syn_state are shaped (B, D). If the
    checkpoint was saved with a different batch size than the one used at
    restore time, the warm state is discarded for that population and a
    warning is logged. This is correct behavior because each batch row
    represents an independent sequence, and averaging or tiling membrane
    state across unrelated sequences produces physically meaningless
    values. Episode-specific membrane continuity belongs to the Cognitive
    Kernel's session management, not to this checkpoint format.

FORMAT VERSIONING:
    Every checkpoint carries a "state_version" integer. If the format
    changes (fields added, renamed, or removed), the version increments
    and the restore path applies migration logic before walking the module
    tree. This prevents silent data loss when loading old checkpoints into
    new code.

DOCUMENTATION STANDARD:
    This file follows the CognitiveKernel Code Documentation Standard.
    Every function has a docstring. Every design decision is explained.
    No biological citations here because this is pure engineering.
"""

from __future__ import annotations

import hashlib
import json
import logging
import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Current checkpoint format version. Increment when the schema changes.
STATE_VERSION = 2


# =========================================================================
# Architecture Hash
# =========================================================================

# These are the config fields that determine tensor shapes throughout the
# model. If any of them change, weight matrices are incompatible and a
# checkpoint from the old architecture cannot be loaded into the new one.
_STRUCTURAL_FIELDS = (
    "d_model", "n_heads", "d_ff", "n_experts", "vocab_size",
    "sensory_layers", "association_layers", "executive_layers",
    "memory_size", "n_clusters", "top_k_experts", "max_seq_len",
    "T", "T_slow", "resonance_top_k", "memory_n_read_heads",
)


def compute_architecture_hash(config) -> str:
    """
    Compute a deterministic hash from the structurally critical fields
    of a TimmyConfig (or any object with matching attribute names).

    The hash is a truncated SHA-256 of the sorted JSON representation
    of the structural fields. Two configs that produce the same hash
    are guaranteed to have compatible tensor shapes. Two configs that
    produce different hashes are guaranteed to be incompatible.

    Args:
        config: a TimmyConfig instance or a dict with the structural
            field names as keys.

    Returns:
        A 16-character hex string.
    """
    if isinstance(config, dict):
        vals = {k: config.get(k, None) for k in _STRUCTURAL_FIELDS}
    else:
        vals = {k: getattr(config, k, None) for k in _STRUCTURAL_FIELDS}

    # Deterministic serialization: sorted keys, no whitespace.
    payload = json.dumps(vals, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class ArchitectureMismatchError(Exception):
    """Raised when a checkpoint's architecture hash does not match the model."""
    pass


# =========================================================================
# Weight Health Diagnostics
# =========================================================================

def _snapshot_weight_health(model: nn.Module) -> Dict[str, Dict[str, float]]:
    """
    Capture mean and std of projection weights in every named zone block.

    This snapshot is stored in the checkpoint at save time and compared
    against the live model at load time to detect weight collapse or
    explosion between training runs.

    Specifically captures:
        - Every block's resonance.W_v (the attention value projection,
          which is also the target of STDP updates)
        - Every block's ffn or moe up/down projections
        - The lm_head output projection

    Args:
        model: the TimmyModel.

    Returns:
        dict mapping parameter path -> {"mean": float, "std": float}.
    """
    snapshot = {}
    targets = ("resonance.W_v", "ffn.up", "ffn.down", "lm_head")

    for name, param in model.named_parameters():
        if any(t in name for t in targets):
            with torch.no_grad():
                w = param.float()
                snapshot[name] = {
                    "mean": w.mean().item(),
                    "std": w.std().item(),
                    "min": w.min().item(),
                    "max": w.max().item(),
                }
    return snapshot


def _check_weight_health(
    saved_snapshot: Dict[str, Dict[str, float]],
    current_snapshot: Dict[str, Dict[str, float]],
) -> List[str]:
    """
    Compare saved and current weight snapshots and return a list of
    warning strings for any parameters that look unhealthy.

    Conditions checked:
        - Dead zone: std < 1e-6 (weights have collapsed to near-constant).
        - Explosion: std > 10.0 or abs(mean) > 5.0 (weights have diverged).
        - Severe drift: std changed by more than 10x between save and load
          (indicates the checkpoint may have been corrupted or the model
          was modified after loading).

    Args:
        saved_snapshot: from the checkpoint.
        current_snapshot: from the live model after loading cold state.

    Returns:
        List of human-readable warning strings (empty if healthy).
    """
    warnings = []
    for name, current in current_snapshot.items():
        std = current["std"]
        mean = current["mean"]

        if std < 1e-6:
            warnings.append(
                f"DEAD ZONE: '{name}' has std={std:.2e}. Weights have "
                f"collapsed. This layer will not produce meaningful spikes."
            )
        elif std > 10.0 or abs(mean) > 5.0:
            warnings.append(
                f"EXPLOSION: '{name}' has mean={mean:.4f}, std={std:.4f}. "
                f"Weights have diverged. Expect saturated firing rates."
            )

        if name in saved_snapshot:
            saved_std = saved_snapshot[name]["std"]
            if saved_std > 0 and (std / (saved_std + 1e-12)) > 10.0:
                warnings.append(
                    f"DRIFT: '{name}' std changed from {saved_std:.4f} to "
                    f"{std:.4f} (>{10}x). Possible corruption or post-load "
                    f"modification."
                )

    return warnings


# =========================================================================
# Collection Helpers
# =========================================================================

def _collect_lif_states(model: nn.Module) -> Dict[str, Dict[str, Any]]:
    """
    Walk the full module tree and collect warm state from every
    AssociativeLIF instance.

    Uses the fully qualified module name as the dictionary key (e.g.
    "association_blocks.0.resonance.lif_q"). This makes the checkpoint
    robust to minor architecture changes as long as the module naming
    stays consistent. If a module is renamed between versions, the
    migration logic in _migrate_checkpoint handles the key remapping.

    Args:
        model: the TimmyModel (or any nn.Module containing AssociativeLIF
            instances anywhere in its subtree).

    Returns:
        dict mapping module path -> serialized neuron state dict.
        Each value contains: v_mem_state, i_syn_state, firing_rate_ema,
        step_counter, v_threshold_raw, beta_mem_raw, beta_syn_raw,
        neighbor_weights, cluster_gain, n_neurons, persistent.
    """
    from timmy_neuron import AssociativeLIF

    states = {}
    for name, module in model.named_modules():
        if isinstance(module, AssociativeLIF):
            state = module.get_state()
            state["n_neurons"] = module.n_neurons
            state["persistent"] = module.persistent
            states[name] = state
    return states


def _collect_moe_states(model: nn.Module) -> Dict[str, Tensor]:
    """
    Collect expert_counts_ema from every SpikeDrivenMoE instance.

    The EMA tracks per-expert utilization over training. Losing it on
    checkpoint restore means the load balance loss starts from a uniform
    prior, which can cause a brief period of unbalanced routing until
    the EMA re-converges. Preserving it avoids that transient.

    Args:
        model: the TimmyModel.

    Returns:
        dict mapping module path -> expert_counts_ema tensor (CPU clone).
    """
    from timmy_experts import SpikeDrivenMoE

    states = {}
    for name, module in model.named_modules():
        if isinstance(module, SpikeDrivenMoE):
            states[name] = module.expert_counts_ema.cpu().clone()
    return states


def _collect_stdp_state(stdp_engine) -> Dict[str, Any]:
    """
    Serialize the STDP engine's internal Python-level scalars.

    These are not nn.Parameters or registered buffers, so state_dict()
    misses them entirely. Without this, the loss EMA baseline resets to
    10.0 on every checkpoint restore, causing a spike in STDP reward
    magnitude until the EMA re-converges.

    Args:
        stdp_engine: the STDPEngine instance from TimmyModel.

    Returns:
        dict with loss_ema, ema_decay, max_update_norm, external_reward,
        reward_scale, a_plus, a_minus, tau_plus, tau_minus, w_max, w_min,
        allowed (as a sorted list for JSON serialization).
    """
    return {
        "loss_ema": stdp_engine._loss_ema,
        "ema_decay": stdp_engine._ema_decay,
        "max_update_norm": stdp_engine.max_update_norm,
        "external_reward": stdp_engine._external_reward,
        "reward_scale": stdp_engine.reward_scale,
        "a_plus": stdp_engine.a_plus,
        "a_minus": stdp_engine.a_minus,
        "tau_plus": stdp_engine.tau_plus,
        "tau_minus": stdp_engine.tau_minus,
        "w_max": stdp_engine.w_max,
        "w_min": stdp_engine.w_min,
        "allowed": sorted(stdp_engine.allowed),
    }


# =========================================================================
# Restoration Helpers
# =========================================================================

def _restore_lif_states(
    model: nn.Module,
    lif_states: Dict[str, Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    """
    Walk the module tree and restore warm state to every AssociativeLIF
    whose path exists in the checkpoint.

    Returns two lists: (restored, skipped). Restored contains the module
    paths that were successfully rehydrated. Skipped contains paths that
    exist in the checkpoint but could not be matched to a module in the
    current model (indicating a rename or architecture change).

    Batch size mismatch is handled per-population: if the saved v_mem_state
    has a different batch dimension than the current buffer, the warm state
    for that population is discarded and initialized to zero with a warning.
    This is correct because each batch row represents an independent
    sequence, and averaging or tiling membrane state across unrelated
    sequences produces physically meaningless values.

    Args:
        model: the TimmyModel.
        lif_states: dict from _collect_lif_states (loaded from checkpoint).

    Returns:
        (restored_paths, skipped_paths).
    """
    from timmy_neuron import AssociativeLIF

    module_map = {
        name: module for name, module in model.named_modules()
        if isinstance(module, AssociativeLIF)
    }

    restored = []
    skipped = []

    for path, saved_state in lif_states.items():
        if path not in module_map:
            skipped.append(path)
            logger.warning(
                "LIF state for '%s' exists in checkpoint but no matching "
                "module found in model. State discarded.", path
            )
            continue

        module = module_map[path]

        # Validate neuron count matches.
        saved_n = saved_state.get("n_neurons", None)
        if saved_n is not None and saved_n != module.n_neurons:
            skipped.append(path)
            logger.warning(
                "LIF '%s': checkpoint has n_neurons=%d but model has %d. "
                "State discarded (architecture mismatch).",
                path, saved_n, module.n_neurons,
            )
            continue

        # Check batch size compatibility for persistent state.
        if module.persistent:
            saved_v = saved_state.get("v_mem_state", None)
            if saved_v is not None and saved_v.shape[0] != module._v_mem_state.shape[0]:
                logger.warning(
                    "LIF '%s': checkpoint batch size %d != current %d. "
                    "Membrane state will reinitialize on next forward call.",
                    path, saved_v.shape[0], module._v_mem_state.shape[0],
                )
                # Still restore the learnable parameters (threshold, tau, etc.)
                # even though the membrane state does not transfer.
                device = module._v_mem_state.device
                module.v_threshold_raw.data = saved_state["v_threshold_raw"].to(device)
                module.beta_mem_raw.data = saved_state["beta_mem_raw"].to(device)
                module.beta_syn_raw.data = saved_state["beta_syn_raw"].to(device)
                module.neighbor_weights.data = saved_state["neighbor_weights"].to(device)
                module.cluster_gain.data = saved_state["cluster_gain"].to(device)
                module._firing_rate_ema = saved_state["firing_rate_ema"].to(device)
                module._step_counter.fill_(saved_state["step_counter"])
                restored.append(path)
                continue

        # Full restore via the existing AssociativeLIF.restore_state method.
        module.restore_state(saved_state)
        restored.append(path)

    # Log modules with no saved state.
    for path in module_map:
        if path not in lif_states:
            logger.info(
                "LIF '%s': no saved state in checkpoint. "
                "Will initialize fresh.", path
            )

    return restored, skipped


def _restore_moe_states(
    model: nn.Module,
    moe_states: Dict[str, Tensor],
) -> None:
    """
    Restore expert_counts_ema to every SpikeDrivenMoE instance.

    Args:
        model: the TimmyModel.
        moe_states: dict from _collect_moe_states (loaded from checkpoint).
    """
    from timmy_experts import SpikeDrivenMoE

    for name, module in model.named_modules():
        if isinstance(module, SpikeDrivenMoE) and name in moe_states:
            saved = moe_states[name]
            if saved.shape == module.expert_counts_ema.shape:
                module.expert_counts_ema.copy_(saved.to(module.expert_counts_ema.device))
            else:
                logger.warning(
                    "MoE '%s': expert_counts_ema shape mismatch "
                    "(%s vs %s). Skipping.",
                    name, saved.shape, module.expert_counts_ema.shape,
                )


def _restore_stdp_state(stdp_engine, saved: Dict[str, Any]) -> None:
    """
    Restore STDP engine scalars from checkpoint.

    Only overwrites values that exist in the saved dict, so older
    checkpoints that lack newer fields degrade gracefully.

    Args:
        stdp_engine: the STDPEngine instance.
        saved: dict from _collect_stdp_state.
    """
    if "loss_ema" in saved:
        stdp_engine._loss_ema = float(saved["loss_ema"])
    if "ema_decay" in saved:
        stdp_engine._ema_decay = float(saved["ema_decay"])
    if "max_update_norm" in saved:
        stdp_engine.max_update_norm = float(saved["max_update_norm"])
    if "external_reward" in saved:
        stdp_engine._external_reward = saved["external_reward"]
    if "reward_scale" in saved:
        stdp_engine.reward_scale = float(saved["reward_scale"])
    if "a_plus" in saved:
        stdp_engine.a_plus = float(saved["a_plus"])
    if "a_minus" in saved:
        stdp_engine.a_minus = float(saved["a_minus"])
    if "tau_plus" in saved:
        stdp_engine.tau_plus = float(saved["tau_plus"])
    if "tau_minus" in saved:
        stdp_engine.tau_minus = float(saved["tau_minus"])
    if "w_max" in saved:
        stdp_engine.w_max = float(saved["w_max"])
    if "w_min" in saved:
        stdp_engine.w_min = float(saved["w_min"])
    if "allowed" in saved:
        stdp_engine.allowed = set(saved["allowed"])


# =========================================================================
# Migration
# =========================================================================

def _migrate_checkpoint(checkpoint: Dict, from_version: int) -> Dict:
    """
    Apply sequential migrations to bring an old checkpoint up to the
    current STATE_VERSION.

    Each migration is a function that takes the checkpoint dict and
    returns the modified dict. Migrations are applied in order from
    from_version+1 to STATE_VERSION.

    Args:
        checkpoint: the raw loaded checkpoint dict.
        from_version: the state_version stored in the checkpoint.

    Returns:
        Migrated checkpoint dict with state_version == STATE_VERSION.
    """
    migrations = {
        # v1 -> v2: added architecture_hash and weight_health fields.
        2: _migrate_v1_to_v2,
    }

    version = from_version
    while version < STATE_VERSION:
        next_version = version + 1
        if next_version in migrations:
            checkpoint = migrations[next_version](checkpoint)
            logger.info("Migrated checkpoint from v%d to v%d.", version, next_version)
        version = next_version

    checkpoint["state_version"] = STATE_VERSION
    return checkpoint


def _migrate_v1_to_v2(checkpoint: Dict) -> Dict:
    """
    Migrate a v1 checkpoint (soul_version=1) to v2 (state_version=2).

    v2 added: architecture_hash, weight_health. v2 also renamed
    "soul_version" to "state_version" and the public API from
    save_soul/load_soul to save_timmy_state/load_timmy_state.

    For v1 checkpoints, architecture_hash and weight_health are set to
    None, which causes the loader to skip the hash check and the health
    comparison (with a warning).
    """
    # Rename version key.
    if "soul_version" in checkpoint:
        checkpoint["state_version"] = checkpoint.pop("soul_version")

    # Backfill missing fields.
    if "architecture_hash" not in checkpoint:
        checkpoint["architecture_hash"] = None
    if "weight_health" not in checkpoint:
        checkpoint["weight_health"] = None

    return checkpoint


# =========================================================================
# Public API
# =========================================================================

def save_timmy_state(
    model: nn.Module,
    path: str,
    config_dict: Optional[Dict] = None,
    optimizer_state: Optional[Dict] = None,
    scheduler_state: Optional[Dict] = None,
    training_step: Optional[int] = None,
    extra: Optional[Dict] = None,
) -> str:
    """
    Save a complete Timmy state checkpoint to disk.

    Captures all three state layers (cold, warm, hot), the architecture
    hash for structural verification, and weight health diagnostics for
    post-load safety checks.

    Args:
        model: the TimmyModel instance.
        path: filesystem path for the checkpoint file. Convention is
            "timmy_step_{N}.state" or "timmy_latest.state".
        config_dict: serialized TimmyConfig (output of vars(cfg) with
            non-serializable fields removed). If None, attempts to
            read model.cfg and serialize it automatically.
        optimizer_state: output of optimizer.state_dict(). Optional.
        scheduler_state: output of scheduler.state_dict(). Optional.
        training_step: current global training step. Optional.
        extra: arbitrary additional metadata (dataset path, git hash,
            training loss at save time, etc.). Optional.

    Returns:
        The path written to (same as input path).
    """
    # Config: serialize if not provided.
    if config_dict is None and hasattr(model, "cfg"):
        config_dict = {
            k: v for k, v in vars(model.cfg).items()
            if not callable(v) and not k.startswith("_")
        }
        # torch.dtype is not pickle-safe in all contexts. Store as string.
        if "dtype" in config_dict:
            config_dict["dtype"] = str(config_dict["dtype"])

    # Architecture hash from the live config.
    arch_hash = None
    if config_dict is not None:
        arch_hash = compute_architecture_hash(config_dict)
    elif hasattr(model, "cfg"):
        arch_hash = compute_architecture_hash(model.cfg)

    # Cold layer: full state_dict (weights + registered buffers).
    cold = model.state_dict()

    # Warm layer: per-population LIF membrane and biophysical state.
    warm_lif = _collect_lif_states(model)

    # Hot layer: STDP engine, MoE EMA, and other ephemeral scalars.
    hot_stdp = _collect_stdp_state(model.stdp)
    hot_moe = _collect_moe_states(model)

    # Weight health snapshot for post-load diagnostics.
    weight_health = _snapshot_weight_health(model)

    # Assemble the checkpoint.
    checkpoint = {
        "state_version": STATE_VERSION,
        "architecture_hash": arch_hash,
        "cold": cold,
        "warm_lif": warm_lif,
        "hot_stdp": hot_stdp,
        "hot_moe": hot_moe,
        "weight_health": weight_health,
        "config": config_dict,
        "training_step": training_step,
    }

    if optimizer_state is not None:
        checkpoint["optimizer"] = optimizer_state
    if scheduler_state is not None:
        checkpoint["scheduler"] = scheduler_state
    if extra is not None:
        checkpoint["extra"] = extra

    # Count what we captured for the log.
    n_lif = len(warm_lif)
    n_moe = len(hot_moe)
    n_params = sum(p.numel() for p in model.parameters())

    torch.save(checkpoint, path)

    logger.info(
        "State saved to '%s': %d params, %d LIF populations, %d MoE modules, "
        "STDP loss_ema=%.4f, step=%s, arch_hash=%s, version=%d.",
        path, n_params, n_lif, n_moe,
        hot_stdp.get("loss_ema", float("nan")),
        training_step, arch_hash, STATE_VERSION,
    )

    return path


def load_timmy_state(
    model: nn.Module,
    path: str,
    strict_cold: bool = False,
    device: Optional[str] = None,
    skip_hash_check: bool = False,
) -> Dict[str, Any]:
    """
    Restore a Timmy state checkpoint into a live TimmyModel.

    Applies all three state layers in order: cold (weights), warm
    (membrane dynamics), hot (STDP/MoE scalars). Verifies the
    architecture hash before loading, and runs weight health diagnostics
    after loading.

    Args:
        model: the TimmyModel instance (already constructed with the
            correct config). Weights will be overwritten.
        path: filesystem path to the .state checkpoint.
        strict_cold: if True, require exact key match for state_dict.
            Default False for safe loading across minor changes.
        device: target device for restored tensors. If None, uses the
            device of model's first parameter.
        skip_hash_check: if True, bypass the architecture hash
            verification. Use only for debugging or when intentionally
            loading a checkpoint from a modified architecture.

    Returns:
        dict with restoration metadata:
            "state_version": int, version of the loaded checkpoint.
            "training_step": int or None.
            "architecture_hash_match": bool or None (if hash unavailable).
            "lif_restored": list of module paths successfully restored.
            "lif_skipped": list of module paths that could not be matched.
            "weight_warnings": list of health warning strings.
            "has_optimizer": bool.
            "has_scheduler": bool.
            "config": the saved config dict (for validation).
            "extra": the saved extra dict (if any).
            "optimizer": optimizer state dict (if present).
            "scheduler": scheduler state dict (if present).

    Raises:
        ArchitectureMismatchError: if the architecture hash does not
            match and skip_hash_check is False.
    """
    if device is None:
        device = next(model.parameters()).device

    map_location = device if isinstance(device, str) else str(device)
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    # Version check and migration.
    saved_version = checkpoint.get(
        "state_version", checkpoint.get("soul_version", 0)
    )
    if saved_version < STATE_VERSION:
        logger.info(
            "Checkpoint version %d is older than current %d. Migrating.",
            saved_version, STATE_VERSION,
        )
        checkpoint = _migrate_checkpoint(checkpoint, saved_version)
    elif saved_version > STATE_VERSION:
        logger.warning(
            "Checkpoint version %d is newer than code version %d. "
            "Loading anyway, but some fields may be ignored.",
            saved_version, STATE_VERSION,
        )

    # Architecture hash verification.
    saved_hash = checkpoint.get("architecture_hash", None)
    hash_match = None

    if saved_hash is not None:
        current_hash = None
        if hasattr(model, "cfg"):
            current_hash = compute_architecture_hash(model.cfg)
        elif "config" in checkpoint and checkpoint["config"] is not None:
            current_hash = compute_architecture_hash(checkpoint["config"])

        if current_hash is not None:
            hash_match = (saved_hash == current_hash)
            if not hash_match and not skip_hash_check:
                raise ArchitectureMismatchError(
                    f"Architecture hash mismatch: checkpoint has '{saved_hash}' "
                    f"but model has '{current_hash}'. The checkpoint was saved "
                    f"from a model with different structural dimensions "
                    f"(d_model, n_heads, n_layers, etc.). Loading would produce "
                    f"a model with random values in every mismatched layer. "
                    f"Pass skip_hash_check=True to override."
                )
            elif not hash_match:
                logger.warning(
                    "Architecture hash mismatch (saved=%s, current=%s) but "
                    "skip_hash_check=True. Proceeding with best-effort load.",
                    saved_hash, current_hash,
                )
    else:
        logger.warning(
            "No architecture hash in checkpoint (pre-v2 format). "
            "Skipping structural verification."
        )

    # Cold layer: model weights and registered buffers.
    cold = checkpoint.get("cold", checkpoint.get("model_state_dict", {}))
    missing, unexpected = model.load_state_dict(cold, strict=strict_cold)
    if missing:
        logger.warning("Cold restore missing keys: %s", missing)
    if unexpected:
        logger.warning("Cold restore unexpected keys: %s", unexpected)

    # Warm layer: per-population LIF state.
    warm_lif = checkpoint.get("warm_lif", {})
    lif_restored, lif_skipped = _restore_lif_states(model, warm_lif)

    # Hot layer: STDP.
    hot_stdp = checkpoint.get("hot_stdp", {})
    if hot_stdp:
        _restore_stdp_state(model.stdp, hot_stdp)
    else:
        # Backward compat: old serialize_state stored stdp_loss_ema flat.
        if "stdp_loss_ema" in checkpoint:
            model.stdp._loss_ema = float(checkpoint["stdp_loss_ema"])

    # Hot layer: MoE expert EMA.
    hot_moe = checkpoint.get("hot_moe", {})
    if hot_moe:
        _restore_moe_states(model, hot_moe)

    # Weight health diagnostics.
    weight_warnings = []
    saved_health = checkpoint.get("weight_health", None)
    current_health = _snapshot_weight_health(model)
    if saved_health is not None and current_health:
        weight_warnings = _check_weight_health(saved_health, current_health)
        for w in weight_warnings:
            logger.warning("Weight health: %s", w)
    elif saved_health is None:
        logger.info("No weight health snapshot in checkpoint. Skipping diagnostics.")

    # Training step.
    training_step = checkpoint.get("training_step", None)

    logger.info(
        "State restored from '%s': version=%d, step=%s, "
        "arch_hash_match=%s, %d/%d LIF populations restored, %d skipped, "
        "%d weight warnings.",
        path, saved_version, training_step, hash_match,
        len(lif_restored), len(lif_restored) + len(lif_skipped),
        len(lif_skipped), len(weight_warnings),
    )

    return {
        "state_version": saved_version,
        "training_step": training_step,
        "architecture_hash_match": hash_match,
        "lif_restored": lif_restored,
        "lif_skipped": lif_skipped,
        "weight_warnings": weight_warnings,
        "has_optimizer": "optimizer" in checkpoint,
        "has_scheduler": "scheduler" in checkpoint,
        "config": checkpoint.get("config", None),
        "extra": checkpoint.get("extra", None),
        "optimizer": checkpoint.get("optimizer", None),
        "scheduler": checkpoint.get("scheduler", None),
    }


# =========================================================================
# Diagnostic: Diff Two Checkpoints
# =========================================================================

def diff_timmy_states(path_a: str, path_b: str) -> Dict[str, Any]:
    """
    Compare two Timmy state checkpoints and report what changed.

    Useful for debugging training runs: load the checkpoint from step N
    and step N+K, and see which LIF populations drifted, whether STDP
    reward baseline moved, whether any expert collapsed, and whether
    any weight matrices entered a dead or explosion zone.

    Args:
        path_a: path to the earlier checkpoint.
        path_b: path to the later checkpoint.

    Returns:
        dict with per-population firing rate deltas, STDP scalar deltas,
        MoE expert utilization deltas, weight health comparison, and a
        summary string.
    """
    a = torch.load(path_a, map_location="cpu", weights_only=False)
    b = torch.load(path_b, map_location="cpu", weights_only=False)

    report = {
        "lif_deltas": {},
        "stdp_deltas": {},
        "moe_deltas": {},
        "weight_health_a": a.get("weight_health", {}),
        "weight_health_b": b.get("weight_health", {}),
        "weight_warnings": [],
    }

    # LIF firing rate deltas.
    lif_a = a.get("warm_lif", {})
    lif_b = b.get("warm_lif", {})
    shared_lif = set(lif_a.keys()) & set(lif_b.keys())
    for path in sorted(shared_lif):
        fr_a = lif_a[path].get("firing_rate_ema", None)
        fr_b = lif_b[path].get("firing_rate_ema", None)
        if fr_a is not None and fr_b is not None:
            delta = (fr_b.float() - fr_a.float()).mean().item()
            report["lif_deltas"][path] = {
                "firing_rate_a": fr_a.mean().item(),
                "firing_rate_b": fr_b.mean().item(),
                "delta": delta,
                "steps_a": lif_a[path].get("step_counter", 0),
                "steps_b": lif_b[path].get("step_counter", 0),
            }

    # STDP deltas.
    stdp_a = a.get("hot_stdp", {})
    stdp_b = b.get("hot_stdp", {})
    for key in set(stdp_a.keys()) | set(stdp_b.keys()):
        va = stdp_a.get(key, None)
        vb = stdp_b.get(key, None)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            report["stdp_deltas"][key] = {"a": va, "b": vb, "delta": vb - va}

    # MoE deltas.
    moe_a = a.get("hot_moe", {})
    moe_b = b.get("hot_moe", {})
    for path in set(moe_a.keys()) & set(moe_b.keys()):
        ea = moe_a[path]
        eb = moe_b[path]
        if ea.shape == eb.shape:
            report["moe_deltas"][path] = {
                "expert_load_a": ea.tolist(),
                "expert_load_b": eb.tolist(),
                "delta": (eb - ea).tolist(),
            }

    # Cross-checkpoint weight health comparison.
    health_a = a.get("weight_health", {})
    health_b = b.get("weight_health", {})
    if health_a and health_b:
        report["weight_warnings"] = _check_weight_health(health_a, health_b)

    # Summary line.
    step_a = a.get("training_step", "?")
    step_b = b.get("training_step", "?")
    n_drifted = sum(
        1 for v in report["lif_deltas"].values() if abs(v["delta"]) > 0.005
    )
    report["summary"] = (
        f"Steps {step_a} -> {step_b}: "
        f"{len(shared_lif)} LIF populations compared, "
        f"{n_drifted} drifted >0.5% firing rate, "
        f"{len(report['weight_warnings'])} weight health warnings."
    )

    return report
