"""
smoke_test.py
End-to-End Smoke Test for the TimmyArray Training Pipeline

Verifies that the full import chain, model construction, forward pass,
Phase 1 training loop (5 steps), critical period probe, Phase 2 training
loop (5 steps), and array monitor all run without error on CPU with a
tiny model configuration.

NOT a correctness test. Does not verify that the model learns anything.
Verifies only that the code runs without crashing. Run this before
committing compute on the 129M training run.

Usage:
    python smoke_test.py

Expected output:
    All checks print PASS. Any FAIL or traceback indicates a bug to fix
    before starting the real training run.

This script uses a deliberately tiny config (d_model=32, vocab_size=256,
2 specialists, 2 layers per zone, seq_len=16, batch=2) so it completes
in under 60 seconds on CPU.
"""

from __future__ import annotations

import sys
import traceback
import os
import tempfile

import torch


def check(label: str, fn):
    """Run fn(), print PASS or FAIL with the label."""
    try:
        result = fn()
        print(f"  PASS  {label}")
        return result
    except Exception as e:
        print(f"  FAIL  {label}")
        traceback.print_exc()
        return None


def main():
    print()
    print("=" * 60)
    print("TIMMY ARRAY SMOKE TEST")
    print("=" * 60)
    print()

    # -------------------------------------------------------------------------
    # 1. Imports
    # -------------------------------------------------------------------------
    print("[ IMPORTS ]")

    timmy_model_mod = check("import timmy_model", lambda: __import__("timmy_model"))
    check("import timmy_neuron", lambda: __import__("timmy_neuron"))
    check("import timmy_encoder", lambda: __import__("timmy_encoder"))
    check("import timmy_attention", lambda: __import__("timmy_attention"))
    check("import timmy_experts", lambda: __import__("timmy_experts"))
    check("import timmy_memory", lambda: __import__("timmy_memory"))
    check("import timmy_blocks", lambda: __import__("timmy_blocks"))
    check("import timmy_plasticity", lambda: __import__("timmy_plasticity"))
    check("import timmy_state", lambda: __import__("timmy_state"))
    check("import timmy_data", lambda: __import__("timmy_data"))
    create_mod = check("import CreateTimmyArray", lambda: __import__("CreateTimmyArray"))
    probe_mod = check("import timmy_criticalperiodprobe", lambda: __import__("timmy_criticalperiodprobe"))
    monitor_mod = check("import array_monitor", lambda: __import__("array_monitor"))
    train_mod = check("import train_array", lambda: __import__("train_array"))

    if any(m is None for m in [timmy_model_mod, create_mod, probe_mod, monitor_mod, train_mod]):
        print("\nCritical import failed. Cannot continue.")
        sys.exit(1)

    print()

    # -------------------------------------------------------------------------
    # 2. Tiny config
    # -------------------------------------------------------------------------
    print("[ TINY CONFIG ]")

    from timmy_model import TimmyConfig, TimmyModel
    from CreateTimmyArray import TimmyArray, TimmyArrayConfig, COLUMN_NAMES

    def make_tiny_cfg():
        col_cfg = TimmyConfig(
            vocab_size=256,
            d_model=32,
            n_heads=2,
            n_layers=6,
            d_ff=64,
            max_seq_len=16,
            T=2,
            T_slow=1,
            n_experts=2,
            top_k_experts=1,
            n_clusters=8,
            cascade_radius=1,
            sensory_layers=1,
            association_layers=1,
            executive_layers=1,
            memory_size=16,
            memory_n_read_heads=2,
            episodic_memory=False,  # kernel not present in smoke test
            max_steps=5,
            batch_size=2,
            grad_accum=1,
            log_every=2,
            save_every=5,
            warmup_steps=1,
            lif_freeze_steps=0,
            tokenizer_id=None,  # not needed; we supply raw token ids
        )
        arr_cfg = TimmyArrayConfig(
            column_cfg=col_cfg,
            num_specialists=2,
            routing_top_k_train=1,
            coordinate_dim=8,
            perforant_rank=2,
        )
        return arr_cfg

    arr_cfg = check("build TimmyArrayConfig", make_tiny_cfg)
    if arr_cfg is None:
        print("\nConfig construction failed. Cannot continue.")
        sys.exit(1)

    print()

    # -------------------------------------------------------------------------
    # 3. Model construction
    # -------------------------------------------------------------------------
    print("[ MODEL CONSTRUCTION ]")

    array = check("construct TimmyArray", lambda: TimmyArray(arr_cfg))
    if array is None:
        print("\nModel construction failed. Cannot continue.")
        sys.exit(1)

    check(
        "count params",
        lambda: print(f"         Prime params: {array.prime.count_params().splitlines()[0]}")
        or True,
    )
    print()

    # -------------------------------------------------------------------------
    # 4. Forward pass
    # -------------------------------------------------------------------------
    print("[ FORWARD PASS ]")

    device = torch.device("cpu")
    array = array.to(device)

    token_ids = torch.randint(0, 256, (2, 15))  # (B=2, S=15); targets are S+1

    def _forward():
        array.eval()
        with torch.no_grad():
            logits, stats = array.prime(token_ids)
        assert logits.shape == (2, 15, 256), f"Unexpected logits shape: {logits.shape}"
        assert "avg_spike_rate" in stats
        return logits, stats

    fwd = check("prime forward pass", _forward)

    def _array_forward():
        array.eval()
        with torch.no_grad():
            prime_logits, kernel_coords, stats = array(token_ids)
        assert prime_logits.shape == (2, 15, 256)
        assert kernel_coords.shape[0] == 2
        assert "active_columns" in stats
        return stats

    arr_fwd = check("full array forward pass", _array_forward)
    if arr_fwd is not None:
        check(
            f"active columns: {arr_fwd.get('active_columns', [])}",
            lambda: True,
        )
    print()

    # -------------------------------------------------------------------------
    # 5. Probe construction
    # -------------------------------------------------------------------------
    print("[ CRITICAL PERIOD PROBE ]")

    from timmy_criticalperiodprobe import ProbeConfig, ProbeState, probe_step

    probe_cfg = check(
        "construct ProbeConfig",
        lambda: ProbeConfig(stability_window=2, entropy_collapse_floor=0.0),
    )
    probe_state = check("construct ProbeState", lambda: ProbeState(cfg=probe_cfg))

    def _probe_step():
        array.prime.eval()
        ready, signals = probe_step(array.prime, probe_state, training_step=0)
        array.prime.train()
        status = probe_state.get_status_dict()
        assert "threshold_variance" in status
        assert "ready" in status
        return status

    check("probe_step runs", _probe_step)
    check("get_status_dict", lambda: probe_state.get_status_dict())
    check("is_ready()", lambda: isinstance(probe_state.is_ready(), bool))
    print()

    # -------------------------------------------------------------------------
    # 6. Monitor construction
    # -------------------------------------------------------------------------
    print("[ ARRAY MONITOR ]")

    from array_monitor import MonitorConfig, MonitorState, monitor_step, summarize

    mon_cfg = check("construct MonitorConfig", lambda: MonitorConfig(
        log_every=2,
        entropy_collapse_floor=0.0,
        divergence_expect_step=99999,
    ))
    mon_state = check("construct MonitorState", lambda: MonitorState(cfg=mon_cfg))

    def _monitor_step_cheap():
        array.eval()
        snapshot = monitor_step(array, mon_state, training_step=0)
        array.train()
        assert "router_entropy" in snapshot
        return snapshot

    check("monitor_step (cheap, no probe)", _monitor_step_cheap)

    def _monitor_step_full():
        probe_ids = token_ids[:1]
        array.eval()
        snapshot = monitor_step(array, mon_state, training_step=1,
                                probe_token_ids=probe_ids)
        array.train()
        return snapshot

    check("monitor_step (full subspace probe)", _monitor_step_full)
    check("format_line", lambda: isinstance(mon_state.format_line(), str))
    check("summarize", lambda: isinstance(summarize(mon_state, last_n=5), str))
    print()

    # -------------------------------------------------------------------------
    # 7. Phase 1 training loop (5 steps)
    # -------------------------------------------------------------------------
    print("[ PHASE 1 TRAINING (5 steps, synthetic data) ]")

    from train_array import train_phase1

    with tempfile.TemporaryDirectory() as tmpdir:
        # Monkey-patch make_data_iter to return synthetic batches
        # so we do not need a real dataset or tokenizer.
        import train_array as ta

        def _synthetic_iter(source, cfg, text_column="text", split="train"):
            while True:
                yield torch.randint(0, 256, (cfg.batch_size, cfg.max_seq_len))

        original_iter = ta.make_data_iter
        ta.make_data_iter = _synthetic_iter

        try:
            p1_probe_cfg = ProbeConfig(
                stability_window=2,
                entropy_collapse_floor=0.0,
                threshold_variance_delta_max=999.0,
                entropy_delta_max=999.0,
                load_imbalance_max=999.0,
            )
            result = check(
                "train_phase1 runs 5 steps",
                lambda: train_phase1(
                    array=array,
                    data_source="synthetic",
                    output_dir=tmpdir,
                    device=device,
                    probe_every=2,
                    probe_cfg=p1_probe_cfg,
                ),
            )
        finally:
            ta.make_data_iter = original_iter

    print()

    # -------------------------------------------------------------------------
    # 8. Phase 2 training loop (5 steps)
    # -------------------------------------------------------------------------
    print("[ PHASE 2 TRAINING (5 steps, synthetic data) ]")

    from train_array import train_phase2

    specialist_names = array.column_names[1:]
    column_sources = {"prime": "synthetic"}
    column_sources.update({name: "synthetic" for name in specialist_names})

    with tempfile.TemporaryDirectory() as tmpdir:
        import train_array as ta
        ta.make_data_iter = _synthetic_iter

        try:
            mon_cfg_p2 = MonitorConfig(
                log_every=2,
                entropy_collapse_floor=0.0,
                divergence_expect_step=99999,
            )
            result2 = check(
                "train_phase2 runs 5 steps",
                lambda: train_phase2(
                    array=array,
                    column_data_sources=column_sources,
                    output_dir=tmpdir,
                    device=device,
                    monitor_every=2,
                    monitor_subspace_every=4,
                    monitor_cfg=mon_cfg_p2,
                ),
            )
        finally:
            ta.make_data_iter = original_iter

    print()

    # -------------------------------------------------------------------------
    # 9. Summary
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("SMOKE TEST COMPLETE")
    print("If all checks above show PASS, the pipeline is ready to train.")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
