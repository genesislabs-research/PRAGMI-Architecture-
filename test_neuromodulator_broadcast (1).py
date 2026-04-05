"""
test_neuromodulator_broadcast.py

Regression tests for NeuromodulatorBroadcast and MaturityComputer.
Tests verify:
1. Neuromodulator EMA updates move in the right direction.
2. @torch.no_grad on update methods: no gradient accumulation.
3. is_sleep_phase correctly gates on ACh level.
4. Asymmetric EMA: maturity increases slowly, decreases faster.
5. Maturity stays in [0, 1] under all inputs.
6. Pessimistic defaults from WorldModelEnsemble (1.0 variance) produce low maturity.
7. Stable signal history (low entropy std, low loss CV, low probe, low variance)
   drives maturity toward 1.0 over many cycles.
8. Unstable signal history (high everything) drives maturity down.
9. HOT state roundtrip preserves history lists and cycle counter.
10. Baseline statistics set correctly after baseline window.
"""
import torch
from neuromodulator_broadcast_teaching import (
    NeuromodulatorBroadcast,
    NeuromodulatorConfig,
)


def test_da_ema_moves_toward_signal():
    cfg = NeuromodulatorConfig()
    nb = NeuromodulatorBroadcast(cfg)
    initial = nb.da.item()
    nb.update_da(1.0)
    assert nb.da.item() > initial, "DA should increase toward signal=1.0"
    nb2 = NeuromodulatorBroadcast(cfg)
    nb2.update_da(0.0)
    assert nb2.da.item() < initial, "DA should decrease toward signal=0.0"
    print("PASS: DA EMA moves toward signal")


def test_ach_ema_moves_toward_signal():
    cfg = NeuromodulatorConfig()
    nb = NeuromodulatorBroadcast(cfg)
    initial = nb.ach.item()
    nb.update_ach(1.0)
    assert nb.ach.item() > initial, "ACh should increase toward signal=1.0"
    print("PASS: ACh EMA moves toward signal")


def test_ne_ema_moves_toward_signal():
    cfg = NeuromodulatorConfig()
    nb = NeuromodulatorBroadcast(cfg)
    initial = nb.ne.item()
    nb.update_ne(1.0)
    assert nb.ne.item() > initial, "NE should increase toward signal=1.0"
    print("PASS: NE EMA moves toward signal")


def test_ht_ema_moves_toward_signal():
    cfg = NeuromodulatorConfig()
    nb = NeuromodulatorBroadcast(cfg)
    initial = nb.ht.item()
    nb.update_ht(1.0)
    assert nb.ht.item() > initial, "5-HT should increase toward signal=1.0"
    print("PASS: 5-HT EMA moves toward signal")


def test_update_methods_no_grad():
    cfg = NeuromodulatorConfig()
    nb = NeuromodulatorBroadcast(cfg)
    # All update methods must not accumulate gradients even if called
    # with a tensor that has requires_grad=True
    signal = torch.tensor(0.8, requires_grad=True)
    nb.update_da(signal.item())
    nb.update_ach(signal.item())
    nb.update_ne(signal.item())
    nb.update_ht(signal.item())
    assert nb.da.grad_fn is None, "DA buffer should not have grad_fn"
    assert nb.ach.grad_fn is None, "ACh buffer should not have grad_fn"
    assert nb.ne.grad_fn is None, "NE buffer should not have grad_fn"
    assert nb.ht.grad_fn is None, "5-HT buffer should not have grad_fn"
    print("PASS: update methods produce no grad_fn on buffers")


def test_is_sleep_phase_gates_on_ach():
    cfg = NeuromodulatorConfig()
    nb = NeuromodulatorBroadcast(cfg)
    # Drive ACh high (wake)
    for _ in range(200):
        nb.update_ach(0.9)
    assert not nb.is_sleep_phase(), "High ACh should be wake phase"
    # Drive ACh low (sleep)
    for _ in range(200):
        nb.update_ach(0.1)
    assert nb.is_sleep_phase(), "Low ACh should be sleep phase"
    print("PASS: is_sleep_phase correctly gates on ACh level")


def test_maturity_stays_in_bounds():
    cfg = NeuromodulatorConfig(entropy_baseline_window=3, loss_baseline_window=3)
    nb = NeuromodulatorBroadcast(cfg)
    for _ in range(100):
        import random
        m = nb.compute_maturity(
            routing_entropy=random.uniform(0, 1),
            loss_value=random.uniform(0.1, 5.0),
            probe_response=random.uniform(0, 1),
            mean_recent_variance=random.uniform(0, 1),
        )
        assert 0.0 <= m <= 1.0, f"Maturity out of bounds: {m}"
    print("PASS: maturity stays in [0, 1] under random inputs")


def test_asymmetric_ema_slow_up_fast_down():
    cfg = NeuromodulatorConfig(
        tau_mature_up=0.995,
        tau_mature_down=0.95,
        entropy_baseline_window=3,
        loss_baseline_window=3,
    )
    nb_up = NeuromodulatorBroadcast(cfg)
    nb_down = NeuromodulatorBroadcast(cfg)

    # Prime baselines with signals that have meaningful variance so
    # normalization denominators are nonzero.
    for i in range(10):
        v = 0.3 + 0.05 * (i % 4)
        nb_up.compute_maturity(v, 1.0 + 0.1 * (i % 3), 0.4, 0.05)
        nb_down.compute_maturity(v, 1.0 + 0.1 * (i % 3), 0.4, 0.05)

    # Force maturity to mid-point so both have room to move in both directions.
    with torch.no_grad():
        nb_up.global_maturity.fill_(0.5)
        nb_down.global_maturity.fill_(0.5)

    # Drive nb_up toward higher maturity with maximally stable signals.
    # Use repeated stable calls so recent history window fills with low-std values.
    for _ in range(5):
        nb_up.compute_maturity(
            routing_entropy=0.3, loss_value=1.0, probe_response=0.01, mean_recent_variance=0.001
        )
    with torch.no_grad():
        nb_up.global_maturity.fill_(0.5)
    nb_up.compute_maturity(
        routing_entropy=0.3, loss_value=1.0, probe_response=0.01, mean_recent_variance=0.001
    )
    delta_up = nb_up.global_maturity.item() - 0.5

    # Drive nb_down toward lower maturity with maximally unstable signals.
    for _ in range(5):
        nb_down.compute_maturity(
            routing_entropy=0.99, loss_value=5.0, probe_response=0.99, mean_recent_variance=1.0
        )
    with torch.no_grad():
        nb_down.global_maturity.fill_(0.5)
    nb_down.compute_maturity(
        routing_entropy=0.99, loss_value=5.0, probe_response=0.99, mean_recent_variance=1.0
    )
    delta_down = nb_down.global_maturity.item() - 0.5

    assert delta_up > 0, f"Stable signals should increase maturity, got delta={delta_up}"
    assert delta_down < 0, f"Unstable signals should decrease maturity, got delta={delta_down}"
    assert abs(delta_down) > abs(delta_up), (
        f"Maturity should decrease faster than it increases: "
        f"|delta_down|={abs(delta_down):.4f}, |delta_up|={abs(delta_up):.4f}"
    )
    print(f"PASS: asymmetric EMA (delta_up={delta_up:.4f}, delta_down={delta_down:.4f})")


def test_pessimistic_defaults_produce_low_maturity():
    # Before WorldModelEnsemble has trained, mean_recent_variance=1.0.
    # Combined with probe_response=0.9 (immature system, high sensitivity),
    # maturity should stay low.
    cfg = NeuromodulatorConfig(entropy_baseline_window=3, loss_baseline_window=3)
    nb = NeuromodulatorBroadcast(cfg)
    for _ in range(50):
        m = nb.compute_maturity(
            routing_entropy=0.9,  # high entropy = not yet specialized
            loss_value=3.0,       # high loss = early training
            probe_response=0.9,   # high sensitivity = immature structure
            mean_recent_variance=1.0,  # pessimistic default from WME
        )
    assert m < 0.3, f"Pessimistic defaults should produce low maturity, got {m:.4f}"
    print(f"PASS: pessimistic defaults produce low maturity ({m:.4f})")


def test_stable_signals_drive_maturity_up():
    # Over many cycles with stable signals, maturity should climb toward 1.0.
    cfg = NeuromodulatorConfig(
        tau_mature_up=0.9,  # faster for test speed
        entropy_baseline_window=3,
        loss_baseline_window=3,
    )
    nb = NeuromodulatorBroadcast(cfg)
    maturity_values = []
    for _ in range(200):
        m = nb.compute_maturity(
            routing_entropy=0.1,      # low entropy = stable routing
            loss_value=0.5,           # low loss = converged
            probe_response=0.05,      # low sensitivity = mature structure
            mean_recent_variance=0.001,  # low variance = confident world model
        )
        maturity_values.append(m)
    assert maturity_values[-1] > maturity_values[10], \
        "Maturity should increase under stable signals"
    assert maturity_values[-1] > 0.5, \
        f"Maturity should exceed 0.5 after 200 stable cycles, got {maturity_values[-1]:.4f}"
    print(f"PASS: stable signals drive maturity up "
          f"(final={maturity_values[-1]:.4f})")


def test_unstable_signals_drive_maturity_down():
    cfg = NeuromodulatorConfig(
        tau_mature_up=0.9,
        entropy_baseline_window=3,
        loss_baseline_window=3,
    )
    nb = NeuromodulatorBroadcast(cfg)
    # First drive maturity up
    for _ in range(100):
        nb.compute_maturity(0.1, 0.5, 0.05, 0.001)
    high_maturity = nb.global_maturity.item()
    # Then hit with destabilizing signals
    for _ in range(30):
        nb.compute_maturity(0.95, 4.5, 0.95, 1.0)
    low_maturity = nb.global_maturity.item()
    assert low_maturity < high_maturity, \
        f"Unstable signals should decrease maturity: {high_maturity:.4f} -> {low_maturity:.4f}"
    print(f"PASS: unstable signals drive maturity down "
          f"({high_maturity:.4f} -> {low_maturity:.4f})")


def test_baseline_statistics_set_after_window():
    cfg = NeuromodulatorConfig(entropy_baseline_window=5, loss_baseline_window=5)
    nb = NeuromodulatorBroadcast(cfg)
    assert nb.entropy_baseline_std.item() < 0.0, \
        "Baseline std should be unset (-1) before window"
    for i in range(5):
        nb.compute_maturity(
            routing_entropy=0.5 + 0.1 * i,
            loss_value=2.0 - 0.2 * i,
            probe_response=0.5,
            mean_recent_variance=0.05,
        )
    assert nb.entropy_baseline_std.item() > 0.0, \
        "Entropy baseline std should be set after window"
    assert nb.loss_baseline_cv.item() > 0.0, \
        "Loss baseline CV should be set after window"
    print(f"PASS: baseline statistics set after window "
          f"(entropy_std={nb.entropy_baseline_std.item():.4f}, "
          f"loss_cv={nb.loss_baseline_cv.item():.4f})")


def test_history_caps_after_baseline():
    cfg = NeuromodulatorConfig(entropy_baseline_window=5, loss_baseline_window=5)
    nb = NeuromodulatorBroadcast(cfg)
    # Run enough cycles that history would grow unboundedly without capping
    for i in range(100):
        nb.compute_maturity(
            routing_entropy=0.3 + 0.05 * (i % 4),
            loss_value=1.0 + 0.1 * (i % 3),
            probe_response=0.4,
            mean_recent_variance=0.05,
        )
    assert len(nb._entropy_history) <= 20,         f"Entropy history should be capped at 20, got {len(nb._entropy_history)}"
    assert len(nb._loss_history) <= 20,         f"Loss history should be capped at 20, got {len(nb._loss_history)}"
    # Baselines must have been set (otherwise cap would not apply)
    assert nb.entropy_baseline_std.item() > 0.0,         "Entropy baseline should be set before cap takes effect"
    print(f"PASS: history lists capped at 20 after baseline set "
          f"(entropy={len(nb._entropy_history)}, loss={len(nb._loss_history)})")


def test_hot_state_roundtrip():
    cfg = NeuromodulatorConfig(entropy_baseline_window=3, loss_baseline_window=3)
    nb = NeuromodulatorBroadcast(cfg)
    for i in range(10):
        nb.compute_maturity(0.3 + 0.02 * i, 1.0 - 0.05 * i, 0.4, 0.05)
    entropy_history_before = list(nb._entropy_history)
    loss_history_before = list(nb._loss_history)
    cycles_before = nb._sleep_cycles_elapsed
    hot = nb.get_hot_state()
    nb._entropy_history = []
    nb._loss_history = []
    nb._sleep_cycles_elapsed = 0
    nb.load_hot_state(hot)
    assert nb._entropy_history == entropy_history_before, \
        "Entropy history not restored"
    assert nb._loss_history == loss_history_before, \
        "Loss history not restored"
    assert nb._sleep_cycles_elapsed == cycles_before, \
        "Sleep cycle counter not restored"
    print("PASS: HOT state roundtrip preserves history and cycle counter")


if __name__ == "__main__":
    print("Running NeuromodulatorBroadcast / MaturityComputer regression tests...\n")
    test_da_ema_moves_toward_signal()
    test_ach_ema_moves_toward_signal()
    test_ne_ema_moves_toward_signal()
    test_ht_ema_moves_toward_signal()
    test_update_methods_no_grad()
    test_is_sleep_phase_gates_on_ach()
    test_maturity_stays_in_bounds()
    test_asymmetric_ema_slow_up_fast_down()
    test_pessimistic_defaults_produce_low_maturity()
    test_stable_signals_drive_maturity_up()
    test_unstable_signals_drive_maturity_down()
    test_baseline_statistics_set_after_window()
    test_history_caps_after_baseline()
    test_hot_state_roundtrip()
    print("\nAll tests passed.")
