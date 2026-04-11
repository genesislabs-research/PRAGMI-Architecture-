"""
theo_checkpoint.py
Save and load Theo .soul checkpoint files.

BIOLOGICAL GROUNDING
This file implements no neural dynamics. It is purely an engineering persistence
layer for the TheoCore module defined in Theo_Core_t.py. It serializes the full
module state including the S-CAM engram buffers, the recurrent executor weights,
and the crystallization ledger, then deserializes them with strict
hyperparameter validation to guarantee architectural consistency on load.

The .soul format is named to reflect the project's design intent: the
checkpoint captures not just weights but the identity state of the organism,
including which skills have been crystallized and when.
"""

import torch
import torch.nn as nn
import os
import tempfile
import json
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional

from Theo_Core_t import TheoCore, TheoSCAM, TheoRecurrentExecutor

SOUL_FORMAT_VERSION = "theo_soul_v1"


def save_theo_soul(
    theo_core: "TheoCore",
    ledger: List[Dict],
    path: str
) -> str:
    """
    Save the complete TheoCore state to a .soul file.

    Captures the full module state_dict, the S-CAM engram buffers,
    the crystallization ledger, and the hyperparameters needed to
    reconstruct a compatible module on load.

    Uses an atomic write pattern (write to temp, then os.replace) so
    that an existing checkpoint is never left in a corrupted state if
    the process is interrupted during save.

    Args:
        theo_core: The TheoCore instance to save.
        ledger: List of crystallization event dictionaries.
        path: Destination file path. Will be created or overwritten.

    Returns:
        The resolved absolute path of the saved file.
    """
    # NOTE: TheoRecurrentExecutor does not store coord_dim as an instance
    # attribute. It is recoverable as input_dim - spike_dim because
    # TheoRecurrentExecutor.__init__ sets self.input_dim = spike_dim + coord_dim.
    _coord_dim = theo_core.executor.input_dim - theo_core.scam.spike_dim

    hyperparameters = {
        "spike_dim": theo_core.scam.spike_dim,
        "max_engrams": theo_core.scam.max_engrams,
        "coord_dim": _coord_dim,
        "hidden_dim": 256,
        "num_layers": theo_core.executor.num_layers,
        "beta": theo_core.executor.beta,
        "num_steps": theo_core.executor.num_steps,
        "confidence_threshold": theo_core.confidence_threshold,
    }

    checkpoint = {
        "format_version": SOUL_FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "crystallization_count": len(ledger),
        "theo_core_state_dict": theo_core.state_dict(),
        "scam_buffers": {
            "sensor_keys": theo_core.scam.sensor_keys.clone(),
            "action_values": theo_core.scam.action_values.clone(),
            "is_active": theo_core.scam.is_active.clone(),
        },
        "crystallization_ledger": ledger,
        "hyperparameters": hyperparameters,
    }

    target_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(target_dir, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        dir=target_dir, suffix=".soul.tmp"
    )
    os.close(fd)

    try:
        torch.save(checkpoint, tmp_path, _use_new_zipfile_serialization=True)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    return path


def load_theo_soul(
    path: str,
    theo_core: "TheoCore",
) -> Tuple[List[Dict], Dict]:
    """
    Load a .soul checkpoint into an existing TheoCore instance.

    Validates format version and hyperparameters before modifying any
    module state. If any dimension mismatches, raises ValueError with
    the specific key that is wrong so the caller knows exactly what
    needs to change.

    Args:
        path: Path to the .soul file.
        theo_core: The TheoCore instance that will receive the loaded state.

    Returns:
        (crystallization_ledger, hyperparameters)

    Raises:
        ValueError: If format version or any architecture dimension mismatches.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    stored_version = checkpoint.get("format_version", "unknown")
    if stored_version != SOUL_FORMAT_VERSION:
        raise ValueError(
            f"Soul format mismatch. File has '{stored_version}', "
            f"expected '{SOUL_FORMAT_VERSION}'."
        )

    stored_hp = checkpoint["hyperparameters"]
    # NOTE: TheoRecurrentExecutor does not store coord_dim directly.
    # Derived from input_dim - spike_dim, matching __init__ arithmetic.
    _coord_dim = theo_core.executor.input_dim - theo_core.scam.spike_dim
    current_hp = {
        "spike_dim": theo_core.scam.spike_dim,
        "max_engrams": theo_core.scam.max_engrams,
        "coord_dim": _coord_dim,
        "num_layers": theo_core.executor.num_layers,
    }
    for key, current_val in current_hp.items():
        stored_val = stored_hp.get(key)
        if stored_val is not None and stored_val != current_val:
            raise ValueError(
                f"Architecture mismatch on '{key}': "
                f"checkpoint has {stored_val}, current module has {current_val}."
            )

    theo_core.load_state_dict(checkpoint["theo_core_state_dict"])

    ledger = checkpoint.get("crystallization_ledger", [])
    return ledger, stored_hp


def test_theo_soul_roundtrip():
    """
    Creates a TheoCore, crystallizes one fake skill, saves to temp file,
    creates a fresh TheoCore, loads the checkpoint, and verifies everything
    matches.
    """
    import tempfile

    original = TheoCore(spike_dim=128, coord_dim=64)

    fake_sensor = torch.randn(1, 128)
    fake_action = torch.randn(1, 128)
    original.scam.crystallize_skill(fake_sensor, fake_action)

    ledger = [{
        "skill_id": "test_skill_001",
        "crystallized_at": datetime.now(timezone.utc).isoformat(),
        "source_loss": 0.005,
        "engram_indices": [0],
        "description": "Roundtrip test skill",
    }]

    with tempfile.TemporaryDirectory() as tmpdir:
        soul_path = os.path.join(tmpdir, "test_theo.soul")
        save_theo_soul(original, ledger, soul_path)

        assert os.path.exists(soul_path), "Soul file was not created"
        assert os.path.getsize(soul_path) > 0, "Soul file is empty"

        loaded = TheoCore(spike_dim=128, coord_dim=64)

        loaded_ledger, loaded_hp = load_theo_soul(soul_path, loaded)

        assert len(loaded_ledger) == 1, f"Ledger length wrong: {len(loaded_ledger)}"
        assert loaded_ledger[0]["skill_id"] == "test_skill_001"
        assert loaded_ledger[0]["source_loss"] == 0.005

        assert loaded_hp["spike_dim"] == 128
        assert loaded_hp["max_engrams"] == 1024
        assert loaded_hp["coord_dim"] == 64

        assert torch.equal(original.scam.sensor_keys, loaded.scam.sensor_keys), \
            "S-CAM sensor_keys mismatch"
        assert torch.equal(original.scam.action_values, loaded.scam.action_values), \
            "S-CAM action_values mismatch"
        assert torch.equal(original.scam.is_active, loaded.scam.is_active), \
            "S-CAM is_active mismatch"

        for name, param in original.executor.named_parameters():
            loaded_param = dict(loaded.executor.named_parameters())[name]
            assert torch.equal(param, loaded_param), \
                f"Executor parameter '{name}' mismatch"

        for key in original.state_dict():
            assert torch.equal(
                original.state_dict()[key],
                loaded.state_dict()[key]
            ), f"State dict key '{key}' mismatch"

    print("PASS: test_theo_soul_roundtrip")
    print(f"  Saved and loaded 1 crystallized skill")
    print(f"  All tensors match")
    print(f"  Ledger intact")
    print(f"  Hyperparameters verified")


def test_theo_soul_dimension_mismatch():
    """
    Verifies that loading a checkpoint into a module with wrong dimensions
    raises a clear ValueError.
    """
    import tempfile

    original = TheoCore(spike_dim=128, coord_dim=64)
    ledger = []

    with tempfile.TemporaryDirectory() as tmpdir:
        soul_path = os.path.join(tmpdir, "test_mismatch.soul")
        save_theo_soul(original, ledger, soul_path)

        wrong_core = TheoCore(spike_dim=64, coord_dim=32)

        try:
            load_theo_soul(soul_path, wrong_core)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Architecture mismatch" in str(e), f"Wrong error: {e}"
            print(f"PASS: test_theo_soul_dimension_mismatch")
            print(f"  Correctly rejected mismatched dimensions: {e}")


def test_theo_soul_atomic_save():
    """
    Verifies that if the original .soul file exists, it is not corrupted
    if the save process is interrupted (simulated by checking temp file cleanup).
    """
    import tempfile

    core = TheoCore(spike_dim=128, coord_dim=64)

    with tempfile.TemporaryDirectory() as tmpdir:
        soul_path = os.path.join(tmpdir, "test_atomic.soul")

        save_theo_soul(core, [], soul_path)
        first_size = os.path.getsize(soul_path)
        assert first_size > 0

        core.scam.crystallize_skill(torch.randn(1, 128), torch.randn(1, 128))
        ledger = [{"skill_id": "atomic_test", "crystallized_at": "now",
                    "source_loss": 0.003, "engram_indices": [0],
                    "description": "atomic save test"}]
        save_theo_soul(core, ledger, soul_path)
        second_size = os.path.getsize(soul_path)
        assert second_size > 0

        remaining = [f for f in os.listdir(tmpdir) if f.endswith(".soul.tmp")]
        assert len(remaining) == 0, f"Temp files left behind: {remaining}"

    print("PASS: test_theo_soul_atomic_save")
    print(f"  First save: {first_size} bytes")
    print(f"  Second save: {second_size} bytes")
    print(f"  No temp files left behind")


if __name__ == "__main__":
    test_theo_soul_roundtrip()
    test_theo_soul_dimension_mismatch()
    test_theo_soul_atomic_save()
    print("\nAll Theo checkpoint tests passed.")
