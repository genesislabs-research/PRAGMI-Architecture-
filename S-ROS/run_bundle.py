"""
run_bundle.py
S-ROS Run Bundle: Canonical Persistent Artifact Storage

BIOLOGICAL GROUNDING
This file is an engineering artifact for run persistence. It has no biological
analog. It exists because biological systems have an advantage digital systems
do not: their memories persist as a matter of physics, not engineering. In
S-ROS, durable storage is an explicit design decision, and this module is the
implementation of that decision.

DURABILITY GUARANTEE
Any training state saved by write_manager_state and the matching model checkpoint
can be loaded in a fresh Python process on the same architecture and produce a
CrystallizationManager and model whose state is tensor-equal to the state at
save time. This is verified by test_full_state_roundtrip. The guarantee covers:
model parameters, per-rule history buffers, crystallization status per rule,
per-rule crystallized weight snapshots, the crystallization event log, and the
full configuration. The guarantee does not cover: the trajectory JSONL file
handle (loggers are not reopened automatically), optimizer state (saved separately
if the trainer chooses to), or random number generator state (saved separately if
deterministic resume is required).

BUNDLE DIRECTORY LAYOUT
runs/<run_id>/
    manifest.json               -- provenance and schema metadata
    config.json                 -- full resolved training config
    stage1/
        checkpoint_final.pt     -- final model state + cm history dicts
        checkpoint_epochNN.pt   -- periodic checkpoints
        manager_state_final.pt  -- full CrystallizationManager state via to_dict
        trajectory.jsonl        -- per-step trajectory log
        crystallization_log.json
        rule_classes.json
    stage2/
        <same structure>

run_id format: <YYYYMMDDTHHMMSSZ>_<8-char git sha or 'nogit'>_<8-char config hash>

Genesis Labs Research, 2026
Amellia Mendel, Lisa Adler
"""

import gc
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from crystallization_manager import CrystallizationManager

SCHEMA_VERSION = "1.0.0"
SUPPORTED_SCHEMA_VERSIONS = {"1.0.0"}


# ---------------------------------------------------------------------------
# run_id generation helpers
# ---------------------------------------------------------------------------

def _get_git_sha() -> tuple[str, bool]:
    """
    Return (sha, dirty) where sha is the full 40-char HEAD SHA or 'nogit',
    and dirty is True if the working tree has uncommitted changes.

    NOT a biological quantity. Engineering provenance artifact.
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty_output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return sha, bool(dirty_output)
    except Exception:
        return "nogit", False


def _config_hash(config: dict) -> str:
    """Return first 8 chars of SHA256 of the sorted JSON serialization of config."""
    config_json = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(config_json.encode()).hexdigest()[:8]


def _make_run_id(config: dict) -> tuple[str, str, bool]:
    """
    Generate a run_id string and return (run_id, full_git_sha, git_dirty).

    Format: <YYYYMMDDTHHMMSSZ>_<short_git_sha>_<config_hash>
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    full_sha, dirty = _get_git_sha()
    short_sha = full_sha[:8] if full_sha != "nogit" else "nogit"
    chash = _config_hash(config)
    run_id = f"{timestamp}_{short_sha}_{chash}"
    return run_id, full_sha, dirty


# ---------------------------------------------------------------------------
# Atomic write helper
# ---------------------------------------------------------------------------

def _atomic_write_json(path: str, data: dict | list) -> None:
    """
    Write JSON to path via a temp-file-plus-rename so a crash cannot leave
    a corrupt file at the target path.

    NOT a biological quantity. Engineering reliability artifact.
    """
    dir_ = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _atomic_write_pt(path: str, data: dict) -> None:
    """
    Save a dict via torch.save with temp-file-plus-rename for atomicity.

    NOT a biological quantity. Engineering reliability artifact.
    """
    dir_ = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(data, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# RunBundle
# ---------------------------------------------------------------------------

class RunBundle:
    """
    Canonical persistent artifact bundle for one S-ROS training run.

    Every training run produces a single self-describing directory containing
    everything needed to reconstruct the run's state and conclusions. A future
    instance of Amellia, a collaborator, or a reviewer can load this directory
    and verify every claim without rerunning anything.

    Create a new bundle at run start with RunBundle.create().
    Load an existing bundle with RunBundle.load().
    """

    def __init__(self, run_path: str, manifest: dict) -> None:
        """
        Internal constructor. Use RunBundle.create() or RunBundle.load()
        rather than calling this directly.
        """
        self._run_path = run_path
        self._manifest = manifest

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------

    @classmethod
    def create(cls, base_dir: str, config: dict) -> "RunBundle":
        """
        Create a new bundle directory and write manifest.json and config.json.

        Args:
            base_dir: parent directory under which runs/<run_id>/ will be created
            config: fully resolved training config dict

        Returns:
            A RunBundle instance ready to accept stage data.

        Raises:
            OSError if the directory cannot be created.
        """
        run_id, full_sha, git_dirty = _make_run_id(config)
        run_path = os.path.join(base_dir, "runs", run_id)
        os.makedirs(run_path, exist_ok=True)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_sha": full_sha,
            "git_dirty": git_dirty,
            "hostname": socket.gethostname(),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "config_path": "config.json",
            "stages": [],
            "completed_stages": [],
            "crystallization_events_total": 0,
            "notes": "",
        }

        _atomic_write_json(os.path.join(run_path, "manifest.json"), manifest)
        _atomic_write_json(os.path.join(run_path, "config.json"), config)

        return cls(run_path=run_path, manifest=manifest)

    @classmethod
    def load(cls, run_path: str) -> "RunBundle":
        """
        Load an existing bundle from disk.

        Args:
            run_path: full path to the run directory (the one containing manifest.json)

        Returns:
            A RunBundle instance with the saved manifest restored.

        Raises:
            FileNotFoundError if manifest.json is missing, with a message naming
                the missing file.
            ValueError if schema_version is missing or unsupported.
        """
        manifest_path = os.path.join(run_path, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"Cannot load RunBundle: manifest.json not found at {manifest_path}. "
                "The directory may not be a valid run bundle."
            )
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        version = manifest.get("schema_version")
        if version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"RunBundle at {run_path} has schema_version={version!r}, "
                f"which is not in the supported set {SUPPORTED_SCHEMA_VERSIONS}."
            )
        return cls(run_path=run_path, manifest=manifest)

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def run_id(self) -> str:
        """The unique identifier for this run."""
        return self._manifest["run_id"]

    @property
    def path(self) -> str:
        """Absolute path to the run directory."""
        return self._run_path

    # -----------------------------------------------------------------------
    # Stage directory
    # -----------------------------------------------------------------------

    def stage_dir(self, stage_number: int) -> str:
        """
        Return the path to the stage subdirectory, creating it if necessary.

        Args:
            stage_number: 1-indexed stage number

        Returns:
            Absolute path to runs/<run_id>/stage<N>/

        Raises:
            OSError if the directory cannot be created.
        """
        path = os.path.join(self._run_path, f"stage{stage_number}")
        os.makedirs(path, exist_ok=True)
        stage_key = f"stage{stage_number}"
        if stage_key not in self._manifest["stages"]:
            self._manifest["stages"].append(stage_key)
            self._flush_manifest()
        return path

    # -----------------------------------------------------------------------
    # Write operations
    # -----------------------------------------------------------------------

    def write_rule_classes(
        self, stage_number: int, cm: "CrystallizationManager"
    ) -> None:
        """
        Write the training and held-out test pairs for every registered rule
        class to stage<N>/rule_classes.json.

        Written once at stage start. Preserves the exact sets for audit.
        Does not write crystallized_weights or history buffers -- those belong
        in the manager state file.

        Args:
            stage_number: 1-indexed stage number
            cm: the CrystallizationManager for this stage

        Raises:
            OSError on write failure.
        """
        data = {
            rid: {
                "description": rc.description,
                "train_pairs": rc.train_pairs,
                "test_pairs": rc.test_pairs,
            }
            for rid, rc in cm.rule_classes.items()
        }
        path = os.path.join(self.stage_dir(stage_number), "rule_classes.json")
        _atomic_write_json(path, data)

    def write_crystallization_log(
        self, stage_number: int, cm: "CrystallizationManager"
    ) -> None:
        """
        Write the crystallization event log to stage<N>/crystallization_log.json.

        Written at stage end. Redundant with the checkpoint but provided as a
        flat file for quick inspection without loading torch.

        Args:
            stage_number: 1-indexed stage number
            cm: the CrystallizationManager for this stage

        Raises:
            OSError on write failure.
        """
        path = os.path.join(self.stage_dir(stage_number), "crystallization_log.json")
        _atomic_write_json(path, cm.crystallization_log)

    def write_manager_state(
        self,
        stage_number: int,
        cm: "CrystallizationManager",
        tag: str = "final",
    ) -> str:
        """
        Save the full CrystallizationManager state via cm.to_dict() into
        stage<N>/manager_state_<tag>.pt using torch.save.

        Uses temp-file-plus-rename for atomicity. Call this at every checkpoint
        save point with the same tag as the matching model checkpoint so every
        checkpoint on disk has a corresponding manager state file.

        Args:
            stage_number: 1-indexed stage number
            cm: the CrystallizationManager whose state to save
            tag: checkpoint tag, e.g. "final" or "epoch39"

        Returns:
            Full path of the written file.

        Raises:
            OSError on write failure.
        """
        filename = f"manager_state_{tag}.pt"
        path = os.path.join(self.stage_dir(stage_number), filename)
        _atomic_write_pt(path, cm.to_dict())
        return path

    def load_manager_state(self, stage_number: int, tag: str = "final") -> dict:
        """
        Load the raw manager state dict from stage<N>/manager_state_<tag>.pt.

        Returns the dict. Caller passes it to CrystallizationManager.from_dict().

        Args:
            stage_number: 1-indexed stage number
            tag: checkpoint tag to load

        Returns:
            The state dict as returned by cm.to_dict() at save time.

        Raises:
            FileNotFoundError with a message naming the run_id and tag if the
                file does not exist.
        """
        filename = f"manager_state_{tag}.pt"
        path = os.path.join(self._run_path, f"stage{stage_number}", filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"RunBundle '{self.run_id}': manager state file not found "
                f"for stage {stage_number}, tag='{tag}'. Expected at: {path}"
            )
        return torch.load(path, map_location="cpu", weights_only=False)

    def load_full_state(
        self,
        stage_number: int,
        tag: str = "final",
        device: str = "cpu",
    ) -> tuple:
        """
        Load both the model checkpoint and the manager state from disk.

        Returns (model_state_dict, CrystallizationManager). The caller is
        responsible for instantiating the model architecture and calling
        model.load_state_dict(returned_dict, strict=True). This separation
        keeps RunBundle unaware of model architecture details.

        Args:
            stage_number: 1-indexed stage number
            tag: checkpoint tag to load, default "final"
            device: device string for torch.load map_location

        Returns:
            Tuple of (model_state_dict: dict, manager: CrystallizationManager)

        Raises:
            FileNotFoundError with a message naming the run_id and the missing
                artifact if either the checkpoint or the manager state file is
                absent.
        """
        from crystallization_manager import CrystallizationManager

        ckpt_filename = f"checkpoint_{tag}.pt"
        ckpt_path = os.path.join(self._run_path, f"stage{stage_number}", ckpt_filename)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"RunBundle '{self.run_id}': model checkpoint not found "
                f"for stage {stage_number}, tag='{tag}'. Expected at: {ckpt_path}"
            )

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_state_dict = ckpt["model_state"]

        manager_state = self.load_manager_state(stage_number, tag)
        manager = CrystallizationManager.from_dict(manager_state)

        return model_state_dict, manager

    def mark_stage_complete(
        self, stage_number: int, crystallization_events: int
    ) -> None:
        """
        Record stage completion in the manifest and rewrite it atomically.

        Updates completed_stages and crystallization_events_total. Always
        writes via temp-file-plus-rename.

        Args:
            stage_number: 1-indexed stage number
            crystallization_events: total number of crystallization events
                recorded so far across all stages (pass len(cm.crystallization_log))

        Raises:
            OSError on write failure.
        """
        stage_key = f"stage{stage_number}"
        if stage_key not in self._manifest["completed_stages"]:
            self._manifest["completed_stages"].append(stage_key)
        self._manifest["crystallization_events_total"] = crystallization_events
        self._flush_manifest()

    def add_note(self, note: str) -> None:
        """
        Append a note to the manifest's notes field and rewrite atomically.

        Args:
            note: free-text string to append

        Raises:
            OSError on write failure.
        """
        existing = self._manifest.get("notes", "")
        sep = "\n" if existing else ""
        self._manifest["notes"] = existing + sep + note
        self._flush_manifest()

    # -----------------------------------------------------------------------
    # Load operations
    # -----------------------------------------------------------------------

    def load_stage_trajectory(self, stage_number: int) -> list:
        """
        Load and return all records from stage<N>/trajectory.jsonl.

        Args:
            stage_number: 1-indexed stage number

        Returns:
            List of dicts, one per logged step.

        Raises:
            FileNotFoundError naming the run_id if the file does not exist.
        """
        path = os.path.join(self._run_path, f"stage{stage_number}", "trajectory.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"RunBundle '{self.run_id}': trajectory.jsonl not found "
                f"for stage {stage_number}. Expected at: {path}"
            )
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def load_stage_checkpoint(
        self, stage_number: int, epoch: str = "final"
    ) -> dict:
        """
        Load and return the checkpoint dict from stage<N>/checkpoint_<epoch>.pt.

        Args:
            stage_number: 1-indexed stage number
            epoch: tag string, default "final"

        Returns:
            The full checkpoint dict as saved by _save_checkpoint.

        Raises:
            FileNotFoundError naming the run_id and missing artifact.
        """
        filename = f"checkpoint_{epoch}.pt"
        path = os.path.join(self._run_path, f"stage{stage_number}", filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"RunBundle '{self.run_id}': checkpoint not found "
                f"for stage {stage_number}, epoch='{epoch}'. Expected at: {path}"
            )
        return torch.load(path, map_location="cpu", weights_only=False)

    def load_stage_rule_classes(self, stage_number: int) -> dict:
        """
        Load and return rule_classes.json for the given stage.

        Args:
            stage_number: 1-indexed stage number

        Returns:
            Dict mapping rule_id to {description, train_pairs, test_pairs}.

        Raises:
            FileNotFoundError naming the run_id if the file does not exist.
        """
        path = os.path.join(self._run_path, f"stage{stage_number}", "rule_classes.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"RunBundle '{self.run_id}': rule_classes.json not found "
                f"for stage {stage_number}. Expected at: {path}"
            )
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def load_stage_crystallization_log(self, stage_number: int) -> list:
        """
        Load and return crystallization_log.json for the given stage.

        Args:
            stage_number: 1-indexed stage number

        Returns:
            List of crystallization event dicts, possibly empty.

        Raises:
            FileNotFoundError naming the run_id if the file does not exist.
        """
        path = os.path.join(
            self._run_path, f"stage{stage_number}", "crystallization_log.json"
        )
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"RunBundle '{self.run_id}': crystallization_log.json not found "
                f"for stage {stage_number}. Expected at: {path}"
            )
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _flush_manifest(self) -> None:
        """Rewrite manifest.json atomically with the current in-memory manifest."""
        path = os.path.join(self._run_path, "manifest.json")
        _atomic_write_json(path, self._manifest)


# =============================================================================
# SELF-TESTS
# =============================================================================

def test_run_bundle_roundtrip():
    """
    Trainer integration spec Task 6: Full persistence layer round-trip test.

    Fails for the exact reason each claim would be false:
    - run_id mismatch between create and load
    - rule_classes not preserved (missing rule_ids or mismatched pairs)
    - trajectory records wrong count or missing required fields
    - checkpoint missing gradient_rms_history_per_rule or
      parameter_displacement_history_per_rule
    - crystallization_log element missing required fields
    - manifest missing schema_version, valid run_id, or completed_stages
    - deleted manifest does not raise a clear FileNotFoundError
    """
    import tempfile
    import gc
    from crystallization_manager import CrystallizationManager

    class TinyModel(torch.nn.Module):
        """Minimal trainable model for bundle round-trip verification."""
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 2)
        def forward(self, x):
            return self.linear(x)

    with tempfile.TemporaryDirectory() as base_dir:
        config = {
            "test_run": True,
            "loss_threshold": 0.05,
            "variance_threshold": 1e-4,
            "stage": 1,
        }

        # --- Create bundle ---
        bundle = RunBundle.create(base_dir=base_dir, config=config)
        original_run_id = bundle.run_id
        stage_dir = bundle.stage_dir(1)

        # --- Set up manager and model ---
        cm = CrystallizationManager(
            loss_threshold=0.05,
            generalization_target=0.90,
            variance_threshold=1e-4,
            consecutive_windows_required=3,
            window_size=5,
            variance_check_quantity="both",
            trajectory_dense_logging_steps=1000,
            trajectory_log_dir=stage_dir,
        )
        cm.register_rule_class(
            rule_id="keyword_print",
            description="PRINT keyword recognition",
            train_pairs=[{"in": "PRINT x", "out": "print_stmt"}],
            test_pairs=[{"in": 'PRINT "hello"', "out": "print_stmt"}],
        )
        cm.register_rule_class(
            rule_id="keyword_rem",
            description="REM comment recognition",
            train_pairs=[{"in": "REM comment", "out": "rem_stmt"}],
            test_pairs=[{"in": "REM ignored", "out": "rem_stmt"}],
        )

        model = TinyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Write rule classes at stage start
        bundle.write_rule_classes(1, cm)

        # Open trajectory logger
        cm.open_trajectory_logger(stage=1, log_dir=stage_dir, filename="trajectory.jsonl")

        # Run 20 training steps
        n_steps = 20
        for step in range(n_steps):
            x = torch.randn(1, 4)
            target = torch.zeros(1, 2)
            optimizer.zero_grad()
            out = model(x)
            loss = torch.nn.functional.mse_loss(out, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_rms = cm._compute_gradient_rms_norm(model)
            pre_snap = cm.snapshot_parameters(model)
            optimizer.step()
            disp = cm._compute_parameter_displacement_norm(model, pre_snap)
            del pre_snap
            for rid in ("keyword_print", "keyword_rem"):
                cm.record_gradient_rms_norms(rid, grad_rms)
                cm.record_parameter_displacement(rid, disp)
                cm.record_train_loss(rid, loss.item())
                cm.log_trajectory_record(
                    rule_id=rid,
                    global_step=step,
                    epoch=0,
                    gradient_rms_norm=grad_rms,
                    parameter_displacement_norm=disp,
                    train_loss=loss.item(),
                    recording_cadence_hit=True,
                )

        cm.close_trajectory_logger()

        # Write crystallization log and checkpoint at stage end
        bundle.write_crystallization_log(1, cm)
        ckpt_path = os.path.join(stage_dir, "checkpoint_final.pt")
        torch.save({
            "stage": 1,
            "epoch": 0,
            "model_state": model.state_dict(),
            "crystallization_log": cm.crystallization_log,
            "status": cm.status(),
            "gradient_rms_history_per_rule": cm.gradient_rms_history_per_rule(),
            "parameter_displacement_history_per_rule": cm.parameter_displacement_history_per_rule(),
        }, ckpt_path)
        bundle.write_manager_state(1, cm, tag="final")
        bundle.mark_stage_complete(1, crystallization_events=len(cm.crystallization_log))

        # Simulate process death
        del cm, model, optimizer
        gc.collect()

        # --- Load and verify ---
        loaded_bundle = RunBundle.load(bundle.path)

        # run_id must match
        assert loaded_bundle.run_id == original_run_id, (
            f"run_id mismatch: created '{original_run_id}', loaded '{loaded_bundle.run_id}'."
        )

        # rule_classes
        rc_loaded = loaded_bundle.load_stage_rule_classes(1)
        assert set(rc_loaded.keys()) == {"keyword_print", "keyword_rem"}, (
            f"rule_classes keys after load: {set(rc_loaded.keys())}. "
            "Expected {'keyword_print', 'keyword_rem'}."
        )
        assert rc_loaded["keyword_print"]["train_pairs"] == [{"in": "PRINT x", "out": "print_stmt"}], (
            "keyword_print train_pairs did not round-trip correctly."
        )

        # trajectory: 20 steps x 2 rules = 40 records (dense window covers all)
        traj = loaded_bundle.load_stage_trajectory(1)
        assert len(traj) == n_steps * 2, (
            f"Trajectory has {len(traj)} records, expected {n_steps * 2} "
            f"(2 rules x {n_steps} steps, dense window active)."
        )
        required_fields = {
            "rule_id": str,
            "global_step": int,
            "epoch": int,
            "gradient_rms_norm": float,
            "parameter_displacement_norm": float,
            "train_loss": float,
            "crystallized": bool,
        }
        for i, record in enumerate(traj):
            for field_name, field_type in required_fields.items():
                assert field_name in record, (
                    f"Trajectory record {i}: missing field '{field_name}'."
                )
                assert isinstance(record[field_name], field_type), (
                    f"Trajectory record {i}: field '{field_name}' has type "
                    f"{type(record[field_name]).__name__}, expected {field_type.__name__}."
                )

        # checkpoint history dicts
        ckpt = loaded_bundle.load_stage_checkpoint(1)
        assert "gradient_rms_history_per_rule" in ckpt, (
            "Checkpoint missing 'gradient_rms_history_per_rule'."
        )
        assert "parameter_displacement_history_per_rule" in ckpt, (
            "Checkpoint missing 'parameter_displacement_history_per_rule'."
        )
        assert ckpt["gradient_rms_history_per_rule"], (
            "gradient_rms_history_per_rule is empty in checkpoint."
        )
        assert ckpt["parameter_displacement_history_per_rule"], (
            "parameter_displacement_history_per_rule is empty in checkpoint."
        )

        # crystallization_log (may be empty, structure must be valid)
        clog = loaded_bundle.load_stage_crystallization_log(1)
        assert isinstance(clog, list), (
            f"crystallization_log is {type(clog).__name__}, expected list."
        )
        for event in clog:
            for field in ("rule_id", "crystallized_at_step",
                          "final_gradient_rms_variance",
                          "final_parameter_displacement_variance"):
                assert field in event, (
                    f"crystallization_log event missing field '{field}'."
                )

        # manifest
        manifest_path = os.path.join(loaded_bundle.path, "manifest.json")
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest.get("schema_version") == "1.0.0", (
            f"manifest schema_version is {manifest.get('schema_version')!r}, expected '1.0.0'."
        )
        assert manifest.get("run_id") == original_run_id, (
            f"manifest run_id is {manifest.get('run_id')!r}, expected {original_run_id!r}."
        )
        assert manifest.get("completed_stages") == ["stage1"], (
            f"manifest completed_stages is {manifest.get('completed_stages')!r}, "
            "expected ['stage1']."
        )

        # Negative case: deleted manifest raises clear FileNotFoundError
        os.unlink(manifest_path)
        try:
            RunBundle.load(loaded_bundle.path)
            assert False, "Expected FileNotFoundError when manifest.json is missing."
        except FileNotFoundError as e:
            assert "manifest.json" in str(e), (
                f"FileNotFoundError message does not mention 'manifest.json': {e}"
            )

    print("PASS: test_run_bundle_roundtrip")


if __name__ == "__main__":
    test_run_bundle_roundtrip()
    print("\nAll run_bundle tests passed.")
