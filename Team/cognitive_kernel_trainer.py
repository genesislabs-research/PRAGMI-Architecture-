"""
cognitive_kernel_trainer.py
Training harness for the Cognitive Kernel hippocampal memory system.

BIOLOGICAL GROUNDING
====================
This trainer exercises the CognitiveKernel in a self-supervised
reconstruction objective. The biological rationale for this objective
is as follows: the hippocampus is not merely a retrieval system but a
predictive system. It continuously generates predictions about incoming
experience (the reconstructed coordinate output of the Subiculum) and
compares those predictions to actual cortical input (the EC-normalized
coordinate vector). The discrepancy between prediction and input is
precisely the novelty signal computed by CA1.

Training therefore optimizes the learned projections (perforant path,
mossy fiber, Schaffer collateral, CA1 comparison matrices, subiculum
gate) to produce reconstructions that converge toward the normalized
EC input. This is a form of predictive coding, which is the dominant
computational theory of neocortical and hippocampal function.

Friston KJ (2010). "The free-energy principle: a unified brain theory?"
Nature Reviews Neuroscience, 11(2):127-138.
DOI: 10.1038/nrn2787

Rao RP, Ballard DH (1999). "Predictive coding in the visual cortex:
a functional interpretation of some extra-classical receptive-field
effects." Nature Neuroscience, 2(1):79-87.
DOI: 10.1038/4580

The astrocytic eta multiplier is incorporated into the optimizer learning
rate at each step, implementing the BCM-style sliding threshold:
high population activity raises eta and amplifies plasticity, low
activity dampens it.

Bienenstock EL et al. (1982). "Theory for the development of neuron
selectivity: orientation specificity and binocular interaction in
visual cortex." Journal of Neuroscience, 2(1):32-48.
DOI: 10.1523/JNEUROSCI.02-01-00032.1982

MEMORY TIER SERIALIZATION
==========================
    HOT (soul checkpoint, saved every step):
        - EC short_term_buffer
        - CA1 working_memory
    WARM (saved at sleep boundary):
        - All learned parameters (optimizer state, model weights)
    COLD (saved at sleep boundary + on demand):
        - CA3 stored patterns
        - CA3 W_recurrent
        - Astrocyte glutamate, calcium states
        - Neuron positions (structural, not trained)

DESIGN NOTES
============
W_recurrent and the CA3 distance_mask are registered buffers, not
parameters. They must be excluded from the optimizer explicitly.
store_episode() updates W_recurrent via pseudoinverse recompute, not
gradient descent. The optimizer must not touch it.

The reconstruction loss is computed against EC-normalized coordinates
(ec_output), not raw input coordinates. This is intentional: the network
sees normalized input internally, and the subiculum is learning to
reconstruct exactly what the EC passes forward, not the unnormalized
input that precedes EC LayerNorm. Computing loss against raw coords
would introduce a systematic scale mismatch.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

# ---------------------------------------------------------------------------
# Import kernel modules. Adjust the import path if the files are renamed or
# placed in a package. The trainer has no other runtime dependencies.
# ---------------------------------------------------------------------------
try:
    from cognitive_kernel_base_for_testing_c__1_ import (
        CognitiveKernel,
        CognitiveKernelConfig,
    )
except ImportError:
    # Fallback: try the teaching file variant.
    from cognitive_kernel_base_for_testing_t__1_ import (  # type: ignore
        CognitiveKernel,
        CognitiveKernelConfig,
    )

logger = logging.getLogger(__name__)


# ===========================================================================
# Trainer Configuration
# ===========================================================================

@dataclass
class TrainerConfig:
    """
    Configuration for the Cognitive Kernel training harness.

    Biological parameters are cited. Engineering parameters are labeled.
    """

    # ---- Optimization ----

    # Base learning rate for Adam optimizer.
    # NOT a biological quantity. Engineering hyperparameter.
    # Astrocytic eta multiplies this at every step.
    lr: float = 3e-4

    # Adam epsilon. Standard value, NOT a biological quantity.
    adam_eps: float = 1e-8

    # Adam weight decay. NOT a biological quantity.
    weight_decay: float = 1e-4

    # Gradient clipping maximum norm.
    # NOT a biological quantity. Engineering safety bound.
    grad_clip_norm: float = 1.0

    # ---- Training Loop ----

    # Total number of training steps.
    # NOT a biological quantity.
    n_steps: int = 10_000

    # Batch size (number of coordinate vectors per forward pass).
    # NOT a biological quantity.
    batch_size: int = 32

    # Number of steps between full validation runs.
    # NOT a biological quantity.
    eval_interval: int = 200

    # Number of validation batches per evaluation.
    # NOT a biological quantity.
    eval_batches: int = 20

    # ---- Sleep Cycle ----

    # Number of steps between sleep consolidation events.
    # Biologically, this models the wake-sleep cycle boundary where
    # EC short-term memory is cleared and CA3 patterns are replayed.
    # NOT a biological quantity at this granularity.
    sleep_interval: int = 500

    # ---- Noise and Retrieval Testing ----

    # Gaussian noise sigma applied to coordinates during retrieval tests.
    # Used to evaluate pattern completion quality under degraded input.
    # Biologically analogous to the partial-cue retrieval paradigm.
    # NOT a biological quantity. Engineering evaluation parameter.
    retrieval_noise_sigma: float = 0.3

    # Number of retrieval pairs to test per evaluation round.
    # NOT a biological quantity.
    n_retrieval_test_pairs: int = 8

    # ---- Checkpointing ----

    # Directory for all checkpoints.
    # NOT a biological quantity.
    checkpoint_dir: str = "checkpoints"

    # How often to save a full WARM checkpoint (weights + optimizer).
    # NOT a biological quantity.
    checkpoint_interval: int = 1_000

    # ---- Logging ----

    # How often to print training diagnostics.
    # NOT a biological quantity.
    log_interval: int = 50

    # Whether to emit detailed per-step diagnostics.
    # NOT a biological quantity.
    verbose: bool = True

    # ---- Data ----

    # If None, the trainer generates random coordinate vectors as a
    # synthetic training signal. Supply a Dataset of (coordinate_dim,)
    # tensors for real training.
    # NOT a biological quantity.
    dataset: Optional[Dataset] = field(default=None, repr=False)

    # Random seed for reproducibility.
    # NOT a biological quantity.
    seed: int = 42

    # Device string. Auto-detects CUDA if set to 'auto'.
    # NOT a biological quantity.
    device: str = "auto"


# ===========================================================================
# Synthetic Dataset
# ===========================================================================

class SyntheticCoordinateDataset(Dataset):
    """
    Synthetic dataset of coordinate vectors for trainer smoke-testing.

    Generates structured random coordinate distributions to exercise the
    full hippocampal circuit without requiring a real upstream encoder.
    The coordinates are sampled from a mixture of Gaussians to create
    the kind of clustered structure that would emerge from a real
    perforant path projection.

    NOT a biological quantity. Engineering scaffold for standalone testing.

    Args:
        coordinate_dim: Dimensionality of each coordinate vector.
        n_samples: Total number of samples in the dataset.
        n_clusters: Number of Gaussian cluster centers.
        cluster_spread: Standard deviation within each cluster.
        seed: Random seed.
    """

    def __init__(
        self,
        coordinate_dim: int = 64,
        n_samples: int = 10_000,
        n_clusters: int = 32,
        cluster_spread: float = 0.5,
        seed: int = 42,
    ) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)

        # Generate cluster centers.
        centers = torch.randn(n_clusters, coordinate_dim, generator=gen)
        centers = F.normalize(centers, dim=-1)

        # Sample data points from cluster mixture.
        cluster_idx = torch.randint(0, n_clusters, (n_samples,), generator=gen)
        coords = centers[cluster_idx]
        coords = coords + torch.randn(n_samples, coordinate_dim, generator=gen) * cluster_spread

        self._coords = coords

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self._coords)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return a single coordinate vector."""
        return self._coords[idx]


# ===========================================================================
# Retrieval Evaluator
# ===========================================================================

class RetrievalEvaluator:
    """
    Evaluates pattern completion quality under noisy cue conditions.

    BIOLOGICAL FUNCTION: In the rodent hippocampus, pattern completion
    is assessed via partial-cue retrieval paradigms. A familiar context
    is presented with one or more features occluded or noisy, and the
    animal's navigation behavior indicates whether the full memory was
    retrieved. This class implements the computational analog: store a
    reference coordinate, present a noisy version, and measure cosine
    similarity between the retrieved reconstruction and the original.

    Rolls ET (2013). DOI: 10.3389/fnsys.2013.00074

    Args:
        noise_sigma: Standard deviation of additive Gaussian noise.
        n_pairs: Number of reference-noisy pairs to test.
    """

    def __init__(self, noise_sigma: float = 0.3, n_pairs: int = 8) -> None:
        """
        Args:
            noise_sigma: Gaussian noise sigma for cue degradation.
            n_pairs: Number of test pairs per evaluation call.
        """
        self.noise_sigma = noise_sigma
        self.n_pairs = n_pairs

    @torch.no_grad()
    def evaluate(
        self,
        kernel: CognitiveKernel,
        reference_coords: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, float]:
        """
        Run pattern completion evaluation.

        Stores each reference coordinate into the kernel, then presents
        a noisy version and measures reconstruction fidelity.

        Args:
            kernel: CognitiveKernel to evaluate.
            reference_coords: (n_pairs, coordinate_dim) clean references.
            device: Torch device.

        Returns:
            Dict with mean cosine similarity and mean MSE under noise.
        """
        kernel.eval()
        refs = reference_coords[:self.n_pairs].to(device)

        # Store all references.
        for i in range(len(refs)):
            kernel(refs[i:i+1], store_if_novel=True)

        # Retrieve from noisy cues.
        noisy = refs + torch.randn_like(refs) * self.noise_sigma
        retrieved = kernel.retrieve_from_cue(noisy)

        cos_sims = F.cosine_similarity(retrieved, refs, dim=-1)
        mse_vals = F.mse_loss(retrieved, refs, reduction="none").mean(dim=-1)

        return {
            "retrieval_cos_sim_mean": cos_sims.mean().item(),
            "retrieval_cos_sim_min": cos_sims.min().item(),
            "retrieval_mse_mean": mse_vals.mean().item(),
        }


# ===========================================================================
# Checkpoint Manager
# ===========================================================================

class CheckpointManager:
    """
    Manages three-tier checkpoint serialization for the Cognitive Kernel.

    HOT tier: soul checkpoint. Saved at every step. Contains only the
        transient state (EC buffer, working memory) needed to resume
        immediately if the process is killed mid-step.
    WARM tier: full model checkpoint. Saved at checkpoint_interval.
        Contains model weights, optimizer state, and training metadata.
    COLD tier: memory state checkpoint. Saved at sleep boundaries and
        on explicit request. Contains CA3 patterns, recurrent weights,
        and astrocyte state.

    NOT a biological quantity. Engineering serialization layer.

    Args:
        checkpoint_dir: Root directory for all checkpoint files.
    """

    def __init__(self, checkpoint_dir: str) -> None:
        """
        Args:
            checkpoint_dir: Path to checkpoint directory.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (self.checkpoint_dir / "hot").mkdir(exist_ok=True)
        (self.checkpoint_dir / "warm").mkdir(exist_ok=True)
        (self.checkpoint_dir / "cold").mkdir(exist_ok=True)

    def save_hot(self, kernel: CognitiveKernel, step: int) -> None:
        """
        Save the HOT (soul) checkpoint.

        Contains only transient memory state: EC buffer and working memory.
        Overwrites the previous HOT checkpoint to keep only the latest.

        Args:
            kernel: CognitiveKernel instance.
            step: Current training step.
        """
        path = self.checkpoint_dir / "hot" / "soul.pt"
        state = kernel.get_memory_state()
        torch.save(
            {
                "step": step,
                "short_term_buffer": state["short_term_buffer"],
                "working_memory": state["working_memory"],
                "astro_state": state["astro_state"],
            },
            path,
        )

    def save_warm(
        self,
        kernel: CognitiveKernel,
        optimizer: torch.optim.Optimizer,
        step: int,
        metrics: Dict[str, float],
    ) -> None:
        """
        Save the WARM (model weights + optimizer) checkpoint.

        Args:
            kernel: CognitiveKernel instance.
            optimizer: Optimizer instance.
            step: Current training step.
            metrics: Most recent training metrics to include in manifest.
        """
        path = self.checkpoint_dir / "warm" / f"step_{step:08d}.pt"
        torch.save(
            {
                "step": step,
                "model_state_dict": kernel.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
            },
            path,
        )
        logger.info("Saved WARM checkpoint: %s", path)

    def save_cold(self, kernel: CognitiveKernel, step: int) -> None:
        """
        Save the COLD (long-term memory) checkpoint.

        Contains CA3 stored patterns, recurrent weights, and astrocyte
        state. These survive sleep consolidation; all other memory tiers
        do not.

        Args:
            kernel: CognitiveKernel instance.
            step: Current training step.
        """
        path = self.checkpoint_dir / "cold" / f"memory_{step:08d}.pt"
        memory_state = kernel.get_memory_state()
        torch.save(
            {
                "step": step,
                "ca3_stored_patterns": memory_state["ca3_stored_patterns"],
                "ca3_W_recurrent": memory_state["ca3_W_recurrent"],
                "astro_state": memory_state["astro_state"],
                "n_patterns": len(memory_state["ca3_stored_patterns"]),
            },
            path,
        )
        logger.info(
            "Saved COLD checkpoint: %s (%d patterns)",
            path,
            len(memory_state["ca3_stored_patterns"]),
        )

    def load_latest_warm(
        self,
        kernel: CognitiveKernel,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> int:
        """
        Load the most recent WARM checkpoint.

        Args:
            kernel: CognitiveKernel instance to load into.
            optimizer: Optimizer to restore. If None, optimizer state
                is not restored.

        Returns:
            Step number of the loaded checkpoint, or 0 if none found.
        """
        warm_dir = self.checkpoint_dir / "warm"
        checkpoints = sorted(warm_dir.glob("step_*.pt"))
        if not checkpoints:
            logger.info("No WARM checkpoint found. Starting from scratch.")
            return 0

        path = checkpoints[-1]
        data = torch.load(path, weights_only=False)
        kernel.load_state_dict(data["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(data["optimizer_state_dict"])
        step = data["step"]
        logger.info("Loaded WARM checkpoint: %s (step %d)", path, step)
        return step


# ===========================================================================
# Cognitive Kernel Trainer
# ===========================================================================

class CognitiveKernelTrainer:
    """
    Full training harness for the Cognitive Kernel.

    TRAINING OBJECTIVE
    ------------------
    The primary loss is reconstruction MSE: the subiculum output (returned
    as `reconstructed` from the kernel forward pass) is compared against
    the EC-normalized coordinates (ec_output). The loss teaches the learned
    projections to produce reconstructions that faithfully reflect what the
    entorhinal cortex presented to the hippocampus.

    The loss is NOT computed against raw input coordinates because the kernel
    internally applies EC LayerNorm before any downstream processing. Using
    raw coordinates as the target would introduce a systematic scale mismatch
    between loss gradient and internal representations.

    ASTROCYTIC LEARNING RATE MODULATION
    ------------------------------------
    The astrocytic eta returned by each forward pass is applied as a
    multiplicative scaling to the base learning rate at every step. This
    implements the BCM sliding threshold: high population activity amplifies
    plasticity (eta > 1.0), low activity stabilizes it (eta < 1.0).

    OPTIMIZER EXCLUSIONS
    --------------------
    W_recurrent and distance_mask are registered buffers, not parameters.
    They are excluded from the optimizer automatically because PyTorch's
    optimizer only sees `model.parameters()`. The trainer does not need
    to manually exclude them, but this is documented here for clarity.

    SLEEP CONSOLIDATION
    -------------------
    At each sleep_interval step, the trainer calls kernel.sleep_consolidation(),
    saves a COLD checkpoint before clearing, and resets the EC buffer and
    working memory. CA3 patterns are preserved.

    Args:
        cfg: TrainerConfig.
        kernel_cfg: CognitiveKernelConfig (uses defaults if None).
    """

    def __init__(
        self,
        cfg: TrainerConfig,
        kernel_cfg: Optional[CognitiveKernelConfig] = None,
    ) -> None:
        """
        Args:
            cfg: TrainerConfig.
            kernel_cfg: CognitiveKernelConfig. Uses kernel defaults if None.
        """
        self.cfg = cfg
        self.kernel_cfg = kernel_cfg or CognitiveKernelConfig()

        # Device selection.
        if cfg.device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(cfg.device)

        logger.info("Training on device: %s", self._device)

        # Reproducibility.
        torch.manual_seed(cfg.seed)

        # Build kernel.
        self.kernel = CognitiveKernel(self.kernel_cfg).to(self._device)

        # Optimizer. W_recurrent and distance_mask are buffers, not
        # parameters, so they are automatically excluded here.
        self.optimizer = torch.optim.Adam(
            self.kernel.parameters(),
            lr=cfg.lr,
            eps=cfg.adam_eps,
            weight_decay=cfg.weight_decay,
        )

        # Checkpoint manager.
        self.checkpoint_manager = CheckpointManager(cfg.checkpoint_dir)

        # Retrieval evaluator.
        self.evaluator = RetrievalEvaluator(
            noise_sigma=cfg.retrieval_noise_sigma,
            n_pairs=cfg.n_retrieval_test_pairs,
        )

        # Dataset and dataloader.
        dataset = cfg.dataset
        if dataset is None:
            dataset = SyntheticCoordinateDataset(
                coordinate_dim=self.kernel_cfg.coordinate_dim,
                seed=cfg.seed,
            )
            logger.info(
                "No dataset provided. Using synthetic coordinate dataset "
                "(%d samples).",
                len(dataset),  # type: ignore[arg-type]
            )

        self.dataloader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=self._device.type == "cuda",
        )

        # Training state.
        self._step: int = 0
        self._history: List[Dict[str, float]] = []

    def _compute_loss(
        self,
        coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Run one forward pass and compute the reconstruction loss.

        The loss is MSE between the subiculum reconstruction and the
        EC-normalized coordinates. To recover ec_output for the loss
        target, we run the EC forward separately before the full kernel
        forward. Both calls use the same coords tensor.

        See module docstring for rationale on targeting ec_output rather
        than raw coords.

        Args:
            coords: (B, coordinate_dim) input coordinates.

        Returns:
            loss: Scalar reconstruction loss.
            reconstructed: (B, coordinate_dim) subiculum output.
            novelty: (B,) novelty scalars.
            diagnostics: Kernel diagnostic dict.
        """
        # Compute EC-normalized target WITHOUT updating the EC buffer,
        # so we can use it as a loss target that matches what the kernel
        # internally sees. We run a no-grad EC pass to get the normalized
        # target, then run the full kernel pass.
        with torch.no_grad():
            # Replicate EC normalization: bias + LayerNorm.
            # We read the EC short_term_buffer directly.
            buffer = self.kernel.entorhinal_cortex.short_term_buffer
            biased = coords + 0.1 * buffer.unsqueeze(0)
            ec_target = self.kernel.entorhinal_cortex.input_norm(biased)

        # store_if_novel=False during the gradient-tracked forward pass.
        # Episode storage (which calls _recompute_weights, an in-place
        # buffer write) must not occur inside the autograd graph.
        reconstructed, novelty, diagnostics = self.kernel(
            coords, store_if_novel=False,
        )

        loss = F.mse_loss(reconstructed, ec_target)
        return loss, reconstructed, novelty, diagnostics

    def _apply_astrocytic_lr(self, eta: float) -> None:
        """
        Modulate optimizer learning rate by the astrocytic eta signal.

        This implements the BCM-style sliding plasticity threshold.
        The base LR from TrainerConfig is scaled by eta at every step.
        The effective LR is clamped within reasonable bounds to prevent
        complete shutdown or explosion of learning.

        NOT a biological quantity in its exact implementation. The BCM
        threshold operates on individual synapses in continuous time;
        this approximates it as a global LR multiplier.

        Bienenstock EL et al. (1982). DOI: 10.1523/JNEUROSCI.02-01-00032.1982

        Args:
            eta: Astrocytic eta from the kernel diagnostic dict.
        """
        effective_lr = self.cfg.lr * eta
        # Clamp to prevent degenerate extremes.
        effective_lr = max(self.cfg.lr * 0.01, min(self.cfg.lr * 10.0, effective_lr))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = effective_lr

    def _train_step(self, coords: torch.Tensor) -> Dict[str, float]:
        """
        Execute one training step.

        Args:
            coords: (B, coordinate_dim) batch of coordinate vectors.

        Returns:
            Step metrics dict.
        """
        self.kernel.train()
        coords = coords.to(self._device)

        self.optimizer.zero_grad()

        loss, reconstructed, novelty, diagnostics = self._compute_loss(coords)

        loss.backward()

        # Episode storage must happen AFTER backward() because store_episode()
        # calls _recompute_weights() which modifies W_recurrent in-place.
        # Performing that in-place write before or during backward() corrupts
        # the autograd graph. W_recurrent is a buffer (not a parameter) so it
        # does not accumulate gradients, but PyTorch still tracks in-place
        # modifications to all tensors in the graph.
        # We re-run a no-grad forward pass solely to trigger novel episode
        # storage with the updated weights already stepped.
        # This means storage happens one step behind the gradient update,
        # which is an acceptable training artifact (NOT a biological quantity).

        # Gradient clipping. NOT a biological quantity.
        grad_norm = nn.utils.clip_grad_norm_(
            self.kernel.parameters(), self.cfg.grad_clip_norm,
        )

        # Apply astrocytic LR modulation before stepping.
        eta = diagnostics.get("astro_eta", 1.0)
        self._apply_astrocytic_lr(eta)

        self.optimizer.step()

        # Post-step episode storage: re-run in no_grad to trigger novel
        # episode encoding now that W_recurrent is safe to modify in-place.
        with torch.no_grad():
            _, _, storage_diag = self.kernel(coords, store_if_novel=True)
            diagnostics["num_stored_episodes"] = storage_diag.get(
                "num_stored_episodes", diagnostics.get("num_stored_episodes", 0),
            )
            diagnostics["stored_this_step"] = storage_diag.get("stored_this_step", False)

        metrics = {
            "loss": loss.item(),
            "novelty_mean": novelty.mean().item(),
            "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm),
            "effective_lr": self.optimizer.param_groups[0]["lr"],
        }
        metrics.update(diagnostics)
        return metrics

    @torch.no_grad()
    def _eval_pass(self, eval_coords: torch.Tensor) -> Dict[str, float]:
        """
        Run full evaluation: reconstruction loss and retrieval quality.

        Args:
            eval_coords: (N, coordinate_dim) evaluation coordinates.

        Returns:
            Evaluation metrics dict.
        """
        self.kernel.eval()
        eval_coords = eval_coords.to(self._device)

        # Accumulate reconstruction loss across eval batches.
        total_loss = 0.0
        total_novelty = 0.0
        n_batches = 0

        bs = self.cfg.batch_size
        for start in range(0, min(len(eval_coords), self.cfg.eval_batches * bs), bs):
            batch = eval_coords[start:start + bs]
            if len(batch) == 0:
                break

            buffer = self.kernel.entorhinal_cortex.short_term_buffer
            biased = batch + 0.1 * buffer.unsqueeze(0)
            ec_target = self.kernel.entorhinal_cortex.input_norm(biased)

            reconstructed, novelty, _ = self.kernel(batch, store_if_novel=False)
            total_loss += F.mse_loss(reconstructed, ec_target).item()
            total_novelty += novelty.mean().item()
            n_batches += 1

        n_batches = max(n_batches, 1)

        # Retrieval evaluation.
        ref_size = min(self.cfg.n_retrieval_test_pairs, len(eval_coords))
        retrieval_metrics = self.evaluator.evaluate(
            self.kernel,
            eval_coords[:ref_size],
            self._device,
        )

        eval_metrics = {
            "eval_loss": total_loss / n_batches,
            "eval_novelty_mean": total_novelty / n_batches,
        }
        eval_metrics.update(retrieval_metrics)
        return eval_metrics

    def _data_iterator(self):
        """
        Infinite iterator over the dataloader.

        Resets the dataloader at epoch boundaries, enabling training for
        an arbitrary number of steps independent of dataset size.

        Yields:
            Batches of coordinate tensors.
        """
        while True:
            for batch in self.dataloader:
                yield batch

    def train(self) -> List[Dict[str, float]]:
        """
        Run the full training loop.

        Returns:
            History list of per-step metric dicts.
        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
        )
        logger.info("=== Cognitive Kernel Training ===")
        logger.info(self.kernel.count_params())
        logger.info(
            "Steps: %d | Batch: %d | LR: %.2e | Device: %s",
            self.cfg.n_steps,
            self.cfg.batch_size,
            self.cfg.lr,
            self._device,
        )

        # Attempt to resume from latest WARM checkpoint.
        start_step = self.checkpoint_manager.load_latest_warm(
            self.kernel, self.optimizer,
        )
        self._step = start_step

        # Build a fixed eval set from a new synthetic dataset so we have
        # a clean reference independent of the training dataloader shuffle.
        eval_dataset = SyntheticCoordinateDataset(
            coordinate_dim=self.kernel_cfg.coordinate_dim,
            n_samples=self.cfg.eval_batches * self.cfg.batch_size + self.cfg.n_retrieval_test_pairs,
            seed=self.cfg.seed + 1,
        )
        eval_coords = torch.stack([eval_dataset[i] for i in range(len(eval_dataset))])

        data_iter = self._data_iterator()
        t_start = time.time()

        for _ in range(self.cfg.n_steps - start_step):
            self._step += 1
            batch = next(data_iter)

            # Handle datasets that return (coords,) tuples vs bare tensors.
            if isinstance(batch, (list, tuple)):
                coords = batch[0]
            else:
                coords = batch

            # ---- Train step ----
            metrics = self._train_step(coords)
            metrics["step"] = self._step
            self._history.append(metrics)

            # ---- HOT checkpoint ----
            self.checkpoint_manager.save_hot(self.kernel, self._step)

            # ---- Logging ----
            if self._step % self.cfg.log_interval == 0:
                elapsed = time.time() - t_start
                steps_per_sec = (self._step - start_step) / max(elapsed, 1e-6)
                logger.info(
                    "Step %6d | loss=%.4f | novelty=%.3f | eta=%.3f | "
                    "lr=%.2e | episodes=%d | %.1f steps/s",
                    self._step,
                    metrics["loss"],
                    metrics.get("novelty_mean", 0.0),
                    metrics.get("astro_eta", 1.0),
                    metrics["effective_lr"],
                    int(metrics.get("num_stored_episodes", 0)),
                    steps_per_sec,
                )

            # ---- Evaluation ----
            if self._step % self.cfg.eval_interval == 0:
                eval_metrics = self._eval_pass(eval_coords)
                logger.info(
                    "  [EVAL] step=%d | eval_loss=%.4f | retrieval_cos=%.4f | "
                    "retrieval_mse=%.4f",
                    self._step,
                    eval_metrics["eval_loss"],
                    eval_metrics.get("retrieval_cos_sim_mean", 0.0),
                    eval_metrics.get("retrieval_mse_mean", 0.0),
                )
                self._history[-1].update(eval_metrics)

            # ---- Sleep consolidation ----
            if self._step % self.cfg.sleep_interval == 0:
                logger.info("  [SLEEP] Running sleep consolidation at step %d...", self._step)
                # Save COLD checkpoint before clearing.
                self.checkpoint_manager.save_cold(self.kernel, self._step)
                sleep_diag = self.kernel.sleep_consolidation()
                logger.info(
                    "  [SLEEP] Done. %d CA3 episodes preserved. "
                    "Short-term and working memory cleared.",
                    int(sleep_diag.get("episodes_in_ca3", 0)),
                )

            # ---- WARM checkpoint ----
            if self._step % self.cfg.checkpoint_interval == 0:
                self.checkpoint_manager.save_warm(
                    self.kernel, self.optimizer, self._step, metrics,
                )

        # Final checkpoint.
        self.checkpoint_manager.save_warm(
            self.kernel, self.optimizer, self._step, self._history[-1] if self._history else {},
        )
        self.checkpoint_manager.save_cold(self.kernel, self._step)

        logger.info("Training complete. %d steps total.", self._step)
        return self._history


# ===========================================================================
# Entry Point
# ===========================================================================

def main() -> None:
    """
    Run training with default configuration on synthetic data.

    For production use, replace cfg.dataset with a real Dataset that
    yields (coordinate_dim,) tensors from the Perforant Path encoder.
    """
    kernel_cfg = CognitiveKernelConfig(
        coordinate_dim=64,
        dentate_gyrus_dim=512,
        ca3_dim=512,
        ca1_dim=256,
        subiculum_dim=128,
        dg_sparsity=0.04,
        ca3_max_episodes=64,
    )

    trainer_cfg = TrainerConfig(
        lr=3e-4,
        batch_size=32,
        n_steps=2_000,
        eval_interval=200,
        sleep_interval=500,
        checkpoint_interval=500,
        log_interval=50,
        checkpoint_dir="checkpoints",
        verbose=True,
        seed=42,
    )

    trainer = CognitiveKernelTrainer(cfg=trainer_cfg, kernel_cfg=kernel_cfg)
    history = trainer.train()

    # Print final summary.
    if history:
        last = history[-1]
        print(f"\nFinal training loss:       {last.get('loss', float('nan')):.4f}")
        print(f"Final eval loss:           {last.get('eval_loss', float('nan')):.4f}")
        print(f"Final retrieval cos sim:   {last.get('retrieval_cos_sim_mean', float('nan')):.4f}")
        print(f"CA3 episodes stored:       {last.get('num_stored_episodes', 0)}")
        print(f"Astrocytic eta (final):    {last.get('astro_eta', float('nan')):.4f}")


if __name__ == "__main__":
    main()
