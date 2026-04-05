"""
epistemic_selector_teaching.py

BIOLOGICAL GROUNDING
====================
This file models the active foraging behavior of biological organisms: the
tendency to seek out experiences that are informative rather than passively
accepting whatever data arrives. In active inference theory, agents select
actions that are expected to resolve uncertainty, maximizing information gain
rather than simply minimizing prediction error.

The brain does not passively receive stimuli. Organisms actively orient their
sensory organs toward novel or surprising events (orienting response), explore
unfamiliar environments (epistemic foraging), and seek out situations where
prediction errors are informative rather than merely noisy. This is driven by
the same neuromodulatory systems modeled in NeuromodulatorBroadcast: NE from
locus coeruleus signals unexpected uncertainty and drives exploratory behavior,
while DA encodes the expected value of exploration versus exploitation.

The EpistemicSelector implements a simplified version of this active foraging
principle for batch selection during training. Rather than iterating through
data in fixed order, the system evaluates candidate batches using the
WorldModelEnsemble and selects the batch that maximizes expected information
gain (ensemble variance). A maturity-dependent damping factor prevents the
selector from acting on unreliable world model predictions during early
training: below maturity 0.3 selection is random, between 0.3 and 0.6 it
transitions linearly to variance-driven selection, above 0.6 it is fully
variance-driven.

This module is a stateless method-provider: it has no learnable parameters
and no persistent state. It is separated from WorldModelEnsemble because
batch selection logic is an ActiveDataLoader concern, not a model concern.

Primary papers grounding this file:

Friston K, et al. (2017). "Active inference: a process theory." Neural
Computation, 29(1), 1-49. DOI: 10.1162/NECO_a_00912

Friston K (2010). "The free-energy principle: a unified brain theory."
Nature Reviews Neuroscience, 11(2), 127-138. DOI: 10.1038/nrn2787

Aston-Jones G, Cohen JD (2005). "An integrative theory of locus
coeruleus-norepinephrine function: adaptive gain and optimal performance."
Annual Review of Neuroscience, 28, 403-450.
DOI: 10.1146/annurev.neuro.28.061604.135709
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

if TYPE_CHECKING:
    from world_model_ensemble_teaching import WorldModelEnsemble

# Maturity thresholds for selection damping.
# Below MATURITY_RANDOM: selection is fully random (world model unreliable).
# Above MATURITY_FULL: selection is fully variance-driven.
# Between: linear interpolation.
# NOT biological quantities: engineering hyperparameters.
MATURITY_RANDOM: float = 0.3
MATURITY_FULL: float = 0.6


def select_batch(
    candidate_coords: list[torch.Tensor],
    world_model: "WorldModelEnsemble",
    global_maturity: float,
    rng: torch.Generator | None = None,
) -> Tuple[int, torch.Tensor]:
    """
    Select the highest-information-gain batch from a list of candidates.

    BIOLOGICAL STRUCTURE: Active foraging / epistemic action selection.
    BIOLOGICAL FUNCTION: Biological agents orient toward novel or uncertain
    stimuli rather than passively processing whatever arrives. High ensemble
    variance in the WorldModelEnsemble signals that the current region of the
    coordinate manifold is undersampled: training on this batch will produce
    the largest reduction in world model uncertainty (information gain).
    NE from locus coeruleus drives this orienting behavior by signaling
    unexpected uncertainty and lowering perceptual thresholds.

    Aston-Jones G, Cohen JD (2005). DOI: 10.1146/annurev.neuro.28.061604.135709
    Friston K, et al. (2017). DOI: 10.1162/NECO_a_00912

    MATURITY DAMPING:
    The world model's variance estimates are unreliable during early training
    because the ensemble has not yet learned to predict coordinate-space
    representations accurately. Selecting greedily on unreliable variance
    wastes compute and may introduce sampling bias.

    Three regions:
    - global_maturity < 0.3: fully random selection. World model too young
      to provide reliable epistemic signal.
    - 0.3 <= global_maturity <= 0.6: linear interpolation between random
      and variance-driven. World model partially reliable.
    - global_maturity > 0.6: fully variance-driven selection.

    The interpolation weight alpha = clip((maturity - 0.3) / 0.3, 0, 1)
    gives the probability of selecting the highest-variance batch. With
    probability (1 - alpha) a random batch is selected instead.
    NOT a biological quantity: the threshold values 0.3 and 0.6 are
    engineering hyperparameters matching the maturity gate from the
    width expansion and CuriosityHead damping specifications.

    ANATOMICAL INTERFACE:
        Sending structure: WorldModelEnsemble (evaluate_batch).
        Receiving structure: EpistemicSelector (this function).
        Connection: No direct biological analog. Engineering interface:
            WorldModelEnsemble is queried under torch.no_grad() to score
            candidate batches without updating the model.

    Args:
        candidate_coords: List of N candidate coordinate tensors, each of
            shape (B, coordinate_dim). Must contain at least one entry.
        world_model: WorldModelEnsemble instance. Queried read-only via
            evaluate_batch() under torch.no_grad(). Must not be in training
            mode during selection (the caller is responsible for this if
            batch norm or dropout is present, though neither is used in
            WorldModelEnsemble).
        global_maturity: Current maturity scalar from NeuromodulatorBroadcast,
            in [0, 1]. Controls the interpolation between random and
            variance-driven selection.
        rng: Optional torch.Generator for reproducible random selection.
            If None, uses the default generator.

    Returns:
        selected_idx: Integer index into candidate_coords of the selected batch.
        selected_batch: The selected coordinate tensor, shape (B, coordinate_dim).

    Raises:
        ValueError: If candidate_coords is empty.
    """
    if not candidate_coords:
        raise ValueError("candidate_coords must contain at least one batch")

    n = len(candidate_coords)

    # Compute alpha: the probability of variance-driven selection.
    # Alpha is 0 below MATURITY_RANDOM, 1 above MATURITY_FULL, linear between.
    # NOT a biological quantity: engineering interpolation.
    alpha = float(
        torch.clamp(
            torch.tensor((global_maturity - MATURITY_RANDOM) / (MATURITY_FULL - MATURITY_RANDOM)),
            0.0, 1.0,
        )
    )

    # Decide whether to use variance-driven or random selection.
    # With probability alpha: variance-driven.
    # With probability (1 - alpha): random.
    use_variance = torch.rand(1, generator=rng).item() < alpha

    if not use_variance or n == 1:
        # Random selection. Uniform over candidates.
        selected_idx = int(torch.randint(n, (1,), generator=rng).item())
        return selected_idx, candidate_coords[selected_idx]

    # Variance-driven selection: evaluate all candidates and pick highest variance.
    # WorldModelEnsemble.evaluate_batch() runs under torch.no_grad() internally.
    variances = [
        world_model.evaluate_batch(coords)[1]
        for coords in candidate_coords
    ]
    selected_idx = int(torch.tensor(variances).argmax().item())
    return selected_idx, candidate_coords[selected_idx]


def select_batch_deterministic(
    candidate_coords: list[torch.Tensor],
    world_model: "WorldModelEnsemble",
    global_maturity: float,
) -> Tuple[int, torch.Tensor, list[float]]:
    """
    Deterministic variant for testing and diagnostics. Always scores all
    candidates and returns the variance scores alongside the selection.

    Unlike select_batch, this function:
    - Always scores all candidates regardless of maturity (no random fallback).
    - Returns the full variance score list for inspection.
    - Does not perform the probabilistic alpha-weighted random fallback.

    This is a diagnostic tool, not the live selection path. Use select_batch
    in production.

    NOT a biological quantity: diagnostic engineering utility.

    Args:
        candidate_coords: List of candidate coordinate tensors.
        world_model: WorldModelEnsemble instance (read-only).
        global_maturity: Maturity scalar (used only for logging, not for
            selection in this variant).

    Returns:
        selected_idx: Index of highest-variance candidate.
        selected_batch: The selected coordinate tensor.
        variances: List of variance scores for all candidates.
    """
    if not candidate_coords:
        raise ValueError("candidate_coords must contain at least one batch")

    variances = [
        world_model.evaluate_batch(coords)[1]
        for coords in candidate_coords
    ]
    selected_idx = int(torch.tensor(variances).argmax().item())
    return selected_idx, candidate_coords[selected_idx], variances
