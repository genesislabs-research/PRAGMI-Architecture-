from __future__ import annotations
from typing import TYPE_CHECKING, Tuple
import torch
if TYPE_CHECKING:
    from world_model_ensemble_p import WorldModelEnsemble

MATURITY_RANDOM: float = 0.3
MATURITY_FULL: float = 0.6


def select_batch(candidate_coords: list[torch.Tensor], world_model: 'WorldModelEnsemble', global_maturity: float, rng: torch.Generator | None = None) -> Tuple[int, torch.Tensor]:
    if not candidate_coords:
        raise ValueError('candidate_coords must contain at least one batch')
    n = len(candidate_coords)
    alpha = float(torch.clamp(torch.tensor((global_maturity - MATURITY_RANDOM) / (MATURITY_FULL - MATURITY_RANDOM)), 0.0, 1.0))
    use_variance = torch.rand(1, generator=rng).item() < alpha
    if not use_variance or n == 1:
        selected_idx = int(torch.randint(n, (1,), generator=rng).item())
        return selected_idx, candidate_coords[selected_idx]
    variances = [world_model.evaluate_batch(coords)[1] for coords in candidate_coords]
    selected_idx = int(torch.tensor(variances).argmax().item())
    return selected_idx, candidate_coords[selected_idx]


def select_batch_deterministic(candidate_coords: list[torch.Tensor], world_model: 'WorldModelEnsemble', global_maturity: float) -> Tuple[int, torch.Tensor, list[float]]:
    if not candidate_coords:
        raise ValueError('candidate_coords must contain at least one batch')
    variances = [world_model.evaluate_batch(coords)[1] for coords in candidate_coords]
    selected_idx = int(torch.tensor(variances).argmax().item())
    return selected_idx, candidate_coords[selected_idx], variances
