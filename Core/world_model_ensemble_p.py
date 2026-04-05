from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class WorldModelConfig:
    coordinate_dim: int = 64
    hidden_dim: int = 128
    n_heads: int = 5
    learning_rate: float = 1e-4
    variance_ceiling: float = 0.1
    error_window: int = 50


class PredictionHead(nn.Module):

    def __init__(self, cfg: WorldModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.coordinate_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.coordinate_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords)


class WorldModelEnsemble(nn.Module):

    def __init__(self, cfg: WorldModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.heads = nn.ModuleList([PredictionHead(cfg) for _ in range(cfg.n_heads)])
        self.optimizer = torch.optim.Adam(self.parameters(), lr=cfg.learning_rate)
        self._recent_errors: list[float] = []
        self._recent_variances: list[float] = []

    def forward(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        predictions = torch.stack([head(coords) for head in self.heads], dim=0)
        mean_prediction = predictions.mean(dim=0)
        ensemble_variance = predictions.var(dim=0).mean()
        return predictions, mean_prediction, ensemble_variance

    def update(self, coords: torch.Tensor, next_coords: torch.Tensor) -> float:
        target = next_coords.detach()
        coords_detached = coords.detach()
        self.optimizer.zero_grad()
        predictions, _, _ = self.forward(coords_detached)
        losses = torch.stack([F.mse_loss(predictions[i], target) for i in range(self.cfg.n_heads)])
        total_loss = losses.mean()
        total_loss.backward()
        self.optimizer.step()
        error_val = total_loss.item()
        self._recent_errors.append(error_val)
        if len(self._recent_errors) > self.cfg.error_window:
            self._recent_errors.pop(0)
        return error_val

    @torch.no_grad()
    def evaluate_batch(self, coords: torch.Tensor) -> Tuple[torch.Tensor, float]:
        _, mean_prediction, ensemble_variance = self.forward(coords)
        var_val = ensemble_variance.item()
        self._recent_variances.append(var_val)
        if len(self._recent_variances) > self.cfg.error_window:
            self._recent_variances.pop(0)
        return mean_prediction, var_val

    def mean_recent_error(self) -> float:
        if not self._recent_errors:
            return 1.0
        return float(sum(self._recent_errors) / len(self._recent_errors))

    def mean_recent_variance(self) -> float:
        if not self._recent_variances:
            return 1.0
        return float(sum(self._recent_variances) / len(self._recent_variances))

    def get_hot_state(self) -> dict:
        return {
            'optimizer_state': self.optimizer.state_dict(),
            'recent_errors': list(self._recent_errors),
            'recent_variances': list(self._recent_variances),
        }

    def load_hot_state(self, hot: dict) -> None:
        self.optimizer.load_state_dict(hot['optimizer_state'])
        self._recent_errors = list(hot.get('recent_errors', []))
        self._recent_variances = list(hot.get('recent_variances', []))
