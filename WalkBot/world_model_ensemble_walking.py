# filename: world_model_ensemble_walking.py
from dataclasses import dataclass
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class WorldModelConfig:
    """Configuration for the ensemble world model."""
    coordinate_dim: int = 64
    action_dim: int = 10
    hidden_dim: int = 128
    n_heads: int = 5
    learning_rate: float = 1e-4
    error_window: int = 50
    variance_ceiling: float = 0.1          # for clipping uncertainty


class PredictionHead(nn.Module):
    """Single deterministic predictor in the ensemble."""
    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        input_dim = cfg.coordinate_dim + cfg.action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.coordinate_dim),
        )

    def forward(self, state_action: torch.Tensor) -> torch.Tensor:
        return self.net(state_action)


class WorldModelEnsemble(nn.Module):
    """
    Ensemble of world models for robust next-state prediction + uncertainty estimation.
    Used by the hippocampal core as source of prediction error (novelty).
    """
    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        self.cfg = cfg
        self.heads = nn.ModuleList([PredictionHead(cfg) for _ in range(cfg.n_heads)])

        self.optimizer = torch.optim.Adam(self.parameters(), lr=cfg.learning_rate)

        # Rolling buffers for diagnostics
        self._recent_errors: list[float] = []
        self._recent_variances: list[float] = []

    def forward(self,
                state: torch.Tensor,
                action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            state:  (B, coordinate_dim) or (coordinate_dim,)
            action: (B, action_dim) or (action_dim,)
        Returns:
            predictions: (n_heads, B, coordinate_dim)
            mean_pred:   (B, coordinate_dim)
            ensemble_var:(B,) or scalar
        """
        # Ensure batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        sa = torch.cat([state, action], dim=-1)                    # (B, coord+action)
        predictions = torch.stack([head(sa) for head in self.heads], dim=0)  # (n_heads, B, coord)
        
        mean_pred = predictions.mean(dim=0)                        # (B, coord)
        ensemble_var = predictions.var(dim=0).mean(dim=-1)         # (B,)

        return predictions, mean_pred, ensemble_var

    def update(self,
               state: torch.Tensor,
               action: torch.Tensor,
               next_state: torch.Tensor) -> torch.Tensor:
        """
        Train the ensemble. Returns the loss tensor (for external logging/gradient control).
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        if next_state.dim() == 1:
            next_state = next_state.unsqueeze(0)

        target = next_state.detach()

        self.optimizer.zero_grad()
        predictions, _, _ = self.forward(state, action)            # (n_heads, B, coord)

        # Average MSE across ensemble members
        losses = [F.mse_loss(pred, target) for pred in predictions]
        loss = torch.stack(losses).mean()

        loss.backward()
        self.optimizer.step()

        # Store scalar for diagnostics
        error_val = loss.item()
        self._recent_errors.append(error_val)
        if len(self._recent_errors) > self.cfg.error_window:
            self._recent_errors.pop(0)

        return loss  # return tensor so training loop can decide whether to retain_graph etc.

    @torch.no_grad()
    def predict(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Convenience method used in the main training loop."""
        _, mean_pred, _ = self.forward(state, action)
        return mean_pred.squeeze(0) if mean_pred.shape[0] == 1 else mean_pred

    @torch.no_grad()
    def get_uncertainty(self, state: torch.Tensor, action: torch.Tensor) -> float:
        """Return current ensemble disagreement (used as alternative novelty signal)."""
        _, _, ensemble_var = self.forward(state, action)
        var_val = ensemble_var.mean().item()
        var_val = min(var_val, self.cfg.variance_ceiling)
        self._recent_variances.append(var_val)
        if len(self._recent_variances) > self.cfg.error_window:
            self._recent_variances.pop(0)
        return var_val

    def mean_recent_error(self) -> float:
        return sum(self._recent_errors) / len(self._recent_errors) if self._recent_errors else 1.0

    def mean_recent_variance(self) -> float:
        return sum(self._recent_variances) / len(self._recent_variances) if self._recent_variances else 1.0
