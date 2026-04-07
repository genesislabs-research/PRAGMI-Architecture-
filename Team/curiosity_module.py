import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class CuriosityConfig:
    input_dim: int = 64  # Coordinate manifold dimension from the Bridge
    hidden_dim: int = 128
    epistemic_scale: float = 1.0
    pragmatic_scale: float = 0.3
    learning_rate: float = 1e-4

class CuriosityHead(nn.Module):
    """
    Active Inference module for Timmy V2.
    Evaluates candidate data batches by minimizing expected free energy.
    """
    def __init__(self, cfg: CuriosityConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device # Pinned to the primary device (P40)

        # Generative model of prediction error (The 'World Model' proxy)
        self.net = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, 1) # Outputs predicted surprise
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.learning_rate)

    def forward(self, manifold_coords: torch.Tensor) -> torch.Tensor:
        """Predicts the surprise (loss) for a given set of representations."""
        return self.net(manifold_coords.to(self.device))

    def update_world_model(self, manifold_coords: torch.Tensor, actual_loss: torch.Tensor):
        """
        Trains the curiosity head to predict actual network error.
        This DETACHES the actual loss to ensure curiosity doesn't 
        interfere with the Prime column's gradients.
        """
        self.optimizer.zero_grad()
        
        # We move coordinates to the primary device for calculation
        predicted_surprise = self.forward(manifold_coords)
        
        # Target is the detached actual loss (ground truth surprise)
        target = actual_loss.detach().to(self.device).view(-1, 1)
        
        loss = F.mse_loss(predicted_surprise, target)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def calculate_expected_free_energy(self, manifold_coords: torch.Tensor, task_reward: float = 0.0) -> torch.Tensor:
        """
        G(π) = Epistemic Value (Information Gain) + Pragmatic Value (Task Goal).
        """
        # 1. Epistemic: Predicted surprise (we seek what we don't know)
        predicted_surprise = self.forward(manifold_coords)
        
        # 2. Pragmatic: Proximity to preferred outcomes (reward proxy)
        # Higher reward lowers the expected free energy
        pragmatic_utility = torch.tensor([task_reward], device=self.device)
        
        # Final objective: minimize G
        efe = (self.cfg.epistemic_scale * predicted_surprise) - \
              (self.cfg.pragmatic_scale * pragmatic_utility)
              
        return efe

    @torch.no_grad()
    def update_neuromodulators(self, manifold_coords: torch.Tensor, broadcast: Any):
        """
        Links surprise to Norepinephrine (NE) levels.
        """
        predicted_surprise = self.forward(manifold_coords).mean().item()
        # High surprise spikes NE, lowering expansion thresholds
        broadcast.update_norepinephrine(predicted_surprise)
