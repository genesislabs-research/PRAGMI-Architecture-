# filename: walking_config.py
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class WalkingKinematicsConfig:
    proprioception_dim: int = 24
    vision_dim: int = 0
    touch_dim: int = 0
    action_dim: int = 10
    action_low: List[float] = None
    action_high: List[float] = None
    forward_velocity_weight: float = 1.0
    energy_penalty_weight: float = 0.1
    smoothness_weight: float = 0.05
    fall_penalty: float = -10.0
    prediction_horizon: int = 1
    max_walking_episodes: int = 128
    novelty_threshold: float = 0.3
    crystallization_error_threshold: float = 0.02
    crystallization_confidence_window: int = 100
