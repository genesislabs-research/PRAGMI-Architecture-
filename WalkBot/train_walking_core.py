# filename: train_walking_core.py
"""
Advanced Walking Agent Training Loop
====================================

Architecture:
    - Neocortical Transducer:     raw sensors → 64-dim cognitive coordinate manifold
    - World Model Ensemble:       robust next-state prediction + uncertainty
    - Hippocampal Cognitive Kernel: episodic memory + astrocytic novelty modulation
    - TheoCore:                   crystallized fast procedural skills (when world model is confident)

This loop implements a dual-memory system:
    Plastic (hippocampal) memory for novel/exploratory walking
    Crystallized (neocortical/Theo) memory for reliable, energy-efficient gaits
"""

from dataclasses import dataclass
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np
from collections import deque

# Import all components
from walking_config import WalkingKinematicsConfig
from simulator_plugin import WalkingSimulatorPlugin
from neocortical_transducer_walking import NeocorticalTransducerWalking
from world_model_ensemble_walking import WorldModelEnsemble, WorldModelConfig
from cognitive_kernel_walking import WalkingCognitiveKernel, WalkingCognitiveKernelConfig
from theo_core_walking import WalkingTheoCore


@dataclass
class TrainingConfig:
    """High-level training hyperparameters"""
    total_steps: int = 200_000
    sleep_interval: int = 1000
    log_interval: int = 200
    exploration_std: float = 0.35
    min_explore_steps: int = 5000          # force exploration early
    crystallization_grace_period: int = 10000
    target_forward_velocity: float = 1.2   # m/s - adjust to your simulator
    reward_scale: float = 1.0


def compute_walking_reward(
    obs: torch.Tensor,
    action: torch.Tensor,
    next_obs: torch.Tensor,
    config: WalkingKinematicsConfig,
    target_vel: float = 1.2
) -> Tuple[float, Dict[str, float]]:
    """
    Sophisticated reward function for stable, efficient bipedal/quadrupedal walking.
    """
    # Assume obs contains proprioception; we extract forward velocity from simulator info or approximate
    # In real plugin, you should return velocity in info dict. Here we simulate a clean version.
    forward_vel = 0.0
    if hasattr(obs, 'forward_velocity'):          # if plugin provides it
        forward_vel = obs.forward_velocity
    else:
        # Placeholder: use change in some root position (you'll improve via plugin)
        forward_vel = 0.8  # dummy

    # Components
    velocity_reward = config.forward_velocity_weight * max(0.0, forward_vel)
    energy_penalty = config.energy_penalty_weight * torch.norm(action, p=2).item()
    smoothness_penalty = config.smoothness_weight * torch.norm(action - getattr(compute_walking_reward, 'last_action', action), p=2).item()
    
    # Fall detection (simple height or orientation check - improve with plugin)
    fall_penalty = config.fall_penalty if forward_vel < 0.1 and abs(forward_vel) < 0.05 else 0.0

    total_reward = velocity_reward - energy_penalty - smoothness_penalty + fall_penalty

    compute_walking_reward.last_action = action.clone()

    diagnostics = {
        "velocity_reward": velocity_reward,
        "energy_penalty": energy_penalty,
        "smoothness_penalty": smoothness_penalty,
        "fall_penalty": fall_penalty,
        "forward_vel": forward_vel,
        "total_reward": total_reward,
    }
    return total_reward, diagnostics


def train_walking_core(
    kinematics_config: WalkingKinematicsConfig,
    plugin: WalkingSimulatorPlugin,
    train_cfg: Optional[TrainingConfig] = None,
):
    if train_cfg is None:
        train_cfg = TrainingConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # ====================== INITIALIZE MODULES ======================
    transducer = NeocorticalTransducerWalking(kinematics_config, coordinate_dim=64).to(device)

    wm_config = WorldModelConfig(
        coordinate_dim=64,
        action_dim=kinematics_config.action_dim,
        n_heads=5,
        hidden_dim=128
    )
    world_model = WorldModelEnsemble(wm_config).to(device)

    hip_config = WalkingCognitiveKernelConfig(
        coordinate_dim=64,
        max_episodes=kinematics_config.max_walking_episodes,
        novelty_threshold=kinematics_config.novelty_threshold,
    )
    hip_core = WalkingCognitiveKernel(hip_config).to(device)

    theo_core = WalkingTheoCore(kinematics_config).to(device)

    # Optimizer (world model only for now - plastic policy coming next)
    wm_optimizer = torch.optim.Adam(world_model.parameters(), lr=wm_config.learning_rate)

    # ====================== STATE & BUFFERS ======================
    obs = plugin.reset()
    state = transducer.encode_sensory(obs).to(device)          # (64,)

    error_history = deque(maxlen=500)
    reward_history = deque(maxlen=500)
    crystallized_count = 0
    explore_steps = 0

    print("Starting walking core training...")

    for step in range(train_cfg.total_steps):
        # ------------------- ACTION SELECTION -------------------
        # Priority: Crystallized (Theo) > Plastic Exploration
        theo_action = theo_core(state, error_history[-1] if error_history else 0.5)

        if theo_action is not None and step > train_cfg.crystallization_grace_period:
            action = theo_action
            action_source = "theo_crystallized"
        else:
            # Plastic phase: Gaussian exploration around zero (can be replaced by learned actor later)
            noise = torch.randn(kinematics_config.action_dim, device=device) * train_cfg.exploration_std
            action = torch.clamp(noise, -1.0, 1.0)
            action_source = "plastic_exploration"
            explore_steps += 1

        # ------------------- ENVIRONMENT INTERACTION -------------------
        next_obs, raw_reward, done, info = plugin.step(action)
        next_state = transducer.encode_sensory(next_obs).to(device)

        # ------------------- WORLD MODEL PREDICTION & UPDATE -------------------
        pred_next_state = world_model.predict(state, action)
        prediction_error = F.mse_loss(pred_next_state, next_state).item()

        # Train world model
        wm_loss = world_model.update(state, action, next_state)
        wm_optimizer.zero_grad()
        wm_loss.backward()
        wm_optimizer.step()

        error_history.append(prediction_error)

        # ------------------- HIPPOCAMPAL PLASTIC MEMORY -------------------
        _, hip_diagnostics = hip_core(state, prediction_error)

        # ------------------- CRYSTALLIZATION -------------------
        crystallized_this_step = False
        if (step > train_cfg.crystallization_grace_period and
            prediction_error < kinematics_config.crystallization_error_threshold):
            
            if theo_core.try_crystallize(state, action, prediction_error):
                crystallized_this_step = True
                crystallized_count += 1

        # ------------------- REWARD & DIAGNOSTICS -------------------
        reward, reward_diag = compute_walking_reward(
            obs, action, next_obs, kinematics_config, train_cfg.target_forward_velocity
        )
        reward_history.append(reward)

        # ------------------- LOGGING -------------------
        if step % train_cfg.log_interval == 0 or crystallized_this_step:
            mean_error = world_model.mean_recent_error()
            mean_var = world_model.mean_recent_variance()
            astro_eta = hip_diagnostics.get("astro_eta", 1.0)
            stored_eps = hip_diagnostics["num_stored_episodes"]

            print(f"Step {step:6d} | "
                  f"Src: {action_source:>12} | "
                  f"Error: {prediction_error:.5f} (mean {mean_error:.5f}) | "
                  f"Var: {mean_var:.4f} | "
                  f"Astro η: {astro_eta:.3f} | "
                  f"Stored: {stored_eps:3d} | "
                  f"Cryst: {crystallized_count:3d} | "
                  f"Reward: {reward:6.3f}")

        # ------------------- SLEEP / OFFLINE CONSOLIDATION -------------------
        if step % train_cfg.sleep_interval == 0 and step > 0:
            print(f"\n=== SLEEP PHASE at step {step} ===")
            print(f"  Episodes in CA3: {hip_diagnostics['num_stored_episodes']}")
            print(f"  Crystallized skills: {len(theo_core.crystallized_skills)}")
            print(f"  Mean recent WM error: {world_model.mean_recent_error():.6f}")
            print(f"  Mean recent uncertainty: {world_model.mean_recent_variance():.6f}\n")

            # TODO: Add hippocampal replay + world model fine-tuning here later

        # ------------------- NEXT ITERATION -------------------
        state = next_state
        obs = next_obs

        if done:
            obs = plugin.reset()
            state = transducer.encode_sensory(obs).to(device)
            print(f"Episode terminated at step {step}")

    # ====================== FINAL SUMMARY ======================
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Total steps: {train_cfg.total_steps}")
    print(f"Crystallized walking skills: {crystallized_count}")
    print(f"Final CA3 episodes: {len(hip_core._stored_patterns)}")
    print(f"Final mean prediction error: {world_model.mean_recent_error():.6f}")
    print(f"Final mean ensemble variance: {world_model.mean_recent_variance():.6f}")
    if reward_history:
        print(f"Average reward (last 500): {sum(reward_history)/len(reward_history):.4f}")

    plugin.close()


# ====================== ENTRY POINT ======================
if __name__ == "__main__":
    # Example usage - you will fill in your actual plugin
    kinematics_cfg = WalkingKinematicsConfig()
    # kinematics_cfg.action_low = [...]   # fill from simulator
    # kinematics_cfg.action_high = [...]

    # plugin = YourRealWalkingSimulatorPlugin(...)
    # train_walking_core(kinematics_cfg, plugin)
    
    print("train_walking_core.py loaded successfully.")
    print("Ready for MoJo plugin.")
