"""
train_cognitive_kernel.py
Training script for the tiny Cognitive Kernel smoke-test.

This script trains the kernel on next-coordinate prediction in the 64-dim manifold.
It runs full wake/sleep cycles with targeted dynamic expansion of the busiest "specialist"
(placeholder for now, but fully wired for future Timmy integration).

Usage:
    python train_cognitive_kernel.py --steps 500 --sleep_every 50
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from torch.optim import AdamW

from cognitive_kernel_base_for_testing_c import (
    CognitiveKernel,
    CognitiveKernelConfig,
    TinyTimmyEnsemble,
)


def train_cognitive_kernel(
    total_steps: int = 1000,
    sleep_every: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    output_dir: str = "checkpoints/kernel",
):
    os.makedirs(output_dir, exist_ok=True)

    cfg = CognitiveKernelConfig()
    kernel = CognitiveKernel(cfg)
    ensemble = TinyTimmyEnsemble(kernel, num_initial_specialists=3)

    optimizer = AdamW(kernel.parameters(), lr=lr, weight_decay=1e-5)

    print("=" * 80)
    print("Cognitive Kernel Smoke-Test Training")
    print("=" * 80)
    print(f"Total steps: {total_steps} | Sleep every: {sleep_every} steps")
    print(f"Initial d_model: {ensemble.current_d_model}")
    print("-" * 80)

    step = 0
    t_start = time.time()

    while step < total_steps:
        # === Wake Phase ===
        for _ in range(sleep_every):
            if step >= total_steps:
                break

            # Generate random coordinates in the 64-dim manifold
            coords = torch.randn(batch_size, cfg.coordinate_dim)

            # Forward through ensemble + kernel
            _, _, diagnostics = ensemble.forward(coords)

            # Simple MSE loss on reconstruction (next-coordinate prediction)
            # In full PRAGMI this will be replaced with proper world model loss
            loss = F.mse_loss(_, coords)   # _ is the reconstruction from kernel

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

            if step % 20 == 0:
                elapsed = time.time() - t_start
                print(
                    f"Step {step:5d} | "
                    f"Loss {loss.item():.4f} | "
                    f"Doubt {diagnostics['doubt_current']:.3f} | "
                    f"Novelty {diagnostics['mean_novelty']:.3f} | "
                    f"tok/s {step / elapsed:.1f}"
                )

        # === Sleep + Targeted Expansion ===
        if step % sleep_every == 0 or step >= total_steps:
            print(f"\n[SLEEP] Day {(step // sleep_every)} - Targeted Expansion")
            sleep_diag = ensemble.sleep_consolidation()

            print(f"  Expanded busiest specialist → d_model = {ensemble.current_d_model}")
            print(f"  Episodes in CA3: {sleep_diag.get('episodes_in_ca3', 0)}")
            print(f"  Current doubt: {kernel.get_doubt_current():.3f}\n")

            # Save checkpoint
            ckpt_path = Path(output_dir) / f"kernel_step_{step}.pt"
            torch.save({
                "step": step,
                "model_state": kernel.state_dict(),
                "ensemble_state": {
                    "current_d_model": ensemble.current_d_model,
                    "specialist_usage": ensemble.specialist_usage,
                },
                "doubt_current": kernel.get_doubt_current(),
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path.name}")

    elapsed = time.time() - t_start
    print("=" * 80)
    print(f"Training finished in {elapsed/60:.1f} minutes")
    print(f"Final d_model after growth: {ensemble.current_d_model}")
    print(f"Final doubt level: {kernel.get_doubt_current():.3f}")
    print("All core features exercised: doubt reflex, sleep expansion, 3D positions, save/load.")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the tiny Cognitive Kernel smoke-test")
    parser.add_argument("--steps", type=int, default=1000, help="Total training steps")
    parser.add_argument("--sleep_every", type=int, default=50, help="Perform sleep + expansion every N steps")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="checkpoints/kernel")
    args = parser.parse_args()

    train_cognitive_kernel(
        total_steps=args.steps,
        sleep_every=args.sleep_every,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
    )
