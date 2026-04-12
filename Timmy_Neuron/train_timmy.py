"""
train_timmy.py
Training Script for the Timmy Spiking Language Model

Usage:
    # Train on TinyStories (small, fast, good for validation):
    python train_timmy.py --data roneneldan/TinyStories

    # Train on OpenWebText (large, the real training run):
    python train_timmy.py --data openwebtext --text_column text

    # Train on local text files:
    python train_timmy.py --data /path/to/corpus.txt

    # Resume from checkpoint:
    python train_timmy.py --data roneneldan/TinyStories --resume checkpoints/timmy_step_5000.soul

Requirements:
    pip install torch transformers datasets

The training loop uses:
    - AdamW optimizer with cosine learning rate schedule and warmup
    - Gradient accumulation (default 16 micro-batches per step)
    - Mixed precision (float16) with gradient scaling
    - Spike rate regularization loss (asymmetric, penalizes silence)
    - MoE load balancing loss
    - Optional reward-modulated STDP on executive zone layers
    - Periodic .soul checkpoints with full state round-trip
"""

from __future__ import annotations

import argparse
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from timmy_model import TimmyConfig, TimmyModel
from timmy_data import create_dataloader


# =========================================================================
# Learning Rate Schedule
# =========================================================================

def cosine_lr(step: int, warmup: int, max_steps: int, lr: float, min_lr: float) -> float:
    """
    Cosine learning rate with linear warmup.

    Args:
        step: current training step.
        warmup: number of warmup steps (linear ramp from 0 to lr).
        max_steps: total training steps.
        lr: peak learning rate.
        min_lr: minimum learning rate at end of cosine decay.

    Returns:
        Learning rate for this step.
    """
    if step < warmup:
        return lr * step / max(warmup, 1)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / max(max_steps - warmup, 1)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# =========================================================================
# Training Loop
# =========================================================================

def train(
    data_source: str,
    resume_path: Optional[str] = None,
    text_column: str = "text",
    split: str = "train",
    output_dir: str = "checkpoints",
):
    """
    Main training function.

    Args:
        data_source: HuggingFace dataset name or local file/directory path.
        resume_path: path to a .soul checkpoint to resume from.
        text_column: text field name for HuggingFace datasets.
        split: dataset split for HuggingFace datasets.
        output_dir: directory for saving checkpoints.
    """

    # ---- Configuration ----
    cfg = TimmyConfig()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Data source: {data_source}")

    # ---- Model ----
    model = TimmyModel(cfg)
    model = model.to(device)
    print(model.count_params())

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    # ---- Mixed precision ----
    use_amp = (device.type == "cuda" and cfg.dtype == torch.float16)
    scaler = GradScaler(enabled=use_amp)

    # ---- Data ----
    loader = create_dataloader(
        source=data_source,
        tokenizer_id=cfg.tokenizer_id,
        max_seq_len=cfg.max_seq_len,
        batch_size=cfg.batch_size,
        text_column=text_column,
        split=split,
        shuffle=True,
    )

    # ---- Resume ----
    start_step = 0
    if resume_path is not None:
        print(f"Resuming from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state(checkpoint, strict=False)
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "step" in checkpoint:
            start_step = checkpoint["step"]
        print(f"Resumed at step {start_step}")

    # ---- Output directory ----
    os.makedirs(output_dir, exist_ok=True)

    # ---- Training state ----
    step = start_step
    micro_step = 0
    accum_loss = 0.0
    accum_spike_loss = 0.0
    accum_moe_loss = 0.0
    tokens_seen = 0
    best_loss = float("inf")

    model.train()
    optimizer.zero_grad()

    print(f"\nTraining started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Max steps: {cfg.max_steps}, Batch: {cfg.batch_size}, "
          f"Grad accum: {cfg.grad_accum}, Effective batch: {cfg.batch_size * cfg.grad_accum}")
    print(f"Sequence length: {cfg.max_seq_len}")
    print("-" * 70)

    t_start = time.time()

    data_iter = iter(loader)
    while step < cfg.max_steps:
        # Get next batch (restart iterator if exhausted).
        try:
            token_ids = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            token_ids = next(data_iter)
            model.reset_state()  # reset persistent state between epochs

        token_ids = token_ids.to(device)
        B, S = token_ids.shape

        # Input: all tokens except the last.
        # Target: all tokens except the first (next-token prediction).
        input_ids = token_ids[:, :-1]
        target_ids = token_ids[:, 1:]

        # Forward pass with optional STDP caching.
        enable_stdp = (step >= cfg.lif_freeze_steps and step % 10 == 0)
        with autocast(enabled=use_amp):
            logits, stats = model(input_ids, enable_stdp=enable_stdp)

            # Language modeling loss.
            lm_loss = F.cross_entropy(
                logits.reshape(-1, cfg.vocab_size),
                target_ids.reshape(-1),
            )

            # Auxiliary losses.
            spike_loss = stats.get("spike_loss", torch.tensor(0.0, device=device))
            if isinstance(spike_loss, (int, float)):
                spike_loss = torch.tensor(spike_loss, device=device)
            moe_lb_loss = stats.get("moe_lb_loss", torch.tensor(0.0, device=device))
            if isinstance(moe_lb_loss, (int, float)):
                moe_lb_loss = torch.tensor(moe_lb_loss, device=device)

            total_loss = lm_loss + spike_loss + cfg.moe_load_balance_weight * moe_lb_loss

            # Scale for gradient accumulation.
            scaled_loss = total_loss / cfg.grad_accum

        # Backward pass.
        scaler.scale(scaled_loss).backward()

        # Accumulate stats.
        accum_loss += lm_loss.item()
        accum_spike_loss += spike_loss.item() if torch.is_tensor(spike_loss) else spike_loss
        accum_moe_loss += moe_lb_loss.item() if torch.is_tensor(moe_lb_loss) else moe_lb_loss
        tokens_seen += B * (S - 1)
        micro_step += 1

        # Optimizer step after grad_accum micro-batches.
        if micro_step % cfg.grad_accum == 0:
            # Gradient clipping.
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.max_grad_norm
            )

            # Learning rate schedule.
            current_lr = cosine_lr(step, cfg.warmup_steps, cfg.max_steps, cfg.lr, cfg.min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            # Step optimizer.
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            step += 1

            # STDP update (after optimizer step, using current loss for reward).
            if enable_stdp:
                model.set_last_loss(lm_loss.item())
                model.stdp_update(current_loss=lm_loss.item())

            # ---- Logging ----
            if step % cfg.log_every == 0:
                avg_loss = accum_loss / cfg.grad_accum
                avg_spike = accum_spike_loss / cfg.grad_accum
                avg_moe = accum_moe_loss / cfg.grad_accum
                elapsed = time.time() - t_start
                tok_per_sec = tokens_seen / elapsed if elapsed > 0 else 0

                print(
                    f"step {step:6d} | "
                    f"loss {avg_loss:.4f} | "
                    f"spike_loss {avg_spike:.4f} | "
                    f"moe_lb {avg_moe:.4f} | "
                    f"lr {current_lr:.2e} | "
                    f"grad_norm {grad_norm:.2f} | "
                    f"spike_rate {stats.get('avg_spike_rate', 0):.4f} | "
                    f"tok/s {tok_per_sec:.0f}"
                )

                if avg_loss < best_loss:
                    best_loss = avg_loss

                accum_loss = 0.0
                accum_spike_loss = 0.0
                accum_moe_loss = 0.0

            # ---- Checkpoint ----
            if step % cfg.save_every == 0:
                ckpt_path = os.path.join(output_dir, f"timmy_step_{step}.soul")
                model.save_state(
                    path=ckpt_path,
                    optimizer_state=optimizer.state_dict(),
                    training_step=step,
                    extra={"best_loss": best_loss, "tokens_seen": tokens_seen},
                )
                print(f"  Saved checkpoint: {ckpt_path}")

    # ---- Final checkpoint ----
    final_path = os.path.join(output_dir, "timmy_final.soul")
    model.save_state(
        path=final_path,
        optimizer_state=optimizer.state_dict(),
        training_step=step,
        extra={"best_loss": best_loss, "tokens_seen": tokens_seen},
    )

    elapsed = time.time() - t_start
    print(f"\nTraining complete.")
    print(f"  Steps: {step}")
    print(f"  Tokens: {tokens_seen:,}")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Time: {elapsed/3600:.1f} hours")
    print(f"  Final checkpoint: {final_path}")


# =========================================================================
# CLI
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Timmy spiking language model")
    parser.add_argument(
        "--data", type=str, required=True,
        help="HuggingFace dataset name (e.g. 'roneneldan/TinyStories') "
             "or path to local text file or directory",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a .soul checkpoint to resume training from",
    )
    parser.add_argument(
        "--text_column", type=str, default="text",
        help="Name of the text field in HuggingFace datasets (default: 'text')",
    )
    parser.add_argument(
        "--split", type=str, default="train",
        help="Dataset split to use (default: 'train')",
    )
    parser.add_argument(
        "--output_dir", type=str, default="checkpoints",
        help="Directory for saving .soul checkpoints (default: 'checkpoints')",
    )
    args = parser.parse_args()

    train(
        data_source=args.data,
        resume_path=args.resume,
        text_column=args.text_column,
        split=args.split,
        output_dir=args.output_dir,
    )
