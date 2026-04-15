"""
curriculum_diagnostic_v5.py
Diagnostic Tool for Curriculum-Staged Theo SNN Checkpoints

Loads a checkpoint and displays per-stage, per-rule classification accuracy
with colored terminal output. Shows crystallization status, confusion patterns,
and template rendering accuracy for Stage 3.

Genesis Labs Research, 2026
Amellia Mendel, Lisa Adler
"""

import torch
import torch.nn.functional as F
import argparse
import sys
from typing import Dict, List

from curriculum_trainer_v5 import (
    RuleClassifierSNN, CharVocab, TemplateEngine, C,
    NUM_KEYWORDS, NUM_PATTERNS,
)
from curriculum_data import (
    build_stage1_data, build_stage2_data, build_stage3_data,
    KEYWORDS, PATTERN_NAMES,
)


def diagnose(checkpoint_path: str, device: str = "auto", verbose: bool = False):
    """Load checkpoint and run full diagnostic."""
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    print(f"\n{C.CYAN}{C.BOLD}{'='*66}{C.RESET}")
    print(f"{C.CYAN}{C.BOLD}  CURRICULUM DIAGNOSTIC v5{C.RESET}")
    print(f"{C.CYAN}{C.BOLD}{'='*66}{C.RESET}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device: {dev}")

    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    stage = ckpt["stage"]
    cfg = ckpt["model_config"]
    epoch = ckpt["epoch"]
    global_step = ckpt.get("global_step", "?")

    stage_names = ["KEYWORD RECOGNITION", "SYNTAX PATTERN RECOGNITION", "FULL STATEMENT PROCESSING"]
    print(f"  Stage: {stage+1} ({stage_names[stage]})")
    print(f"  Epoch: {epoch}")
    print(f"  Global step: {global_step}")

    if stage == 0:
        num_classes = NUM_KEYWORDS
        class_names = [f"keyword_{kw.lower()}" for kw in KEYWORDS]
    else:
        num_classes = NUM_PATTERNS
        class_names = PATTERN_NAMES

    vocab = CharVocab()
    model = RuleClassifierSNN(
        vocab_size=vocab.vocab_size,
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_classes=num_classes,
        num_lif_layers=cfg["num_lif_layers"],
        max_len=cfg["max_len"],
        beta=cfg["beta"],
    ).to(dev)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    if stage == 0:
        data = build_stage1_data()
    elif stage == 1:
        data = build_stage2_data()
    else:
        data = build_stage3_data()

    template = TemplateEngine()

    # Per-rule results
    print(f"\n{C.BOLD}  Per-Rule Test Accuracy:{C.RESET}")
    print(f"  {'Rule':35s} {'Train':>8s} {'Test':>8s} {'Status':>10s}")
    print(f"  {'-'*65}")

    status = ckpt.get("status", {})
    total_train_correct = 0
    total_train_count = 0
    total_test_correct = 0
    total_test_count = 0
    confusion = {}

    for rule in data:
        rid = rule["rule_id"]
        # Train accuracy
        train_correct = 0
        train_count = 0
        for pair in rule["train_pairs"]:
            input_ids = vocab.encode(pair["input"], cfg["max_len"]).unsqueeze(0).to(dev)
            with torch.no_grad():
                logits = model(input_ids)
            pred = logits.argmax(dim=-1).item()
            if pred == pair["label"]:
                train_correct += 1
            train_count += 1
        train_acc = train_correct / max(train_count, 1)

        # Test accuracy
        test_correct = 0
        test_count = 0
        for pair in rule["test_pairs"]:
            input_ids = vocab.encode(pair["input"], cfg["max_len"]).unsqueeze(0).to(dev)
            with torch.no_grad():
                logits = model(input_ids)
            pred = logits.argmax(dim=-1).item()
            expected = pair["label"]
            if pred == expected:
                test_correct += 1
            else:
                key = (class_names[expected], class_names[min(pred, len(class_names)-1)])
                confusion[key] = confusion.get(key, 0) + 1
                if verbose:
                    print(f"    {C.RED}MISS{C.RESET} {pair['input'][:50]:50s} "
                          f"expected:{class_names[expected]} got:{class_names[min(pred, len(class_names)-1)]}")
            test_count += 1

            # Stage 3: also check template rendering
            if stage == 2 and pred == expected and "output" in pair:
                rendered = template.render(PATTERN_NAMES[pred], pair["input"])
                if rendered is not None and rendered.strip() != pair["output"].strip():
                    if verbose:
                        print(f"    {C.YELLOW}TEMPLATE MISMATCH{C.RESET} "
                              f"rendered:'{rendered}' expected:'{pair['output']}'")

        test_acc = test_correct / max(test_count, 1)
        total_train_correct += train_correct
        total_train_count += train_count
        total_test_correct += test_correct
        total_test_count += test_count

        cryst = status.get(rid, {}).get("crystallized", False)
        cryst_str = f"{C.GREEN}FROZEN{C.RESET}" if cryst else f"{C.RED}PLASTIC{C.RESET}"
        train_color = C.GREEN if train_acc >= 0.9 else C.YELLOW if train_acc >= 0.5 else C.RED
        test_color = C.GREEN if test_acc >= 0.9 else C.YELLOW if test_acc >= 0.5 else C.RED
        print(f"  {rid:35s} {train_color}{train_acc:7.0%}{C.RESET} {test_color}{test_acc:7.0%}{C.RESET}  {cryst_str}")

    overall_train = total_train_correct / max(total_train_count, 1)
    overall_test = total_test_correct / max(total_test_count, 1)
    print(f"  {'-'*65}")
    print(f"  {'OVERALL':35s} {overall_train:7.0%} {overall_test:7.0%}")

    if confusion:
        print(f"\n{C.BOLD}  Top Confusions:{C.RESET}")
        sorted_conf = sorted(confusion.items(), key=lambda x: -x[1])[:10]
        for (expected, predicted), count in sorted_conf:
            print(f"    {expected:30s} -> {predicted:30s} ({count}x)")

    # Crystallization log
    log = ckpt.get("crystallization_log", [])
    if log:
        print(f"\n{C.BOLD}  Crystallization Events:{C.RESET}")
        for event in log:
            gen = event.get("final_generalization_accuracy", 0)
            step = event.get("crystallized_at_step", "?")
            print(f"    {event['rule_id']:30s} step:{step:>6} gen:{gen:.0%}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curriculum Diagnostic v5")
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--verbose", action="store_true", help="Show individual misses")
    args = parser.parse_args()
    diagnose(args.checkpoint, device=args.device, verbose=args.verbose)
