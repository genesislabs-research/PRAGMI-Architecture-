"""
theo_diagnostic.py
Shows exactly what the model predicts vs ground truth for every pair.

Usage: python3.11 theo_diagnostic.py --checkpoint theo_rote_run3.pt
Or after training: python3.11 theo_diagnostic.py --checkpoint theo_rote_run3_epoch100.pt

This does not train anything. It loads a saved checkpoint and runs
inference on every training pair and every held-out test pair, printing
the model's actual character-level output next to the expected output.
This tells us exactly what the model learned, what it confused, and
where generalization is failing.

Genesis Labs Research, 2026
Amellia Mendel, Lisa Adler
"""

import torch
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hello_world_trainer import RoteLearner, CharVocab
from rote_data_generator import build_rule_classes


def diagnose(checkpoint_path: str, device: str = "auto", max_len: int = 64):
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    vocab = CharVocab()
    model = RoteLearner(vocab_size=vocab.vocab_size, max_len=max_len).to(dev)

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"Step count at save: {ckpt.get('step_count', 'unknown')}")
    print(f"Device: {dev}")
    print()

    rules = build_rule_classes()
    total_exact = 0
    total_pairs = 0
    total_partial = 0

    for rule in rules:
        rid = rule["rule_id"]
        print(f"{'='*70}")
        print(f"RULE: {rid}")
        print(f"  {rule['description']}")
        print(f"{'='*70}")

        for split_name, pairs in [("TRAIN", rule["train_pairs"]), ("TEST", rule["test_pairs"])]:
            for pair in pairs:
                input_ids = vocab.encode(pair["input"], max_len).unsqueeze(0).to(dev)
                with torch.no_grad():
                    logits = model(input_ids)
                pred_ids = logits.argmax(dim=-1).squeeze(0)
                predicted = vocab.decode(pred_ids)
                expected = pair["output"]

                exact = predicted.strip() == expected.strip()
                total_pairs += 1
                if exact:
                    total_exact += 1

                # partial match: count matching characters
                match_chars = sum(1 for a, b in zip(predicted, expected) if a == b)
                max_chars = max(len(predicted.strip()), len(expected.strip()), 1)
                partial_score = match_chars / max_chars
                total_partial += partial_score

                tag = "EXACT" if exact else "WRONG"
                marker = "OK" if exact else "XX"

                print(f"  [{split_name:5s}] [{marker}] partial:{partial_score:.0%}")
                print(f"    INPUT:    {pair['input']}")
                print(f"    EXPECTED: {expected}")
                print(f"    GOT:      {predicted.strip()}")
                if not exact:
                    # show character-level diff
                    diff = []
                    for i, (e, g) in enumerate(zip(expected, predicted)):
                        if e == g:
                            diff.append(" ")
                        else:
                            diff.append("^")
                    print(f"    DIFF:     {''.join(diff)}")
                print()

    print(f"{'='*70}")
    print(f"SUMMARY")
    print(f"  Total pairs:    {total_pairs}")
    print(f"  Exact matches:  {total_exact} ({total_exact/total_pairs:.0%})")
    print(f"  Avg partial:    {total_partial/total_pairs:.1%}")
    print(f"{'='*70}")

    # per-rule summary
    print()
    print(f"{'RULE CLASS':30s} {'TRAIN':>8s} {'TEST':>8s}")
    print(f"{'-'*50}")
    for rule in rules:
        rid = rule["rule_id"]
        train_ok = 0
        test_ok = 0
        for pair in rule["train_pairs"]:
            input_ids = vocab.encode(pair["input"], max_len).unsqueeze(0).to(dev)
            with torch.no_grad():
                logits = model(input_ids)
            pred = vocab.decode(logits.argmax(dim=-1).squeeze(0))
            if pred.strip() == pair["output"].strip():
                train_ok += 1
        for pair in rule["test_pairs"]:
            input_ids = vocab.encode(pair["input"], max_len).unsqueeze(0).to(dev)
            with torch.no_grad():
                logits = model(input_ids)
            pred = vocab.decode(logits.argmax(dim=-1).squeeze(0))
            if pred.strip() == pair["output"].strip():
                test_ok += 1
        nt = len(rule["train_pairs"])
        nh = len(rule["test_pairs"])
        print(f"  {rid:30s} {train_ok}/{nt:>2d}     {test_ok}/{nh:>2d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Theo Training Diagnostic")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-len", type=int, default=64)
    args = parser.parse_args()
    diagnose(args.checkpoint, args.device, args.max_len)
