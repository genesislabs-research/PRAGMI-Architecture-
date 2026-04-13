"""
crystallization_manager.py
Neo: Generalization-Gated Crystallization Monitor for S-ROS

This module does not care about memorization. It cares about understanding.
A pathway crystallizes when the network produces correct output for held-out
instances of a rule class it has never been trained on. That is the test for
genuine learning. Everything else is filing.

BIOLOGICAL GROUNDING
Critical period closure: Turrigiano (2008) synaptic scaling, Abraham (2008)
metaplasticity, Bhatt et al. (2009) astrocytic calcium gating of structural
plasticity windows. Crystallization is the computational analog of critical
period closure: once a rule is learned (generalization confirmed), the pathway
locks and plasticity budget is freed for unconverged rules.

The crystallization criterion is generalization, not convergence on trained
examples. This is the difference between understanding and memorization.

Genesis Labs Research, 2026
Amellia Mendel, Lisa Adler
"""

import torch
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone


@dataclass
class RuleClass:
    """One rule class with its training and held-out test instances."""
    rule_id: str
    description: str
    train_pairs: List[Dict]
    test_pairs: List[Dict]
    weight_delta_history: List[float] = field(default_factory=list)
    train_loss_history: List[float] = field(default_factory=list)
    test_loss_history: List[float] = field(default_factory=list)
    generalization_accuracy_history: List[float] = field(default_factory=list)
    consecutive_generalization_passes: int = 0
    is_crystallized: bool = False
    crystallized_at_step: Optional[int] = None
    crystallized_weights: Optional[Dict[str, torch.Tensor]] = None


class CrystallizationManager:
    """
    Neo: watches the plastic side and decides when a rule has been understood.

    Not a doctor. A thermometer. But the thermometer reads generalization,
    not memorization. A rule crystallizes when:
      1. Training loss on seen instances is below threshold for K windows
      2. Generalization accuracy on UNSEEN held-out instances exceeds target
         for K consecutive evaluation windows
      3. Weight delta variance has stabilized (learning has stopped naturally)

    All three conditions must be met simultaneously. Condition 2 is the one
    that distinguishes this from every other convergence monitor in existence.
    """

    def __init__(
        self,
        loss_threshold: float = 0.01,
        generalization_target: float = 0.90,
        variance_threshold: float = 1e-5,
        consecutive_windows_required: int = 5,
        window_size: int = 50,
    ):
        self.loss_threshold = loss_threshold
        self.generalization_target = generalization_target
        self.variance_threshold = variance_threshold
        self.consecutive_windows_required = consecutive_windows_required
        self.window_size = window_size
        self.rule_classes: Dict[str, RuleClass] = {}
        self.crystallization_log: List[Dict] = []
        self.global_step: int = 0

    def register_rule_class(
        self,
        rule_id: str,
        description: str,
        train_pairs: List[Dict],
        test_pairs: List[Dict],
    ) -> None:
        """Register a rule class with its training and held-out test data."""
        if len(test_pairs) == 0:
            raise ValueError(
                f"Rule class '{rule_id}' has no held-out test pairs. "
                "Cannot measure generalization without held-out data."
            )
        self.rule_classes[rule_id] = RuleClass(
            rule_id=rule_id,
            description=description,
            train_pairs=train_pairs,
            test_pairs=test_pairs,
        )

    def record_train_loss(self, rule_id: str, loss: float) -> None:
        """Record training loss for a rule class after a training step."""
        rc = self.rule_classes[rule_id]
        if rc.is_crystallized:
            return
        rc.train_loss_history.append(loss)

    def record_weight_deltas(self, rule_id: str, delta_norm: float) -> None:
        """Record the norm of weight changes for a rule class."""
        rc = self.rule_classes[rule_id]
        if rc.is_crystallized:
            return
        rc.weight_delta_history.append(delta_norm)

    def evaluate_generalization(
        self,
        rule_id: str,
        model: torch.nn.Module,
        encode_fn,
        decode_fn,
        compare_fn,
    ) -> float:
        """
        Run the model on held-out test pairs for this rule class.
        Returns generalization accuracy (0.0 to 1.0).

        encode_fn(pair) -> model input tensor
        decode_fn(model_output) -> predicted output representation
        compare_fn(predicted, expected_pair) -> bool (correct or not)
        """
        rc = self.rule_classes[rule_id]
        if rc.is_crystallized:
            return 1.0

        correct = 0
        total = len(rc.test_pairs)
        with torch.no_grad():
            for pair in rc.test_pairs:
                input_tensor = encode_fn(pair)
                output = model(input_tensor)
                predicted = decode_fn(output)
                if compare_fn(predicted, pair):
                    correct += 1

        accuracy = correct / total if total > 0 else 0.0
        rc.generalization_accuracy_history.append(accuracy)
        rc.test_loss_history.append(1.0 - accuracy)
        return accuracy

    def check_crystallization(self, rule_id: str) -> bool:
        """
        Check whether a rule class meets all three crystallization criteria.
        Returns True if the rule should crystallize NOW.
        """
        rc = self.rule_classes[rule_id]
        if rc.is_crystallized:
            return False

        k = self.consecutive_windows_required

        train_converged = (
            len(rc.train_loss_history) >= k
            and all(
                l < self.loss_threshold
                for l in rc.train_loss_history[-k:]
            )
        )

        generalization_met = (
            len(rc.generalization_accuracy_history) >= k
            and all(
                a >= self.generalization_target
                for a in rc.generalization_accuracy_history[-k:]
            )
        )

        if len(rc.weight_delta_history) >= self.window_size:
            recent_deltas = rc.weight_delta_history[-self.window_size:]
            delta_variance = torch.var(torch.tensor(recent_deltas)).item()
            weights_stable = delta_variance < self.variance_threshold
        else:
            weights_stable = False

        return train_converged and generalization_met and weights_stable

    def crystallize(
        self,
        rule_id: str,
        model: torch.nn.Module,
        step: int,
    ) -> Dict:
        """
        Freeze the pathway for this rule class. Extract weight snapshot.
        Returns the crystallization event record.
        """
        rc = self.rule_classes[rule_id]
        rc.is_crystallized = True
        rc.crystallized_at_step = step
        rc.crystallized_weights = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
        }

        event = {
            "rule_id": rule_id,
            "description": rc.description,
            "crystallized_at_step": step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "final_train_loss": rc.train_loss_history[-1],
            "final_generalization_accuracy": rc.generalization_accuracy_history[-1],
            "final_weight_delta_variance": torch.var(
                torch.tensor(rc.weight_delta_history[-self.window_size:])
            ).item() if len(rc.weight_delta_history) >= self.window_size else None,
            "total_train_steps": len(rc.train_loss_history),
            "num_train_pairs": len(rc.train_pairs),
            "num_test_pairs": len(rc.test_pairs),
        }
        self.crystallization_log.append(event)
        return event

    def step(
        self,
        rule_id: str,
        train_loss: float,
        delta_norm: float,
        model: torch.nn.Module,
        encode_fn,
        decode_fn,
        compare_fn,
        eval_interval: int = 10,
    ) -> Optional[Dict]:
        """
        All-in-one step: record metrics, evaluate generalization periodically,
        check crystallization, and crystallize if ready. Returns crystallization
        event if one occurred, None otherwise.
        """
        self.global_step += 1
        rc = self.rule_classes[rule_id]
        if rc.is_crystallized:
            return None

        self.record_train_loss(rule_id, train_loss)
        self.record_weight_deltas(rule_id, delta_norm)

        if self.global_step % eval_interval == 0:
            self.evaluate_generalization(
                rule_id, model, encode_fn, decode_fn, compare_fn
            )

            if self.check_crystallization(rule_id):
                return self.crystallize(rule_id, model, self.global_step)

        return None

    def status(self) -> Dict:
        """Summary of all rule classes and their crystallization state."""
        summary = {}
        for rid, rc in self.rule_classes.items():
            summary[rid] = {
                "crystallized": rc.is_crystallized,
                "crystallized_at_step": rc.crystallized_at_step,
                "train_steps": len(rc.train_loss_history),
                "last_train_loss": rc.train_loss_history[-1] if rc.train_loss_history else None,
                "last_generalization": (
                    rc.generalization_accuracy_history[-1]
                    if rc.generalization_accuracy_history else None
                ),
                "consecutive_passes": rc.consecutive_generalization_passes,
            }
        return summary

    def save_log(self, path: str) -> None:
        """Write the crystallization event log to disk."""
        with open(path, "w") as f:
            json.dump(self.crystallization_log, f, indent=2)

    def fraction_crystallized(self) -> float:
        """What fraction of registered rule classes have crystallized."""
        if not self.rule_classes:
            return 0.0
        n_frozen = sum(1 for rc in self.rule_classes.values() if rc.is_crystallized)
        return n_frozen / len(self.rule_classes)


# =========================================================================
# SELF-TESTS
# =========================================================================

def test_register_and_status():
    cm = CrystallizationManager()
    cm.register_rule_class(
        rule_id="print_basic",
        description="PRINT with string literal",
        train_pairs=[{"in": 'PRINT "Hello"', "out": "Print 'Hello' to screen"}],
        test_pairs=[{"in": 'PRINT "World"', "out": "Print 'World' to screen"}],
    )
    s = cm.status()
    assert "print_basic" in s
    assert s["print_basic"]["crystallized"] is False
    print("PASS: test_register_and_status")


def test_no_test_pairs_raises():
    cm = CrystallizationManager()
    try:
        cm.register_rule_class(
            rule_id="bad",
            description="no test data",
            train_pairs=[{"in": "x", "out": "y"}],
            test_pairs=[],
        )
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("PASS: test_no_test_pairs_raises")


def test_crystallization_requires_generalization():
    """Verify that low training loss alone does not trigger crystallization."""
    cm = CrystallizationManager(
        loss_threshold=0.05,
        generalization_target=0.90,
        variance_threshold=1e-4,
        consecutive_windows_required=3,
        window_size=5,
    )
    cm.register_rule_class(
        rule_id="test_rule",
        description="test",
        train_pairs=[{"in": "a", "out": "b"}],
        test_pairs=[{"in": "c", "out": "d"}],
    )
    for _ in range(20):
        cm.record_train_loss("test_rule", 0.001)
        cm.record_weight_deltas("test_rule", 0.0001)
    assert not cm.check_crystallization("test_rule"), (
        "Should NOT crystallize without generalization evaluation"
    )
    print("PASS: test_crystallization_requires_generalization")


def test_full_crystallization_cycle():
    """Simulate a rule class that genuinely generalizes and should crystallize."""
    cm = CrystallizationManager(
        loss_threshold=0.05,
        generalization_target=0.80,
        variance_threshold=1e-3,
        consecutive_windows_required=3,
        window_size=5,
    )
    cm.register_rule_class(
        rule_id="comma_rule",
        description="PRINT with comma separator uses zones",
        train_pairs=[
            {"in": 'PRINT "A","B"', "out": "Print 'A' zone1, 'B' zone2"},
            {"in": 'PRINT "X","Y"', "out": "Print 'X' zone1, 'Y' zone2"},
        ],
        test_pairs=[
            {"in": 'PRINT "C","D"', "out": "Print 'C' zone1, 'D' zone2"},
        ],
    )

    for _ in range(10):
        cm.record_train_loss("comma_rule", 0.01)
        cm.record_weight_deltas("comma_rule", 0.0001)

    rc = cm.rule_classes["comma_rule"]
    for _ in range(5):
        rc.generalization_accuracy_history.append(0.95)

    assert cm.check_crystallization("comma_rule"), (
        "Should crystallize: train converged, generalization met, weights stable"
    )

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor([1.0]))
        def forward(self, x):
            return x * self.w

    event = cm.crystallize("comma_rule", FakeModel(), step=100)
    assert event["rule_id"] == "comma_rule"
    assert event["final_generalization_accuracy"] == 0.95
    assert cm.rule_classes["comma_rule"].is_crystallized is True
    assert cm.fraction_crystallized() == 1.0
    print("PASS: test_full_crystallization_cycle")


if __name__ == "__main__":
    test_register_and_status()
    test_no_test_pairs_raises()
    test_crystallization_requires_generalization()
    test_full_crystallization_cycle()
    print("\nAll crystallization_manager tests passed.")
