from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.nn as nn


@dataclass
class NeuromodulatorConfig:
    da_decay: float = 0.995
    ach_decay: float = 0.98
    ne_decay: float = 0.99
    ht_decay: float = 0.999
    da_baseline_init: float = 0.5
    ach_baseline_init: float = 0.7
    ne_baseline_init: float = 0.4
    ht_baseline_init: float = 0.5
    tau_mature_up: float = 0.995
    tau_mature_down: float = 0.95
    maturity_component_weight: float = 0.25
    entropy_baseline_window: int = 10
    loss_baseline_window: int = 10
    variance_ceiling: float = 0.1


class NeuromodulatorBroadcast(nn.Module):

    def __init__(self, cfg: NeuromodulatorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer('da', torch.tensor(cfg.da_baseline_init))
        self.register_buffer('ach', torch.tensor(cfg.ach_baseline_init))
        self.register_buffer('ne', torch.tensor(cfg.ne_baseline_init))
        self.register_buffer('ht', torch.tensor(cfg.ht_baseline_init))
        self.register_buffer('global_maturity', torch.tensor(0.0))
        self.register_buffer('entropy_baseline_std', torch.tensor(-1.0))
        self.register_buffer('loss_baseline_cv', torch.tensor(-1.0))
        self._entropy_history: list[float] = []
        self._loss_history: list[float] = []
        self._sleep_cycles_elapsed: int = 0

    @torch.no_grad()
    def update_da(self, signal: float) -> None:
        self.da.copy_(self.cfg.da_decay * self.da + (1.0 - self.cfg.da_decay) * signal)

    @torch.no_grad()
    def update_ach(self, signal: float) -> None:
        self.ach.copy_(self.cfg.ach_decay * self.ach + (1.0 - self.cfg.ach_decay) * signal)

    @torch.no_grad()
    def update_ne(self, signal: float) -> None:
        self.ne.copy_(self.cfg.ne_decay * self.ne + (1.0 - self.cfg.ne_decay) * signal)

    @torch.no_grad()
    def update_ht(self, signal: float) -> None:
        self.ht.copy_(self.cfg.ht_decay * self.ht + (1.0 - self.cfg.ht_decay) * signal)

    def is_sleep_phase(self) -> bool:
        return self.ach.item() < 0.5

    def compute_maturity(self, routing_entropy: float, loss_value: float, probe_response: float, mean_recent_variance: float) -> float:
        self._sleep_cycles_elapsed += 1
        self._entropy_history.append(routing_entropy)
        self._loss_history.append(loss_value)
        if self._sleep_cycles_elapsed == self.cfg.entropy_baseline_window and self.entropy_baseline_std.item() < 0.0:
            std_val = float(torch.tensor(self._entropy_history).std())
            self.entropy_baseline_std.copy_(torch.tensor(max(std_val, 1e-6)))
        if self._sleep_cycles_elapsed == self.cfg.loss_baseline_window and self.loss_baseline_cv.item() < 0.0:
            t = torch.tensor(self._loss_history)
            cv_val = float(t.std() / (t.mean() + 1e-9))
            self.loss_baseline_cv.copy_(torch.tensor(max(cv_val, 1e-6)))
        _HISTORY_CAP = 20
        if self.entropy_baseline_std.item() > 0.0 and len(self._entropy_history) > _HISTORY_CAP:
            self._entropy_history = self._entropy_history[-_HISTORY_CAP:]
        if self.loss_baseline_cv.item() > 0.0 and len(self._loss_history) > _HISTORY_CAP:
            self._loss_history = self._loss_history[-_HISTORY_CAP:]
        if self.entropy_baseline_std.item() > 0.0 and len(self._entropy_history) >= 2:
            recent_std = float(torch.tensor(self._entropy_history[-20:]).std())
            routing_component = float(torch.clamp(torch.tensor(1.0 - recent_std / self.entropy_baseline_std.item()), 0.0, 1.0))
        else:
            routing_component = 0.0
        if self.loss_baseline_cv.item() > 0.0 and len(self._loss_history) >= 2:
            recent = torch.tensor(self._loss_history[-20:])
            cv = float(recent.std() / (recent.mean() + 1e-9))
            loss_component = float(torch.clamp(torch.tensor(1.0 - cv / self.loss_baseline_cv.item()), 0.0, 1.0))
        else:
            loss_component = 0.0
        probe_component = float(torch.clamp(torch.tensor(1.0 - probe_response), 0.0, 1.0))
        ensemble_agreement_component = float(torch.clamp(torch.tensor(1.0 - mean_recent_variance / max(self.cfg.variance_ceiling, 1e-9)), 0.0, 1.0))
        w = self.cfg.maturity_component_weight
        new_maturity = w * routing_component + w * loss_component + w * probe_component + w * ensemble_agreement_component
        current = self.global_maturity.item()
        if new_maturity > current:
            updated = self.cfg.tau_mature_up * current + (1.0 - self.cfg.tau_mature_up) * new_maturity
        else:
            updated = self.cfg.tau_mature_down * current + (1.0 - self.cfg.tau_mature_down) * new_maturity
        updated = float(torch.clamp(torch.tensor(updated), 0.0, 1.0))
        with torch.no_grad():
            self.global_maturity.copy_(torch.tensor(updated))
        return updated

    def get_hot_state(self) -> dict:
        return {
            'entropy_history': list(self._entropy_history),
            'loss_history': list(self._loss_history),
            'sleep_cycles_elapsed': self._sleep_cycles_elapsed,
        }

    def load_hot_state(self, hot: dict) -> None:
        self._entropy_history = list(hot.get('entropy_history', []))
        self._loss_history = list(hot.get('loss_history', []))
        self._sleep_cycles_elapsed = hot.get('sleep_cycles_elapsed', 0)
