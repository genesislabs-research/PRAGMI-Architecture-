from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class CorticalBufferConfig:
    d_model: int = 256
    buffer_dim: int = 32
    injection_scale: float = 0.1
    buffer_tau: float = 0.8


class CorticalBuffer(nn.Module):

    def __init__(self, cfg: CorticalBufferConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.compress = nn.Linear(cfg.d_model, cfg.buffer_dim, bias=False)
        self.expand = nn.Linear(cfg.buffer_dim, cfg.d_model, bias=False)
        self.register_buffer('state', torch.zeros(cfg.buffer_dim))
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.compress.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.expand.weight, mean=0.0, std=0.01)

    def get_injection(self) -> torch.Tensor:
        return self.cfg.injection_scale * self.expand(self.state)

    def update(self, v_mem: torch.Tensor) -> None:
        with torch.no_grad():
            v = v_mem.detach()
            if v.dim() == 2:
                v = v.mean(dim=0)
            compressed = self.compress(v)
            self.state.copy_(
                self.cfg.buffer_tau * self.state + (1.0 - self.cfg.buffer_tau) * compressed
            )

    def expand_for_new_d_model(self, new_d_model: int) -> None:
        old_d_model = self.cfg.d_model
        delta = new_d_model - old_d_model
        assert delta > 0, f'new_d_model {new_d_model} must exceed current {old_d_model}'
        old_compress_weight = self.compress.weight.data
        new_cols = torch.zeros(self.cfg.buffer_dim, delta, device=old_compress_weight.device, dtype=old_compress_weight.dtype)
        new_cols.uniform_(0.0, 1e-8)
        new_compress_weight = torch.cat([old_compress_weight, new_cols], dim=1)
        new_compress = nn.Linear(new_d_model, self.cfg.buffer_dim, bias=False)
        new_compress.weight = nn.Parameter(new_compress_weight)
        self.compress = new_compress
        old_expand_weight = self.expand.weight.data
        new_rows = torch.zeros(delta, self.cfg.buffer_dim, device=old_expand_weight.device, dtype=old_expand_weight.dtype)
        new_rows.uniform_(0.0, 1e-8)
        new_expand_weight = torch.cat([old_expand_weight, new_rows], dim=0)
        new_expand = nn.Linear(self.cfg.buffer_dim, new_d_model, bias=False)
        new_expand.weight = nn.Parameter(new_expand_weight)
        self.expand = new_expand
        self.cfg = CorticalBufferConfig(d_model=new_d_model, buffer_dim=self.cfg.buffer_dim, injection_scale=self.cfg.injection_scale, buffer_tau=self.cfg.buffer_tau)

    def reset(self) -> None:
        with torch.no_grad():
            self.state.zero_()

    def get_hot_state(self) -> dict:
        return {'state': self.state.detach().cpu().clone()}

    def load_hot_state(self, hot: dict) -> None:
        with torch.no_grad():
            self.state.copy_(hot['state'].to(self.state.device))
