
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class CognitiveKernelConfig:

    coordinate_dim: int = 64

    dentate_gyrus_dim: int = 512

    ca3_dim: int = 512

    ca1_dim: int = 256

    subiculum_dim: int = 128

    dg_sparsity: float = 0.04

    ca3_max_episodes: int = 64

    ca3_settle_steps: int = 5

    ca3_recurrent_gain: float = 0.7

    ca1_novelty_threshold: float = 0.3

    ec_buffer_tau: float = 0.5

    astro_glutamate_tau: float = 0.95

    astro_calcium_gain: float = 4.0

    astro_calcium_target: float = 0.5

    astro_eta_min: float = 0.1
    astro_eta_max: float = 3.0

    anatomical_layout: bool = True

    ca3_distance_sigma: float = 2.0

class NeuronPositionRegistry(nn.Module):

    def __init__(self, cfg: CognitiveKernelConfig) -> None:
        super().__init__()

        total_neurons = (
            cfg.coordinate_dim       # EC input layer
            + cfg.dentate_gyrus_dim  # DG
            + cfg.ca3_dim            # CA3
            + cfg.ca1_dim            # CA1
            + cfg.subiculum_dim      # Subiculum
        )

        if cfg.anatomical_layout:
            positions = self._anatomical_positions(cfg)
        else:
            positions = torch.rand(total_neurons, 3) * 10.0

        self.register_buffer("positions", positions)

        self._boundaries = self._compute_boundaries(cfg)
        self.total_neurons = total_neurons

    @staticmethod
    def _compute_boundaries(
        cfg: CognitiveKernelConfig,
    ) -> Dict[str, Tuple[int, int]]:
        offset = 0
        boundaries = {}
        for name, size in [
            ("ec", cfg.coordinate_dim),
            ("dg", cfg.dentate_gyrus_dim),
            ("ca3", cfg.ca3_dim),
            ("ca1", cfg.ca1_dim),
            ("subiculum", cfg.subiculum_dim),
        ]:
            boundaries[name] = (offset, offset + size)
            offset += size
        return boundaries

    @staticmethod
    def _anatomical_positions(
        cfg: CognitiveKernelConfig,
    ) -> torch.Tensor:
        positions = []
        gen = torch.Generator().manual_seed(0)

        n_ec = cfg.coordinate_dim
        ec_pos = torch.zeros(n_ec, 3)
        ec_pos[:, 0] = torch.linspace(-3.0, 3.0, n_ec)
        ec_pos[:, 1] = 0.0 + torch.randn(n_ec, generator=gen) * 0.2
        ec_pos[:, 2] = torch.randn(n_ec, generator=gen) * 0.5
        positions.append(ec_pos)

        n_dg = cfg.dentate_gyrus_dim
        dg_pos = torch.zeros(n_dg, 3)
        dg_pos[:, 0] = torch.randn(n_dg, generator=gen) * 2.0
        dg_pos[:, 1] = 2.0 + torch.randn(n_dg, generator=gen) * 0.3
        dg_pos[:, 2] = torch.randn(n_dg, generator=gen) * 1.5
        positions.append(dg_pos)

        n_ca3 = cfg.ca3_dim
        ca3_pos = torch.zeros(n_ca3, 3)
        theta = torch.linspace(0, math.pi, n_ca3)
        ca3_pos[:, 0] = 3.0 * torch.cos(theta)
        ca3_pos[:, 1] = 4.0 + torch.sin(theta) * 1.0
        ca3_pos[:, 2] = torch.randn(n_ca3, generator=gen) * 0.8
        ca3_pos += torch.randn_like(ca3_pos) * 0.15
        positions.append(ca3_pos)

        n_ca1 = cfg.ca1_dim
        ca1_pos = torch.zeros(n_ca1, 3)
        ca1_pos[:, 0] = torch.linspace(-2.5, 2.5, n_ca1)
        ca1_pos[:, 1] = 6.0 + torch.randn(n_ca1, generator=gen) * 0.25
        ca1_pos[:, 2] = torch.randn(n_ca1, generator=gen) * 0.6
        positions.append(ca1_pos)

        n_sub = cfg.subiculum_dim
        sub_pos = torch.zeros(n_sub, 3)
        sub_pos[:, 0] = torch.linspace(-2.0, 2.0, n_sub)
        sub_pos[:, 1] = 7.5 + torch.randn(n_sub, generator=gen) * 0.2
        sub_pos[:, 2] = torch.randn(n_sub, generator=gen) * 0.4
        positions.append(sub_pos)

        return torch.cat(positions, dim=0)

    def get_population_positions(self, name: str) -> torch.Tensor:
        start, end = self._boundaries[name]
        return self.positions[start:end]

    def pairwise_distances(self, name: str) -> torch.Tensor:
        pos = self.get_population_positions(name)
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)
        return torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)

class DentateGyrus(nn.Module):

    def __init__(self, cfg: CognitiveKernelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.perforant_path = nn.Linear(
            cfg.coordinate_dim, cfg.dentate_gyrus_dim, bias=True,
        )

        self.norm = nn.LayerNorm(cfg.dentate_gyrus_dim)

        self.k_active = max(1, int(cfg.dentate_gyrus_dim * cfg.dg_sparsity))

    def forward(self, ec_input: torch.Tensor) -> torch.Tensor:
        projected = self.norm(F.relu(self.perforant_path(ec_input)))

        topk_vals, topk_idx = projected.topk(self.k_active, dim=-1)
        sparse = torch.zeros_like(projected)
        sparse.scatter_(1, topk_idx, topk_vals)

        return sparse

class CA3AttractorNetwork(nn.Module):

    def __init__(
        self, cfg: CognitiveKernelConfig, positions: NeuronPositionRegistry,
    ) -> None:
        super().__init__()
        self.cfg = cfg

        self.mossy_fiber = nn.Linear(cfg.dentate_gyrus_dim, cfg.ca3_dim, bias=False)

        self.register_buffer(
            "W_recurrent",
            torch.zeros(cfg.ca3_dim, cfg.ca3_dim),
        )

        ca3_distances = positions.pairwise_distances("ca3")
        distance_mask = torch.exp(-ca3_distances / cfg.ca3_distance_sigma)
        self.register_buffer("distance_mask", distance_mask)

        self._stored_patterns: List[torch.Tensor] = []

        self.schaffer_collateral = nn.Linear(cfg.ca3_dim, cfg.ca1_dim, bias=True)

    def store_episode(self, dg_pattern: torch.Tensor) -> int:
        pattern = dg_pattern.detach().clone()
        pattern = pattern / (pattern.norm() + 1e-8)

        if len(self._stored_patterns) >= self.cfg.ca3_max_episodes:
            self._stored_patterns.pop(0)

        self._stored_patterns.append(pattern)
        self._recompute_weights()
        return len(self._stored_patterns)

    def _recompute_weights(self) -> None:
        if not self._stored_patterns:
            self.W_recurrent.zero_()
            return

        device = self.W_recurrent.device
        patterns = torch.stack(
            [p.to(device) for p in self._stored_patterns], dim=0,
        )
        W = patterns.T @ patterns
        W.fill_diagonal_(0.0)
        W = W * self.distance_mask
        self.W_recurrent.copy_(W)

    def retrieve(self, cue: torch.Tensor) -> torch.Tensor:
        state = cue
        gain = self.cfg.ca3_recurrent_gain

        for _ in range(self.cfg.ca3_settle_steps):
            recurrent = F.linear(state, self.W_recurrent)
            state = torch.tanh(gain * recurrent + (1.0 - gain) * cue)

        return state

    def forward(
        self, dg_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ca3_cue = self.mossy_fiber(dg_output)
        ca3_state = self.retrieve(ca3_cue)
        ca1_input = self.schaffer_collateral(ca3_state)

        return ca3_state, ca1_input

class CA1Comparator(nn.Module):

    def __init__(self, cfg: CognitiveKernelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.temporoammonic = nn.Linear(
            cfg.coordinate_dim, cfg.ca1_dim, bias=True,
        )

        self.compare_schaffer = nn.Linear(cfg.ca1_dim, cfg.ca1_dim, bias=False)
        self.compare_direct = nn.Linear(cfg.ca1_dim, cfg.ca1_dim, bias=False)

        self.output_gate = nn.Linear(cfg.ca1_dim, cfg.ca1_dim, bias=True)

    def forward(
        self,
        schaffer_input: torch.Tensor,
        ec_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        direct = self.temporoammonic(ec_input)

        s_proj = self.compare_schaffer(schaffer_input)
        d_proj = self.compare_direct(direct)

        cos_sim = F.cosine_similarity(s_proj, d_proj, dim=-1)
        novelty = (1.0 - cos_sim).clamp(0.0, 1.0)

        novelty_gate = novelty.unsqueeze(-1)
        combined = novelty_gate * direct + (1.0 - novelty_gate) * schaffer_input
        ca1_output = torch.tanh(self.output_gate(combined))

        return ca1_output, novelty

class Subiculum(nn.Module):

    def __init__(self, cfg: CognitiveKernelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.ca1_to_sub = nn.Linear(cfg.ca1_dim, cfg.subiculum_dim, bias=True)
        self.sub_to_ec = nn.Linear(
            cfg.subiculum_dim, cfg.coordinate_dim, bias=True,
        )
        self.gate = nn.Linear(cfg.subiculum_dim, cfg.subiculum_dim, bias=True)

    def forward(self, ca1_output: torch.Tensor) -> torch.Tensor:
        sub_state = F.relu(self.ca1_to_sub(ca1_output))
        gate_signal = torch.sigmoid(self.gate(sub_state))
        gated = sub_state * gate_signal
        reconstructed = self.sub_to_ec(gated)
        return reconstructed

class AstrocyticRegulator(nn.Module):

    def __init__(self, cfg: CognitiveKernelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.register_buffer(
            "extrasynaptic_glutamate", torch.tensor(0.0),
        )
        self.register_buffer(
            "astrocytic_calcium", torch.tensor(cfg.astro_calcium_target),
        )

    @torch.no_grad()
    def update(self, population_activity: torch.Tensor) -> float:
        activity = population_activity.detach().abs().mean().item()

        tau = self.cfg.astro_glutamate_tau
        self.extrasynaptic_glutamate.copy_(
            tau * self.extrasynaptic_glutamate + (1.0 - tau) * activity
        )

        calcium_drive = torch.sigmoid(
            torch.tensor(
                self.cfg.astro_calcium_gain
                * (self.extrasynaptic_glutamate.item()
                   - self.cfg.astro_calcium_target)
            )
        ).item()
        self.astrocytic_calcium.copy_(
            tau * self.astrocytic_calcium + (1.0 - tau) * calcium_drive
        )

        deviation = self.astrocytic_calcium.item() - self.cfg.astro_calcium_target
        eta = 1.0 + deviation
        eta = max(self.cfg.astro_eta_min, min(self.cfg.astro_eta_max, eta))
        return eta

    def get_state(self) -> Dict[str, float]:
        return {
            "glutamate": self.extrasynaptic_glutamate.item(),
            "calcium": self.astrocytic_calcium.item(),
        }

class EntorhinalCortex(nn.Module):

    def __init__(self, cfg: CognitiveKernelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.register_buffer(
            "short_term_buffer",
            torch.zeros(cfg.coordinate_dim),
        )

        self.input_norm = nn.LayerNorm(cfg.coordinate_dim)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            batch_mean = coords.detach().mean(dim=0)
            tau = self.cfg.ec_buffer_tau
            self.short_term_buffer.copy_(
                tau * self.short_term_buffer + (1.0 - tau) * batch_mean
            )

        biased = coords + 0.1 * self.short_term_buffer.unsqueeze(0)
        return self.input_norm(biased)

    def reset_short_term(self) -> None:
        with torch.no_grad():
            self.short_term_buffer.zero_()

class CognitiveKernel(nn.Module):

    def __init__(self, cfg: Optional[CognitiveKernelConfig] = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = CognitiveKernelConfig()
        self.cfg = cfg

        self.neuron_positions = NeuronPositionRegistry(cfg)

        self.entorhinal_cortex = EntorhinalCortex(cfg)
        self.dentate_gyrus = DentateGyrus(cfg)
        self.ca3 = CA3AttractorNetwork(cfg, self.neuron_positions)
        self.ca1 = CA1Comparator(cfg)
        self.subiculum = Subiculum(cfg)
        self.astrocyte = AstrocyticRegulator(cfg)

        self.register_buffer(
            "working_memory", torch.zeros(cfg.ca1_dim),
        )

    def forward(
        self, coords: torch.Tensor, store_if_novel: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        diagnostics: Dict[str, float] = {}

        ec_output = self.entorhinal_cortex(coords)

        dg_sparse = self.dentate_gyrus(ec_output)
        dg_sparsity = (dg_sparse == 0).float().mean().item()
        diagnostics["dg_sparsity"] = dg_sparsity

        ca3_state, schaffer_to_ca1 = self.ca3(dg_sparse)

        ca1_output, novelty = self.ca1(schaffer_to_ca1, ec_output)

        with torch.no_grad():
            self.working_memory.copy_(ca1_output.detach().mean(dim=0))

        reconstructed = self.subiculum(ca1_output)

        eta = self.astrocyte.update(ca3_state)
        diagnostics["astro_eta"] = eta
        diagnostics.update(self.astrocyte.get_state())

        mean_novelty = novelty.mean().item()
        diagnostics["mean_novelty"] = mean_novelty
        diagnostics["num_stored_episodes"] = len(self.ca3._stored_patterns)

        if store_if_novel and mean_novelty > self.cfg.ca1_novelty_threshold:
            mean_dg = dg_sparse.detach().mean(dim=0)
            mossy_pattern = self.ca3.mossy_fiber(mean_dg)
            n_stored = self.ca3.store_episode(mossy_pattern)
            diagnostics["num_stored_episodes"] = n_stored
            diagnostics["stored_this_step"] = True
        else:
            diagnostics["stored_this_step"] = False

        return reconstructed, novelty, diagnostics

    def retrieve_from_cue(self, partial_coords: torch.Tensor) -> torch.Tensor:
        ec_output = self.entorhinal_cortex(partial_coords)
        dg_sparse = self.dentate_gyrus(ec_output)
        ca3_state, schaffer = self.ca3(dg_sparse)
        ca1_output, _ = self.ca1(schaffer, ec_output)
        return self.subiculum(ca1_output)

    def sleep_consolidation(self) -> Dict[str, float]:
        diagnostics: Dict[str, float] = {}

        self.entorhinal_cortex.reset_short_term()

        with torch.no_grad():
            self.working_memory.zero_()

        diagnostics["episodes_in_ca3"] = len(self.ca3._stored_patterns)
        diagnostics["short_term_cleared"] = True
        diagnostics["working_memory_cleared"] = True

        return diagnostics

    def get_neuron_positions(self) -> Dict[str, torch.Tensor]:
        return {
            name: self.neuron_positions.get_population_positions(name)
            for name in ["ec", "dg", "ca3", "ca1", "subiculum"]
        }

    def get_memory_state(self) -> Dict[str, object]:
        return {
            "short_term_buffer": (
                self.entorhinal_cortex.short_term_buffer.detach().cpu().clone()
            ),
            "working_memory": self.working_memory.detach().cpu().clone(),
            "ca3_stored_patterns": [
                p.detach().cpu().clone() for p in self.ca3._stored_patterns
            ],
            "ca3_W_recurrent": self.ca3.W_recurrent.detach().cpu().clone(),
            "astro_state": self.astrocyte.get_state(),
        }

    def load_memory_state(self, state: Dict[str, object]) -> None:
        with torch.no_grad():
            device = self.ca3.W_recurrent.device

            if "short_term_buffer" in state:
                self.entorhinal_cortex.short_term_buffer.copy_(
                    state["short_term_buffer"].to(device)
                )
            if "working_memory" in state:
                self.working_memory.copy_(
                    state["working_memory"].to(device)
                )
            if "ca3_stored_patterns" in state:
                self.ca3._stored_patterns = [
                    p.to(device) for p in state["ca3_stored_patterns"]
                ]
            if "ca3_W_recurrent" in state:
                self.ca3.W_recurrent.copy_(
                    state["ca3_W_recurrent"].to(device)
                )

    def count_params(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        ec = sum(
            p.numel() for n, p in self.named_parameters()
            if "entorhinal" in n
        )
        dg = sum(
            p.numel() for n, p in self.named_parameters()
            if "dentate" in n
        )
        ca3 = sum(
            p.numel() for n, p in self.named_parameters()
            if "ca3" in n
        )
        ca1 = sum(
            p.numel() for n, p in self.named_parameters()
            if "ca1" in n
        )
        sub = sum(
            p.numel() for n, p in self.named_parameters()
            if "subiculum" in n
        )
        return (
            f"Cognitive Kernel: {total / 1e3:.1f}K parameters\n"
            f"  Entorhinal Cortex: {ec / 1e3:.1f}K\n"
            f"  Dentate Gyrus:     {dg / 1e3:.1f}K\n"
            f"  CA3 Attractor:     {ca3 / 1e3:.1f}K\n"
            f"  CA1 Comparator:    {ca1 / 1e3:.1f}K\n"
            f"  Subiculum:         {sub / 1e3:.1f}K\n"
            f"  3D Positions:      "
            f"{self.neuron_positions.total_neurons} neurons"
        )

if __name__ == "__main__":
    print("=" * 60)
    print("Cognitive Kernel Core: Self-Test")
    print("=" * 60)

    cfg = CognitiveKernelConfig()
    kernel = CognitiveKernel(cfg)
    print(kernel.count_params())

    coords = torch.randn(4, 64)
    recon, novelty, diag = kernel(coords)
    assert recon.shape == (4, 64), f"Expected (4,64), got {recon.shape}"
    assert novelty.shape == (4,), f"Expected (4,), got {novelty.shape}"
    print(f"\n[PASS] Forward: recon={tuple(recon.shape)}, "
          f"novelty={novelty.mean():.4f}")
    print(f"       DG sparsity: {diag['dg_sparsity']:.4f}")
    print(f"       Astro eta:   {diag['astro_eta']:.4f}")
    print(f"       Stored:      {diag['num_stored_episodes']} episodes")

    for i in range(10):
        c = torch.randn(4, 64) * (1.0 + i * 0.1)
        _, _, d = kernel(c)
    print(f"\n[PASS] After 10 steps: {d['num_stored_episodes']} episodes stored")

    full = torch.randn(1, 64)
    kernel(full, store_if_novel=True)
    noisy = full + torch.randn_like(full) * 0.3
    retrieved = kernel.retrieve_from_cue(noisy)
    assert retrieved.shape == (1, 64)
    print(f"[PASS] Retrieval from noisy cue: shape={tuple(retrieved.shape)}")

    positions = kernel.get_neuron_positions()
    for name, pos in positions.items():
        assert pos.shape[1] == 3, f"{name} positions not 3D"
    total_neurons = sum(p.shape[0] for p in positions.values())
    print(f"[PASS] 3D positions: {total_neurons} neurons across "
          f"{len(positions)} populations")

    sleep_diag = kernel.sleep_consolidation()
    assert kernel.entorhinal_cortex.short_term_buffer.abs().sum() == 0.0
    assert kernel.working_memory.abs().sum() == 0.0
    print(f"[PASS] Sleep: short-term cleared, working memory cleared, "
          f"{sleep_diag['episodes_in_ca3']} episodes preserved in CA3")

    state = kernel.get_memory_state()
    kernel.sleep_consolidation()
    kernel.load_memory_state(state)
    assert len(kernel.ca3._stored_patterns) == len(state["ca3_stored_patterns"])
    print(f"[PASS] Memory state round-trip: "
          f"{len(kernel.ca3._stored_patterns)} patterns restored")

    coords = torch.randn(2, 64)
    kernel(coords)
    st = kernel.entorhinal_cortex.short_term_buffer.norm().item()
    wm = kernel.working_memory.norm().item()
    lt = len(kernel.ca3._stored_patterns)
    print(f"[PASS] Three tiers active: "
          f"short-term={st:.4f}, working={wm:.4f}, "
          f"long-lasting={lt} episodes")

    print(f"\n{'=' * 60}")
    print("All tests passed.")
    print(f"{'=' * 60}")
