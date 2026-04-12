"""
timmy/experts.py
Spike-Driven Mixture of Experts with Cluster-Based Routing

BIOLOGICAL GROUNDING:
This file implements the Mixture of Experts (MoE) layer used in Timmy's
association zone. In biological cortex, the association areas contain
functionally specialized subpopulations that process different aspects of
input (e.g. different object features, spatial relations, semantic
categories). Incoming activity is routed to the relevant subpopulation
based on the content of the input, not uniformly broadcast.

The routing mechanism uses minicolumn cluster firing rates (from the
AssociativeLIF cascade amplification) as the routing signal. This means
expert selection is driven by population-level spike statistics, not by
a separate learned routing network operating on raw activations. The
cluster-to-expert mapping divides the minicolumn array evenly across
experts, so each expert "owns" a contiguous block of clusters.

The load balancing loss prevents expert collapse (where the router learns
to send all tokens to one or two experts and the rest are never used).

Key grounding papers:
1. Shazeer N, Mirhoseini A, Maziarz K, Davis A, Le Q, Hinton G, Dean J
   (2017). "Outrageously large neural networks: The sparsely-gated mixture
   of experts layer." ICLR 2017. DOI: 10.48550/arXiv.1701.06538

2. Felleman DJ, Van Essen DC (1991). "Distributed hierarchical processing
   in the primate cerebral cortex." Cerebral Cortex, 1(1):1-47.
   DOI: 10.1093/cercor/1.1.1
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Tuple

from timmy_neuron import AssociativeLIF, NeuronConfig


# =========================================================================
# Spiking Expert Group
# =========================================================================

class SpikingExpertGroup(nn.Module):
    """
    Bank of parallel feed-forward expert networks with LIF nonlinearities.

    Each expert is a standard up-projection / down-projection FFN where both
    the hidden and output activations pass through LIF neurons. The dispatch
    loops over experts (not tokens), which avoids materializing all expert
    outputs simultaneously (FIX A: memory-efficient dispatch).

    NOT a biological structure. Engineering implementation of parallel
    specialized processing.

    Reference: Shazeer N et al. (2017). "Outrageously large neural networks."
    ICLR 2017. DOI: 10.48550/arXiv.1701.06538
    """

    def __init__(self, n_experts: int, d_model: int, d_ff: int, neuron_cfg: NeuronConfig = None):
        """
        Args:
            n_experts: number of parallel experts.
            d_model: input/output dimensionality.
            d_ff: total feed-forward width (divided evenly across experts).
            neuron_cfg: neuron config for expert LIF populations.
        """
        super().__init__()
        if neuron_cfg is None:
            neuron_cfg = NeuronConfig()
        self.n_experts = n_experts
        self.expert_ff = d_ff // n_experts
        ef = self.expert_ff
        self.up = nn.ModuleList([nn.Linear(d_model, ef, bias=False) for _ in range(n_experts)])
        self.down = nn.ModuleList([nn.Linear(ef, d_model, bias=False) for _ in range(n_experts)])
        self.lif_hidden = AssociativeLIF(ef, neuron_cfg)
        self.lif_output = AssociativeLIF(d_model, neuron_cfg)

    def forward(self, x: Tensor, expert_indices: Tensor, expert_weights: Tensor) -> Tensor:
        """
        Dispatch tokens to experts and aggregate weighted outputs.

        Args:
            x: (T, N, D) input tensor (N = B*S flattened).
            expert_indices: (N, top_k) per-token expert assignments.
            expert_weights: (N, top_k) per-token expert weights (softmax).

        Returns:
            (T, N, D) weighted sum of expert outputs.
        """
        T, N, D = x.shape
        top_k = expert_indices.shape[1]
        output = torch.zeros_like(x)

        # Loop over experts, not tokens (FIX A: memory-efficient dispatch).
        # For each expert, find which tokens are assigned to it, process only
        # those tokens, and scatter the weighted results back.
        for e in range(self.n_experts):
            mask = torch.zeros(N, device=x.device, dtype=x.dtype)
            for k in range(top_k):
                is_e = (expert_indices[:, k] == e).to(x.dtype)
                mask = mask + is_e * expert_weights[:, k]
            if mask.sum() == 0:
                continue
            active = mask > 0
            if not active.any():
                continue
            active_x = x[:, active, :]
            Ta, Na, Da = active_x.shape
            h = self.up[e](active_x.reshape(Ta * Na, Da)).reshape(Ta, Na, -1)
            h, _ = self.lif_hidden(h)
            o = self.down[e](h.reshape(Ta * Na, -1)).reshape(Ta, Na, Da)
            o, _ = self.lif_output(o)
            w = mask[active].unsqueeze(0).unsqueeze(-1)
            output[:, active, :] += o * w

        return output


# =========================================================================
# Spike-Driven Mixture of Experts
# =========================================================================

class SpikeDrivenMoE(nn.Module):
    """
    Mixture of Experts router that uses minicolumn spike rates for
    expert selection.

    BIOLOGICAL STRUCTURE: Association cortex with functionally specialized
    subpopulations. Input is routed to the relevant subpopulation based on
    the pattern of population-level activity.

    BIOLOGICAL FUNCTION: The routing signal comes from minicolumn cluster
    firing rates (the same clusters used by cascade amplification in
    AssociativeLIF). Each expert "owns" a contiguous block of clusters:
    expert e receives clusters [e*cpe, (e+1)*cpe) where cpe = n_clusters //
    n_experts. The mean firing rate across an expert's owned clusters
    determines how strongly that expert is activated for a given token.

    Reference: Felleman DJ, Van Essen DC (1991). "Distributed hierarchical
    processing in the primate cerebral cortex." Cerebral Cortex, 1(1):1-47.
    DOI: 10.1093/cercor/1.1.1

    Load balancing loss (FIX G): penalizes uneven expert utilization.
    Without this, the router collapses to sending all tokens to one or
    two experts (winner-take-all instability).
    Reference: Shazeer N et al. (2017). DOI: 10.48550/arXiv.1701.06538
    """

    def __init__(self, cfg, neuron_cfg: NeuronConfig = None):
        """
        Args:
            cfg: model-level config providing n_experts, top_k_experts,
                n_clusters, d_model, d_ff, moe_route_temperature,
                moe_load_balance_weight.
            neuron_cfg: neuron config for the routing LIF and expert LIFs.
        """
        super().__init__()
        if neuron_cfg is None:
            neuron_cfg = NeuronConfig()
        self.cfg = cfg
        self.n_experts = cfg.n_experts
        self.top_k = cfg.top_k_experts
        self.clusters_per_expert = cfg.n_clusters // cfg.n_experts
        self.expert_group = SpikingExpertGroup(
            cfg.n_experts, cfg.d_model, cfg.d_ff, neuron_cfg
        )
        # Routing LIF: produces the spike trains from which cluster firing
        # rates are computed for expert scoring.
        self.route_lif = AssociativeLIF(cfg.d_model, neuron_cfg)
        self.expert_bias = nn.Parameter(torch.zeros(cfg.n_experts))
        self.register_buffer(
            "expert_counts_ema",
            torch.ones(cfg.n_experts) / cfg.n_experts,
        )

    def _compute_expert_scores(self, spikes: Tensor) -> Tensor:
        """
        Compute per-token expert scores from minicolumn firing rates.

        Takes the mean firing rate across timesteps, aggregates into cluster
        rates, groups clusters by expert ownership, and produces a score
        per expert per token.

        Args:
            spikes: (T, N, D) spike tensor from the routing LIF.

        Returns:
            (N, n_experts) expert score logits.
        """
        fr = spikes.mean(dim=0)  # (N, D)
        N, D = fr.shape
        nc = self.cfg.n_clusters
        cid = torch.arange(D, device=fr.device) % nc

        # Per-cluster mean firing rate.
        cr = torch.zeros(N, nc, device=fr.device, dtype=fr.dtype)
        cr.scatter_add_(1, cid.unsqueeze(0).expand(N, -1), fr)
        cr = cr / max(D // nc, 1)

        # Group clusters by expert: mean rate per expert's cluster block.
        es = cr.reshape(N, self.n_experts, self.clusters_per_expert).mean(dim=-1)
        es = es / max(self.cfg.moe_route_temperature, 0.01)
        return es + self.expert_bias.to(es.dtype)

    def _load_balance_loss(self, scores: Tensor, top_idx: Tensor) -> Tensor:
        """
        Load balancing loss (FIX G): n_experts * sum(fraction * probability).

        Penalizes configurations where a few experts receive most tokens.

        Args:
            scores: (N, n_experts) expert score logits.
            top_idx: (N, top_k) selected expert indices.

        Returns:
            Scalar loss.
        """
        N = scores.shape[0]
        ef = torch.zeros(self.n_experts, device=scores.device)
        for e in range(self.n_experts):
            ef[e] = (top_idx == e).float().sum() / (N * self.top_k)
        rp = F.softmax(scores, dim=-1).mean(dim=0)
        loss = self.n_experts * (ef * rp).sum()
        with torch.no_grad():
            self.expert_counts_ema.lerp_(ef, 0.01)
        return loss

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict]:
        """
        Route input through the MoE layer.

        Args:
            x: (T, B, S, D) spiking tensor from previous block.

        Returns:
            output: (T, B, S, D) expert-processed tensor.
            stats: dict with routing entropy, load balance loss, per-expert load.
        """
        T, B, S, D = x.shape
        N = B * S
        xf = x.reshape(T, N, D)

        # Routing: pass through dedicated LIF, compute cluster-based scores.
        rs, _ = self.route_lif(xf)
        es = self._compute_expert_scores(rs)
        ts, ti = torch.topk(es, self.top_k, dim=-1)
        tw = F.softmax(ts.float(), dim=-1).to(x.dtype)

        # Dispatch to experts.
        output = self.expert_group(xf, ti, tw).reshape(T, B, S, D)

        # Load balancing loss and diagnostics.
        lb = self._load_balance_loss(es, ti)
        stats = {
            "moe_route_entropy": -(
                F.softmax(es, dim=-1) * F.log_softmax(es + 1e-8, dim=-1)
            ).sum(-1).mean().item(),
            "moe_load_balance_loss": lb,
        }
        with torch.no_grad():
            for e in range(self.n_experts):
                stats[f"expert_{e}_load"] = self.expert_counts_ema[e].item()

        return output, stats
