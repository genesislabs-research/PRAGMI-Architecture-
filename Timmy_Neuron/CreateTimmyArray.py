"""
CreateTimmyArray.py
The Cortical Column Ensemble: TimmyPrime and Specialist Columns
Sharing a Unified Cognitive Kernel

BIOLOGICAL GROUNDING:
This file assembles the TimmyArray, a cortical column ensemble in which one
broadband integration column (TimmyPrime) and a set of named specialist
columns share a single Cognitive Kernel as their common hippocampal memory
substrate. The architecture is grounded in the Thousand Brains Theory and
the empirical literature on coordinated neuronal ensembles in cortical columns.

The biological model is the neocortical column ensemble. Each cortical column
is a complete sensory-motor modeling system that builds its own model of the
world by assigning sensory data to locations in reference frames. Columns
develop orientation-specific or domain-specific tuning through
experience-dependent plasticity, not through architectural constraints. The
integration column (Prime) maintains the broadest receptive field and
coordinates the ensemble by providing the routing signal that determines
which specialists are engaged for any given input.

The voting and coordination mechanism follows Hawkins et al. (2019): columns
share information through long-range horizontal connections to reach consensus,
but each column runs its own complete model in parallel. Here Prime's
post-MemoryCortex representation serves as the coordination signal because
the MemoryCortex (modeled on prefrontal delay-period neurons) integrates
the full temporal context before the executive decision is committed. This
extracts routing information that is richer than raw MoE cluster assignments
but does not force Prime's executive zone to complete before specialists
begin, preserving executive-zone parallelism.

The specialist columns are named by the temporal integration scale and
processing mode they develop tuning for. Column identity is defined by
connectivity fingerprint and receptive field statistics, exactly as
biological cortical columns are defined. Specialization does not emerge
from training-phase data assignment. It emerges from sleep-cycle pruning
and consolidation after deployment, via firstday.py and sleep.py. At
construction time all specialists are uninitialized; clone_prime_to_specialists()
copies Prime's trained weights into every specialist simultaneously before
the first day cycle begins. This is the initialization event, not a training
phase. The sleep cycle then drives divergence from that shared starting point
through experience-dependent synaptic tagging and overnight consolidation.

    prime     -- Always-active integration column. Broadband receptive field.
                 Coordinates the ensemble. Analog: prefrontal coordination
                 of posterior cortical areas.
    proximal  -- Tuned to near-term temporal context, tight dependency
                 windows, fine syntactic/sequential structure. Analog:
                 premotor cortex handling immediate sequential planning.
    distal    -- Tuned to long-range dependency resolution, discourse
                 coherence, semantic integration across separated spans.
                 Analog: posterior parietal cortex integrating over extended
                 temporal windows.
    affective -- Tuned to valenced content, social reasoning, motivational
                 framing. Provides the emotional metadata that PRAGMI attaches
                 to episodic memory during reconsolidation. Analog: cingulate
                 cortex mediating affective evaluation of experience.
    somatic   -- Tuned to grounded, sensorimotor, embodied content. Primary
                 interface for Robby's embodied experience stream.
                 Analog: somatosensory/motor cortex.
    structural -- Tuned to formal relational structure: code, mathematical
                  schemas, logical composition, compositional hierarchies.
                  Analog: left inferior frontal gyrus handling hierarchical
                  syntactic structure.

The PerforantPath receives a list of column outputs and reduces them to
a single coordinate-space vector via learned attention-weighted pooling.
This preserves the 64-dimensional coordinate manifold regardless of how
many specialists are active. The biological grounding is the Semedo et al.
communication subspace: the hippocampus receives the integrated output of
all open communication subspaces, not raw zone activity.

Each active specialist delivers its output through its own PerforantPathBridge
instance. The attention weights are learned by a single-head attention
module whose query is derived from Prime's post-MemoryCortex representation,
making Prime the coordinator of what the kernel hears.

CA1 mismatch feedback is sent to exactly two recipients: Prime (always, because
Prime's routing decisions require ongoing reinforcement) and the specialist
whose routing weight was highest for the current episode. This enforces the
one-writer-per-episode constraint that keeps the CA3 memory stable.

Three grounding papers for the full file:

1. Hawkins J, Lewis M, Klukas M, Purdy S, Ahmad S (2019). "A framework
   for intelligence and cortical function based on grid cells in the
   neocortex." Frontiers in Neural Circuits, 12:121.
   DOI: 10.3389/fncir.2018.00121
   (Thousand Brains Theory: every column is a complete world model; columns
   vote to reach consensus; Prime as the broadband integration column.)

2. See JZ, Atencio CA, Sohal VS, Schreiner CE (2018). "Coordinated neuronal
   ensembles in primary auditory cortical columns." eLife, 7:e35587.
   DOI: 10.7554/eLife.35587
   (Column identity is defined by higher-order correlation structure, not
   pairwise distance or receptive field overlap alone. Ensemble membership
   is detected via PCA/ICA on the spike correlation matrix. cNE events
   carry ~10% more mutual information than individual spikes. Routing based
   on cluster firing rates is grounded here.)

3. Semedo JD, Zandvakili A, Machens CK, Yu BM, Kohn A (2019). "Cortical
   areas interact through a communication subspace." Neuron, 102(1):249-259.
   DOI: 10.1016/j.neuron.2019.01.026
   (The kernel receives only the predictive dimensions of each column's
   activity, not the raw population vector. The attention-weighted PerforantPath
   pooling implements a dynamic version of this: the query from Prime selects
   which column's predictive dimensions dominate the kernel input at each step.)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from timmy_model import TimmyModel, TimmyConfig
from timmy_state import save_timmy_state, load_timmy_state, ArchitectureMismatchError


# =========================================================================
# Column Identity Registry
# =========================================================================

# The six named specialist columns plus Prime. These names are stable
# identifiers for checkpoint keys, ModuleDict keys, and DataLoader
# assignment. They do NOT change when training data changes.
# NOT a biological quantity. Engineering registry.
COLUMN_NAMES: List[str] = [
    "prime",
    "proximal",
    "distal",
    "affective",
    "somatic",
    "structural",
]

# Prime is always active. Specialists are conditionally active based on
# the routing signal. The prime index is 0 by convention.
PRIME_INDEX: int = 0


# =========================================================================
# Configuration
# =========================================================================

@dataclass
class TimmyArrayConfig:
    """
    Configuration for the TimmyArray cortical column ensemble.

    Every parameter that has a biological analog cites its source.
    Parameters that are engineering approximations are labeled explicitly.
    """

    # ---- Column configuration ----

    # Shared TimmyConfig applied to all columns. Prime and all specialists
    # are structurally identical at construction. After train_prime.py
    # completes, firstday.py calls clone_prime_to_specialists() to copy
    # Prime's trained weights into every specialist simultaneously.
    # Functional differentiation then emerges through sleep-cycle pruning
    # and consolidation, not through training-phase data assignment.
    # This mirrors the canonical cortical column: structurally uniform,
    # functionally differentiated through experience.
    # Reference: Mountcastle VB (1997). "The columnar organization of the
    # neocortex." Brain, 120(4):701-722. DOI: 10.1093/brain/120.4.701
    column_cfg: TimmyConfig = field(default_factory=TimmyConfig)

    # Number of specialist columns (excludes Prime).
    # NOT a biological quantity. Engineering choice. Default 5 gives the
    # six named columns: prime + proximal + distal + affective + somatic
    # + structural.
    num_specialists: int = 5

    # ---- Routing ----

    # Rank of the routing projection from Prime's post-MemoryCortex
    # representation to the routing score vector. Low rank enforces
    # the communication subspace constraint: specialists hear only the
    # predictive dimensions of Prime's working memory state.
    # Reference: Semedo JD et al. (2019). DOI: 10.1016/j.neuron.2019.01.026
    routing_rank: int = 4

    # Temperature for routing softmax. Lower values produce harder routing
    # (more winner-take-all). Higher values produce softer routing
    # (more ensemble averaging).
    # NOT a biological quantity. Training hyperparameter.
    routing_temperature: float = 1.0

    # Number of specialists active per forward pass during training.
    # At inference, top-1 is used. During training, top-k with
    # straight-through gradients allows multiple specialists to receive
    # gradient signal.
    # NOT a biological quantity. Training artifact.
    routing_top_k_train: int = 2

    # ---- PerforantPath symphony pooling ----

    # Dimensionality of the coordinate manifold. Must match
    # HippocampalConfig.coordinate_dim in the Cognitive Kernel.
    # NOT a biological quantity. Engineering choice.
    # Reference: Sainburg T et al. (2021). DOI: 10.1162/neco_a_01434
    coordinate_dim: int = 64

    # Rank of each column-to-kernel communication subspace. Each active
    # column has its own PerforantPathBridge with this rank. Rank 3 follows
    # Semedo et al. (2019) finding of rank 2-3 subspaces between V1 and V2.
    # Reference: Semedo JD et al. (2019). DOI: 10.1016/j.neuron.2019.01.026
    perforant_rank: int = 3

    # ---- CA1 feedback routing ----

    # Scale applied to the novelty scalar before injecting it into the
    # winning specialist's STDPEngine via set_external_reward().
    # NOT a biological quantity. Engineering gain.
    ca1_reward_scale: float = 1.0

    # CA1 feedback always goes to Prime regardless of routing outcome.
    # This flag is exposed to allow ablation studies.
    # NOT a biological quantity. Engineering switch.
    prime_always_receives_ca1: bool = True

    # ---- Load balancing ----

    # Per-specialist bias terms for routing, analogous to DeepSeek-V3's
    # auxiliary-loss-free load balancing. Adjusted dynamically during training
    # to prevent routing collapse without corrupting the routing gradient.
    # Reference: Wang P et al. (2024). "Auxiliary-loss-free load balancing
    # strategy for mixture of experts." arXiv: 2408.15664.
    # DOI: {To be added later.}
    # NOT a biological quantity. Training stability mechanism.
    load_balance_gamma: float = 0.001


# =========================================================================
# Perforant Path Symphony Bridge
# =========================================================================

class PerforantPathSymphonyBridge(nn.Module):
    """
    Multi-column communication subspace projecting N column outputs to a
    single coordinate-space vector for the Cognitive Kernel.

    BIOLOGICAL STRUCTURE: Perforant Path, the axonal bundle from entorhinal
    cortex layer II to dentate gyrus and CA3. In the TimmyArray, each active
    column is an area of neocortex projecting to the entorhinal-hippocampal
    interface through its own low-rank communication subspace.

    BIOLOGICAL FUNCTION: The hippocampus does not receive the full population
    activity of each cortical area. It receives the integrated output of all
    open communication subspaces, termed the chorus. The chorus dimensionality
    is much lower than the sum of all zone activities. The query from Prime's
    working memory state selects which column's predictive dimensions dominate
    the chorus at any moment, without closing other channels.

    INTERFACE BOUNDARY:
        SENDING:    TimmyArray column outputs (neocortex, source populations)
        RECEIVING:  Cognitive Kernel coordinate manifold (allocortex/hippocampus)
        CONNECTION: Perforant Path (entorhinal cortex layer II projections)

    Each column has its own (U_send, channel_gains, V_receive) projection.
    The outputs are attention-pooled using a query derived from Prime's
    post-MemoryCortex representation, so Prime coordinates what the kernel
    hears without being a sequential bottleneck on specialist processing.

    References:
        Semedo JD et al. (2019). DOI: 10.1016/j.neuron.2019.01.026
        Witter MP et al. (2000). "Cortico-hippocampal communication by way
        of parallel parahippocampal-subicular pathways." Hippocampus,
        10(4):398-410. DOI: 10.1002/1098-1063(2000)10:4<398::AID-HIPO6>3.0.CO;2-K
        Rolls ET (2013). "The mechanisms for pattern completion and pattern
        separation in the hippocampus." Frontiers in Systems Neuroscience,
        7:74. DOI: 10.3389/fnsys.2013.00074
    """

    def __init__(self, cfg: TimmyArrayConfig):
        """
        Args:
            cfg: TimmyArrayConfig providing d_model, coordinate_dim,
                perforant_rank, num_specialists.
        """
        super().__init__()
        self.cfg = cfg
        n_columns = 1 + cfg.num_specialists  # Prime + specialists
        d = cfg.column_cfg.d_model
        r = cfg.perforant_rank
        c = cfg.coordinate_dim

        # Per-column low-rank projections: (d_model, rank) @ (rank, coord_dim).
        # Each column has independent U_send, channel_gains, V_receive.
        # Initialized with QR decomposition for orthogonal columns, following
        # perforant_path.py. NOT a biological quantity, initialization choice.
        self.U_send = nn.ParameterList()
        self.channel_gains = nn.ParameterList()
        self.V_receive = nn.ParameterList()
        for _ in range(n_columns):
            u = torch.empty(d, r)
            nn.init.orthogonal_(u)
            self.U_send.append(nn.Parameter(u))
            self.channel_gains.append(nn.Parameter(torch.ones(r)))
            v = torch.randn(r, c) * (1.0 / math.sqrt(r))
            self.V_receive.append(nn.Parameter(v))

        # Single-head attention pooling. Query is derived from Prime's
        # post-MemoryCortex output (d_model -> 1 attention weight per column).
        # This is a learned selection mechanism, not a fixed weighting.
        # NOT a biological structure. Engineering pooling mechanism.
        self.attn_query_proj = nn.Linear(d, c, bias=False)
        self.attn_scale = 1.0 / math.sqrt(c)

    def forward(
        self,
        column_outputs: List[Tensor],
        prime_memory_output: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Project each active column's output through its communication subspace
        and attention-pool to a single coordinate vector.

        INTERFACE BOUNDARY:
            SENDING:    column_outputs from TimmyArray (neocortex)
            RECEIVING:  coordinate vector for Cognitive Kernel CA3 input
            CONNECTION: Perforant Path

        Args:
            column_outputs: list of (batch, d_model) spike-rate tensors, one
                per active column. Index 0 is always Prime.
            prime_memory_output: (batch, d_model) Prime's post-MemoryCortex
                representation, used as the attention query. This is extracted
                before Prime's executive zone so that executive-zone processing
                runs in parallel with specialist columns.

        Returns:
            coords: (batch, coordinate_dim) kernel input vector.
            attn_weights: (batch, n_active_columns) attention weights for
                diagnostics and CA1 feedback routing.
        """
        # Project each column output through its communication subspace.
        # Cost: O(B * d * r + B * r * c) per column, versus O(B * d * c) dense.
        # Reference: Semedo JD et al. (2019). DOI: 10.1016/j.neuron.2019.01.026
        projections = []
        for i, col_out in enumerate(column_outputs):
            scaled_U = self.U_send[i] * self.channel_gains[i].unsqueeze(0)
            subspace = col_out @ scaled_U          # (B, rank)
            proj = subspace @ self.V_receive[i]    # (B, coordinate_dim)
            projections.append(proj)

        # Stack projections: (B, n_active, coordinate_dim).
        stacked = torch.stack(projections, dim=1)

        # Attention query from Prime's working memory state.
        # (B, coordinate_dim) -> (B, 1, coordinate_dim) for bmm.
        query = self.attn_query_proj(prime_memory_output).unsqueeze(1)

        # Scaled dot-product attention over column projections.
        # attn_scores: (B, 1, n_active)
        attn_scores = torch.bmm(query, stacked.transpose(1, 2)) * self.attn_scale
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, 1, n_active)

        # Weighted sum to single coordinate vector.
        # coords: (B, coordinate_dim)
        coords = torch.bmm(attn_weights, stacked).squeeze(1)
        attn_weights_out = attn_weights.squeeze(1)  # (B, n_active)

        return coords, attn_weights_out


# =========================================================================
# Routing Signal Extractor
# =========================================================================

class ColumnRouter(nn.Module):
    """
    Extracts the routing signal from Prime's post-MemoryCortex representation
    and produces a probability distribution over specialist columns.

    BIOLOGICAL STRUCTURE: Long-range horizontal connections between cortical
    columns with similar functional properties. Prime's broadband working
    memory state activates specialists whose tuning is most relevant to the
    current input.

    BIOLOGICAL FUNCTION: In the Thousand Brains Theory, columns vote to reach
    consensus by sharing information through long-range connections. Here
    Prime produces a routing distribution that determines which specialists
    are engaged. The routing is driven by population-level spike statistics
    in Prime's working memory (the cluster firing rates from MemoryCortex),
    paralleling the spike-driven routing in SpikeDrivenMoE.

    The routing uses a low-rank projection (communication subspace) from
    Prime's d_model representation to a routing score per specialist. This
    enforces the constraint that only the predictive dimensions of Prime's
    state influence routing, consistent with Semedo et al. (2019).

    Routing collapse is prevented by per-specialist bias terms that are
    adjusted dynamically during training without auxiliary loss, following
    Wang et al. (2024). This preserves the purity of the routing gradient
    while maintaining balanced specialist utilization.

    References:
        Hawkins J et al. (2019). DOI: 10.3389/fncir.2018.00121
        Semedo JD et al. (2019). DOI: 10.1016/j.neuron.2019.01.026
        Wang P et al. (2024). arXiv: 2408.15664. DOI: {To be added later.}
        See JZ et al. (2018). DOI: 10.7554/eLife.35587
        (cNE cluster firing rates as routing signal, Section 2.)
    """

    def __init__(self, cfg: TimmyArrayConfig):
        """
        Args:
            cfg: TimmyArrayConfig.
        """
        super().__init__()
        self.cfg = cfg
        d = cfg.column_cfg.d_model
        r = cfg.routing_rank
        n = cfg.num_specialists

        # Low-rank routing projection. U: (d, rank), V: (rank, n_specialists).
        # Product is a (d, n_specialists) matrix of rank at most routing_rank.
        # Reference: Semedo JD et al. (2019). DOI: 10.1016/j.neuron.2019.01.026
        self.U_route = nn.Parameter(torch.empty(d, r))
        nn.init.orthogonal_(self.U_route)
        self.V_route = nn.Parameter(torch.randn(r, n) * (1.0 / math.sqrt(r)))

        # Per-specialist bias terms for load balancing.
        # Adjusted by load_balance_gamma * sign(load_deviation) each step.
        # NOT a biological quantity. Auxiliary-loss-free load balancing.
        # Reference: Wang P et al. (2024). arXiv: 2408.15664.
        self.register_buffer("routing_bias", torch.zeros(n))

        # Exponential moving average of specialist utilization.
        # Tracks which specialists are over or under-utilized for bias update.
        # NOT a biological quantity. Training monitoring.
        self.register_buffer("specialist_load_ema", torch.ones(n) / n)

    def forward(
        self,
        prime_memory_output: Tensor,
        top_k: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute routing distribution over specialists from Prime's working
        memory state.

        Args:
            prime_memory_output: (batch, d_model) Prime's post-MemoryCortex
                representation. Extracted before executive blocks so that
                specialist executive zones run in parallel with Prime's.
            top_k: if provided, zero out all but top-k routing weights
                (straight-through for gradient flow). At inference, top_k=1.
                During training, top_k=cfg.routing_top_k_train.

        Returns:
            routing_weights: (batch, num_specialists) soft routing
                distribution (sums to 1 over specialists).
            routing_indices: (batch, top_k) integer indices of selected
                specialists, sorted by weight descending.
        """
        # Low-rank projection: (B, d) -> (B, rank) -> (B, n_specialists).
        scores = (prime_memory_output @ self.U_route) @ self.V_route

        # Add per-specialist bias for load balancing.
        scores = scores + self.routing_bias.unsqueeze(0)

        # Temperature scaling before softmax.
        # NOT a biological quantity. Training hyperparameter.
        scores = scores / self.cfg.routing_temperature

        routing_weights = F.softmax(scores, dim=-1)  # (B, n_specialists)

        if top_k is not None and top_k < self.cfg.num_specialists:
            # Straight-through top-k: keep top-k values, zero the rest,
            # but let gradients flow through the zero entries too.
            # This allows gradient signal to reach non-selected specialists
            # during training, preventing them from collapsing.
            # NOT a biological quantity. Training artifact.
            topk_vals, topk_idx = torch.topk(routing_weights, top_k, dim=-1)
            mask = torch.zeros_like(routing_weights)
            mask.scatter_(-1, topk_idx, 1.0)
            # Straight-through: forward uses masked weights, backward sees full.
            routing_weights = routing_weights * mask + (
                routing_weights - routing_weights.detach()
            )
            routing_indices = topk_idx
        else:
            routing_indices = torch.argsort(
                routing_weights, dim=-1, descending=True
            )

        return routing_weights, routing_indices

    @torch.no_grad()
    def update_load_balance_bias(self, routing_weights: Tensor) -> None:
        """
        Adjust per-specialist bias terms based on observed routing distribution.

        Specialists receiving more than average load get their bias decreased
        (making them less likely to be selected). Specialists receiving less
        than average load get their bias increased. This prevents routing
        collapse without auxiliary loss.

        Called once per training step after the forward pass. Gradient is
        detached; this is a meta-update, not part of the computational graph.

        Args:
            routing_weights: (batch, num_specialists) routing distribution
                from the current forward pass.

        NOT a biological quantity. Auxiliary-loss-free load balancing.
        Reference: Wang P et al. (2024). arXiv: 2408.15664.
        """
        mean_load = routing_weights.mean(dim=0)  # (n_specialists,)
        self.specialist_load_ema.lerp_(mean_load, 0.01)
        balanced = torch.ones_like(self.specialist_load_ema) / self.cfg.num_specialists
        deviation = self.specialist_load_ema - balanced
        self.routing_bias -= self.cfg.load_balance_gamma * deviation.sign()


# =========================================================================
# TimmyArray
# =========================================================================

class TimmyArray(nn.Module):
    """
    Cortical column ensemble: TimmyPrime and named specialist columns sharing
    a unified Cognitive Kernel.

    BIOLOGICAL STRUCTURE: Neocortical column ensemble projecting to a shared
    hippocampal formation. Each column is a TimmyModel instance (a complete
    spiking neural network language model). Prime is the always-active
    broadband integration column. Specialists are conditionally active based
    on Prime's routing signal.

    BIOLOGICAL FUNCTION: Every column builds a complete world model through
    experience-dependent plasticity. Columns vote to reach consensus by sharing
    information through the PerforantPathSymphonyBridge, which delivers a
    single coordinate vector to the Cognitive Kernel regardless of how many
    columns are active. The kernel's CA1 mismatch signal propagates back to
    Prime and the winning specialist as reward for STDP.

    The threshold adaptation finding from Huang et al. (2022) is directly
    relevant here: columns trained on different input distributions will
    develop different threshold dynamics and effective connectivity structures,
    not just different weight matrices. The per-neuron learnable threshold in
    AssociativeLIF (v_threshold_raw, trained via backpropagation) enables this
    specialization to emerge naturally. Diagnostics tracking per-column
    threshold statistics are provided to monitor specialization progress.

    FORWARD PASS DEPENDENCY STRUCTURE:
        1. Prime: encoder -> sensory -> association -> MemoryCortex
           (extract routing signal and PerforantPath query here)
        2. Parallel: Prime executive + all selected specialist full passes
        3. PerforantPathSymphonyBridge: pool active column outputs to
           kernel coordinate input
        4. Cognitive Kernel forward (external, not in this file)
        5. CA1 feedback: inject novelty scalar to Prime and winning specialist
           via STDPEngine.set_external_reward()

    Steps 2 and 3 can be parallelized across devices. Steps 1 and 4-5
    are sequential.

    CHECKPOINT FORMAT:
        The .soul file produced by save_array_state() follows the three-layer
        COLD/WARM/HOT format from timmy_state.py, applied once to Prime and
        once per specialist. The Cognitive Kernel state is saved at the root
        level under "cognitive_kernel" and is NOT nested under any column.
        The array architecture hash includes num_specialists and column names
        to detect structural mismatches on load.

    References:
        Hawkins J et al. (2019). DOI: 10.3389/fncir.2018.00121
        See JZ et al. (2018). DOI: 10.7554/eLife.35587
        Semedo JD et al. (2019). DOI: 10.1016/j.neuron.2019.01.026
        Mountcastle VB (1997). DOI: 10.1093/brain/120.4.701
        Huang C, Zeldenrust F, Celikel T (2022). "Cortical representation of
        touch in silico." Neuroinformatics, 20:1013-1039.
        DOI: 10.1007/s12021-022-09576-5
        (Threshold adaptation and effective connectivity as primary
        mechanisms of column differentiation, not weight matrices alone.)
        Pérez-Ortega J, Alejandre-García T, Yuste R (2021). "Long-term
        stability of cortical ensembles." eLife, 10:e64449.
        DOI: 10.7554/eLife.64449
        (Ensemble identity is carried by the core connectivity structure,
        not by fixed membership. ~68% of stable ensemble neurons persist
        across 46 days; the rest rotate. Spontaneous activity rehearses the
        same functional circuit elements as evoked activity.)
    """

    def __init__(self, cfg: TimmyArrayConfig):
        """
        Initialize TimmyPrime, all specialist columns, the ColumnRouter,
        and the PerforantPathSymphonyBridge.

        All columns start from identical TimmyConfig. Differentiation
        emerges through training. No weights are frozen at initialization.

        Args:
            cfg: TimmyArrayConfig defining column architecture and routing.
        """
        super().__init__()
        self.cfg = cfg
        self.column_names: List[str] = COLUMN_NAMES[: 1 + cfg.num_specialists]

        # Prime column: always active, broadband integration.
        # Index 0 in all column lists.
        self.prime = TimmyModel(cfg.column_cfg)

        # Specialist columns, accessed by semantic name via ModuleDict.
        # Named access survives checkpoint serialization and is used for
        # DataLoader assignment and CA1 feedback routing.
        specialist_names = self.column_names[1:]  # excludes prime
        self.specialists = nn.ModuleDict(
            {name: TimmyModel(cfg.column_cfg) for name in specialist_names}
        )

        # Routing signal extractor: Prime's working memory -> specialist
        # probability distribution.
        self.router = ColumnRouter(cfg)

        # Perforant Path symphony bridge: N column outputs -> single
        # coordinate vector for the Cognitive Kernel.
        self.perforant_bridge = PerforantPathSymphonyBridge(cfg)

        # Per-specialist expert load EMA for diagnostics.
        # Tracks how often each specialist is selected as the winning column.
        # Used by save_array_state() to populate the HOT layer.
        # NOT a biological quantity. Training monitoring.
        self.register_buffer(
            "specialist_selection_ema",
            torch.ones(cfg.num_specialists) / cfg.num_specialists,
        )

    # ---- Accessors ----

    def specialist_list(self) -> List[TimmyModel]:
        """Return specialists in name order (matches column_names[1:])."""
        return [self.specialists[n] for n in self.column_names[1:]]

    def all_columns(self) -> List[TimmyModel]:
        """Return [prime] + specialists in name order."""
        return [self.prime] + self.specialist_list()


    def clone_prime_to_specialists(self) -> None:
        """
        Copy Prime's trained weights into every specialist simultaneously.

        BIOLOGICAL STRUCTURE: Experience-dependent cortical column divergence
        from a shared developmental template.
        PLAIN ENGLISH: After Prime is trained on the full input distribution,
        this method copies its complete learned state (weights, LIF thresholds,
        synaptic time constants, MoE cluster assignments, STDP scalars) into
        every specialist column at once. All specialists begin as exact replicas
        of Prime. Functional differentiation then emerges over subsequent sleep
        cycles through specialty-directed synaptic pruning and consolidation
        in sleep.py and firstday.py, not through separate training phases.

        The biological analog is the developmental period in which cortical
        columns with identical genetic substrate diverge through experience-
        dependent plasticity after a shared progenitor template is established.
        The clone event ends the template phase. The sleep cycle begins
        divergence.

        Reference: Rakic P (1988). "Specification of cerebral cortical areas."
        Science, 241(4862):170-176. DOI: 10.1126/science.3291116
        (Radial unit hypothesis: cortical columns share a common progenitor
        lineage and identical initial connectivity; areal identity emerges
        from thalamic input statistics and activity-dependent refinement,
        not from intrinsic architectural differences at birth.)

        Reference: Bi GQ, Poo MM (1998). "Synaptic modifications in cultured
        hippocampal neurons: dependence on spike timing, synaptic strength,
        and postsynaptic cell type." Journal of Neuroscience, 18(24):10464-10472.
        DOI: 10.1523/JNEUROSCI.18-24-10464.1998
        (STDP state is copied with weights so each specialist begins with
        Prime's full plasticity history intact, not a cold STDP state.)

        Implementation note: state_dict() / load_state_dict() with strict=True
        guarantees every parameter and buffer is transferred. The router and
        perforant_bridge are NOT cloned; they are shared coordination
        infrastructure, not column-specific state.

        NOT a biological quantity: the simultaneous copy to all specialists
        in a single loop is a computational convenience. Biology performs
        the equivalent through cell division and axonal projection over
        developmental time.

        After weight copy, each specialist's LIF membrane state is reset to
        rest via reset_state(). Prime's membrane potential at training end
        reflects Prime's last input sequence and is not a valid starting
        state for a specialist entering its first day cycle.

        Raises:
            RuntimeError: if any specialist architecture does not exactly
                match Prime's. Cannot occur if all columns were constructed
                from the same TimmyConfig, but checked defensively.
        """
        prime_state = self.prime.state_dict()
        for name in self.column_names[1:]:
            specialist = self.specialists[name]
            try:
                specialist.load_state_dict(prime_state, strict=True)
            except RuntimeError as e:
                raise RuntimeError(
                    f"clone_prime_to_specialists(): architecture mismatch for "
                    f"specialist '{name}'. All specialists must be built from "
                    f"the same TimmyConfig as Prime. Original error: {e}"
                )
        # Reset membrane states after copy. Weight copy is permanent;
        # dynamic state (v_mem, i_syn) should begin from biological rest,
        # not inherited from Prime's last training sequence.
        # NOT a biological quantity. Computational initialization artifact.
        for name in self.column_names[1:]:
            self.specialists[name].reset_state()

    def column_by_name(self, name: str) -> TimmyModel:
        """Return the column for the given name. 'prime' returns self.prime."""
        if name == "prime":
            return self.prime
        return self.specialists[name]

    # ---- State management ----

    def reset_state(self) -> None:
        """
        Reset persistent membrane state for all columns.

        Call between unrelated sequences (e.g., between documents in a
        training batch). Delegates to TimmyModel.reset_state() for each
        column.

        The membrane state carries not just memory context but also the
        excitability regime that gates whether new inputs register at all
        (Huang et al., 2022, DOI: 10.1007/s12021-022-09576-5). Cold-starting
        between unrelated sequences is therefore important for both memory
        correctness and input gating correctness.
        """
        self.prime.reset_state()
        for col in self.specialist_list():
            col.reset_state()

    # ---- Forward pass ----

    def forward(
        self,
        token_ids: Tensor,
        float_embeds: Optional[Tensor] = None,
        enable_stdp: bool = False,
        inference: bool = False,
    ) -> Tuple[Tensor, Tensor, Dict]:
        """
        Full array forward pass.

        Step 1: Run Prime through encoder, sensory, association, and
                MemoryCortex. Extract routing signal and PerforantPath query.
        Step 2: Compute routing distribution. Select specialist(s).
        Step 3: Run selected specialists and Prime executive in parallel
                (sequential here; parallel dispatch across devices is left
                to the training harness).
        Step 4: Pool active column outputs via PerforantPathSymphonyBridge
                to produce the kernel coordinate input.

        CA1 feedback (Step 5) is applied externally after the kernel forward
        pass via inject_ca1_feedback().

        Args:
            token_ids: (B, S) integer token indices.
            float_embeds: optional (B, S, float_embed_dim) external continuous
                embeddings (MEM 3, passed through to each column's encoder).
            enable_stdp: if True, cache pre/post spike tensors in each active
                column for a subsequent stdp_update() call.
            inference: if True, use top-1 routing (winner-take-all).
                If False, use top-k routing with straight-through gradients.

        Returns:
            prime_logits: (B, S, vocab_size) output logits from Prime.
                The LM head is on Prime because Prime has the broadest
                receptive field and serves as the primary token predictor.
                Specialist logits are available via the stats dict if needed.
            kernel_coords: (B, coordinate_dim) coordinate input for the
                Cognitive Kernel, ready to pass to PerforantPathBridge.forward()
                in cognitive_kernel.py.
            stats: dict containing routing weights, attn weights, per-column
                spike rates, MoE load balance losses, and specialist selection
                indices.
        """
        stats: Dict = {}

        # ---- Step 1: Prime through MemoryCortex ----
        # Run Prime's encoder, sensory, association, and MemoryCortex.
        # The post-MemoryCortex representation carries working memory context
        # for routing without requiring Prime's executive zone to complete first.
        prime_memory_out, prime_partial_stats = self._run_prime_to_memory(
            token_ids, float_embeds=float_embeds, enable_stdp=enable_stdp
        )
        stats.update({f"prime_{k}": v for k, v in prime_partial_stats.items()})

        # ---- Step 2: Routing ----
        top_k = 1 if inference else self.cfg.routing_top_k_train
        routing_weights, routing_indices = self.router(
            prime_memory_out, top_k=top_k
        )
        stats["routing_weights"] = routing_weights.detach()
        stats["routing_indices"] = routing_indices.detach()

        # Update load balance biases (no gradient).
        if self.training:
            self.router.update_load_balance_bias(routing_weights.detach())

        # Update selection EMA for diagnostics and HOT-layer checkpointing.
        with torch.no_grad():
            self.specialist_selection_ema.lerp_(
                routing_weights.detach().mean(dim=0), 0.01
            )

        # ---- Step 3: Complete Prime executive + run selected specialists ----
        prime_logits, prime_exec_stats = self._run_prime_executive(
            prime_memory_out, enable_stdp=enable_stdp
        )
        stats.update({f"prime_{k}": v for k, v in prime_exec_stats.items()})

        # Collect active column outputs for the PerforantPath bridge.
        # Prime's executive output is always included (index 0).
        # Shape: (B, d_model) after averaging over sequence and time.
        prime_pool = self._pool_for_perforant(prime_exec_stats["exec_output"])
        active_column_outputs: List[Tensor] = [prime_pool]
        active_column_names: List[str] = ["prime"]
        winning_specialist_idx: int = routing_indices[:, 0].mode().values.item()

        specialist_list = self.specialist_list()
        selected_indices = routing_indices[:, 0].unique().tolist()
        for spec_idx in sorted(selected_indices):
            if spec_idx >= len(specialist_list):
                continue
            spec_name = self.column_names[1 + spec_idx]
            spec_col = specialist_list[spec_idx]
            _, spec_stats = spec_col(
                token_ids,
                float_embeds=float_embeds,
                enable_stdp=enable_stdp,
            )
            spec_pool = self._pool_for_perforant(spec_stats.get("exec_output", None))
            if spec_pool is not None:
                active_column_outputs.append(spec_pool)
                active_column_names.append(spec_name)
            stats.update(
                {f"{spec_name}_{k}": v for k, v in spec_stats.items()}
            )

        stats["active_columns"] = active_column_names

        # ---- Step 4: PerforantPath symphony pooling ----
        kernel_coords, attn_weights = self.perforant_bridge(
            active_column_outputs, prime_memory_out
        )
        stats["perforant_attn_weights"] = attn_weights.detach()
        stats["winning_specialist_idx"] = winning_specialist_idx

        return prime_logits, kernel_coords, stats

    # ---- CA1 feedback injection ----

    def inject_ca1_feedback(
        self,
        novelty_scalar: float,
        stats: Dict,
    ) -> None:
        """
        Deliver CA1 mismatch novelty signal as external STDP reward.

        Sent to exactly two recipients:
            1. Prime: always, because Prime's routing decisions require
               ongoing reinforcement regardless of which specialist won.
            2. The winning specialist: the column whose routing weight was
               highest for the current episode. This is the column that wrote
               the episode to CA3 memory; it should receive the mismatch signal
               that reflects whether that write was appropriate.

        This enforces the one-writer-per-episode constraint. Broadcasting
        to all active specialists would cause redundant CA3 writes from
        multiple columns seeing the same novelty signal.

        The Izhikevich (2007) three-factor learning rule grounds this:
        the reward signal is delivered to the synapse that contributed to
        the outcome, not to the whole brain.
        Reference: Izhikevich EM (2007). "Solving the distal reward problem
        through linkage of STDP and dopamine signaling." Cerebral Cortex,
        17(10):2443-2452. DOI: 10.1093/cercor/bhl152

        Args:
            novelty_scalar: float in [-1, 1]. Positive values indicate novel
                experience worth strengthening. Negative values indicate
                predicted experience (mismatch in the other direction).
            stats: the stats dict from the most recent forward() call.
                Used to identify the winning specialist index.
        """
        scaled_reward = novelty_scalar * self.cfg.ca1_reward_scale

        # Prime always receives feedback.
        if self.cfg.prime_always_receives_ca1:
            self.prime.stdp.set_external_reward(scaled_reward)

        # Winning specialist receives feedback.
        winning_idx = stats.get("winning_specialist_idx", None)
        if winning_idx is not None and winning_idx < self.cfg.num_specialists:
            winning_name = self.column_names[1 + winning_idx]
            self.specialists[winning_name].stdp.set_external_reward(scaled_reward)

    # ---- Internal helpers ----

    def _run_prime_to_memory(
        self,
        token_ids: Tensor,
        float_embeds: Optional[Tensor],
        enable_stdp: bool,
    ) -> Tuple[Tensor, Dict]:
        """
        Run Prime's encoder, sensory, association, and MemoryCortex.
        Return the post-MemoryCortex representation and partial stats.

        The post-MemoryCortex representation is extracted here because:
        (a) MemoryCortex slow-decaying LIF neurons (tau_mem=0.99) integrate
            the full association-zone context, giving routing richer signal
            than raw association output.
        (b) Extracting here allows Prime's executive zone to run in parallel
            with specialist columns, avoiding a sequential bottleneck.

        Reference: Wang XJ (2001). "Synaptic reverberation underlying mnemonic
        persistent activity." Trends in Neurosciences, 24(8):455-463.
        DOI: 10.1016/S0166-2236(00)01868-3
        (PFC slow-decaying neurons as the substrate for working memory context.)

        Returns:
            memory_out: (B, d_model) averaged post-MemoryCortex representation.
            partial_stats: dict of sensory/association/memory diagnostics.
        """
        # This method directly accesses Prime's internal modules.
        # It replicates the first half of TimmyModel.forward() up to and
        # including MemoryCortex. The second half (executive + readout) is
        # completed in _run_prime_executive().
        T_t = self.prime.cfg.T_total
        D = self.prime.cfg.d_model
        B, S = token_ids.shape
        stats: Dict = {}
        moe_lb = torch.tensor(0.0, device=token_ids.device)

        cur = self.prime.encoder(token_ids, float_embeds=float_embeds)
        isp, _ = self.prime.input_lif(cur)
        isp = isp.reshape(T_t, B, S, D)
        x = isp

        if enable_stdp:
            self.prime._stdp_cache["input"] = isp.detach()

        for i, bl in enumerate(self.prime.sensory_blocks):
            x, bs = bl(x)
            lb = bs.pop("moe_load_balance_loss", None)
            if lb is not None:
                moe_lb = moe_lb + lb
            for k, v in bs.items():
                stats[f"sensory_{i}_{k}"] = v

        for i, bl in enumerate(self.prime.association_blocks):
            prev = x.detach() if enable_stdp else None
            x, bs = bl(x)
            lb = bs.pop("moe_load_balance_loss", None)
            if lb is not None:
                moe_lb = moe_lb + lb
            for k, v in bs.items():
                stats[f"assoc_{i}_{k}"] = v
            if enable_stdp and prev is not None:
                self.prime._stdp_cache[f"assoc_{i}_pre"] = prev
                self.prime._stdp_cache[f"assoc_{i}_post"] = x.detach()

        x, ms = self.prime.memory_cortex(x)
        stats.update(ms)
        stats["moe_lb_loss"] = moe_lb

        # Pool over T and S to get (B, D) routing query.
        # Mean pooling over temporal and sequence dimensions.
        # NOT a biological operation. Engineering reduction for routing.
        memory_out = x.mean(dim=(0, 2))  # (T, B, S, D) -> (B, D)

        # Store x for the executive continuation.
        self._prime_memory_x = x

        return memory_out, stats

    def _run_prime_executive(
        self,
        prime_memory_out: Tensor,
        enable_stdp: bool,
    ) -> Tuple[Tensor, Dict]:
        """
        Complete Prime's forward pass from MemoryCortex output through
        executive blocks, readout, and LM head.

        Called after routing is computed. Runs in parallel with specialist
        column passes in the full forward() method.

        Args:
            prime_memory_out: stored from _run_prime_to_memory(), not used
                directly here (the stored self._prime_memory_x is used).
            enable_stdp: cache executive-zone pre/post tensors.

        Returns:
            logits: (B, S, vocab_size).
            exec_stats: dict with executive spike rates and exec_output tensor.
        """
        x = self._prime_memory_x
        T_t = self.prime.cfg.T_total
        B_S = x.shape[1] * x.shape[2]
        D = self.prime.cfg.d_model
        stats: Dict = {}

        for i, bl in enumerate(self.prime.executive_blocks):
            prev = x.detach() if enable_stdp else None
            x, bs = bl(x)
            for k, v in bs.items():
                stats[f"exec_{i}_{k}"] = v
            if enable_stdp and prev is not None:
                self.prime._stdp_cache[f"executive_{i}_pre"] = prev
                self.prime._stdp_cache[f"executive_{i}_post"] = x.detach()

        # Store executive output for PerforantPath pooling.
        stats["exec_output"] = x

        xf = x.reshape(T_t, B_S, D)
        rsp, vm = self.prime.readout_lif(xf)
        a = self.prime.readout_ema_decay
        ema = torch.zeros(B_S, D, device=x.device, dtype=vm.dtype)
        for t in range(T_t):
            ema = a * ema + (1 - a) * vm[t]

        B = x.shape[1]
        S = x.shape[2]
        vs = ema.reshape(B, S, D)
        sm = rsp.mean(dim=0).reshape(B, S, D)
        ro = vs + sm

        xn = F.layer_norm(
            ro.float(),
            self.prime.readout_norm.normalized_shape,
            self.prime.readout_norm.weight.float()
            if self.prime.readout_norm.weight is not None else None,
            self.prime.readout_norm.bias.float()
            if self.prime.readout_norm.bias is not None else None,
            self.prime.readout_norm.eps,
        ).to(ro.dtype)
        logits = self.prime.lm_head(xn)

        return logits, stats

    def _pool_for_perforant(self, exec_output: Optional[Tensor]) -> Optional[Tensor]:
        """
        Reduce a column's executive output to a (B, d_model) vector for
        PerforantPath input.

        The PerforantPathSymphonyBridge expects (batch, d_model) spike-rate
        tensors. Executive output has shape (T, B, S, D); mean pooling over
        T and S gives the population average rate vector.

        NOT a biological operation. Engineering reduction for the bridge.

        Args:
            exec_output: (T, B, S, D) or None.

        Returns:
            (B, D) tensor, or None if exec_output is None.
        """
        if exec_output is None:
            return None
        return exec_output.mean(dim=(0, 2))  # (T, B, S, D) -> (B, D)

    # ---- Checkpoint ----

    def save_array_state(
        self,
        path_prefix: str,
        optimizer_states: Optional[Dict[str, Dict]] = None,
        training_step: Optional[int] = None,
        kernel_state: Optional[Dict] = None,
        extra: Optional[Dict] = None,
    ) -> Dict[str, str]:
        """
        Save a complete array checkpoint.

        Applies the three-layer COLD/WARM/HOT format from timmy_state.py
        to each column independently. The Cognitive Kernel state is saved
        once at the root level under "cognitive_kernel", not nested under
        any column, to avoid redundant storage.

        The checkpoint also records:
        - column_names: list of column identity strings
        - num_specialists: for architectural hash verification on load
        - specialist_selection_ema: HOT layer tracking of specialist utilization
        - routing_bias: HOT layer for load-balancing state continuity

        Args:
            path_prefix: base path. Files are written as
                {prefix}_prime.state, {prefix}_proximal.state, etc.
            optimizer_states: dict mapping column name to optimizer state_dict.
                Optional. Keys must match column_names.
            training_step: current global training step.
            kernel_state: the Cognitive Kernel's serialized state dict.
                Saved once at the root level.
            extra: arbitrary metadata (dataset path, git hash, etc.).

        Returns:
            dict mapping column name to the path written.
        """
        written: Dict[str, str] = {}

        # Save per-column state.
        for name, col in zip(self.column_names, self.all_columns()):
            col_path = f"{path_prefix}_{name}.state"
            opt_state = (optimizer_states or {}).get(name, None)
            col.save_state(
                path=col_path,
                optimizer_state=opt_state,
                training_step=training_step,
                extra={
                    "column_name": name,
                    "num_specialists": self.cfg.num_specialists,
                    "column_names": self.column_names,
                    "specialist_selection_ema": (
                        self.specialist_selection_ema.cpu().tolist()
                    ),
                    "routing_bias": self.router.routing_bias.cpu().tolist(),
                    **(extra or {}),
                },
            )
            written[name] = col_path

        # Save Cognitive Kernel state once at root.
        if kernel_state is not None:
            kernel_path = f"{path_prefix}_cognitive_kernel.pt"
            torch.save(kernel_state, kernel_path)
            written["cognitive_kernel"] = kernel_path

        return written

    # ---- Diagnostics ----

    def specialization_report(self) -> str:
        """
        Report per-column mean spike threshold and firing rate EMA as
        proxies for specialization progress.

        Columns that have developed distinct threshold dynamics are
        specializing as expected (Huang et al., 2022). Columns with
        similar threshold distributions to Prime are still in the
        generalist regime.

        Returns:
            Human-readable string report.
        """
        lines = ["TimmyArray Specialization Report"]
        lines.append(f"{'Column':<14} {'Mean Threshold':>16} {'Firing Rate EMA':>16}")
        lines.append("-" * 50)
        for name, col in zip(self.column_names, self.all_columns()):
            thresh = col.input_lif.v_threshold.mean().item()
            rate = col.input_lif._firing_rate_ema.mean().item()
            lines.append(f"{name:<14} {thresh:>16.4f} {rate:>16.4f}")
        lines.append("")
        lines.append("Specialist selection EMA:")
        for i, name in enumerate(self.column_names[1:]):
            load = self.specialist_selection_ema[i].item()
            lines.append(f"  {name:<12} {load:.4f}")
        return "\n".join(lines)

    def count_params(self) -> str:
        """Human-readable parameter count for the full array."""
        total = sum(p.numel() for p in self.parameters())
        prime_p = sum(p.numel() for p in self.prime.parameters())
        spec_p = sum(
            p.numel()
            for col in self.specialist_list()
            for p in col.parameters()
        )
        router_p = sum(p.numel() for p in self.router.parameters())
        bridge_p = sum(p.numel() for p in self.perforant_bridge.parameters())
        return (
            f"TimmyArray total:    {total/1e6:.1f}M\n"
            f"  Prime:             {prime_p/1e6:.1f}M\n"
            f"  Specialists ({self.cfg.num_specialists}x): {spec_p/1e6:.1f}M\n"
            f"  Router:            {router_p/1e3:.1f}K\n"
            f"  PerforantBridge:   {bridge_p/1e3:.1f}K"
        )
