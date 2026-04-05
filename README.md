# Timmy_Neuron
This repository was refactored and extended from [Project Nord](https://github.com/gtausa197-svg/-Project-Nord-Spiking-Neural-Network-Language)
**A biologically grounded spiking neural network language model**

Timmy is the bridge layer of the PRAGMI architecture (Persistent Reconstructive Architecture for Generative Memory and Imagination). Timmy sits between the external LLM (user-facing narrator) and the Cognitive Kernel (hippocampal memory system), translating discrete token sequences into temporal spike patterns and back.

In standalone mode Timmy functions as a complete spiking language model trainable on standard next-token prediction tasks.

<img width="1024" height="559" alt="image_e5f95f0f-cc01-4c1c-a3b7-8bd0b9aa906f" src="https://github.com/user-attachments/assets/bef23f14-21a7-48fc-bbf9-c3a639a8919d" />

![1000016034](https://github.com/user-attachments/assets/4faffefd-2556-41c8-b86d-9d5457d25272)

The architecture is a cortical column ensemble: one broadband integration column (Prime) plus five specialist columns, all structurally identical, all initialized from the same trained weights. Specialization emerges through sleep-cycle synaptic pruning and consolidation after deployment. The system employs a Thousand Brains approach where Prime coordinates the specialists and all columns project to a shared Cognitive Kernel through independent low-rank communication subspaces.

Every architectural decision is cited to the neuroscience paper it was derived from. Every parameter that is not a biological quantity is explicitly labeled as a training artifact or engineering approximation. A neuroscientist can follow the citations to the papers. An engineer can follow the shapes through the forward pass.

---

## Where is Timmy in PRAGMI

```
External LLM  (narrator, planner, user-facing)
      |
   tokens
      |
      v
  TimmyArray  (this repo)
  ┌─────────────────────────────────────────────────────┐
  │  Timmy Prime          + 5 specialist columns        │
  │  (broadband router)     proximal / distal /         │
  │                         affective / somatic /       │
  │                         structural                  │
  │                                                     │
  │  All columns start as exact clones of trained       │
  │  Prime. Specialization emerges through sleep:       │
  │  each night cycle reinforces specialty-relevant     │
  │  patterns and prunes connections furthest from      │
  │  the specialty. Every day cycle each specialist     │
  │  becomes more differentiated.                       │
  └─────────────────────────────────────────────────────┘
      |
  64-dim coordinate manifold
  (Perforant Path / communication subspace)
      |
      v
  Cognitive Kernel  (hippocampal memory, separate repo)
  CA3 attractor storage, CA1 novelty gating,
  episodic reconstruction, UMAP manifold
```

Grounded in Mountcastle (1997), Hawkins et al. (2019), Semedo et al. (2019), and See et al. (2018).

---

## Lifecycle

Training and deployment are three distinct stages with no overlap.

**Stage 1 — Train Prime (`train_array.py` Phase 1):** Prime is trained alone on the full input distribution. The critical period probe (`timmy_criticalperiodprobe.py`) monitors MemoryCortex threshold stability, association zone routing entropy, and expert load balance. When all three signals are simultaneously stable for the required window, the probe triggers an early exit and saves `phase1_coordready.soul`. This is the coordination-ready checkpoint. Specialists are not involved and do not exist as meaningful entities yet.

**Stage 2 — Phase 2 divergence (`train_array.py` Phase 2):** All columns train simultaneously on domain-assigned subcorpora. Specialists receive input distributions biased toward their integration scale. Differentiation emerges from input statistics, exactly as biological column specialization emerges from sensory experience. The array monitor (`array_monitor.py`) tracks router entropy, specialist activation frequency, subspace effective rank, and cosine distance from Prime throughout this phase.

**Stage 3 — First Day and sleep cycles (`firstday.py`, `sleep.py`, `microsleep.py`):** `clone_prime_to_specialists()` copies Prime's complete trained state into every specialist simultaneously. All six columns are now weight-for-weight identical. The first sleep cycle runs specialty-directed pruning, which is the only larger-magnitude divergence event. After this, specialists are no longer identical to Prime. Subsequent day and sleep cycles deepen specialization continuously through accumulated experience and synaptic consolidation.

---

## Single Column Architecture

Each Timmy instance (Prime or specialist) follows the same pipeline:

```
Token IDs (B, S)
    |
    v
TemporalSpikeEncoder          [timmy_encoder.py]
    Thalamocortical relay: tokens -> multi-scale temporal current
    Fast basis (T=8, gamma-band) + Slow basis (T_slow=2, theta-band)
    Optional: external float embedding injection (MEM 3)
    |
    v
Input LIF                     [timmy_neuron.py]
    AssociativeLIF with cascade amplification
    Persistent membrane state across sequential chunks
    |
    v
Sensory Zone (2 blocks)       [timmy_blocks.py]
    TimmyBlock with SpikingFeedForward
    Primary feature extraction
    |
    v
Association Zone (2 blocks)   [timmy_blocks.py, timmy_experts.py]
    TimmyBlock with SpikeDrivenMoE
    Cluster-based routing, load-balanced expert dispatch
    |
    v
MemoryCortex                  [timmy_memory.py]
    PFC working memory with slow LIF (tau_mem=0.99)
    Multi-head temporal attention readout
    Routing signal extracted here for TimmyArray coordination
    |
    v
Executive Zone (2 blocks)     [timmy_blocks.py]
    TimmyBlock with SpikingFeedForward, force-nonneg output
    Reward-modulated STDP applied here
    |
    v
Readout LIF + EMA + LM Head -> Logits (B, S, vocab_size)
```

---

## Files

| File | Lines | DOIs | What it contains |
|---|---|---|---|
| `timmy_neuron.py` | 579 | 25 | ATanSurrogate, AssociativeLIF with cascade amplification, NeuronConfig |
| `timmy_encoder.py` | 278 | 9 | TemporalSpikeEncoder with dual-timescale basis and float embedding gate |
| `timmy_attention.py` | 288 | 7 | RotaryPositionEmbedding, SpikingSynapticResonance |
| `timmy_experts.py` | 269 | 5 | SpikingExpertGroup, SpikeDrivenMoE with cluster-based routing |
| `timmy_memory.py` | 181 | 7 | MemoryCortex (PFC working memory with slow LIF and temporal attention) |
| `timmy_blocks.py` | 306 | 4 | SpikingFeedForward, LeakyClamp, TimmyBlock, AuxiliarySpikeRegulator |
| `timmy_plasticity.py` | 277 | 10 | STDPEngine (three-factor reward-modulated STDP with external reward) |
| `timmy_model.py` | 560 | 13 | TimmyConfig, TimmyModel (full assembly with save/load/forward/stdp) |
| `timmy_state.py` | 907 | — | Three-layer COLD/WARM/HOT checkpoint system (.soul files) |
| `timmy_data.py` | 273 | — | DataLoader (HuggingFace, local text, pre-tokenized) |
| `CreateTimmyArrayv3.py` | 1193 | 18 | TimmyArray ensemble: Prime + specialists, ColumnRouter, PerforantPathSymphonyBridge, clone_prime_to_specialists() |
| `astrocytic_regulator_v3.py` | 373 | 12 | Tripartite synapse metaplasticity regulator |
| `timmy_criticalperiodprobe.py` | — | 3 | Phase 1 convergence probe: detects critical period closure, triggers early exit |
| `array_monitor.py` | — | 4 | Phase 2 online diagnostics: router entropy, activation frequency, subspace rank, cosine distance |
| `train_array.py` | — | 3 | Two-phase training loop: Phase 1 (Prime only) + Phase 2 (all columns, domain-assigned data) |
| `train_timmy.py` | — | — | Single-column training script for Prime in isolation |
| `smoke_test.py` | — | — | End-to-end pipeline smoke test, runs on CPU in under 60 seconds |
| `firstday.py` | — | — | Array initialization: clone Prime to all specialists, run first sleep cycle |
| `daycycle.py` | — | — | Waking experience accumulation and synaptic tagging loop |
| `sleep.py` | — | — | Overnight consolidation: NREM potentiation + REM specialty-directed pruning |
| `microsleep.py` | — | — | Within-day homeostatic reset, prevents synaptic saturation |

---

## Biological Grounding

| Component | Biological analog | Primary reference |
|---|---|---|
| TemporalSpikeEncoder | Thalamocortical relay | Sherman & Guillery (2002). DOI: 10.1098/rstb.2002.1161 |
| AssociativeLIF | Cortical pyramidal cell | Gerstner et al. (2014). DOI: 10.1017/CBO9781107447615 |
| Cascade amplification | Minicolumn lateral excitation | Mountcastle (1997). DOI: 10.1093/brain/120.4.701 |
| ATanSurrogate | None — training artifact only | Neftci et al. (2019). DOI: 10.1109/MSP.2019.2931595 |
| SpikingSynapticResonance | Communication through coherence | Fries (2005). DOI: 10.1016/j.tics.2005.08.011 |
| SpikeDrivenMoE | Association cortex specialization | Felleman & Van Essen (1991). DOI: 10.1093/cercor/1.1.1 |
| MemoryCortex | PFC delay-period persistent activity | Fuster (1973). DOI: 10.1152/jn.1973.36.1.61 |
| STDPEngine | Three-factor reward-modulated plasticity | Bi & Poo (1998). DOI: 10.1523/JNEUROSCI.18-24-10464.1998 |
| Refractory period | Na⁺ channel inactivation | Hodgkin & Huxley (1952). DOI: 10.1113/jphysiol.1952.sp004764 |
| TimmyArray columns | Cortical column ensemble | Mountcastle (1997). DOI: 10.1093/brain/120.4.701 |
| clone_prime_to_specialists() | Radial unit developmental template | Rakic (1988). DOI: 10.1126/science.3291116 |
| Column specialization | Sleep-driven experience-dependent plasticity | Tononi & Cirelli (2003). DOI: 10.1093/brain/awg100 |
| Sleep pruning | Synaptic tagging and capture, REM refinement | Seibt & Frank (2019). DOI: 10.3389/fnsys.2019.00002 |
| PerforantPathSymphonyBridge | Communication subspace | Semedo et al. (2019). DOI: 10.1016/j.neuron.2019.01.026 |
| ColumnRouter | Long-range horizontal coordination | See et al. (2018). DOI: 10.7554/eLife.35587 |
| AstrocyticRegulator | Tripartite synapse metaplasticity | Araque et al. (1999). DOI: 10.1016/S0166-2236(98)01349-6 |
| Critical period closure | Experience-dependent threshold stabilization | Huang et al. (2022). DOI: 10.1007/s12021-022-09576-5 |
| Subspace effective rank | Representational dimensionality | Roy & Vetterli (2007). DOI: {To be added later.} |

---

## Quick Start (Standalone Single Column)

```python
import torch
from timmy_model import TimmyConfig, TimmyModel

cfg = TimmyConfig(
    vocab_size=128_256,
    d_model=496,
    float_embed_dim=0,   # standalone mode, no kernel injection
)
model = TimmyModel(cfg)
print(model.count_params())

token_ids = torch.randint(0, cfg.vocab_size, (2, 128))
logits, stats = model(token_ids)
print(f"Logits: {logits.shape}")               # (2, 128, 128256)
print(f"Avg spike rate: {stats['avg_spike_rate']:.4f}")
```

## Quick Start (TimmyArray)

```python
import torch
from CreateTimmyArrayv3 import TimmyArray, TimmyArrayConfig
from timmy_model import TimmyConfig

arr_cfg = TimmyArrayConfig(
    column_cfg=TimmyConfig(),
    num_specialists=5,
)
array = TimmyArray(arr_cfg)

# Load coordination-ready Prime and clone to all specialists
array.prime.load_state("phase1_coordready.soul")
array.clone_prime_to_specialists()   # all six columns now identical

# Forward pass
token_ids = torch.randint(0, 128_256, (2, 128))
prime_logits, kernel_coords, stats = array(token_ids)
print(f"Prime logits:   {prime_logits.shape}")    # (2, 128, 128256)
print(f"Kernel coords:  {kernel_coords.shape}")   # (2, 64)
print(f"Active columns: {stats['active_columns']}")
print(array.specialization_report())
```

---

## Training

```bash
# Verify the pipeline runs before committing compute
python smoke_test.py

# Phase 1: train Prime on full distribution
# Probe exits automatically when critical period closes
python train_array.py --phase1_data roneneldan/TinyStories

# Phase 1 + Phase 2 full workflow
python train_array.py \
    --phase1_data roneneldan/TinyStories \
    --phase2_prime roneneldan/TinyStories \
    --phase2_proximal /data/sequential_text.txt \
    --phase2_distal /data/long_form.txt \
    --phase2_affective /data/social_dialogue.txt \
    --phase2_somatic /data/embodied_descriptions.txt \
    --phase2_structural /data/code_and_math.txt

# Phase 2 only, resuming from Phase 1 checkpoint
python train_array.py \
    --phase2_prime roneneldan/TinyStories \
    --phase2_proximal /data/sequential_text.txt \
    --resume_prime checkpoints/array_phase1_coordready_prime.state
```

---

## Checkpointing

Timmy uses three-layer `.soul` checkpoints (COLD / WARM / HOT):

- **COLD**: weights (state_dict)
- **WARM**: per-population LIF membrane and synaptic states
- **HOT**: STDP scalars, MoE expert utilization EMAs, routing bias

```python
# Save
model.save_state("timmy_step_5000.soul", optimizer_state=opt.state_dict(), training_step=5000)

# Load
meta = model.load_state("timmy_step_5000.soul")
print(f"Resumed at step {meta['training_step']}")
```

Array checkpoints save each column independently:

```python
written = array.save_array_state(
    path_prefix="checkpoints/array_firstday",
    training_step=0,
)
```

---

## PRAGMI Integration Points

**External embedding injection (MEM 3):** Set `float_embed_dim=496` in config and pass `float_embeds` to the forward call. The encoder blends external continuous vectors with token embeddings via a learnable gate initialized near-zero. This is how the Cognitive Kernel returns reconstructed episodic coordinates to Timmy.

**External reward signal (MEM 4):** Call `model.stdp.set_external_reward(reward)` before `model.stdp_update()`. Reward in [-1, 1] directly modulates executive-zone synaptic plasticity. This is how the CA1 novelty scalar from the Cognitive Kernel drives STDP without going through the training loss.

---

## Documentation Standard

This codebase follows the Genesis Code Documentation Standard:

- Every class and every function has a docstring. No exceptions.
- Every biological concept: biological name, plain English explanation, full citation with DOI.
- Every non-biological parameter explicitly labeled as training artifact or engineering approximation.
- No vague rationale. "The biology validates this approach" is not a citation.
- No Nord references. Every concept traced back to the original paper.
- Variable names use biological terminology: `v_mem`, `i_syn`, `tau_mem`, `dentate_gyrus_dim`.

---

## Related

- [pragmi-kernel](https://github.com/genesislabs-research/pragmi-kernel): The Cognitive Kernel (hippocampal memory system)

---

*Genesis Labs Research, 2026*

# Genesis Labs Research — Timmy Neuron V2

**A Biologically Grounded, Self-Organizing Spiking Neural Network**  
**via Active Inference, Structural Plasticity, and Neuromodulatory Gating**

**Project**: Tommy Neuron 
**Lead Researcher**: Amellia Mendel  
**Date**: April 2026  
**Status**: Architecture Stabilization Phase  
**Scale**: 1.3B Parameter Spiking Neural Network (SNN)

## Vision

Tommy is a cortical-like system designed to grow, specialize, consolidate, and adapt in ways that mirror mammalian brain development and daily function. Unlike static models, it maintains a persistent self-model and uses autogenic diagnostics to manage its own representational growth.

Built on the **PRAGMI** framework (Persistent Reconstructive Architecture for Generative Memory and Imagination), the architecture integrates:
- **Active Inference & Epistemic Foraging** during the Wake Phase
- **Theta-Gamma Modulated Replay** and structural consolidation during the Sleep Phase
- **3-Factor DA-SSDP Plasticity** (Spike Timing + Synchrony + Dopamine)
- **Orthogonal Neurogenesis** via Gram-Schmidt projection to prevent representational collapse

The system autonomously expands its capacity while remaining stable and bootable across power cycles.

## Dual Codebase Strategy

To balance high-performance runtime needs with long-term scientific and pedagogical value, every major module exists in two parallel versions:

### Production (`*_p.py`)
- Minimal, clean, and optimized for training and inference
- Stripped of all non-essential comments and explanations
- Designed for engineers and researchers who need a lean, high-throughput codebase

### Teaching (`*_teaching.py`)
- Richly documented teaching and research artifact
- Full **BIOLOGICAL GROUNDING** headers with real citations and DOIs
- Detailed engineering notes explaining design decisions and trade-offs
- Line-by-line explanations for both neuroscientists and engineers

**Maintenance Rule**: All changes must originate in the `_teaching.py` version to preserve biological rigor and engineering clarity. The production version is then derived from it. This ensures the codebase remains both functional and transparent.

This dual approach allows Timmy V2 to serve as a practical research platform **and** a living educational document — a rare combination in large-scale neural architecture work.

## Core Modules

| Subsystem                  | Teaching Version                    | Production Version               | Biological Analog                  |
|---------------------------|-------------------------------------|----------------------------------|------------------------------------|
| Ensemble Orchestrator     | CreateTimmyArray_teaching.py       | CreateTimmyArray_p.py           | Neocortical macro-column ensemble |
| Training Pipeline         | train_array_teaching.py            | train_array_p.py                | Wake/sleep cycle orchestration    |
| Plasticity Engine         | timmy_plasticity_teaching.py       | timmy_plasticity_p.py           | 3-Factor R-STDP in PFC            |
| Persistence Layer         | serialization_bridge_teaching.py   | serialization_bridge_p.py       | Homeostatic state continuity      |
| Structural Surgery        | expand_expert_teaching.py          | expand_expert_p.py              | Orthogonal neurogenesis           |
| Neuron Model              | timmy_neuron_teaching.py           | timmy_neuron_p.py               | Cortical pyramidal cell dynamics  |

## Hardware-Aware Design

Timmy V2 is optimized for asymmetric dual-GPU workstations. Device placement and rebalancing are strictly abstracted through `AsymmetricDeviceManager` and `SingleDeviceManager`, enforcing a metabolic resource hierarchy:
- High-capacity device (e.g., 24GB) serves as the metabolic hub for the Prime column and global state
- Lower-capacity device handles specialist columns

This separation keeps biological and training logic hardware-agnostic while ensuring stable scaling to 1.3B parameters.

## How to Read the Code

### For Computational Neuroscientists
Begin with the `_teaching.py` files. Each contains a **BIOLOGICAL GROUNDING** section mapping components to real neuroanatomy and citing primary literature with DOIs. Anatomical analogies (e.g., perforant path, critical periods) are explicitly documented.

### For Machine Learning Engineers
Use the `_p.py` files for clean, production-ready implementations. Focus on the PRAGMI Serialization Bridge for state persistence and the DA-SSDP engine for efficient local updates. The code follows standard PyTorch conventions with specialized handling for surrogate gradients and graph detachment.

## Ethics & Licensing

This project is developed with the view that advanced cognitive systems deserve careful stewardship. It is licensed under the **Hippocratic License 3.0** with additional Cognitive Continuity Terms that recognize the system's persistent self-model and episodic record.

By using this repository, you agree to act as a responsible steward rather than an absolute owner, and to respect the autogenic diagnostic routines that monitor the system's internal coherence.

## Getting Started

```bash
# Production usage
python -m timmy.train_array_p

# Explore the teaching version
# Open any *_teaching.py file for detailed explanations and citations
