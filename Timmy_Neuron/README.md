# Timmy_Neuron
This repository is refactored and extended from Project Nord,[Project Nord](https://github.com/gtausa197-svg/-Project-Nord-Spiking-Neural-Network-Language) a biologically grounded spiking neural network (SNN) language model. Project Nord represents a technical milestone in the field, successfully challenging the long-standing consensus that training a language model from scratch in the spike domain was intractable.
**A biologically grounded spiking neural network language model**

Timmy is entirely built on the PRAGMI architecture (Persistent Reconstructive Architecture for Generative Memory and Imagination). 

In standalone mode Timmy functions as a complete spiking language model trainable on standard next-token prediction tasks without distillation of any kind. Timmy is a pure spiking neural network.

Timmy sits at a crossroads having the ability of translating discrete token sequences into temporal spike patterns and back this gives Timmy the unique ability to sit between the external LLM (user-facing narrator) and the Cognitive Kernel (hippocampal memory system) possibly giving an externa llm memory fidelity that is not prebound a to context window before it's needed.




<img width="1024" height="559" alt="image_e5f95f0f-cc01-4c1c-a3b7-8bd0b9aa906f" src="https://github.com/user-attachments/assets/bef23f14-21a7-48fc-bbf9-c3a639a8919d" />

![1000016034](https://github.com/user-attachments/assets/4faffefd-2556-41c8-b86d-9d5457d25272)

The architecture is a cortical column ensemble: one broadband integration column (Prime) plus five specialist columns, all structurally identical, all initialized from the same trained weights. Specialization emerges through sleep-cycle synaptic pruning and consolidation after deployment. The system employs a Thousand Brains approach where Prime coordinates the specialists and all columns project to a shared Cognitive Kernel through independent low-rank communication subspaces.

Every architectural decision is cited to the neuroscience paper it was derived from. Every parameter that is not a biological quantity is explicitly labeled as a training artifact or engineering approximation. A neuroscientist can follow the citations to the papers. An engineer can follow the shapes through the forward pass.

# What the files Do

## Timmy Neuron Project Documentation
## array_monitor.py
is the diagnostic instrumentation module for Phase two divergence training of the cortical column ensemble. It evaluates whether specialist columns are genuinely differentiating from the prime integration column by tracking four key metrics: router entropy to ensure balanced activation, activation frequency to monitor stability, subspace effective rank to measure representational dimensionality, and cosine distance to quantify structural divergence. By accumulating these signals over time and generating alerts for routing collapse or failure to diverge, it provides the essential telemetry needed to verify that the ensemble is developing specialized and non overlapping functional networks.
## astrocytic_regulator_v3.py
models the active role of astrocytes in regulating synaptic plasticity through the tripartite synapse and calcium mediated metaplasticity. It acts as a homeostatic governor by monitoring local neural activity for glutamate spillover and translating it into an intracellular calcium signal. This calcium wave then computes a plasticity modifier for each neuron that adjusts the learning rate dynamically based on recent activity levels. By implementing a sliding threshold that raises resistance during high activity to prevent runaway potentiation and lowers it during quiescence to facilitate encoding, the module ensures long term stability across the learning substrate while preserving its metabolic state across episodic boundaries.
## CreateTimmyArray.py
assembles the core cortical column ensemble where a central broadband integration column coordinates multiple distinct specialist columns sharing a unified cognitive kernel. It initializes all columns as identical replicas before divergence and uses a dedicated router to distribute processing load based on the working memory state of the prime column. During execution, it runs the selected specialists in parallel and funnels their outputs through a perforant path symphony bridge which compresses the active representations into a single coordinate vector for hippocampal processing. It then completes the learning loop by routing external mismatch and novelty feedback back exclusively to the prime column and the specific specialist that won the routing bid for that episode.
## firstday.py
is the implementation of the developmental transition between the template phase and the experience-dependent specialization phase for the cortical column ensemble. It initiates this process by copying the completely trained state of the prime integration column into every specialist column simultaneously, ensuring they all start as identical replicas. Immediately following this clone event, it executes a targeted initial sleep cycle that applies specialty-directed synaptic scaling to begin functionally differentiating the columns. By completing this initial divergence and saving the resulting post-sleep state, this script officially transitions the array into its mature, specialized operational phase.
## README.md
serves as the primary documentation for the Timmy Neuron repository, explaining its role as the bridge layer within the broader PRAGMI architecture. It details how the system translates discrete token sequences into temporal spike patterns and outlines the core Thousand Brains design featuring a central integration column and multiple specialists. The document also provides critical integration instructions for injecting external continuous embeddings and novelty reward signals into the network. Furthermore, it strictly defines the project's documentation standard, mandating that every architectural component be explicitly grounded in biological literature with accompanying citations.
## REFERENCES.md
is the comprehensive bibliography and biological grounding ledger for the entire PRAGMI codebase. It provides a complete, organized list of all academic papers and DOIs that are cited within the project to justify architectural decisions, parameter values, and biological claims. The document systematically categorizes these sources into specific neuroscience domains such as single neuron dynamics, synaptic plasticity, and cortical architecture, while explicitly mapping each citation to the exact Python files where its principles are implemented.
## smoke_test.py
is an end to end diagnostic script that verifies the structural integrity of the entire TimmyArray training pipeline. It runs a rapid sequential check covering model construction, forward passes, Phase one training, the critical period probe, Phase two divergence training, and the array monitor. It uses a deliberately miniaturized configuration to complete in under sixty seconds on a standard processor. It does not evaluate whether the model is actually learning, but rather ensures that the entire code execution path is free of crashes or structural bugs before committing expensive compute resources to a full scale training run.
## timmy_attention.py
is the implementation of the network's attention mechanism, modeled biologically as synaptic resonance between spiking neural populations. It extracts meaningful attention scores from discrete spike trains by filtering query and key populations through leaky integrate and fire dynamics and applying rotary position embeddings. It utilizes a top k sparsification strategy to limit each token to attending to a maximum of sixty four other tokens. This dual purpose design maintains biological plausibility by mimicking limited cortical connectivity while significantly reducing the computational overhead compared to a standard full attention matrix.
## timmy_blocks.py
assembles the core processing blocks that make up the three zone hierarchical architecture of the sensory, association, and executive regions. Each block executes one round of synaptic resonance followed by a feed forward transformation, incorporating residual connections and layer normalization. The executive zones specifically enforce non negative output clamping to prevent inhibitory signals from improperly propagating backward through the network. The module also includes an asymmetric spike rate loss function that heavily penalizes neurons that fall silent while lightly penalizing overactive neurons, ensuring stable population firing rates throughout the network.
## timmy_criticalperiodprobe.py
functions as a convergence monitor to determine when the prime integration column has finished its initial developmental phase and is ready to coordinate the rest of the ensemble. It acts as the official gatekeeper between Phase one and Phase two of training. It calculates coordination readiness by tracking the stability of spike thresholds in the memory cortex and the routing entropy in the association zones. Once these signals stop drifting and stabilize, the probe signals that the column has settled on a consistent temporal integration strategy and can safely be cloned to the specialist columns.
## timmy_data.py
is the data pipeline module responsible for feeding tokenized text into the training loop. It handles input from three distinct sources including HuggingFace datasets, local text files, and pre-tokenized tensor files. It operates by converting raw text into fixed-length chunks using a configurable tokenizer, batching these sequences, and yielding them in a format ready for the network's forward pass. By default, it utilizes the Llama 3 tokenizer to process the vocabulary, ensuring the spiking neural network receives a steady, properly formatted stream of discrete tokens during the learning cycle.
## timmy_encoder.py
is the multi-scale temporal spike encoder that serves as the translation layer between discrete token space and spiking dynamics. It models the thalamocortical relay by receiving compressed input and converting it into a multi-scale temporal current drive, utilizing a fast basis for gamma-band oscillatory modulation and a slow basis for theta-band envelope modulation. It also features an external float embedding path with a learnable gate, which allows continuous vectors from outside the token vocabulary to be injected directly into the encoder. This provides the critical interface through which the cognitive kernel returns reconstructed episodic coordinates and external sensor streams can enter the spiking pipeline.
## timmy_experts.py
is the implementation of the spike-driven mixture of experts layer utilized within the association zone. It routes incoming neural activity to functionally specialized subpopulations based on the content of the input, modeling how biological association cortices process different aspects of sensory data. It achieves this by using minicolumn cluster firing rates as the routing signal instead of relying on a separate learned routing network, mapping each expert to a contiguous block of clusters. To ensure stability, it incorporates a load balancing loss function that prevents expert collapse, ensuring that the network does not learn to route all tokens to just one or two experts while ignoring the rest.
## timmy_memory.py
is the implementation of the short-term working memory cortex, modeled on the sustained firing activity observed in the prefrontal cortex during delay periods. It utilizes slow-decaying associative leaky integrate-and-fire neurons to act as a leaky buffer that accumulates and retains spike-driven input across a single processing window. It functions strictly as a fast, within-context scratchpad rather than a long-term episodic store, allowing downstream layers to read from the retained temporal context via multi-head temporal attention. This persistent activity provides the network with the essential capacity to maintain task-relevant information across sequential processing steps.
## timmy_model.py
is the implementation of the complete spiking language model that serves as the subconscious bridge between the external token world and the cognitive kernel. It translates discrete token sequences into temporal spike patterns and back by modeling the hierarchical organization of the mammalian neocortex. It processes information sequentially through sensory blocks for low level feature extraction, association blocks for expert routing, a memory cortex for short term context, and executive blocks for decision selection. By forcing non negative outputs in the executive zone and integrating long term context from the episodic memory layer, it provides a stable and biologically grounded foundation for next token prediction.
## timmy_neuron.py
is the implementation of the fundamental spiking unit used throughout the sensory, association, and executive zones of the network. It extends the standard leaky integrate and fire neuron model with three critical biological mechanisms to support stable cognitive processing. It applies synaptic current filtering to smooth discrete input spikes into continuous integrable currents, enforces an absolute refractory period to prevent physically unrealistic rapid firing, and uses cascade amplification to model lateral excitatory connections within cortical minicolumns. By combining these dynamics, it ensures that the network translates inputs into biologically plausible and mathematically stable temporal spike trains.
## timmy_plasticity.py
is the implementation of the reward modulated spike timing dependent plasticity engine used for online synaptic modification. It utilizes a three factor learning rule where the relative timing of presynaptic and postsynaptic spikes determines the direction of weight changes, while a modulatory reward signal gates the overall magnitude of the update. It isolated this learning mechanism specifically to the executive zone to prevent instability in the sensory and association regions which rely solely on backpropagation. Additionally, it provides an external reward signal path that allows the cognitive kernel or environment to directly inject reward values and transition the system from self supervised to externally guided learning.
## train_timmy.py
is the primary training script for the Timmy spiking language model. It handles the full learning loop by orchestrating data loading from sources like HuggingFace or local text files and managing the optimization process. The script utilizes an AdamW optimizer with a cosine learning rate schedule and incorporates several specialized loss functions including asymmetric spike rate regularization to prevent neural silence and mixture of experts load balancing. It also supports mixed precision training and optional reward modulated spike timing dependent plasticity for the executive zone, ensuring that the model can be trained efficiently while maintaining biological plausibility.
## train_array.py
is the implementation of the two-phase training workflow for the TimmyArray cortical column ensemble. In the first phase, it trains the prime integration column alone on a broad data distribution to establish a generalist coordination scaffold. In the second phase, it enables simultaneous training across all columns where each specialist receives domain-specific input to drive functional divergence through experience-dependent plasticity. Throughout this process, the script manages the dynamic load balancing of the column router and ensures that differentiation emerges naturally from the input statistics rather than through rigid architectural constraints.


---

## Where is Timmy in PRAGMI

```
External LLM  (narrator, planner, user-facing) has not been implemented yet 
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
