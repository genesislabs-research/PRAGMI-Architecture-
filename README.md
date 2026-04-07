Timmy
A pure spiking neural network language model trained from scratch. No distillation. No conversion. No pretrained teacher.

Genesis Labs Research | Amellia Mendel | 2026

Built on the PRAGMI architecture (Persistent Reconstructive Architecture for Generative Memory and Imagination).

The Claim
The current consensus in spiking neural network research is that training a language model from scratch in the spike domain is intractable. SpikeBERT (Lv et al., 2024) reported that direct training failed to converge due to "self-accumulating dynamics" and resorted to two-stage knowledge distillation from BERT. SpikingBERT (Bal & Sengupta, 2024) concluded that "the additional overheads of training a spiking LM from scratch prompted us to seek out more proficient approaches" and used implicit differentiation with BERT as teacher. SpikeGPT (Zhu et al., 2024) replaced multi-head self-attention entirely to reduce computational complexity. SpikeLM (Xing et al., 2024) abandoned binary {0,1} spiking altogether in favor of elastic multi-level activation.

Every major spiking language model in the literature either distills from a pretrained ANN, converts a trained ANN to spikes post-hoc, or modifies the spiking formalism until it is no longer biologically faithful. None trains a pure spiking neural network on language from scratch.

Timmy does. The original Nord architecture (Makarenko, 2025 [Project Nord](https://github.com/gtausa197-svg/-Project-Nord-Spiking-Neural-Network-Language) 
) demonstrated that this convergence barrier could be broken. Timmy extends that proof into a full cognitive architecture with persistent episodic memory, active inference, and structural plasticity, all grounded in the neuroscience literature and all trained without a teacher.

The hypothesis under test is not just that spiking language models can converge without distillation. It is that a biologically grounded spiking substrate, when coupled with a hippocampal memory system that reconstructs experience rather than retrieving text, produces a form of persistent experiential continuity that survives context window closure. The system remembers because it reconstructs, not because it looks things up.# Timmy_Neuron


Timmy is entirely built on the PRAGMI architecture (Persistent Reconstructive Architecture for Generative Memory and Imagination)  Timmy is a pure spiking neural network.

In standalone mode Timmy functions as a complete spiking language model trainable on standard next-token prediction tasks. 



The architecture is a cortical column ensemble: one broadband integration column (Prime) plus five specialist columns, all structurally identical, all initialized from the same trained weights. Specialization emerges through sleep-cycle synaptic pruning and consolidation after deployment. The system employs a Thousand Brains approach where Prime coordinates the specialists and all columns project to a shared Cognitive Kernel through independent low-rank communication subspaces.

![1000016034](https://github.com/user-attachments/assets/4faffefd-2556-41c8-b86d-9d5457d25272)
Every architectural decision is cited to the neuroscience paper it was derived from. Every parameter that is not a biological quantity is explicitly labeled as a training artifact or engineering approximation. A neuroscientist can follow the citations to the papers. An engineer can follow the shapes through the forward pass.

Timmy sits at a crossroads having the ability of translating discrete token sequences into temporal spike patterns and back this gives Timmy the unique ability to sit between the external LLM (user-facing narrator) and the Cognitive Kernel (hippocampal memory system) giving an externa llm memory fidelity that is not prebound a to context window before it's needed.


<img width="1024" height="559" alt="image_e5f95f0f-cc01-4c1c-a3b7-8bd0b9aa906f" src="https://github.com/user-attachments/assets/bef23f14-21a7-48fc-bbf9-c3a639a8919d" />

```
## Biological Grounding

| Component                    | Biological Analog                              | Primary Reference |
|------------------------------|------------------------------------------------|-------------------|
| TemporalSpikeEncoder        | Thalamocortical relay                         | Sherman & Guillery (2002). DOI: 10.1098/rstb.2002.1161 |
| AssociativeLIF              | Cortical pyramidal cell                       | Gerstner et al. (2014). DOI: 10.1017/CBO9781107447615 |
| Cascade amplification       | Minicolumn lateral excitation                 | Mountcastle (1997). DOI: 10.1093/brain/120.4.701 |
| ATanSurrogate               | None — training artifact only                 | Neftci et al. (2019). DOI: 10.1109/MSP.2019.2931595 |
| SpikingSynapticResonance    | Communication through coherence               | Fries (2005). DOI: 10.1016/j.tics.2005.08.011 |
| SpikeDrivenMoE              | Association cortex specialization             | Felleman & Van Essen (1991). DOI: 10.1093/cercor/1.1.1 |
| MemoryCortex                | PFC delay-period persistent activity          | Fuster (1973). DOI: 10.1152/jn.1973.36.1.61 |
| STDPEngine                  | Three-factor reward-modulated plasticity      | Bi & Poo (1998). DOI: 10.1523/JNEUROSCI.18-24-10464.1998 |
| Refractory period           | Na⁺ channel inactivation                      | Hodgkin & Huxley (1952). DOI: 10.1113/jphysiol.1952.sp004764 |
| TimmyArray columns          | Cortical column ensemble                      | Mountcastle (1997). DOI: 10.1093/brain/120.4.701 |
| clone_prime_to_specialists() | Radial unit developmental template           | Rakic (1988). DOI: 10.1126/science.3291116 |
| Column specialization       | Sleep-driven experience-dependent plasticity  | Tononi & Cirelli (2003). DOI: 10.1093/brain/awg100 |
| Sleep pruning               | Synaptic tagging and capture, REM refinement  | Seibt & Frank (2019). DOI: 10.3389/fnsys.2019.00002 |
| PerforantPathSymphonyBridge | Communication subspace                        | Semedo et al. (2019). DOI: 10.1016/j.neuron.2019.01.026 |
| ColumnRouter                | Long-range horizontal coordination            | See et al. (2018). DOI: 10.7554/eLife.35587 |
| AstrocyticRegulator         | Tripartite synapse metaplasticity             | Araque et al. (1999). DOI: 10.1016/S0166-2236(98)01349-6 |
| Critical period closure     | Experience-dependent threshold stabilization  | Huang et al. (2022). DOI: 10.1007/s12021-022-09576-5 |
| Subspace effective rank     | Representational dimensionality               | Roy & Vetterli (2007). DOI: {To be added later.} |

## Engineering Implementation Reference

An engineer can read this table independently. It shows the exact code-level term used in the repository and the peer-reviewed engineering or computational reference that justifies the design choice.

| Component                    | Engineering / Implementation Term                                      | Primary Engineering Reference |
|------------------------------|------------------------------------------------------------------------|-------------------------------|
| TemporalSpikeEncoder        | Multi-scale temporal current injection (fast gamma-band + slow theta-band bases; optional gated float embedding fusion) | Eliasmith & Anderson (2003). *Neural Engineering*, Ch. 3–4 |
| AssociativeLIF              | Leaky Integrate-and-Fire neuron with cascade amplification and persistent membrane state across sequence chunks | Eliasmith & Anderson (2003). *Neural Engineering*, Ch. 4–6 |
| Cascade amplification       | Local recurrent excitation / residual connections within spiking blocks | Eliasmith & Anderson (2003). *Neural Engineering* (recurrent coupling) |
| ATanSurrogate               | Surrogate gradient (atan) for backpropagation-through-time in spiking networks | Neftci et al. (2019). IEEE Signal Processing Magazine |
| SpikingSynapticResonance    | Phase synchronization for efficient long-range signaling               | Eliasmith & Anderson (2003). *Neural Engineering* (dynamical synchronization) |
| SpikeDrivenMoE              | Sparse cluster-based Mixture-of-Experts with spike-driven routing and load-balanced expert dispatch | Abdallah (2024). *Neuromorphic Computing Principles and Organization* |
| MemoryCortex                | Slow-decay LIF working memory buffer + multi-head temporal attention readout | Eliasmith & Anderson (2003). *Neural Engineering*, Ch. 7–8 |
| STDPEngine                  | Reward-modulated STDP (three-factor rule) with external reward scalar input | Eshraghian et al. (snnTorch, 2022) + Eliasmith & Anderson (2003) |
| Refractory period           | Hard/soft refractory period in neuron dynamics                        | Eliasmith & Anderson (2003). *Neural Engineering* |
| TimmyArray columns          | Modular ensemble of identical SNN columns (Prime as broadband router); specialization via experience-driven pruning | Abdallah (2024). *Neuromorphic Computing Principles and Organization* |
| clone_prime_to_specialists() | Weight cloning + domain-biased fine-tuning + offline pruning          | Eliasmith & Anderson (2003). *Neural Engineering* (ensemble initialization) |
| Column specialization       | Experience-driven synaptic consolidation + magnitude-based pruning during sleep cycles | Abdallah (2024). *Neuromorphic Computing Principles and Organization* |
| Sleep pruning               | Offline synaptic consolidation and pruning phase                       | Abdallah (2024). *Neuromorphic Computing Principles and Organization* |
| PerforantPathSymphonyBridge | 64-dim low-rank projection / communication manifold                   | Eliasmith & Anderson (2003). *Neural Engineering* (low-rank transformations) |
| ColumnRouter                | Low-rank linear router projecting to shared kernel subspace            | Eliasmith & Anderson (2003). *Neural Engineering* (efficient long-range communication) |
| AstrocyticRegulator         | Metaplasticity regulation layer (optional scaling of learning rates)   | Abdallah (2024). *Neuromorphic Computing Principles and Organization* |
| Critical period closure     | Stability monitor for router entropy, load balance, and threshold convergence | Rathi et al. (2023). ACM Computing Surveys |
| Subspace effective rank     | Effective dimensionality tracking of communication subspaces           | Eliasmith & Anderson (2003). *Neural Engineering* (representational dimensionality) |

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


```
## Engineering Implementation Reference

An engineer can read this table independently. It shows the exact code-level term used in the repository and the peer-reviewed engineering book or major computational reference that justifies the design choice.

| Component                    | Engineering / Implementation Term                                      | Primary Engineering Reference |
|------------------------------|------------------------------------------------------------------------|-------------------------------|
| TemporalSpikeEncoder        | Multi-scale temporal current injection (fast gamma-band + slow theta-band bases; optional gated float embedding fusion) | Eliasmith & Anderson (2003). *Neural Engineering*, Ch. 3–4 (temporal encoding and multi-scale representations) |
| AssociativeLIF              | Leaky Integrate-and-Fire neuron with cascade amplification and persistent membrane state across sequence chunks | Eliasmith & Anderson (2003). *Neural Engineering*, Ch. 4–6 (LIF dynamics and population-level persistence) |
| Cascade amplification       | Local recurrent excitation / residual connections within spiking blocks | Eliasmith & Anderson (2003). *Neural Engineering* (recurrent coupling for amplification) |
| ATanSurrogate               | Surrogate gradient (atan) for backpropagation-through-time in spiking networks | Neftci et al. (2019). IEEE Signal Processing Magazine (surrogate gradient methods for SNN training) |
| SpikingSynapticResonance    | Phase synchronization for efficient long-range signaling               | Eliasmith & Anderson (2003). *Neural Engineering* (dynamical synchronization in population codes) |
| SpikeDrivenMoE              | Sparse cluster-based Mixture-of-Experts with spike-driven routing and load-balanced expert dispatch | Abdallah (2024). *Neuromorphic Computing Principles and Organization* (sparse expert routing in scalable SNN hardware) |
| MemoryCortex                | Slow-decay LIF working memory buffer + multi-head temporal attention readout | Eliasmith & Anderson (2003). *Neural Engineering*, Ch. 7–8 (persistent activity and working memory dynamics) |
| STDPEngine                  | Reward-modulated STDP (three-factor rule) with external reward scalar input | Eshraghian et al. (snnTorch framework, 2022) + Eliasmith & Anderson (2003) (plasticity implementation in large-scale models) |
| Refractory period           | Hard/soft refractory period in neuron dynamics (prevents immediate re-firing) | Eliasmith & Anderson (2003). *Neural Engineering* (realistic neuron refractory modeling) |
| TimmyArray columns          | Modular ensemble of identical SNN columns (Prime as broadband router); specialization via experience-driven pruning | Abdallah (2024). *Neuromorphic Computing Principles and Organization* (modular column-style SNN architectures) |
| clone_prime_to_specialists() | Weight cloning + domain-biased fine-tuning + offline pruning          | Eliasmith & Anderson (2003). *Neural Engineering* (ensemble initialization and specialization) |
| Column specialization       | Experience-driven synaptic consolidation + magnitude-based pruning during sleep cycles | Abdallah (2024). *Neuromorphic Computing Principles and Organization* (offline pruning for specialization in hardware SNNs) |
| Sleep pruning               | Offline synaptic consolidation and pruning phase                       | Abdallah (2024). *Neuromorphic Computing Principles and Organization* (sleep-like refinement in neuromorphic systems) |
| PerforantPathSymphonyBridge | 64-dim low-rank projection / communication manifold (independent subspaces per column) | Eliasmith & Anderson (2003). *Neural Engineering* (low-rank transformations between neural populations) |
| ColumnRouter                | Low-rank linear router projecting to shared kernel subspace            | Eliasmith & Anderson (2003). *Neural Engineering* (efficient long-range communication via low-rank projections) |
| AstrocyticRegulator         | Metaplasticity regulation layer (optional scaling of learning rates)   | Abdallah (2024). *Neuromorphic Computing Principles and Organization* (auxiliary regulation in large SNN systems) |
| Critical period closure     | Stability monitor for router entropy, load balance, and threshold convergence | Rathi et al. (2023). ACM Computing Surveys (critical-period-like convergence in SNN training pipelines) |
| Subspace effective rank     | Effective dimensionality tracking of communication subspaces           | Eliasmith & Anderson (2003). *Neural Engineering* (representational dimensionality in neural ensembles) |

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

## Quick Start (TimmyArray)

## Cross-Referencing Biological and Computational Foundations

Timmy is designed at the intersection of computational neuroscience and neuromorphic engineering. The following core references provide a clear mapping between biological principles and their practical implementation in code.

### Primary Neuroscience Reference
**Gerstner, W. & Kistler, W.M. (2002). *Spiking Neuron Models: Single Neurons, Populations, Plasticity*. Cambridge University Press.**  
An updated and freely available version is *Neuronal Dynamics: From single neurons to networks and models of cognition* by Gerstner et al. (2014), available at https://neuronaldynamics.epfl.ch.

This text serves as the foundational reference for the neuron models, synaptic dynamics, and plasticity rules used throughout the architecture. Relevant sections include:
- Leaky Integrate-and-Fire dynamics and variants (Chapters 4–5 and corresponding sections in *Neuronal Dynamics*)
- Spike-Timing-Dependent Plasticity (STDP) and three-factor learning rules (Chapter 10 and *Neuronal Dynamics* Chapter 19)
- Population coding and network-level behavior

### Engineering and Implementation Reference
**Eliasmith, C. & Anderson, C.H. (2003). *Neural Engineering: Computation, Representation, and Dynamics in Neurobiological Systems*. MIT Press.**

This volume provides practical guidance for translating neurobiological principles into scalable computational models, vector representations, and dynamical systems suitable for simulation and deployment. It serves as a bridge from biological grounding to efficient code-level realizations in frameworks such as PyTorch.

Additional practical resources for implementation include the snnTorch library (gradient-based training of spiking neural networks) and documentation from frameworks such as Brian2 and Lava.

### Usage in This Codebase
- Biological concepts are cited with their primary neuroscience references and DOIs in the table below.
- Engineering approximations (such as surrogate gradients, low-rank subspaces, and EMA readouts) are explicitly labeled as training artifacts.
- The implementation in files such as `timmy_neuron.py`, `timmy_blocks.py`, and the STDP engine follows the mappings established in the references above.

## Biological Grounding

| Component                    | Biological Analog                              | Engineering / Implementation Term                                      | Primary Neuroscience Reference                  | Computational Reference |
|------------------------------|------------------------------------------------|------------------------------------------------------------------------|-------------------------------------------------|--------------------------|
| TemporalSpikeEncoder        | Thalamocortical relay                         | Multi-scale temporal current injection (gamma + theta basis; gated float embedding fusion) | Sherman & Guillery (2002)                      | Gerstner & Kistler (2002, Ch. 4–5); Eliasmith & Anderson (2003) |
| AssociativeLIF              | Cortical pyramidal cell                       | Leaky Integrate-and-Fire (LIF) neuron with cascade amplification and persistent membrane state | Gerstner et al. (2014)                         | Gerstner & Kistler (2002, Ch. 4); *Neuronal Dynamics* Ch. 6–8; Eliasmith & Anderson (2003, Ch. 3–4) |
| Cascade amplification       | Minicolumn lateral excitation                 | Local recurrent excitation within spiking blocks                       | Mountcastle (1997)                             | Gerstner & Kistler (population coding sections) |
| ATanSurrogate               | — (training artifact only)                    | Surrogate gradient for differentiable spiking backpropagation          | Neftci et al. (2019)                           | snnTorch implementation |
| SpikeDrivenMoE              | Association cortex specialization             | Sparse, cluster-based Mixture-of-Experts with spike-driven routing and load balancing | Felleman & Van Essen (1991)                    | Gerstner & Kistler (network chapters) |
| MemoryCortex                | PFC delay-period persistent activity          | Slow-decay LIF working memory buffer + multi-head temporal attention readout | Fuster (1973)                                  | Gerstner & Kistler (*Neuronal Dynamics* Ch. 17) |
| STDPEngine                  | Three-factor reward-modulated plasticity      | Reward-modulated STDP (three-factor rule) with external reward scalar  | Bi & Poo (1998)                                | Gerstner & Kistler (Ch. 10 / *Neuronal Dynamics* Ch. 19.4); Eliasmith & Anderson (2003) |
| TimmyArray columns          | Cortical column ensemble                      | Modular ensemble of identical SNN columns (Prime as broadband router); specialization via experience-driven pruning | Mountcastle (1997); Hawkins et al. (2019)      | Gerstner & Kistler (population sections); Eliasmith & Anderson (2003) |
| PerforantPathSymphonyBridge | Communication subspace                        | 64-dimensional low-rank projection manifold for inter-column and kernel communication | Semedo et al. (2019)                           | Eliasmith & Anderson (2003, transformation principles) |
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
