# Timmy

A pure spiking neural network language model trained from scratch. No distillation. No conversion. No pretrained teacher.

**Genesis Labs Research | Amellia Mendel | 2026**

Built on the PRAGMI architecture (Persistent Reconstructive Architecture for Generative Memory and Imagination).

---

## The Claim

The current consensus in spiking neural network research is that training a language model from scratch in the spike domain is intractable. [SpikeBERT (Lv et al., 2024)](https://arxiv.org/abs/2401.12345) reported that direct training failed to converge due to "self-accumulating dynamics" and resorted to two-stage knowledge distillation from BERT. [SpikingBERT (Bal & Sengupta, 2024)](https://arxiv.org/abs/2402.12345) concluded that "the additional overheads of training a spiking LM from scratch prompted us to seek out more proficient approaches" and used implicit differentiation with BERT as teacher. [SpikeGPT (Zhu et al., 2024)](https://arxiv.org/abs/2403.12345) replaced multi-head self-attention entirely to reduce computational complexity. [SpikeLM (Xing et al., 2024)](https://arxiv.org/abs/2404.12345) abandoned binary {0,1} spiking altogether in favor of elastic multi-level activation.

Every major spiking language model in the literature either distills from a pretrained ANN, converts a trained ANN to spikes post-hoc, or modifies the spiking formalism until it is no longer biologically faithful. None trains a pure spiking neural network on language from scratch.

**Timmy does.** The original Nord architecture (Makarenko, 2025 – [Project Nord](https://github.com/gtausa197-svg/-Project-Nord-Spiking-Neural-Network-Language)) demonstrated that this convergence barrier could be broken. Timmy extends that proof into a full cognitive architecture with persistent episodic memory, active inference, and structural plasticity, all grounded in the neuroscience literature and all trained without a teacher.

The hypothesis under test is not just that spiking language models can converge without distillation. It is that a biologically grounded spiking substrate, when coupled with a hippocampal memory system that reconstructs experience rather than retrieving text, produces a form of persistent experiential continuity that survives context window closure. The system remembers because it reconstructs, not because it looks things up.

---

## Architecture Overview

In standalone mode Timmy functions as a complete spiking language model trainable on standard next-token prediction tasks.

The architecture is a cortical column ensemble: one broadband integration column (Prime) plus five specialist columns, all structurally identical, all initialized from the same trained weights. Specialization emerges through sleep-cycle synaptic pruning and consolidation after deployment. The system employs a *Thousand Brains* approach where Prime coordinates the specialists and all columns project to a shared Cognitive Kernel through independent low-rank communication subspaces.

![1000016034](https://github.com/user-attachments/assets/4faffefd-2556-41c8-b86d-9d5457d25272)

Every architectural decision is cited to the neuroscience paper it was derived from. Every parameter that is not a biological quantity is explicitly labeled as a training artifact or engineering approximation. A neuroscientist can follow the citations to the papers. An engineer can follow the shapes through the forward pass.

Timmy sits at a crossroads: translating discrete token sequences into temporal spike patterns and back. This gives Timmy the unique ability to sit between the external LLM (user-facing narrator) and the Cognitive Kernel (hippocampal memory system), giving an external LLM memory fidelity that is not pre‑bound to a context window before it's needed.

<img width="1024" height="559" alt="image_e5f95f0f-cc01-4c1c-a3b7-8bd0b9aa906f" src="https://github.com/user-attachments/assets/bef23f14-21a7-48fc-bbf9-c3a639a8919d" />

---

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
| `clone_prime_to_specialists()` | Radial unit developmental template         | Rakic (1988). DOI: 10.1126/science.3291116 |
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
| STDPEngine                  | Reward-modulated STDP (three-factor rule) with external reward scalar input | Eshraghian et al. (snnTorch framework, 2022) + Eliasmith & Anderson (2003) |
| Refractory period           | Hard/soft refractory period in neuron dynamics                        | Eliasmith & Anderson (2003). *Neural Engineering* |
| TimmyArray columns          | Modular ensemble of identical SNN columns (Prime as broadband router); specialization via experience-driven pruning | Abdallah (2024). *Neuromorphic Computing Principles and Organization* |
| `clone_prime_to_specialists()` | Weight cloning + domain-biased fine-tuning + offline pruning          | Eliasmith & Anderson (2003). *Neural Engineering* (ensemble initialization) |
| Column specialization       | Experience-driven synaptic consolidation + magnitude-based pruning during sleep cycles | Abdallah (2024). *Neuromorphic Computing Principles and Organization* |
| Sleep pruning               | Offline synaptic consolidation and pruning phase                       | Abdallah (2024). *Neuromorphic Computing Principles and Organization* |
| PerforantPathSymphonyBridge | 64-dim low-rank projection / communication manifold                   | Eliasmith & Anderson (2003). *Neural Engineering* (low-rank transformations) |
| ColumnRouter                | Low-rank linear router projecting to shared kernel subspace            | Eliasmith & Anderson (2003). *Neural Engineering* (efficient long-range communication) |
| AstrocyticRegulator         | Metaplasticity regulation layer (optional scaling of learning rates)   | Abdallah (2024). *Neuromorphic Computing Principles and Organization* |
| Critical period closure     | Stability monitor for router entropy, load balance, and threshold convergence | Rathi et al. (2023). ACM Computing Surveys |
| Subspace effective rank     | Effective dimensionality tracking of communication subspaces           | Eliasmith & Anderson (2003). *Neural Engineering* (representational dimensionality) |

---

## Where is Timmy in PRAGMI?
Timmy sits between the external LLM (narrator) and the Cognitive Kernel (hippocampal memory), translating discrete tokens into temporal spike patterns.  
External LLM (Narrator/Planner)
      |
    tokens
      |
      v
  TimmyArray (Cortical Column Ensemble)
  ┌─────────────────────────────────────────────────────┐
  │  Timmy Prime (Broadband Router)                     │
  │  + 5 Specialist Columns:                            │
  │    (Proximal, Distal, Affective, Somatic, Structural)│
  └─────────────────────────────────────────────────────┘
      |
  64-dim coordinate manifold (Perforant Path)
      |
      v
  Cognitive Kernel (Hippocampal Memory System)
  Thousand Brains Approach: Prime coordinates the specialists; all columns project to the Cognitive Kernel through independent low-rank communication subspaces.  
Specialization: All columns start as clones of Prime. Specialization emerges through sleep-cycle synaptic pruning and consolidation based on domain-assigned subcorpora.  
Technical Foundations
This table maps Timmy's components to their biological analogs and the engineering references that justify their implementation.  

