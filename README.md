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
