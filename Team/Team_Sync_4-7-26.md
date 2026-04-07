

## PRAGMI / Timmy — Team Sync Ai generated (obviously) 4_4_26 

## living record of architectural decisions, their reasoning, and the state of every file in the codebase
every working session one is created by AI summary which {in theory} should capture the reasoning for every decision {hopefully} not just the conclusion. Anyone reading this document should be able to pick up without re-litigating settled ground or wonder why a decision was made.

Chronology rule: when extractions contradict each other, the more recent one wins. When anything contradicts decisions made the newest wins. This document supersedes Team_Sync_4-4-26.md on any point of conflict.

## Repository Structure


Primary repository: https://github.com/genesislabs-research/PRAGMI-Architecture-

```
PRAGMI-Architecture-/
├── README.md
├── READMEded.md
├── Teaching/
│   ├── cognitive_kernel_core_t.py      ← hippocampal memory system (1,403 lines, 7 tests)
│   ├── cortical_buffer_t.py            ← PFC working memory (11 tests)
│   ├── world_model_ensemble_t.py       ← predictive coding ensemble (10 tests)
│   ├── neuromodulator_broadcast_t.py   ← DA/NE/ACh/5-HT + maturity (14 tests)
│   └── epistemic_selector_t.py         ← active batch selection (7 tests)
├── Core/
│   ├── cognitive_kernel_core_c.py      ← stripped production kernel (NEW this session)
│   ├── cortical_buffer_p.py
│   ├── world_model_ensemble_p.py
│   ├── neuromodulator_broadcast_p.py
│   └── epistemic_selector_p.py
└── Team/
    ├── Team_Sync_4-4-26.md
    ├── Team_Sync_4-7-26.md             ← this document
    ├── small_core_train.py             ← integration harness (NEW this session)
    ├── cognitive_kernel_trainer.py     ← standalone kernel trainer
    ├── cognitive_kernel_base_for_testing_c.py  ← original kernel delivery for Vovo
    ├── cognitive_kernel_base_for_testing_t.py  ← original kernel teaching version
    └── Training/
        └── ded/
            ├── Train_tiny_Tim.py
            └── tiny_tim.py
```

Legacy repositories (reference only, not active development):
- https://github.com/genesislabs-research/Timmy_Neuron (previous Timmy, Timmy SNN column files live here)
- https://github.com/gtausa197-svg/-Project-Nord-Spiking-Neural-Network-Language-Model (original Nord)

## What Was Built This Session (April 7, 2026)


**Deliverable 1: cognitive_kernel_core_c.py (609 lines)**

Production version of the cognitive kernel teaching file. Stripped of all docstrings, comments, and annotations. All 7 self-tests pass. 699.1K parameters, same distribution as teaching version. One defect identified during Opus review: the total_neurons computation on line 55 was mangled during stripping (inline comments removed from a multi-line addition, leaving an empty tuple). The non-anatomical fallback path would crash and self.total_neurons reports "()" instead of 1472. Fix required before upload: replace the empty tuple with the explicit sum of population dimensions. This is a one-line fix. The anatomical path (default) works correctly, which is why all 7 tests pass despite the bug.

**Deliverable 2: small_core_train.py (1,479 lines, 18 integration tests)**

Integration training harness wiring together all five PRAGMI subsystems into a single training loop. This file was described in the April 5 session report as a deliverable but was never actually produced until now. Opus wrote the complete spec, Sonnet executed it, Opus reviewed the output.

The harness uses a LightweightEncoder (small feedforward network, engineering stand-in for TimmyModel) and a FixedCoordinateProjection (random orthonormal matrix via QR decomposition, stand-in for PerforantPathSymphonyBridge) to generate 64-dim coordinate vectors from text data. The coordinates feed through the full subsystem chain: CognitiveKernel for storage/reconstruction, WorldModelEnsemble for prediction/uncertainty, NeuromodulatorBroadcast for DA/NE global signals, EpistemicSelector for diagnostic batch evaluation, CorticalBuffer instantiated but idle (needs TimmyModel membrane state).

Two separate optimizers: encoder optimizer (Adam, encoder params only) and kernel optimizer (Adam, kernel learnable params only, W_recurrent and distance_mask excluded as buffers). WorldModelEnsemble uses its internal optimizer. FixedCoordinateProjection contributes zero gradient.

Training loop per step: encode tokens, project to coordinates (no grad), kernel forward, kernel loss (MSE against EC-normalized target), kernel backprop, encoder loss (MSE against detached reconstruction), encoder backprop, world model update (delay-one target), DA update from normalized loss signal, NE update from ensemble variance, astrocytic eta modulates kernel LR, epistemic selector diagnostic every eval_interval steps, sleep consolidation every sleep_interval steps, checkpoint every 1000 steps.

Data: FineWeb-Edu primary, TinyStories fallback, synthetic random fallback if both fail. Character-level tokenization (no external tokenizer dependency). CLI with --test-only flag.

18 integration tests validate all subsystem contracts before training starts. Tests cover: encoder shape, projection shape and determinism, gradient isolation, kernel forward/accumulation/sleep/serialization, world model forward/convergence/HOT round-trip, neuromodulator updates/maturity/HOT round-trip, epistemic selector randomness at low maturity and scoring, cortical buffer update/reset, full pipeline end-to-end.

## Decisions Made This Session


**Integration harness designed to run without TimmyModel.**
Reason: The Timmy SNN files (timmy_neuron.py, timmy_encoder.py, timmy_attention.py, timmy_experts.py, timmy_memory.py, timmy_blocks.py, timmy_plasticity.py, timmy_state.py, timmy_model.py) live in the Timmy_Neuron repo and have not been migrated to PRAGMI-Architecture-. Waiting for migration would block all integration work. The LightweightEncoder and FixedCoordinateProjection are explicitly labeled as stand-ins with replacement instructions in their docstrings.

**Encoder loss is MSE against detached kernel reconstruction.**
Reason: The encoder needs a training signal but its gradient must not flow through the kernel. The kernel's reconstruction (subiculum output) is the natural target because it represents what the hippocampal system believes the input should look like. Detaching prevents gradient coupling. When TimmyModel replaces the encoder, the gradient path will be different (TimmyModel has its own STDP-based plasticity, not backprop through the reconstruction).

**CorticalBuffer is instantiated but idle in the integration harness.**
Reason: The cortical buffer requires TimmyModel's membrane state (v_mem) to produce meaningful state. Without the SNN, updating the buffer from feedforward encoder activations would train it on the wrong signal distribution. It is exercised in the test suite to validate its contracts but left idle during training.

**Opus review workflow established for Sonnet-produced code.**
Reason: Sonnet executes mechanical code generation effectively from detailed specs but does not catch subtle interface mismatches (e.g., the total_neurons stripping bug). Opus writes specs, Sonnet executes, Amellia brings results back to Opus for review before repo upload. This workflow minimizes Opus usage while maintaining code quality.

**Usage allocation decided: Opus for decisions and review, Sonnet (PRAGMI project) for code execution, blank Sonnet for self-contained tasks, Haiku not used (reasoning ceiling risk), big Opus (full research library) reserved for new biological grounding decisions only.**

## Open Questions Carried Forward


All open questions from Team_Sync_4-4-26.md remain open unless explicitly closed above. Key items:

1. Maturity value ownership: resolved. Owned by NeuromodulatorBroadcast. Four-component weighted sum at 0.25 each, asymmetric EMA (tau_up=0.995, tau_down=0.95).

2. Maturity directionality: still open. Can maturity decrease when encountering genuinely novel domain?

3. Pragmatic signal source in EFE: still open. What provides the task_reward signal?

4. Prime double forward pass: still open. Intentional or structural leftover?

5. d_model: resolved. Seed 256, dynamic expansion to max 1024, increment 64.

6. forward_with_spike_cache: still open.

7. Critical period decay target: still open.

8. Manifold integration consensus logic: still open.

9. Expert detection threshold: still open. Current value 0.78, not confirmed final.

10. Checkpoint resume with dynamic specialists: still open.

11. specialist_ages tracking location: still open.

## New Open Question


**12. Timmy SNN file migration.**
Should the Timmy column files (timmy_neuron.py, timmy_encoder.py, timmy_attention.py, timmy_experts.py, timmy_memory.py, timmy_blocks.py, timmy_plasticity.py, timmy_state.py, timmy_model.py, CreateTimmyArrayv3.py) be migrated from Timmy_Neuron repo into PRAGMI-Architecture-? The integration harness currently runs without them using stand-ins. When TimmyModel integration begins, they need to be importable. Options: migrate into PRAGMI-Architecture- (single repo), keep separate and add to Python path, or create a shared package. Not yet decided.

## What Comes Next


1. Fix the total_neurons bug in cognitive_kernel_core_c.py (one-line fix, Sonnet task).
2. Upload cognitive_kernel_core_c.py to Core/ and small_core_train.py to Team/.
3. Upload this Team Sync to Team/.
4. Run the integration harness on real data (TinyStories) on Amellia's machine to validate end-to-end training loop convergence.
5. Begin TimmyModel integration: decide on file migration strategy (open question 12), write the integration spec for replacing LightweightEncoder with TimmyModel, wire CorticalBuffer to real membrane state.

## File Status


| File | Location | Status | Tests |
|---|---|---|---|
| cognitive_kernel_core_t.py | Teaching/ | Complete | 7/7 pass |
| cognitive_kernel_core_c.py | Core/ (pending upload) | Bug fix needed (total_neurons) | 7/7 pass (default path) |
| cortical_buffer_t.py | Teaching/ | Complete | 11/11 pass |
| cortical_buffer_p.py | Core/ | Complete | 11/11 pass |
| world_model_ensemble_t.py | Teaching/ | Complete | 10/10 pass |
| world_model_ensemble_p.py | Core/ | Complete | 10/10 pass |
| neuromodulator_broadcast_t.py | Teaching/ | Complete | 14/14 pass |
| neuromodulator_broadcast_p.py | Core/ | Complete | 14/14 pass |
| epistemic_selector_t.py | Teaching/ | Complete | 7/7 pass |
| epistemic_selector_p.py | Core/ | Complete | 7/7 pass |
| small_core_train.py | Team/ (pending upload) | Complete, Opus reviewed | 18/18 pass |
| cognitive_kernel_trainer.py | Team/ | Complete | Standalone |

*Genesis Labs Research, 2026*  Amellia Mendel / LM Adler 
