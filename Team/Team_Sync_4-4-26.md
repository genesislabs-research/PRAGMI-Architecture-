# PRAGMI / Timmy — Team Sync Ai generated 4_4_26 

## living record of architectural decisions, their reasoning, and the state of every file in the codebase
every working session one is created by AI summary which {in theory} should capture the reasoning for every decision {hopefully} not just the conclusion. Anyone reading this document should be able to pick up without re-litigating settled ground.

Chronology rule: when extractions contradict each other, the more recent one wins. When anything contradicts decisions made the newest wins. 

---
## Repository Structure


Primary repository: https://github.com/genesislabs-research/PRAGMI-Architecture-

```
PRAGMI-Architecture-/
├── README.md
├── Teaching/
│   ├── cortical_buffer_t.py            ← PFC working memory
│   ├── world_model_ensemble_t.py       ← predictive coding ensemble
│   ├── neuromodulator_broadcast_t.py   ← DA/NE/ACh/5-HT + maturity
│   └── epistemic_selector_t.py         ← active batch selection
├── Core/
│   ├── cortical_buffer_p.py
│   ├── world_model_ensemble_p.py
│   ├── neuromodulator_broadcast_p.py
│   └── epistemic_selector_p.py
└── Team/
    ├── Team_Sync_4-4-26.md             ← this document
    ├── Train_tiny_Tim.py
    └── tiny_tim.py
```

---


## Hardware Abstraction (SETTLED)


Hardware configuration is fully quarantined in three files:

- `device_allocator_base.py` — abstract base class `DeviceAllocatorBase` with three-method interface: `get_prime_device()`, `get_optimal_device_for_specialist()`, `can_spawn_new_column()`. Also contains `get_device_manager()` factory.
- `device_allocator_single.py` — `SingleDeviceManager(DeviceAllocatorBase)`, one GPU or CPU fallback.
- `device_allocator_asymmetric.py` — `AsymmetricDeviceManager(DeviceAllocatorBase)`, dual GPU unequal VRAM.

Factory selects implementation via `PRAGMI_DEVICE_CONFIG` environment variable or auto-detection. No other file in the codebase contains a device index, `.to('cuda:0')`, or hardware-specific logic. Core modules accept `DeviceAllocatorBase` and call its interface. Adding a new hardware configuration means adding one new allocator file, nothing else.

Current hardware: Dell Precision T7910, dual Xeon E5-2683 v4, 128GB RAM, AORUS GTX 1080 Ti 11GB, Tesla P40 24GB expected. P40 is primary (Prime column), 1080 Ti is secondary (specialist offloading).

Status: COMPLETE. All three files written and verified.

---

## Decisions Log (Settled — Do Not Re-Litigate)


**CuriosityHead uses a small ensemble (5 MLPs) not a single MLP.**
Reason: A deterministic point-prediction MLP cannot express uncertainty. Epistemic value in active inference is specifically expected information gain, which requires the model to know something about its own confidence. An ensemble of five small MLPs gives variance across predictions for free at negligible compute cost. High variance means world model uncertain about this manifold region, go train here. Self-corrects as world model learns.

**CuriosityHead has a global maturity value.**
Reason: In early training the world model is chasing a moving target and predictions are unreliable. Without a maturity signal, active batch selection is noise early on while adding compute overhead. Maturity starts low, dampening the selection signal. As maturity increases and world model becomes reliable, active selection kicks in and speedup benefit arrives. Also gives principled shared signal for expansion thresholds, NE sensitivity, critical period probe. Biological grounding: critical period closure, Huang et al. (2022). DOI: 10.1007/s12021-022-09576-5

**Device handling removed from CuriosityHead.**
Reason: Violates hardware abstraction layer. All device logic goes through DeviceAllocatorBase.

**Optimizer lives inside CuriosityHead intentionally.**
Reason: Curiosity head has its own independent learning dynamic separate from main training loop. Deliberate architectural choice for training speedup. Optimizer state must be explicitly handled in checkpointing, goes into HOT layer alongside STDP scalars and MoE EMAs.

**ExpandExpert implemented as native nn.Module.**
Reason: Allows sleep phase to be a true part of the network lifecycle and correctly integrates with PyTorch autograd engine.

**Cholinergic wake/sleep switch implemented.**
Reason: Physically isolates consolidation phase (sleep) from sensory encoder (wake) by toggling feedforward vs recurrent/feedback weights. Without this ACh only modulates hippocampal feedback, not all zones as spec requires.

**Raw prediction error used as surprise proxy.**
Reason: Calculating exact variational free energy in high-dimensional spiking latent spaces is computationally toxic to memory. Raw prediction error is the defensible approximation.

**ActiveDataLoader with skim-and-select candidate buffer.**
Reason: Running full 10-timestep LIF forward-and-backward on entire dataset to find surprise would take years. Skim-and-select evaluates candidates cheaply before committing full compute.

**newborn_lr_multiplier applied to newborn column parameters.**
Reason: Newborn column requires grace period to catch up to mature consolidated experts before routing algorithm starves it. Value: 2.5 (updated from initial 2.0).

**Dynamic expansion: one new orthogonal specialist per sleep cycle.**
Reason: Prevents chaotic topology shifts. Triggered by dopamine-weighted coherence and routing load.

**Momentum buffers manually zeroed after structural projection.**
Reason: After Gram-Schmidt or soft re-orthogonalization, optimizer momentum fights the correction if not cleared.

**CreateTimmyArray: PerforantPathSymphonyBridge needs add_column().**
Reason: Bridge has fixed ParameterList sizes set at init. New columns would be silently ignored or crash.

**CreateTimmyArray: ColumnRouter needs add_column().**
Reason: V_route is fixed shape (r, num_cols) at init. New columns have no routing weights.

**MicroglialPruner rejected as standalone module.**
Reason: Computing separate complement tag for every synapse adds too much compute overhead during sleep. Integrated into existing STDP threshold dynamics instead.

**Exact Gram-Schmidt via SVD rejected for every new column.**
Reason: Dense matrix operations for exact orthogonality are computationally toxic and bottleneck the GPU. Soft approximation used instead.

**Unprotected newborn column instantiation rejected.**
Reason: Mature expert column lateral inhibition instantly suppresses newborn before it learns.

**Separate isolated MicroglialPruner rejected.**
Reason: Too much overhead. Integrated into STDP threshold logic.

**Meta-learning inner loops rejected.**
Reason: Double-backward passes through multi-column SNN shatter 24GB VRAM ceiling.

**Exact variational free energy rejected.**
Reason: Computationally toxic in high-dimensional latent spaces.

**Standard static PyTorch DataLoader rejected.**
Reason: Passive iteration prevents Active Inference.

**Raw MSE loss against CuriosityHead sigmoid output using standard cross-entropy rejected.**
Reason: Cross-entropy regularly outputs values above 1.0, making sigmoid target mathematically impossible and causing immediate weight saturation.

---

## Open Questions (Must Be Resolved Before Writing Code)


**1. Maturity value ownership.**
Where does the global maturity value live and who owns it? Options: neuromodulator broadcast (reads like ACh, global state multiple systems consume), or TimmyArray itself as a registered buffer updated each training step. Not yet decided.

**2. Maturity directionality.**
Is maturity monotonically increasing with training steps, or can it decrease when system encounters genuinely novel domain? Affects implementation significantly. Not yet decided.

**3. Pragmatic signal source in EFE.**
calculate_expected_free_energy takes task_reward: float = 0.0 as argument, meaning pragmatic term is almost always zero. What actually provides this signal in the architecture? Options: CA1 novelty scalar from Cognitive Kernel, neuromodulator broadcast DA level, something else. Not yet decided.

**4. Prime double forward pass.**
Current forward calls Prime in parallel column loop AND again at end with external_context. Two full forward passes per batch. Intentional or structural leftover? Not yet resolved.

**5. d_model.**
496 (inherited from Nord v4) vs 768 (Gemini sessions). Since we are starting fresh and not building off v4, this needs a fresh decision. Not yet decided.

**6. forward_with_spike_cache implementation details.**
How to expose pre/post spikes cleanly in individual LIF/MoE blocks. Currently marked TODO.

**7. Critical period decay target.**
Should be applied to A_plus/A_minus amplitudes or to per-column learning rate multiplier. Not yet decided.

**8. Manifold integration consensus logic.**
Inside forward_with_spike_cache. Currently marked TODO.

**9. Expert detection threshold.**
Exact trigger threshold theta_coherence for expert detection in sleep phase. Current value 0.78, updated from 0.75, but not confirmed final.

**10. Checkpoint resume with dynamic specialists.**
When resuming a checkpoint that has N mature specialists, how to handle load_state_dict so fresh TimmyArray (which initializes with 0 specialists) dynamically scales up architecture to match checkpoint before weights are loaded. Not yet decided.

**11. specialist_ages tracking.**
Should specialist_ages buffer (1D tensor incrementing each wake cycle, appending 0 when ExpandExpert spawns) live in CreateTimmyArray or in ExpandExpert. Not yet decided.

---
- Amellia Mendel / LM Adler 
