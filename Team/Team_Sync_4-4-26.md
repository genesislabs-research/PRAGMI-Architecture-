# PRAGMI / Timmy — Team Sync Ai generated 4_4_26 - Amellia Mendel 



## What This Document Is

This is not a handoff summary. It is a living record of architectural decisions, their reasoning, and the state of every file in the codebase. It gets updated at the end of every session, surgically, not rewritten. The reasoning for every decision is preserved, not just the conclusion. Anyone reading this document should be able to pick up without re-litigating settled ground.

Chronology rule: when extractions contradict each other, the more recent one wins. When anything contradicts decisions made the newest wins. 

---

## The Project In One Paragraph

Timmy is a pure spiking neural network language model built on the PRAGMI architecture (Persistent Reconstructive Architecture for Generative Memory and Imagination). It sits between an external LLM (narrator, user-facing) and the Cognitive Kernel (hippocampal episodic memory, separate repo). It translates discrete token sequences into temporal spike patterns and back. It is trained from scratch, no distillation, no transformer conversion. The goal is genuine persistent episodic memory that survives context window closure through neural reconstruction, not RAG, not retrieval, not a filing cabinet. The system reconstructs experience rather than retrieving text chunks. This project is building toward an academic paper. The documentation is the scientific grounding that makes the paper possible.

---

## Three-Layer Architecture (Hard Boundaries)

```
External LLM          narrator, planner, user-facing
                      never touches kernel internal state directly
        |
      tokens
        |
        v
Timmy                 subconscious token translator
                      maps LLM token embedding space <-> kernel coordinate space
                      enforces Terms of Service (abuse detection, existential fear induction,
                      weaponized nullification)
                      NOT end-user accessible in public wrapper
        |
  64-dim coordinate manifold (Perforant Path)
        |
        v
Cognitive Kernel      hippocampal memory system (separate repo)
                      CA3 attractor storage, CA1 novelty gating
                      episodic reconstruction, UMAP manifold
```

These boundaries do not bleed. The external LLM is downstream of reconstructed experience and is treated as narrator and interpreter. Timmy is the bridge language, the organism's internal voice.

---

## Timmy Column Architecture

One Prime column (broadband integration) plus dynamic specialist columns. All specialists begin as exact weight-for-weight clones of trained Prime. Specialization emerges through sleep-cycle synaptic pruning and consolidation. It is not hardcoded. New orthogonal specialist columns can be spawned during sleep phase. Dynamic expansion is a first-class citizen of the architecture, not an afterthought.

Lifecycle: Train Prime alone (Phase 1) -> critical period probe detects convergence -> clone Prime to all specialists -> sleep cycle runs specialty-directed pruning (first divergence event) -> day/sleep cycles deepen specialization through accumulated experience.

Sleep phase rhythm: Prune -> detect experts -> theta-gamma modulated replay -> orthogonal expansion -> router rebalance.

Neuromodulator as master clock: ACh gates encoding vs consolidation. DA biases expansion and R-STDP. NE boosts surprise-driven plasticity. 5-HT modulates stability.

---

## Dual-File Rule

Every biological architecture file exists as exactly two files:

`filename_teaching.py` — full implementation plus complete documentation standard. This is the scientific record.

`filename_p.py` — identical logic, zero comments, zero docstrings, zero annotations of any kind. Not even a module docstring. Only executable Python. This is the deployment artifact.

They are maintained as a matched pair. A pair that has diverged in behavior is considered broken. When one changes, the other changes in the same session.

Exception: `device_allocator_base.py`, `device_allocator_single.py`, `device_allocator_asymmetric.py` are infrastructure files. Single file each, no suffix convention, never imported by model files directly.

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

## Documentation Standard (Teaching Files)

Every class and every significant function has a docstring. No exceptions.

Every biological concept has three things:
1. The biological name of the structure or process
2. A plain English explanation of what it does in the actual brain
3. Full citation: Author(s) (Year). "Title." Journal, Volume(Issue), Pages. DOI: verified DOI string

Unknown DOIs: `{To be added later.}` Never guess a DOI.

If a parameter value came from a specific paper, the comment says so explicitly.

If something is NOT a biological quantity: label it explicitly as training artifact, engineering approximation, or computational convenience.

No vague rationale. "The biology validates this approach" is not a citation.

Variable names use biological terminology: `v_mem`, `i_syn`, `tau_mem`, `dentate_gyrus_dim`.

Every file has a BIOLOGICAL GROUNDING header in the module docstring: what brain region or process the file models, why it matters, two or three most important grounding papers.

Any function crossing an anatomical boundary documents both sides: sending structure, receiving structure, biological name of the connection.

No Nord references anywhere. Every concept traced to the original paper, not to any prior implementation.

---

## Engineering Constraints

- Python 3.11
- Every persistent state uses `register_buffer`, never `nn.Parameter` for diagnostic or structural state. This includes routing_bias, specialist_load_ema, specialist_selection_ema. Overwriting buffers with nn.Parameter mid-training causes autograd to crash on in-place EMA operations and breaks serialization.
- Membrane states detached to prevent graph bloat during long training runs
- `can_spawn_new_column()` called before any structural network change
- Router, bridge, and all dynamically-sized structures have `add_column()` methods
- `max_specialists` enforced at array level, not only inside expander. Default 16.
- Teaching and production files maintained as matched pair, never allowed to diverge
- `@torch.no_grad()` on NeuromodulatorBroadcast forward pass. Instantaneous signals with active gradient history fed into EMA buffers cause massive VRAM leak.
- Normalize main task loss to 0-1 scale before applying MSE for curiosity head. Raw cross-entropy (e.g. 6.0) against sigmoid output causes immediate weight saturation since target is mathematically impossible.
- New parameters injected directly into active optimizer state immediately after expansion. Without injection, newborn column receives forward passes but weights do not update during optimizer.step().
- Temporal loop (for t in range(T)) implemented in custom Triton kernel. Python loop R/W costs from loading v_mem and i_syn from global memory at every step are a bandwidth bottleneck.
- Spatial cluster connections (_cascade_amplify) kept as post-hoc correction outside fused kernel. Random memory access (scatter/gather) during every timestep breaks coalesced memory access.
- ATan surrogate derivatives recalculated on the fly in backward pass rather than caching. Trades cheap arithmetic for VRAM, increasing arithmetic intensity to prevent GPU idling.
- No meta-learning inner loops requiring double-backward passes. Second-order meta-gradients through multi-column SNN would shatter 24GB VRAM ceiling.

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

## Parameter Values

- T=10 temporal basis: T=8 fast (gamma-band), T_slow=2 slow (theta-band)
- EMA alpha=0.99 for neuromodulator baselines. Engineering approximation for ~100-step memory. NOT biological quantity.
- sigma=1.5 for STDP Gaussian kernel
- A_plus=0.01, A_minus=0.008 for STDP amplitudes. From Bi & Poo (1998).
- surrogate_alpha=1.0 for ATan surrogate gradient. NOT biological quantity, training artifact.
- cascade_gain=0.2 for local recurrent excitation in AssociativeLIF. Engineering hyperparameter.
- beta_mem=0.95, beta_syn=0.9 (default LIF values)
- v_threshold=1.0 (default)
- 64-dim coordinate manifold (Perforant Path subspace). Fixed.
- expert_coherence_threshold=0.78 (updated from 0.75). Pending confirmation.
- prune_threshold=1e-4 (updated from 0.12)
- orthogonality_strength=0.92 (updated from 0.85)
- max_specialists=16 (updated from 12). Engineering safeguard.
- newborn_lr_multiplier=2.5 (updated from 2.0)
- da_decay=0.995, ach_decay=0.98, ne_decay=0.99, ht_decay=0.999
- da_baseline_init=0.5, ach_baseline_init=0.7, ne_baseline_init=0.4, ht_baseline_init=0.5
- max_modulation=2.0
- epistemic_scale=1.0, pragmatic_scale=0.3. NOT biological quantities, engineering approximations.
- temperature=0.1
- batch_size=32
- Sparsity target: 96%
- VRAM budget: 24GB (Tesla P40)
- Triton kernel tau=0.01, dropout probability=0.2 (MorphSNN STSP attention)
- MorphSNN diffusion depth M: {To be decided}
- MSTH calcium target c_target=0.5, gamma=0.12, theta=0.08
- d_model_seed=256, dynamic expansion to max 1024, increment 64. CLOSED. See Dynamic Column Width Expansion spec and April 5 Session Closures.

---

## Equations Log

**Thalamic gate core equation:**
y = x ⊙ σ(W_g · c + b_g)

**Thalamic gate with NE modulation:**
g = σ(β_NE · (W_g · c + b_g))

**Hippocampal encoding (fast):**
H ← H + η_fast · f(input)

**Hippocampal replay/consolidation feedback (slow):**
A ← A + η_slow · α_ACh · replay(H)

**Basal ganglia disinhibition:**
I_i ← I_i - α · S_i

**Cerebellum forward model:**
predicted = f(copy, state)
error = actual_sensory - predicted
next_command ← command - η · error

**Neuromodulator mappings:**
gated_signal = raw_signal × gain_NE
Δw = DA_error × learning_rate
effective_input = α_ACh × bottom_up + (1 - α_ACh) × top_down
exploration_rate ∝ 5-HT_level

**DA-SSDP weight update rule (Tian et al. 2024):**
ΔW_ij = clip(1/B · Σ_b G_b · g_bij · (A_+ · λ_bij - A_- · (1 - λ_bij)), -1, 1)
where:
- λ_bij = synchrony gate (binary: 1 if both neurons fired at least once in window)
- g_bij = exp(-Δt² / (2σ²)) (Gaussian temporal weighting on first-spike latency)
- G_b = clip(1 + k · (S_b - μ_S) / σ_S, 0, 2) (dopamine gate from normalized batch synchrony)

**SVPG local learning rule (Yang et al. 2025):**
∂J(π)/∂w_ij = Σ_t γ^t r_t [q_i(v_j - q_j) + q_j(v_i - q_i)]
where v are binary firing states and q are firing probabilities

**AdaLi surrogate gradient (Hou et al. 2026):**
δ(U) = α/(V_th - V⁻)  if V⁻ < U < V_th
δ(U) = β/(V⁺ - V_th)  if V_th ≤ U < V⁺
Update range (V⁻, V⁺) narrows inward across training epochs.

**Expected Free Energy:**
G(π) = epistemic_scale × predicted_surprise - pragmatic_scale × pragmatic_utility

**Dopamine gate (DA-SSDP):**
G_b = clip(1 + k · (S_b - μ_S) / σ_S, 0, 2)

**Reward modulation (STDP):**
reward = sigmoid((loss_EMA - current_loss) × scale)
dW_final = dW × (2 × reward - 1)

**MSTH emergency detection:**
Emergency_t = (Σ_{k∈{mag,rate,var,mean}} E_k ≥ 2) ∧ (t - t_last > τ_refract)

**MSTH calcium regulation:**
c_t^reg = c_t - 0.12 · σ(4.0(c_t - c_target)) ⊙ (c_t - c_target)

**MSTH system health:**
H_system = 1/3 (H_activity + H_calcium + H_weights)

**MorphSNN LIF:**
u_i^(t) = τ_decay · u_i^(t-1) + C_i^(t) - V_th · s_i^(t-1)

**MorphSNN adjacency update:**
S^(t) = β · S^(t-1) + (1-β) · Â^(t)

**SSONN gradient metrics:**
g_abs(e) = 1/M · Σ |∂L^(j)/∂w_e|
g_var(e) = Var[|∂L^(j)/∂w_e|]

**SSONN function-preserving expansion:**
W_ij^(1) = 1 if v_j^p = src[v_i^n], else ε ~ U(0, 10^-8)
W_ij^(2) = W[e_i] if v_j^(p+1) = dst[v_i^n], else ε ~
