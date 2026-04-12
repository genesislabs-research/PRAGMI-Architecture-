# PRAGMI Architecture: Complete Mathematical Reference
**Every equation for every arrow, feedback loop, timing, and component**
*Amellia Mendel – Lisa Adler*
*March 30, 2026*

---

### Contents
1. Thalamic Gate (Input → Primary Sensory Cortex)
2. Dorsal/Ventral Stream Split (after Primary Sensory Cortex)
3. PFC + ACC → Thalamus Top-Down Feedback Loop
4. Hippocampus → Association Cortex Memory Feedback Loop
5. Basal Ganglia Disinhibitory Gating Loop
6. Cerebellum Efference Copy + Error Loop
7. Neuromodulator Broadcasts (Global Operating-Mode Signals)
8. ACC Conflict Detection
9. Association Cortex Bidirectional Feedback
10. Primary Sensory Cortex (V1/A1)
* Appendix A: Module-to-Biology Mapping
* Appendix B: Log-Dynamic Timing Hierarchy

---

### 1. Thalamic Gate (Input → Primary Sensory Cortex)
**Arrow:** Input → Thalamus → Primary Sensory Cortex (active gate, not passive relay)

$$y = x \odot \sigma(W_{g}c + b_{g}) \quad (1)$$
$$g = \sigma(\beta_{NE} \cdot (W_{g}c + b_{g})) \quad (2)$$

* **Variables:** $x$ = raw sensory input; $c$ = control signal from PFC/ACC (goals + conflict); $\beta_{NE}$ = norepinephrine gain scalar; $W_{g}$, $b_{g}$ = learnable gate parameters.
* **Timing:** Top-down control signal $c$ from PFC/ACC arrives before the sensory signal reaches V1/A1. NE scales precision in real time every cycle.
* *Halassa, M.M. and Kastner, S. (2017). "Thalamic functions in distributed cognitive control." Nature Neuroscience, 20(12), 1669-1679. DOI: 10.1038/s41593-017-0020-1*

### 2. Dorsal/Ventral Stream Split (after Primary Sensory Cortex)
**Arrow:** Primary Sensory Cortex → Dorsal Stream & Ventral Stream (parallel split)

**Ventral Stream** (object identity, meaning, value):
$$Ventral = f_{v}(S) \quad (3)$$

**Dorsal Stream** (spatial location, motion, action guidance):
$$Dorsal = f_{d}(S) \quad (4)$$

* **Timing:** Early split immediately after V1/A1. This split is a prerequisite for correct PFC subdivision. Without it, ventromedial PFC and dorsolateral PFC receive mismatched inputs and cannot specialize.
* *Ungerleider, L.G. and Mishkin, M. (1982). "Two cortical visual systems." In Ingle, Goodale and Mansfield (Eds.), Analysis of Visual Behavior. MIT Press, 549-586.*

### 3. PFC + ACC → Thalamus Top-Down Feedback Loop
**Arrow:** PFC + ACC → Thalamus (Layer 6 corticothalamic feedback)

$$c = PFC_{goal} + \beta \cdot ACC_{conflict} \quad (5)$$

* **Variables:** $c$ = composite control signal; $PFC_{goal}$ = working memory/goal vector; $ACC_{conflict}$ = entropy-based conflict scalar; $\beta$ = coupling weight.
* **Timing:** Layer 6 corticothalamic feedback arrives before the next sensory cycle. This is a continuous loop, not a one-shot signal.
* *Halassa, M.M. and Kastner, S. (2017). "Thalamic functions in distributed cognitive control." Nature Neuroscience, 20(12), 1669-1679.*

### 4. Hippocampus → Association Cortex Memory Feedback Loop
**Arrow:** Hippocampus → Association Cortex (bidirectional replay and consolidation)

**Fast Encoding:**
$$H \leftarrow H + \eta_{fast} \cdot f(input) \quad (6)$$

**Slow Consolidation:**
$$A \leftarrow A + \eta_{slow} \cdot \alpha_{ACh} \cdot replay(H) \quad (7)$$

* **Variables:** $H$ = hippocampal sparse representation; $A$ = association cortex representation; $\eta_{fast}$ = high learning rate (encoding); $\eta_{slow}$ = small learning rate (consolidation); $\alpha_{ACh}$ = acetylcholine modulation scalar.
* **Timing:** Fast update during active input. Slow replay during offline and consolidation phases. ACh is high during encoding (write mode) and lower during replay (read mode).
* *McClelland, J.L., McNaughton, B.L. and O'Reilly, R.C. (1995). "Why there are complementary learning systems in the hippocampus and neocortex." Psychological Review, 102(3), 419-457.*

### 5. Basal Ganglia Disinhibitory Gating Loop
**Arrow:** Basal Ganglia → Thalamus (disinhibitory selection, not excitation)

**Direct Pathway (chosen channel):**
$$I_{i} \leftarrow I_{i} - \alpha \cdot S_{i} \cdot (1 + d) \quad (8)$$

**Indirect Pathway (competitors):**
$$I_{j} \leftarrow I_{j} + \beta \cdot S_{j} \cdot (1 - d) \quad \forall j \ne i \quad (9)$$

* **Variables:** $I$ = tonic inhibition from GPi/SNr on thalamus; $S$ = striatal activation for different actions; $d$ = dopamine reward prediction error scalar.
* **Timing:** DA-modulated selection happens before thalamic relay to cortex. The hyperdirect pathway provides fast broad suppression before fine selection begins.
* *Mink, J. W. (1996). "The basal ganglia: focused selection and inhibition of competing motor programs." Progress in Neurobiology, 50(4), 381-425.*

### 6. Cerebellum Efference Copy + Error Loop
**Arrow:** Cortex → Cerebellum (efference copy) → Cortex (error refinement)

$$pred = f_{cerebellum}(cmd, state) \quad (10)$$
$$error = actual - pred \quad (11)$$
$$cmd_{next} = cmd - \eta \cdot error \quad (12)$$

* **Variables:** $cmd$ = cortical command (efference copy); $pred$ = predicted sensory consequence; $error$ = mismatch used for calibration.
* **Timing:** Efference copy is sent before command execution. NE modulates the gain on error computation, especially under arousal or uncertainty.
* *Wolpert, D.M., Miall, R.C. and Kawato, M. (1998). "Internal models in the cerebellum." Trends in Cognitive Sciences, 2(9), 338-347.*

### 7. Neuromodulator Broadcasts (Global Operating-Mode Signals)
**Arrows:** DA, NE, ACh, 5-HT broadcast to all modules simultaneously.

**Dopamine (plasticity and reinforcement):**
$$\Delta w \leftarrow \Delta w \cdot (1 + d) \quad (13)$$

**Norepinephrine (arousal and gating gain):**
$$g \leftarrow g \cdot ne \quad (14)$$

**Acetylcholine (balance):**
$$input_{eff} = ach \cdot bottom\_up + (1 - ach) \cdot top\_down \quad (15)$$

**Serotonin (uncertainty tolerance):**
$$exploration\_rate \propto (1 + s) \quad (16)$$

* **Timing:** All neuromodulators broadcast in real time and update every cycle. Without explicit neuromodulation, the system has fixed hyperparameters and cannot shift between operating modes.
* *Yu, A.J. and Dayan, P. (2005). "Uncertainty, neuromodulation, and attention." Neuron, 46(4), 681-692.*

### 8. ACC Conflict Detection
**Arrow:** Competing PFC activations → ACC → control signal to Thalamus and NE system

$$conflict = \sum_{i \ne j} |r_{i} \cdot r_{j}| \quad \text{or} \quad H(r) = -\sum_{i} p_{i} \log p_{i} \quad (17)$$
$$control\_signal = \beta \cdot conflict \quad (18)$$

* **Variables:** $r$ = vector of competing response activations; $\beta$ = coupling coefficient to NE gain and thalamic gate precision.
* **Timing:** Continuous scalar signal updated every cycle. Modulates thalamic gate gain and NE broadcast in real time. Functions as an uncertainty estimate, not a binary error flag.
* *Botvinick, M.M. et al. (2001). "Conflict monitoring and cognitive control." Psychological Review, 108(3), 624-652.*

### 9. Association Cortex Bidirectional Feedback
**Arrow:** Lower areas → Association Cortex → Higher areas (MoE binding with feedback)

$$A = MoE(concat(S_{1}, S_{2}, \dots, S_{n}) + top\_down\_feedback) \quad (19)$$

* **Variables:** $S_{1} \dots S_{n}$ = inputs from different sensory streams; $top\_down\_feedback$ = projection from PFC and ACC back into association cortex.
* **Constraint:** MoE experts mimic regional specialization. The critical requirement is bidirectional connectivity: convergent input from lower areas and divergent feedback back down.
* *Mesulam, M.M. (1998). "From sensation to cognition." Brain, 121(6), 1013-1052.*

### 10. Primary Sensory Cortex (V1/A1)
**Arrow:** Thalamus → Primary Sensory Cortex (first cortical processing stage)

$$R(x,y) = G_{\sigma_{c}}(x,y) - G_{\sigma_{s}}(x,y) \quad (20)$$

* **Variables:** $G_{\sigma_{c}}$ = center Gaussian (excitatory); $G_{\sigma_{s}}$ = surround Gaussian (inhibitory). Neurons fire strongly to preferred features (oriented edges, spatial frequencies) and weakly to uniform stimulation. Organized into orientation columns.
* **Note:** No semantics or context at this stage. Raw feature extraction only. This is the input to the dorsal/ventral stream split.
* *Hubel, D.H. and Wiesel, T.N. (1968). "Receptive fields and functional architecture of monkey striate cortex." Journal of Physiology, 195(1), 215-243.*

---

### Summary: Implementation Priority
1. **Thalamic Gate** (highest priority: everything else depends on it)
2. **Hippocampus feedback loop** (resolves catastrophic forgetting)
3. **Basal Ganglia disinhibitory gating** (replaces softmax with biological selection)
4. **Cerebellum efference copy loop** (completes predictive coding)
5. **ACC conflict monitoring** (connects uncertainty to attentional precision)
6. **Neuromodulator broadcasts** (enables real-time operating mode shifts)

### Appendix A: Module-to-Biology Mapping

| Module                  | Biological Analog          | Functional Role |
|-------------------------|----------------------------|-----------------|
| **ThalamicRouter**      | Thalamus                   | Multiplicative gating layer; primary input filter. |
| **CLS_Module**          | Hippocampus/Cortex         | Fast-path encoding and slow-path consolidation replay. |
| **StriatalGatingSelector** | Basal Ganglia           | Disinhibitory selector; releases winning action from tonic bias. |
| **CerebellarCorrection**| Cerebellum                 | Parallel forward model; efference copy vs. sensory actuals. |
| **ACC_Monitor**         | Anterior Cingulate         | Entropy-based conflict detection modulating $\beta_{NE}$ |

### Appendix B: Log-Dynamic Timing Hierarchy

**Cross-Frequency Coupling Rule:** Gamma bursts are gated by the phase of the Theta cycle. Feature binding only occurs when the network is in an excitable state prepared by the slower rhythms.

| Regime  | Frequency   | Functional Role |
|---------|-------------|-----------------|
| **Delta** | 1-4 Hz    | Master Scheduler: global state transitions and consolidation phases. |
| **Theta** | 4-10 Hz   | Phase Reference: sequential trajectory encoding in Memory Cortex. |
| **Beta**  | 13-30 Hz  | Status Quo: maintains PFC goal states and suppresses irrelevant updates. |
| **Gamma** | 30-100 Hz | Local Binding: high-frequency bursts for cortical feature unification. |
