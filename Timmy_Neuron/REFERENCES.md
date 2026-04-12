# PRAGMI Research References

Complete list of all papers cited across the PRAGMI codebase. Every DOI has been used to ground at least one architectural decision, parameter value, or biological claim in the code. Papers marked with the file(s) where they appear.

---

## Neuroscience: Single Neuron Dynamics

**Hodgkin AL, Huxley AF (1952).** "A quantitative description of membrane current and its application to conduction and excitation in nerve." *Journal of Physiology*, 117(4):500-544.  
DOI: [10.1113/jphysiol.1952.sp004764](https://doi.org/10.1113/jphysiol.1952.sp004764)  
*Used in:* `timmy_neuron.py` — refractory period (Na⁺ channel inactivation), membrane reset dynamics.

**Fuster JM (1973).** "Unit activity in prefrontal cortex during delayed-response performance: neuronal correlates of transient memory." *Journal of Neurophysiology*, 36(1):61-78.  
DOI: [10.1152/jn.1973.36.1.61](https://doi.org/10.1152/jn.1973.36.1.61)  
*Used in:* `timmy_memory.py` — MemoryCortex as PFC delay-period persistent activity.

**Stafstrom CE, Schwindt PC, Crill WE (1982).** "Negative slope conductance due to a persistent subthreshold sodium current in cat neocortical neurons in vitro." *Nature*, 297(5865):406-408.  
DOI: [10.1038/297406a0](https://doi.org/10.1038/297406a0)  
*Used in:* `timmy_neuron.py` — persistent sodium current and threshold dynamics.

**Goldman-Rakic PS (1995).** "Cellular basis of working memory." *Neuron*, 14(3):477-485.  
DOI: [10.1016/0896-6273(95)90304-6](https://doi.org/10.1016/0896-6273(95)90304-6)  
*Used in:* `timmy_memory.py` — PFC working memory as the biological model for MemoryCortex.

**Wang XJ (2001).** "Synaptic reverberation underlying mnemonic persistent activity." *Trends in Neurosciences*, 24(8):455-463.  
DOI: [10.1016/S0166-2236(00)01868-3](https://doi.org/10.1016/S0166-2236(00)01868-3)  
*Used in:* `timmy_memory.py`, `CreateTimmyArray.py` — slow-decaying LIF neurons (tau_mem=0.99) as substrate for working memory. Post-MemoryCortex representation as routing signal.

---

## Neuroscience: Synaptic Plasticity

**Bi GQ, Poo MM (1998).** "Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and postsynaptic cell type." *Journal of Neuroscience*, 18(24):10464-10472.  
DOI: [10.1523/JNEUROSCI.18-24-10464.1998](https://doi.org/10.1523/JNEUROSCI.18-24-10464.1998)  
*Used in:* `timmy_plasticity.py` — STDP time constants and learning window shape.

**Bienenstock EL, Cooper LN, Munro PW (1982).** "Theory for the development of neuron selectivity: orientation specificity and binocular interaction in visual cortex." *Journal of Neuroscience*, 2(1):32-48.  
DOI: [10.1523/JNEUROSCI.02-01-00032.1982](https://doi.org/10.1523/JNEUROSCI.02-01-00032.1982)  
*Used in:* `astrocytic_regulator_v3.py` — BCM sliding threshold as the theoretical basis for metaplasticity modulation.

**Izhikevich EM (2007).** "Solving the distal reward problem through linkage of STDP and dopamine signaling." *Cerebral Cortex*, 17(10):2443-2452.  
DOI: [10.1093/cercor/bhl152](https://doi.org/10.1093/cercor/bhl152)  
*Used in:* `timmy_plasticity.py`, `CreateTimmyArray.py` — three-factor reward-modulated STDP. CA1 feedback delivered to the synapse that contributed to the outcome, not broadcast to all.

**Fremaux N, Gerstner W (2016).** "Neuromodulated spike-timing-dependent plasticity, and theory of three-factor learning rules." *Frontiers in Neural Circuits*, 9:85.  
DOI: [10.3389/fncir.2015.00085](https://doi.org/10.3389/fncir.2015.00085)  
*Used in:* `timmy_plasticity.py`, `train_array.py` — three-factor learning rule grounding LM-loss-modulated STDP.

---

## Neuroscience: Cortical Architecture and Columns

**Mountcastle VB (1997).** "The columnar organization of the neocortex." *Brain*, 120(4):701-722.  
DOI: [10.1093/brain/120.4.701](https://doi.org/10.1093/brain/120.4.701)  
*Used in:* `timmy_neuron.py`, `timmy_model.py`, `CreateTimmyArray.py` — minicolumn cascade amplification, structural uniformity of cortical columns at initialization, column identity defined by connectivity not content.

**Felleman DJ, Van Essen DC (1991).** "Distributed hierarchical processing in the primate cerebral cortex." *Cerebral Cortex*, 1(1):1-47.  
DOI: [10.1093/cercor/1.1.1](https://doi.org/10.1093/cercor/1.1.1)  
*Used in:* `timmy_blocks.py`, `timmy_model.py` — three-zone neocortical hierarchy (sensory, association, executive).

**Sherman SM, Guillery RW (2002).** "The role of the thalamus in the flow of information to the cortex." *Philosophical Transactions of the Royal Society B*, 357(1428):1695-1708.  
DOI: [10.1098/rstb.2002.1161](https://doi.org/10.1098/rstb.2002.1161)  
*Used in:* `timmy_encoder.py` — thalamocortical relay as biological model for TemporalSpikeEncoder.

**Fries P (2005).** "A mechanism for cognitive dynamics: neuronal communication through neuronal coherence." *Trends in Cognitive Sciences*, 9(10):474-480.  
DOI: [10.1016/j.tics.2005.08.011](https://doi.org/10.1016/j.tics.2005.08.011)  
*Used in:* `timmy_attention.py` — communication-through-coherence as biological model for SpikingSynapticResonance.

**Hawkins J, Lewis M, Klukas M, Purdy S, Ahmad S (2019).** "A framework for intelligence and cortical function based on grid cells in the neocortex." *Frontiers in Neural Circuits*, 12:121.  
DOI: [10.3389/fncir.2018.00121](https://doi.org/10.3389/fncir.2018.00121)  
*Used in:* `CreateTimmyArray.py` — Thousand Brains Theory: every column is a complete world model, columns vote to reach consensus, Prime as broadband integration column.

**Huang C, Zeldenrust F, Celikel T (2022).** "Cortical representation of touch in silico." *Neuroinformatics*, 20:1013-1039.  
DOI: [10.1007/s12021-022-09576-5](https://doi.org/10.1007/s12021-022-09576-5)  
*Used in:* `CreateTimmyArray.py`, `train_array.py` — threshold adaptation and effective connectivity as primary mechanisms of column differentiation. Membrane state gates input registration, not just memory. Justifies Phase 2 domain assignment without weight freezing.

---

## Neuroscience: Population Coding and Ensembles

**Semedo JD, Zandvakili A, Machens CK, Yu BM, Kohn A (2019).** "Cortical areas interact through a communication subspace." *Neuron*, 102(1):249-259.  
DOI: [10.1016/j.neuron.2019.01.026](https://doi.org/10.1016/j.neuron.2019.01.026)  
*Used in:* `CreateTimmyArray.py` — low-rank communication subspace (rank 3) per column, attention-weighted pooling to 64-dim coordinate manifold, routing projection in predictive subspace only.

**See JZ, Atencio CA, Sohal VS, Schreiner CE (2018).** "Coordinated neuronal ensembles in primary auditory cortical columns." *eLife*, 7:e35587.  
DOI: [10.7554/eLife.35587](https://doi.org/10.7554/eLife.35587)  
*Used in:* `CreateTimmyArray.py` — column identity defined by higher-order correlation structure. cNE cluster firing rates as routing signal. Ensemble membership not predicted by pairwise distance or receptive field overlap alone.

**Pérez-Ortega J, Alejandre-García T, Yuste R (2021).** "Long-term stability of cortical ensembles." *eLife*, 10:e64449.  
DOI: [10.7554/eLife.64449](https://doi.org/10.7554/eLife.64449)  
*Used in:* `CreateTimmyArray.py`, `train_array.py` — ensemble identity carried by connectivity structure of stable core (~68% of neurons persist 46 days). Spontaneous activity rehearses same functional elements as evoked activity. Directly validates PRAGMI theoretical core.

---

## Neuroscience: Hippocampal Formation

**Witter MP, Naber PA, van Haeften T, Machielsen WC, Rombouts SA, Barkhof F, Scheltens P, Lopes da Silva FH (2000).** "Cortico-hippocampal communication by way of parallel parahippocampal-subicular pathways." *Hippocampus*, 10(4):398-410.  
DOI: [10.1002/1098-1063(2000)10:4<398::AID-HIPO6>3.0.CO;2-K](https://doi.org/10.1002/1098-1063(2000)10:4<398::AID-HIPO6>3.0.CO;2-K)  
*Used in:* `CreateTimmyArray.py` — perforant path anatomy grounding PerforantPathSymphonyBridge.

**Sainburg T, McPherson MJ, Bresin K, Lopez M, McAuley JD (2021).** "Parametric UMAP embeddings for representation and semisupervised learning." *Neural Computation*, 33(11):2884-2907.  
DOI: [10.1162/neco_a_01434](https://doi.org/10.1162/neco_a_01434)  
*Used in:* `CreateTimmyArray.py` — coordinate_dim=64 manifold as engineering choice for kernel interface.

---

## Neuroscience: Glial Biology

**Araque A, Parpura V, Sanzgiri RP, Haydon PG (1999).** "Tripartite synapses: glia, the unacknowledged partner." *Trends in Neurosciences*, 22(5):208-215.  
DOI: [10.1016/S0166-2236(98)01349-6](https://doi.org/10.1016/S0166-2236(98)01349-6)  
*Used in:* `astrocytic_regulator_v3.py` — foundational tripartite synapse model. Astrocyte as active computational participant.

**Stellwagen D, Malenka RC (2006).** "Synaptic scaling mediated by glial TNF-alpha." *Nature*, 440(7087):1054-1059.  
DOI: [10.1038/nature04671](https://doi.org/10.1038/nature04671)  
*Used in:* `astrocytic_regulator_v3.py` — astrocyte-derived signaling directly regulates synaptic strength homeostasis.

**Turrigiano GG (2008).** "The self-tuning neuron: synaptic scaling of excitatory synapses." *Cell*, 135(3):422-435.  
DOI: [10.1016/j.cell.2008.10.008](https://doi.org/10.1016/j.cell.2008.10.008)  
*Used in:* `astrocytic_regulator_v3.py` — synaptic scaling as homeostatic mechanism. eta_modifier as computational implementation of scaling signal.

**Tzingounis AV, Wadiche JI (2007).** "Glutamate transporters: confining runaway excitation by shaping synaptic transmission." *Nature Reviews Neuroscience*, 8(12):935-947.  
DOI: [10.1038/nrn2274](https://doi.org/10.1038/nrn2274)  
*Used in:* `astrocytic_regulator_v3.py` — glutamate transporter clearance kinetics (GLT-1, GLAST) grounding the decay_rate parameter.

---

## Machine Learning: Spiking Neural Networks

**Neftci EO, Mostafa H, Zenke F (2019).** "Surrogate gradient learning in spiking neural networks: bringing the power of gradient-based optimization to spiking neural networks." *IEEE Signal Processing Magazine*, 36(6):51-63.  
DOI: [10.1109/MSP.2019.2931595](https://doi.org/10.1109/MSP.2019.2931595)  
*Used in:* `timmy_neuron.py` — ATanSurrogate gradient. Explicitly labeled NOT a biological quantity, training artifact only.

**Fang W, Yu Z, Chen Y, Masquelier T, Huang T, Tian Y (2021).** "Incorporating learnable membrane time constants to enhance learning of spiking neural networks." *ICCV 2021*, pp. 2661-2671.  
DOI: [10.1109/ICCV48922.2021.00266](https://doi.org/10.1109/ICCV48922.2021.00266)  
*Used in:* `timmy_neuron.py` — learnable membrane time constant via parametric tau_mem.

---

## Machine Learning: Transformers and Attention

**Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez AN, Kaiser L, Polosukhin I (2017).** "Attention is all you need." *NeurIPS 2017*.  
DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)  
*Used in:* `timmy_attention.py` — scaled dot-product attention base. theta=10000 frequency base for position encoding.

**Su J, Lu Y, Pan S, Murtadha A, Wen B, Liu Y (2024).** "RoFormer: enhanced transformer with rotary position embedding." *Neurocomputing*, 568:127063.  
DOI: [10.1016/j.neucom.2023.127063](https://doi.org/10.1016/j.neucom.2023.127063)  
*Used in:* `timmy_attention.py` — rotary position embedding (RoPE).

**Shazeer N, Mirhoseini A, Maziarz K, Davis A, Le Q, Hinton G, Dean J (2017).** "Outrageously large neural networks: the sparsely-gated mixture-of-experts layer." *ICLR 2017*.  
DOI: [10.48550/arXiv.1701.06538](https://doi.org/10.48550/arXiv.1701.06538)  
*Used in:* `timmy_experts.py` — sparse MoE routing architecture.

**He K, Zhang X, Ren S, Sun J (2016).** "Deep residual learning for image recognition." *CVPR 2016*, pp. 770-778.  
DOI: [10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90)  
*Used in:* `timmy_blocks.py` — residual connections.

**Wang P, Li L, Shao Y, Xu R, Dai X, Li P, Chen Q, Sui Z, Wang B (2024).** "Auxiliary-loss-free load balancing strategy for mixture of experts." arXiv: 2408.15664.  
DOI: [10.48550/arXiv.2408.15664](https://doi.org/10.48550/arXiv.2408.15664)  
*Used in:* `CreateTimmyArray.py`, `train_array.py` — per-specialist bias terms adjusted dynamically to prevent routing collapse without auxiliary loss.

---

## Computational Neuroscience: Reference Texts

**Gerstner W, Kistler WM, Naud R, Paninski L (2014).** *Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition.* Cambridge University Press.  
DOI: [10.1017/CBO9781107447615](https://doi.org/10.1017/CBO9781107447615)  
*Used in:* `timmy_neuron.py` — membrane time constant biological ranges, LIF neuron formulation, synaptic current kinetics.

**Buzsaki G (2006).** *Rhythms of the Brain.* Oxford University Press.  
DOI: [10.1093/acprof:oso/9780195301069.001.0001](https://doi.org/10.1093/acprof:oso/9780195301069.001.0001)  
*Used in:* `timmy_model.py`, `timmy_encoder.py` — gamma-band (T=8) and theta-band (T_slow=2) temporal basis grounding.

---

## Summary

| Category | Papers | Files |
|---|---|---|
| Single neuron dynamics | 5 | timmy_neuron.py, timmy_memory.py |
| Synaptic plasticity | 4 | timmy_plasticity.py, astrocytic_regulator_v3.py, CreateTimmyArray.py |
| Cortical architecture | 5 | timmy_blocks.py, timmy_encoder.py, timmy_attention.py, CreateTimmyArray.py |
| Population coding | 3 | CreateTimmyArray.py, train_array.py |
| Hippocampal formation | 2 | CreateTimmyArray.py |
| Glial biology | 4 | astrocytic_regulator_v3.py |
| SNN training | 2 | timmy_neuron.py |
| Transformers / MoE | 5 | timmy_attention.py, timmy_experts.py, timmy_blocks.py, CreateTimmyArray.py |
| Reference texts | 2 | timmy_neuron.py, timmy_model.py, timmy_encoder.py |
| **Total** | **32 unique papers** | |

---

*Genesis Labs Research, 2026*
