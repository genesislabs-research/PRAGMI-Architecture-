# Contains various de novo snn's
# doing various things because reasons
# This is a public development space and nothing should be considered finished
PRAGMI (PRAG-mee)
**Persistent Reconstructive Architecture for Generative Memory and Imagination**


## here's some stuff I used:
Thalamic Gating and Top-Down Attention



DOI: 10.1038/s41593-017-0020-1
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

### Have an amazing day!
