# S-ROS (Spiking Robot Operating System)
## Project Goals

1) Reach Crystallization Threshold: Prove that a de novo spiking neural network can reach a transition prediction loss of < 0.007 (0.7% MSE) to safely lock in autonomous skills.

2) Procedural Execution: Prove that a spiking neural network can learn procedural, syntactic code execution (BASIC-style logic) without pre-training or distillation.

3) Pushdown Automaton (PDA) Emulation: Overcome the limits of standard Finite State Machines by implementing a discrete Spiking Call Stack, allowing the network to natively track GOSUB recursion and nested loops without state aliasing.

4) The Triple-Core Bridge: Establish a biologically grounded neuro-symbolic bridge that completely isolates exploratory learning from rigid, near zero-hallucination robotic actuation.

####  Chatbots are a solved problem, but standard robotic operating systems are rigid message-passers that cannot adapt. S-ROS marries the absolute, deterministic certainty of procedural logic (BASIC/ROS) with the associative, real-time learning capabilities of a pure spiking neural network.
S-ROS strips out the bloated probability matrices required to parse natural language and replaces them with a hyper-efficient, neuro-symbolic state machine. It does not predict the next word in a sentence; it predicts the next logical machine state, remembers the outcome, and physically rewires its own execution pathways based on success and failure.


### 1. Tiny_Tim (The Logic Engine)
A radically lightweight cortical column ensemble designed as a small testing kernel. Without the burden of learning human grammar, Tiny_Tim functions as a self-wiring biological logic gate. It receives environmental spike vectors, processes them through its spiking layers, and outputs command spike vectors. It learns causal, procedural sequencing through reward-modulated Spiking-Timing-Dependent Plasticity (STDP).

### 2. The Neuro-Symbolic Translator (The Rigid Interface)
S-ROS abandons massive tokenizers in favor of a dynamicslly expanding lookup table. Every valid environmental state (SENSOR_BLOCKED) and procedural command (MOTOR_REVERSE) maps 1-to-1 to a fixed, orthogonal spike pattern.
 * **Zero ambiguity.**
 * **Zero hallucination.**
 * **Pure deterministic translation** from ASCII/bytes to spikes, and spikes to machine commands.

### 3. S-CAM (Spiking Content-Addressable Memory 64K Procedural RAM)

The associative "hard drive" of 65,536-slot procedural execution block. S-CAM conceptually mirrors the 64-Kilobyte addressable memory architecture of classic 8-bit microcomputers (like the Commodore 64), but built entirely out of spiking neural tensors.
It acts as the physical RAM for the execution core. Instead of just holding isolated associative reflexes, it stores contiguous blocks of procedural logic mapped into 128-dimensional spike vectors. Using strict one-hot encoding for FSM stability, it hybridizes associative memory (content-addressable jumps and hardware interrupts) with sequential execution (a spiking program counter).
This allows S-ROS to store, recall, and natively step through complex, nested procedural code entirely within a biologically grounded spiking manifold, guaranteeing near zero-hallucination hardware execution.
The associative "hard drive." S-CAM acts as a biological database for execution memory, replacing the larger Cognitive Kernel for the Theo side of the neocortex.


### The Execution Loop
![S-ROS Architecture Diagram](./assets/s-ros-1775765475261.jpg)

Sense: Hardware reads the environment.
Encode: Translator converts the exact state into a spike vector.
Recall: S-CAM retrieves the associative memory of past actions in this state.
Decide: Tiny_Tim sequences the logical response.
Decode: Translator maps the output spikes back to strict procedural commands.
Act: Hardware executes the command.
Adapt: Environmental feedback drives neuromodulators (Dopamine/Norepinephrine equivalents). Success locks the synaptic pathway via STDP; failure triggers a novelty mismatch and penalizes the pathway.

