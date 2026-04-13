# S-ROS (Spiking Robot Operating System)
## Project Goals

1) Prove that a de novo spiking neural network can reach a transition prediction loss of < 0.007 (0.7% MSE) 

2) Prove that a de novo spiking neural network can learn procedural, syntactic code execution (BASIC-style logic) without pre-training or distillation.

3) Prove that a de novo spiking neural network can overcome the limits of standard Finite State Machines by implementing a discrete Spiking Call Stack, allowing the network to natively track GOSUB recursion and nested loops without state aliasing.
  
4) Establish a biologically grounded neuro-symbolic bridge that completely isolates exploratory learning from rigid, near zero-hallucination robotic actuation laying the ground work for learning systems which can reach an Interal Crystallization Threshold: 0.7% MSE while also containing one shot learning mechanics.

# Chatbots are a solved problem, 
#### but standard robotic operating systems are rigid message-passers that cannot adapt. S-ROS marries the absolute, deterministic certainty of procedural logic (BASIC/ROS) with the associative, real-time learning capabilities of a pure spiking neural network.

## A system that predicts the next token and then stores memories of its own predictions is a system that just remembers what it guessed. 
### It does not understand what it guessed or why. Understanding requires the system to build an internal model where the relationships between elements are represented explicitly enough that the system can answer questions it was never trained on by composing what it knows. When a human learns that PRINT with a comma puts output in the next zone, they do not just memorize the input-output pair. They build a model of what zones are, what commas do as separators, and what PRINT does as a display operation. Those three pieces compose freely: they can predict what PRINT A$, B$, C$ does without ever having seen a three-argument example, because they understand the comma rule, not just the specific case.


## Rote Learning + Crystallization Pipeline

**rote_data_generator.py**  
Generates 15 BASIC rule classes and splits data into Training Pairs and Held-Out Test Pairs.

**hello_world_trainer.py**  
1. Encodes BASIC lines  
2. Runs RoteLearner (LIF Network)  
3. Computes cross-entropy loss  
4. Backpropagates & updates weights  

After every N training steps, it hands control to:

**crystallization_manager.py (Neo)**  
Tests the network on the Held-Out Test Pairs.  
Checks 3 conditions for K consecutive windows:  
• Training loss < threshold  
• Weight delta variance stabilized  
• Generalization accuracy > target  

If all 3 pass → **Rule Crystallized!** (skipped in future epochs)  
If not → continue training next epoch.

**Final output:** Checkpoint + Crystallization Log

## Rote Learning + Crystallization Pipeline
┌────────────────────────────────────────────────────────┐
│               rote_data_generator.py                   │
│     (Generates 15 BASIC rule classes & splits data)    │
└──────┬──────────────────────────────────────────┬──────┘
       │                                          │
       │ (Training Pairs)                         │ (Held-Out Test Pairs)
       ▼                                          │
┌──────────────────────────────────────┐          │
│       hello_world_trainer.py         │          │
│                                      │          │
│  1. Encodes BASIC lines              │          │
│  2. Runs RoteLearner (LIF Network)   │          │
│  3. Computes cross-entropy loss      │          │
│  4. Backpropagates & updates weights │          │
└──────┬───────────────────────────────┘          │
       │                                          │
       │ (After N training steps)                 │
       ▼                                          ▼
┌────────────────────────────────────────────────────────┐
│              crystallization_manager.py                │
│                        (Neo)                           │
│                                                        │
│  Tests network on Held-Out Test Pairs.                 │
│  Checks 3 conditions for K consecutive windows:        │
│    • Training loss < threshold                         │
│    • Weight delta variance stabilized                  │
│    • Generalization accuracy > target                  │
└──────┬─────────────────────────────────────────┬───────┘
       │                                         │
       ▼                                         ▼
  [FAIL / NO]                               [PASS / YES]
       │                                         │
       │ (Rule not learned)                      │ (Rule understood)
       └─────────────────────────────────────────┤
            Continues training next epoch        │
                                                 ▼
                                    ┌─────────────────────────┐
                                    │   Rule Crystallized!    │
                                    │  (Skip rule in future)  │
                                    └────────────┬────────────┘
                                                 │
                                                 ▼
                                    ┌─────────────────────────┐
                                    │  Final Checkpoint &     │
                                    │  Crystallization Log    │
                                    └─────────────────────────┘




