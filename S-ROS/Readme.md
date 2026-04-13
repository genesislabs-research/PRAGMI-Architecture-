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

flowchart TD
    A[<b>rote_data_generator.py</b><br/>Generates 15 BASIC rule classes & splits data]
    
    A -->|Training Pairs| B
    A -->|Held-Out Test Pairs| C

    subgraph Training Engine
        B[<b>hello_world_trainer.py</b><br/>1. Encodes BASIC lines<br/>2. Runs RoteLearner LIF Network<br/>3. Computes cross-entropy loss<br/>4. Backpropagates & updates weights]
    end

    B -->|After N training steps| C

    subgraph Neo
        C{<b>crystallization_manager.py</b><br/>Tests network on Held-Out Test Pairs<br/>Checks 3 conditions for K windows:<br/>1. Loss < threshold<br/>2. Weight delta stabilized<br/>3. Gen accuracy > target}
    end

    C -->|FAIL / NO<br/>Rule not learned| D[Continues training next epoch]
    D -->|Loops back| B
    
    C -->|PASS / YES<br/>Rule understood| E[<b>Rule Crystallized!</b><br/>Skip rule in future]
    E --> F[Final Checkpoint &<br/>Crystallization Log]
    
    classDef default fill:#2b313c,stroke:#5c6370,stroke-width:2px,color:#d7dae0;
    classDef pass fill:#1e4620,stroke:#2ea043,stroke-width:2px,color:#ffffff;
    classDef fail fill:#4a1c1c,stroke:#f85149,stroke-width:2px,color:#ffffff;
    
    class E,F pass;
    class D fail;
    
                                     
        
 
                                     
