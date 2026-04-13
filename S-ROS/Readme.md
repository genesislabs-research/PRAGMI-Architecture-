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

```mermaid
flowchart TD
    %% Data Generator
    A[rote_data_generator.py\nGenerates 15 BASIC rule classes & splits data]

    %% Data flow
    A -->|"Training Pairs"| B
    A -->|"Held-Out Test Pairs"| C

    %% Trainer
    B[hello_world_trainer.py\n\n1. Encodes BASIC lines\n2. Runs RoteLearner (LIF Network)\n3. Computes cross-entropy loss\n4. Backpropagates & updates weights]

    %% Evaluation trigger
    B -->|"After N training steps"| C

    %% Crystallization Manager (Neo)
    C[crystallization_manager.py (Neo)\n\nTests network on Held-Out Test Pairs.\nChecks 3 conditions for K consecutive windows:\n• Training loss < threshold\n• Weight delta variance stabilized\n• Generalization accuracy > target]

    %% Decision branches
    C -->|FAIL / NO\nRule not learned| D[Continues training\nnext epoch]
    C -->|PASS / YES\nRule understood| E[Rule Crystallized!\n(Skip rule in future)]

    %% Loop back to trainer
    D --> B

    %% Final output
    E --> F[Final Checkpoint &\nCrystallization Log]

    %% Styling
    classDef generator fill:#bae6fd,stroke:#0c4a6e,stroke-width:2px,color:#0c4a6e,rx:15,ry:15;
    classDef trainer fill:#99f6e4,stroke:#134e4a,stroke-width:2px,color:#134e4a,rx:15,ry:15;
    classDef neo fill:#c4b5fd,stroke:#4c1d95,stroke-width:3px,color:#4c1d95,rx:15,ry:15;
    classDef success fill:#86efac,stroke:#166534,stroke-width:2px,color:#166534,rx:15,ry:15;
    classDef final fill:#4ade80,stroke:#14532d,stroke-width:2px,color:#14532d,rx:15,ry:15;

    class A generator
    class B trainer
    class C neo
    class E success
    class F final



## Rote Learning + Crystallization Pipeline


```mermaid
flowchart TD
    %% Top level
    A[rote_data_generator.py<br/>Generates 15 BASIC rule classes & splits data] 

    %% Data splits
    A -->|Training Pairs| B
    A -->|Held-Out Test Pairs| C

    %% Trainer
    B[hello_world_trainer.py<br/><br/>1. Encodes BASIC lines<br/>2. Runs RoteLearner (LIF Network)<br/>3. Computes cross-entropy loss<br/>4. Backpropagates & updates weights]

    %% Flow from trainer to evaluation
    B -->|After N training steps| C

    %% Crystallization Manager (Neo)
    C[crystallization_manager.py<br/>(Neo)<br/><br/>Tests network on Held-Out Test Pairs.<br/>Checks 3 conditions for K consecutive windows:<br/>• Training loss < threshold<br/>• Weight delta variance stabilized<br/>• Generalization accuracy > target]

    %% Decision branches
    C -->|FAIL / NO<br/>Rule not learned| D[Continues training next epoch]
    C -->|PASS / YES<br/>Rule understood| E[Rule Crystallized!<br/>(Skip rule in future)]

    %% Loop back
    D --> B

    %% Final output
    E --> F[Final Checkpoint &<br/>Crystallization Log]

    %% Styling
    classDef generator fill:#bae6fd,stroke:#0c4a6e,stroke-width:2px,color:#0c4a6e,rx:10,ry:10;
    classDef trainer fill:#99f6e4,stroke:#134e4a,stroke-width:2px,color:#134e4a,rx:10,ry:10;
    classDef neo fill:#c4b5fd,stroke:#4c1d95,stroke-width:3px,color:#4c1d95,rx:12,ry:12;
    classDef success fill:#86efac,stroke:#166534,stroke-width:2px,color:#166534,rx:10,ry:10;
    classDef final fill:#4ade80,stroke:#14532d,stroke-width:2px,color:#14532d,rx:10,ry:10;

    class A generator
    class B trainer
    class C neo
    class E success
    class F final

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
    
                                     
        
 
                                     
