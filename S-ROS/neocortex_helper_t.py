"""
neocortex_helper_t.py
Neocortical Translation Utilities: Multimodal Binding and Population Decoding

BIOLOGICAL GROUNDING
This file models the association cortices performing the raw mathematical
translation of diverse coordinate spaces. It handles the discrete-to-continuous
population decoding necessary to unify deterministic execution memory spikes
with continuous semantic float embeddings before entorhinal projection.

Mesulam MM (1998). "From sensation to cognition." Brain, 121(6), 1013-1052.
DOI: 10.1093/brain/121.6.1013
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class PopulationDecoder(nn.Module):
    """
    Converts discrete orthogonal spikes to a continuous vector space.
    BIOLOGICAL STRUCTURE: Cortical Population Coding.
    BIOLOGICAL FUNCTION: Extracts continuous variables from the collective
    activity of discrete spiking neurons.
    Eliasmith C, Anderson CH (2003). "Neural engineering: Computation,
    representation, and dynamics in neurobiological systems." MIT Press.
    DOI: 10.7551/mitpress/4934.001.0001
    """
    def __init__(self, spike_dim: int, coordinate_dim: int = 64) -> None:
        super().__init__()
        # NOT a biological quantity: linear weights act as decoding filters.
        self.optimal_linear_decoder = nn.Linear(spike_dim, coordinate_dim, bias=False)

    def forward(self, discrete_spikes: torch.Tensor) -> torch.Tensor:
        return self.optimal_linear_decoder(discrete_spikes)

class SchemaOverlapMetric(nn.Module):
    """
    Computes geometric similarity between memory reconstructions and schemas.
    BIOLOGICAL STRUCTURE: Association Cortex Pattern Matching.
    BIOLOGICAL FUNCTION: Quantifies representational overlap between incoming
    episodes and established semantic knowledge.
    McClelland JL, McNaughton BC, O'Reilly RC (1995). "Why there are
    complementary learning systems in the hippocampus and neocortex."
    Psychological Review. DOI: 10.1037/0033-295X.102.3.419
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, state_a: torch.Tensor, state_b: torch.Tensor) -> torch.Tensor:
        # NOT a biological quantity: cosine similarity is a mathematical abstraction
        # for activation pattern correlation.
        return F.cosine_similarity(state_a, state_b, dim=-1)

class MultimodalBinder(nn.Module):
    """
    Concatenates and projects segregated streams into a shared coordinate space.
    BIOLOGICAL STRUCTURE: Neocortical mixture of experts binding.
    BIOLOGICAL FUNCTION: Integrates inputs from distinct specialized
    networks into a singular manifold for hippocampal processing.
    Fuster JM (2001). "The prefrontal cortex--an update."
    Neuron, 30(2), 319-333. DOI: 10.1016/S0896-6273(01)00285-9
    """
    def __init__(self, coordinate_dim: int = 64) -> None:
        super().__init__()
        self.semantic_projection = nn.Linear(coordinate_dim, coordinate_dim, bias=True)
        self.unification_layer = nn.Linear(coordinate_dim * 2, coordinate_dim, bias=True)

    def forward(self, decoded_action_state: torch.Tensor, ventral_semantic_state: torch.Tensor) -> torch.Tensor:
        semantic_state = torch.tanh(self.semantic_projection(ventral_semantic_state))
        unified = torch.cat([decoded_action_state, semantic_state], dim=-1)
        return torch.tanh(self.unification_layer(unified))
      
