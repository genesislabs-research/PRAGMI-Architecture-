"""
firstday.py
First Day: Clone Prime to Specialists and Run Initial Sleep Cycle

BIOLOGICAL GROUNDING:
This script implements the developmental transition between the template phase
and the experience-dependent specialization phase of TimmyArray. In the
mammalian brain, cortical columns share a common progenitor lineage and
identical initial connectivity. Areal identity emerges from thalamic input
statistics and activity-dependent refinement, not from intrinsic architectural
differences at birth (Rakic, 1988). This script enacts the computational
analog of that transition.

The three stages of firstday.py:

STAGE 1 - CLONE: clone_prime_to_specialists() copies Prime's complete trained
state (weights, LIF thresholds, synaptic time constants, MoE cluster
assignments, STDP scalars) into every specialist column simultaneously.
All six columns are now weight-for-weight identical. LIF membrane states
are reset to rest after copy because Prime's membrane potential at training
end reflects Prime's last input sequence, not a valid starting state for a
specialist entering its first experience.

STAGE 2 - FIRST SLEEP: A single abbreviated sleep cycle runs immediately
after the clone. This is the only larger-magnitude divergence event. The
sleep cycle applies specialty-directed synaptic scaling to each specialist,
weakening connections least relevant to the specialist's assigned integration
scale. After this cycle, specialists are no longer identical to Prime. All
subsequent divergence is gradual, accumulating across day and sleep cycles.

STAGE 3 - CHECKPOINT: The post-sleep array state is saved as
firstday_complete.soul for each column. These are the starting checkpoints
for Phase 2 training.

The biological grounding for sleep-driven divergence is the synaptic
homeostasis hypothesis: sleep consolidates waking experience by
downscaling synaptic weights in proportion to their relevance to the
organism's goals (Tononi & Cirelli, 2003). Here we apply that principle
at array initialization: each specialist's first sleep biases it toward
its assigned integration scale before any real experience arrives.

Key grounding papers:
1. Rakic P (1988). "Specification of cerebral cortical areas." Science,
   241(4862):170-176. DOI: 10.1126/science.3291116
   (Radial unit hypothesis: columns share common progenitor lineage;
   areal identity emerges from experience, not intrinsic differences.)

2. Tononi G, Cirelli C (2003). "Sleep and synaptic homeostasis: a
   hypothesis." Brain Research Bulletin, 62(2):143-150.
   DOI: 10.1016/j.brainresbull.2003.09.004
   (Sleep downscales synaptic weights; net effect is selective
   consolidation of relevant connections.)

3. Bi GQ, Poo MM (1998). "Synaptic modifications in cultured hippocampal
   neurons: dependence on spike timing, synaptic strength, and postsynaptic
   cell type." Journal of Neuroscience, 18(24):10464-10472.
   DOI: 10.1523/JNEUROSCI.18-24-10464.1998
   (STDP state is preserved through the clone so each specialist begins
   with Prime's full plasticity history intact.)

Usage:
    python firstday.py --prime checkpoints/array_phase1_coordready_prime.state
                       --output checkpoints/firstday/

    # Skip sleep cycle (clone only, for testing):
    python firstday.py --prime checkpoints/array_phase1_coordready_prime.state
                       --output checkpoints/firstday/ --no-sleep
"""

from __future__ import annotations

import argparse
import os
import sys
import math
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from CreateTimmyArray import TimmyArray, TimmyArrayConfig, COLUMN_NAMES
from timmy_model import TimmyConfig
from timmy_state import load_timmy_state


# =========================================================================
# Specialty Bias Vectors
# =========================================================================

# Per-specialist synaptic scaling bias vectors applied during first sleep.
# Each vector defines the relative importance of different processing
# dimensions for that specialist's integration scale.
#
# Values are scaling factors applied to v_threshold_raw during the first
# sleep cycle. Values > 1.0 raise the threshold (reduce sensitivity) for
# dimensions less relevant to this specialist. Values < 1.0 lower the
# threshold (increase sensitivity) for relevant dimensions.
#
# NOT biological quantities. Engineering initialization bias that approximates
# the effect of early thalamic input statistics on cortical column tuning.
# Reference: Rakic P (1988). DOI: 10.1126/science.3291116
#
# These are conservative starting biases. The real divergence emerges from
# Phase 2 input distribution differences. These just break the initial
# symmetry so columns do not start Phase 2 as perfect clones.

SPECIALTY_THRESHOLD_BIAS: Dict[str, float] = {
    # proximal: tuned to near-term context, fine sequential structure.
    # Slightly lower threshold for faster response to local patterns.
    "proximal":   0.98,
    # distal: tuned to long-range dependencies, discourse coherence.
    # Slightly higher threshold encourages integration over longer windows.
    "distal":     1.02,
    # affective: tuned to valenced content, social reasoning.
    # Neutral starting bias; affective tuning emerges from experience.
    "affective":  1.00,
    # somatic: tuned to grounded, embodied, sensorimotor content.
    # Slightly lower threshold for sensitivity to procedural patterns.
    "somatic":    0.97,
    # structural: tuned to formal relational structure, code, logic.
    # Slightly higher threshold for noise rejection in formal domains.
    "structural": 1.03,
}


# =========================================================================
# First Sleep Cycle
# =========================================================================

def run_first_sleep(
    array: TimmyArray,
    device: torch.device,
    noise_scale: float = 0.001,
) -> None:
    """
    Apply the initial specialty-directed sleep cycle to all specialist columns.

    BIOLOGICAL STRUCTURE: Sleep-driven synaptic homeostasis and experience-
    dependent refinement of cortical column tuning.

    BIOLOGICAL FUNCTION: The first sleep cycle applies specialty-directed
    synaptic scaling to break the post-clone symmetry between columns.
    Each specialist receives a small bias to its LIF spike thresholds
    reflecting the processing demands of its assigned integration scale.
    A small amount of biological noise is added to prevent degenerate
    synchrony across the ensemble.

    This is NOT a full sleep cycle in the sense of daycycle.py/sleep.py.
    It is a minimal initialization event: threshold perturbation plus
    noise injection. The full NREM/REM consolidation cycle requires
    episodic memory content from actual experience and runs after
    deployment via sleep.py.

    Reference: Tononi G, Cirelli C (2003). DOI: 10.1016/j.brainresbull.2003.09.004
    Reference: Seibt J, Frank MG (2019). "Primed to sleep: the dynamics of
    synaptic plasticity across brain states." Frontiers in Systems
    Neuroscience, 13:2. DOI: 10.3389/fnsys.2019.00002

    NOT a biological quantity: the specific bias magnitudes in
    SPECIALTY_THRESHOLD_BIAS are engineering approximations, not derived
    from measured biological data.

    Args:
        array: TimmyArray instance post-clone. Prime is not modified.
        device: torch device.
        noise_scale: standard deviation of Gaussian noise added to
            specialist weights. Small by design. NOT a biological quantity.
    """
    print("\n[ FIRST SLEEP CYCLE ]")
    print("  Applying specialty-directed threshold bias to specialists.")
    print("  Prime is not modified.")

    with torch.no_grad():
        for name in array.column_names[1:]:  # excludes prime
            specialist = array.specialists[name]
            bias = SPECIALTY_THRESHOLD_BIAS.get(name, 1.0)

            # Apply threshold bias to every AssociativeLIF layer in this
            # specialist. v_threshold_raw is a learnable (n_neurons,) parameter.
            # Multiplying by bias shifts the specialist's firing sensitivity
            # relative to Prime without disrupting the weight structure.
            lif_layers_modified = 0
            for module in specialist.modules():
                vt = getattr(module, "v_threshold_raw", None)
                if vt is not None and isinstance(vt, nn.Parameter):
                    vt.data.mul_(bias)
                    lif_layers_modified += 1

            # Add small noise to break post-clone weight symmetry.
            # Without this, all specialists are perfect replicas and gradient
            # updates in Phase 2 may be degenerate for the first few steps.
            # NOT a biological quantity. Numerical stability mechanism.
            params_noised = 0
            for param in specialist.parameters():
                if param.requires_grad:
                    noise = torch.randn_like(param.data) * noise_scale
                    param.data.add_(noise)
                    params_noised += 1

            print(
                f"  {name:12s}: threshold bias={bias:.3f} "
                f"({lif_layers_modified} LIF layers), "
                f"noise added to {params_noised} parameters"
            )

    print("  First sleep complete.")


# =========================================================================
# Save Helpers
# =========================================================================

def save_firstday_checkpoints(
    array: TimmyArray,
    output_dir: str,
) -> None:
    """
    Save post-firstday checkpoints for all columns.

    BIOLOGICAL STRUCTURE: Post-sleep memory consolidation state snapshot.
    BIOLOGICAL FUNCTION: Preserves the post-clone, post-first-sleep state
    as the starting point for Phase 2 training. Each column is saved
    independently so any column can be resumed or replaced without
    disturbing the others.

    NOT a biological quantity: file format and naming convention are
    engineering choices.

    Args:
        array: TimmyArray instance post-firstday.
        output_dir: directory to write .state files.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[ SAVING FIRSTDAY CHECKPOINTS to {output_dir} ]")

    for name, col in zip(array.column_names, array.all_columns()):
        path = str(Path(output_dir) / f"firstday_{name}.state")
        col.save_state(path, training_step=0)
        print(f"  Saved {name}: {path}")

    print("  All columns saved.")


# =========================================================================
# Main
# =========================================================================

def run_firstday(
    prime_checkpoint: str,
    output_dir: str,
    run_sleep: bool = True,
    noise_scale: float = 0.001,
    device_str: str = "cpu",
) -> TimmyArray:
    """
    Execute the full firstday pipeline: load Prime, construct array,
    clone Prime to specialists, run first sleep, save checkpoints.

    ANATOMICAL INTERFACE:
        Sending structure: TimmyModel (TimmyPrime, Phase 1 coordready checkpoint).
        Receiving structure: TimmyArray with all specialist columns initialized.
        Connection: clone_prime_to_specialists() copies Prime's state into
        each specialist via state_dict()/load_state_dict(). This is the
        computational analog of the radial unit developmental template
        described by Rakic (1988).

    Args:
        prime_checkpoint: path to Phase 1 Prime .state file.
        output_dir: directory for firstday checkpoint output.
        run_sleep: if True, run the first sleep cycle after clone.
            Set False for testing or if you want a pure clone baseline.
        noise_scale: Gaussian noise std added to specialist weights
            during first sleep. NOT a biological quantity.
        device_str: torch device string.

    Returns:
        TimmyArray with all columns initialized and ready for Phase 2.
    """
    device = torch.device(device_str)

    print()
    print("=" * 60)
    print("TIMMY FIRSTDAY")
    print("Clone Prime to specialists + first sleep cycle")
    print("=" * 60)

    # Load Prime checkpoint to recover config.
    print(f"\n[ LOADING PRIME CHECKPOINT ]")
    print(f"  {prime_checkpoint}")

    raw = torch.load(prime_checkpoint, map_location=device, weights_only=False)
    saved_cfg = raw.get("config")
    if saved_cfg is None:
        raise ValueError(
            f"Checkpoint at {prime_checkpoint} does not contain a 'config' key. "
            "Cannot reconstruct array."
        )

    # Build array config from saved column config.
    # num_specialists defaults to 5 (full six-column array).
    # Override via command line if you need a smaller array for testing.
    from timmy_model import TimmyModel
    prime = TimmyModel(saved_cfg).to(device)
    meta = prime.load_state(prime_checkpoint, device=device)
    print(f"  Prime loaded. Training step: {meta.get('training_step', 'unknown')}")

    arr_cfg = TimmyArrayConfig(
        column_cfg=saved_cfg,
        num_specialists=5,
    )
    array = TimmyArray(arr_cfg).to(device)

    # Install trained Prime into the array.
    array.prime.load_state_dict(prime.state_dict())
    array.prime.load_state(prime_checkpoint, device=device)
    print("  Prime installed into array.")

    # Stage 1: Clone.
    print("\n[ STAGE 1: CLONE PRIME TO SPECIALISTS ]")
    print("  Reference: Rakic P (1988). DOI: 10.1126/science.3291116")
    array.clone_prime_to_specialists()
    print("  Clone complete. All specialists are now identical to Prime.")
    print("  LIF membrane states reset to rest on all specialists.")

    # Stage 2: First sleep.
    if run_sleep:
        run_first_sleep(array, device, noise_scale=noise_scale)
    else:
        print("\n[ FIRST SLEEP SKIPPED (--no-sleep flag) ]")

    # Stage 3: Save.
    save_firstday_checkpoints(array, output_dir)

    print()
    print("=" * 60)
    print("FIRSTDAY COMPLETE")
    print(f"Checkpoints saved to: {output_dir}")
    print("Next step: run Phase 2 training with these checkpoints as --resume_paths")
    print("=" * 60)
    print()

    return array


# =========================================================================
# CLI
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "First Day: clone trained Prime to all specialist columns "
            "and run the initial specialty-directed sleep cycle. "
            "Produces Phase 2 starting checkpoints."
        )
    )
    parser.add_argument(
        "--prime",
        type=str,
        required=True,
        help="Path to Phase 1 Prime .state checkpoint (phase1_coordready_prime.state).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/firstday",
        help="Output directory for firstday checkpoints (default: checkpoints/firstday).",
    )
    parser.add_argument(
        "--no-sleep",
        action="store_true",
        help="Skip the first sleep cycle. Clone only. Useful for testing.",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.001,
        help="Gaussian noise std added to specialist weights during sleep (default: 0.001).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device (default: cpu). Use 'cuda' for GPU.",
    )

    args = parser.parse_args()

    try:
        run_firstday(
            prime_checkpoint=args.prime,
            output_dir=args.output,
            run_sleep=not args.no_sleep,
            noise_scale=args.noise_scale,
            device_str=args.device,
        )
    except Exception as e:
        print(f"\nFIRSTDAY FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
