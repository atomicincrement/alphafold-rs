#!/usr/bin/env python3
"""
Validation framework for comparing alphafold-rs outputs with reference AlphaFold2.

This script provides utilities to:
1. Run the reference AlphaFold2 implementation
2. Extract intermediate values (Evoformer outputs, pair/single representations)
3. Compare with alphafold-rs outputs

The reference implementation is in alphafold-reference/ as a git submodule.

Setup:
    pip install dm-tree tensorflow jax dm-haiku absl-py

Running reference AlphaFold:
    python alphafold-reference/run_alphafold.py \\
        --fasta_paths=sequence.fasta \\
        --output_dir=reference_output/ \\
        --model_preset=monomer \\
        --max_template_date=2024-01-01
"""

import json
import sys
from pathlib import Path

# Try to import reference AlphaFold modules
try:
    sys.path.insert(0, str(Path(__file__).parent / "alphafold-reference"))
    from alphafold.model import model
    from alphafold.data import pipeline
    REFERENCE_AVAILABLE = True
except ImportError as e:
    REFERENCE_AVAILABLE = False
    print(f"Note: Reference AlphaFold not available ({e})")


def extract_evoformer_outputs(result_dict):
    """Extract Evoformer intermediate representations from reference output."""
    # The reference implementation stores intermediate values in the result
    # Look for Evoformer state tensors
    features = {}
    
    if "single" in result_dict:
        single = result_dict["single"]
        features["single_shape"] = single.shape
        features["single_mean"] = float(single.mean())
        features["single_std"] = float(single.std())
    
    if "pair" in result_dict:
        pair = result_dict["pair"]
        features["pair_shape"] = pair.shape
        features["pair_mean"] = float(pair.mean())
        features["pair_std"] = float(pair.std())
    
    return features


def compare_outputs(rust_coords, ref_coords, tolerance=0.5):
    """
    Compare predicted Cα coordinates from rust vs reference.
    
    Args:
        rust_coords: List of [x, y, z] from alphafold-rs
        ref_coords: List of [x, y, z] from reference
        tolerance: Maximum allowed difference in Ångströms
    
    Returns:
        Dict with comparison metrics
    """
    import numpy as np
    
    rust_arr = np.array(rust_coords)
    ref_arr = np.array(ref_coords)
    
    if rust_arr.shape != ref_arr.shape:
        return {
            "error": f"Shape mismatch: rust {rust_arr.shape} vs ref {ref_arr.shape}"
        }
    
    # Calculate per-residue distances
    distances = np.linalg.norm(rust_arr - ref_arr, axis=1)
    
    return {
        "num_residues": len(distances),
        "rmsd": float(np.sqrt(np.mean(distances**2))),
        "mae": float(np.mean(distances)),
        "max_distance": float(np.max(distances)),
        "min_distance": float(np.min(distances)),
        "within_tolerance": int(np.sum(distances <= tolerance)),
        "percent_within_tolerance": 100.0 * np.sum(distances <= tolerance) / len(distances),
        "distances": distances.tolist(),
    }


def main():
    """Main validation routine."""
    print("AlphaFold-rs Validation Framework")
    print("=" * 50)
    
    # Check if reference implementation is available
    if not REFERENCE_AVAILABLE:
        print("\nReference AlphaFold2 not available.")
        print("To enable validation:")
        print("  1. Install dependencies:")
        print("     pip install dm-tree tensorflow jax dm-haiku absl-py")
        print("  2. Install localcolabfold (recommended):")
        print("     pip install localcolabfold")
        print("\nAlternatively, compare with pre-computed reference outputs.")
        return
    
    print("\nReference AlphaFold2 is available!")
    print("Use: python alphafold-reference/run_alphafold.py")
    print("     to generate reference outputs for comparison.")


if __name__ == "__main__":
    main()
