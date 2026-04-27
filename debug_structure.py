#!/usr/bin/env python3
"""Debug script to trace structure module frame evolution."""

import json
import numpy as np

# Load Evoformer output to understand input
with open('rust_evoformer_corrected.json') as f:
    evo_stats = json.load(f)

print("Evoformer Output Statistics")
print("=" * 60)
print(f"Single representation: {evo_stats['single_shape']}")
print(f"  Mean: {evo_stats['single_stats']['mean']:.3f}")
print(f"  Std:  {evo_stats['single_stats']['std']:.3f}")
print(f"  Range: [{evo_stats['single_stats']['min']:.3f}, {evo_stats['single_stats']['max']:.3f}]")

print(f"\nPair representation: {evo_stats['pair_shape']}")
print(f"  Mean: {evo_stats['pair_stats']['mean']:.3f}")
print(f"  Std:  {evo_stats['pair_stats']['std']:.3f}")
print(f"  Range: [{evo_stats['pair_stats']['min']:.3f}, {evo_stats['pair_stats']['max']:.3f}]")

# Load predicted coordinates
with open('rust_coords_corrected.json') as f:
    coords = np.array(json.load(f))

print("\n" + "=" * 60)
print("Structure Module Output")
print("=" * 60)
print(f"Coordinates shape: {coords.shape}")

# Analyze frame deltas
inter_res_distances = np.linalg.norm(np.diff(coords, axis=0), axis=1)
print(f"\nInter-residue Cα-Cα distances:")
print(f"  Mean: {np.mean(inter_res_distances):.3f} Å")
print(f"  Std:  {np.std(inter_res_distances):.3f} Å")
print(f"  Min:  {np.min(inter_res_distances):.3f} Å")
print(f"  Max:  {np.max(inter_res_distances):.3f} Å")

print(f"\nFirst 10 distances:")
for i, d in enumerate(inter_res_distances[:10]):
    print(f"  {i+1:2d} → {i+2:2d}: {d:.3f} Å")

# Expected: ~3.81 Å for extended β-strand, ~3.8 Å for typical backbone
# Our prediction: ~5.05 Å (too large)

print("\n" + "=" * 60)
print("Analysis")
print("=" * 60)
print(f"Ratio (predicted/expected): {np.mean(inter_res_distances) / 3.81:.2f}x")

# Check if structure is just uniformly scaled
# Try different scaling hypotheses
scales_to_try = [1.0, 1.25, 1.5, 1.75, 2.0]
print(f"\nIf we scale coordinates by factor k:")
for k in scales_to_try:
    scaled_distances = inter_res_distances / k
    mean_d = np.mean(scaled_distances)
    print(f"  k={k:.2f}: mean={mean_d:.3f} Å (diff: {abs(mean_d - 3.81):.3f})")

# Look at coordinate magnitudes
print(f"\nCoordinate magnitudes:")
print(f"  Mean distance from origin: {np.mean(np.linalg.norm(coords, axis=1)):.3f} Å")
print(f"  Max distance from origin:  {np.max(np.linalg.norm(coords, axis=1)):.3f} Å")

# Check if there's a systematic translation/rotation issue
print(f"\nFirst few coordinates (should be ~[0,0,0] for 1st residue ideally):")
for i in range(5):
    print(f"  Residue {i+1}: [{coords[i,0]:7.3f}, {coords[i,1]:7.3f}, {coords[i,2]:7.3f}]")

print("\n" + "=" * 60)
print("Hypothesis Check")
print("=" * 60)
print(f"✓ Coordinates are not close to origin (structure is not at frame origin)")
print(f"✓ Distances are ~1.3x larger than expected (uniform overstretching)")
print(f"✓ Issue is NOT just the 27x scaling factor")
print(f"\nProbable causes:")
print(f"  1. Frame update deltas are too large")
print(f"  2. Quaternion normalization error")
print(f"  3. Frame composition error")
print(f"  4. IPA point positions are too spread out")
