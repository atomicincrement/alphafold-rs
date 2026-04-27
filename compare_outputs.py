#!/usr/bin/env python3
"""Compare Rust AlphaFold implementation against experimental structure (PDB 2F4K)."""

import json
import numpy as np
from scipy.spatial.distance import cdist

def load_coords(filepath):
    """Load coordinates from JSON file."""
    with open(filepath) as f:
        coords = json.load(f)
    return np.array(coords)

def load_pdb_ca_coords(pdb_file):
    """Extract CA coordinates from PDB file."""
    coords = []
    with open(pdb_file) as f:
        for line in f:
            if line.startswith('ATOM') and ' CA ' in line:
                # PDB format: x is cols 30-38, y is 38-46, z is 46-54
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                except ValueError:
                    continue
    return np.array(coords)

def kabsch_alignment(P, Q):
    """
    Align point cloud P to Q using Kabsch algorithm.
    Returns RMSD and aligned P.
    """
    # Center both clouds
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    
    # Compute covariance matrix
    H = P_centered.T @ Q_centered
    
    # SVD
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Align P to Q
    P_aligned = (R @ P_centered.T).T + centroid_Q
    
    # Compute RMSD
    rmsd = np.sqrt(np.mean(np.sum((P_aligned - Q_centered - centroid_Q) ** 2, axis=1)))
    
    return rmsd, P_aligned

def compute_metrics(predicted, experimental):
    """Compute comparison metrics between predicted and experimental structures."""
    
    # Pad or trim to same length
    n = min(len(predicted), len(experimental))
    pred = predicted[:n]
    exp = experimental[:n]
    
    print(f"\n{'='*60}")
    print(f"Comparing {n} residues")
    print(f"{'='*60}\n")
    
    # 1. Raw distances before alignment
    raw_distances = np.linalg.norm(pred - exp, axis=1)
    print("Raw Distances (before alignment):")
    print(f"  Mean: {np.mean(raw_distances):.3f} Å")
    print(f"  Std:  {np.std(raw_distances):.3f} Å")
    print(f"  Min:  {np.min(raw_distances):.3f} Å")
    print(f"  Max:  {np.max(raw_distances):.3f} Å")
    
    # 2. Align using Kabsch
    rmsd, pred_aligned = kabsch_alignment(pred, exp)
    aligned_distances = np.linalg.norm(pred_aligned - exp, axis=1)
    
    print(f"\nAfter Kabsch Alignment:")
    print(f"  RMSD: {rmsd:.3f} Å")
    print(f"  Mean distance: {np.mean(aligned_distances):.3f} Å")
    print(f"  Std:  {np.std(aligned_distances):.3f} Å")
    print(f"  Min:  {np.min(aligned_distances):.3f} Å")
    print(f"  Max:  {np.max(aligned_distances):.3f} Å")
    
    # 3. Percentage within tolerance
    tolerances = [0.5, 1.0, 2.0, 3.0]
    print(f"\nResidues within tolerance (after alignment):")
    for tol in tolerances:
        pct = 100 * np.sum(aligned_distances <= tol) / len(aligned_distances)
        print(f"  ≤ {tol} Å: {pct:.1f}%")
    
    # 4. Inter-residue distances (backbone geometry)
    if n >= 2:
        pred_inter = np.linalg.norm(np.diff(pred, axis=0), axis=1)
        exp_inter = np.linalg.norm(np.diff(exp, axis=0), axis=1)
        inter_diff = pred_inter - exp_inter
        
        print(f"\nBackbone Ca-Ca distances (inter-residue):")
        print(f"  Predicted: mean={np.mean(pred_inter):.3f}, std={np.std(pred_inter):.3f} Å")
        print(f"  Experimental: mean={np.mean(exp_inter):.3f}, std={np.std(exp_inter):.3f} Å")
        print(f"  Difference: mean={np.mean(np.abs(inter_diff)):.3f} Å")
    
    # 5. Per-residue breakdown
    print(f"\nPer-residue distances (after alignment):")
    print(f"{'Res':<5} {'Distance (Å)':<15} {'Inter-res (Å)':<15}")
    print("-" * 35)
    for i in range(min(10, len(aligned_distances))):  # Show first 10
        inter = ""
        if i < len(pred_inter):
            inter = f"{inter_diff[i]:+.3f}"
        print(f"{i+1:<5} {aligned_distances[i]:>6.3f}          {inter:>14}")
    if len(aligned_distances) > 10:
        print(f"... ({len(aligned_distances) - 10} more)")
    
    return {
        'raw_rmsd': np.sqrt(np.mean(raw_distances ** 2)),
        'rmsd': rmsd,
        'mean_distance': np.mean(aligned_distances),
        'within_1A': np.sum(aligned_distances <= 1.0) / len(aligned_distances)
    }

if __name__ == '__main__':
    print("Loading structures...")
    
    try:
        # Try corrected version first, fall back to old version
        try:
            predicted = load_coords('rust_coords_corrected.json')
            print(f"✓ Rust output (corrected): {predicted.shape[0]} residues")
        except FileNotFoundError:
            predicted = load_coords('rust_coords.json')
            print(f"✓ Rust output: {predicted.shape[0]} residues")
    except FileNotFoundError:
        print("✗ rust_coords*.json not found. Run: ./target/release/alphafold-rs generate sequence.fasta rust_coords.json")
        exit(1)
    
    try:
        experimental = load_pdb_ca_coords('2f4k.pdb')
        print(f"✓ Experimental (PDB 2F4K): {experimental.shape[0]} residues")
    except FileNotFoundError:
        print("✗ 2f4k.pdb not found")
        exit(1)
    
    metrics = compute_metrics(predicted, experimental)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"RMSD vs experimental: {metrics['rmsd']:.3f} Å")
    print(f"Residues within 1 Å: {metrics['within_1A']*100:.1f}%")
    
    if metrics['rmsd'] < 1.0:
        print("✓ GOOD: Structure aligns well with experimental data")
    elif metrics['rmsd'] < 2.0:
        print("⚠ FAIR: Reasonable structural agreement")
    else:
        print("✗ POOR: Significant deviation from experimental structure")
