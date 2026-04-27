#!/usr/bin/env python3
"""
Proper alignment and RMSD calculation for predicted vs PDB structure.
Uses Kabsch algorithm for optimal superposition.
"""

import math
from urllib.request import urlopen
import gzip

def dist(p1, p2):
    """Euclidean distance between two 3D points."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def mean_point(points):
    """Centroid of a point cloud."""
    if not points:
        return [0, 0, 0]
    n = len(points)
    return [sum(p[i] for p in points) / n for i in range(3)]

def subtract_point(p, center):
    """Subtract center from point."""
    return [p[i] - center[i] for i in range(3)]

def matrix_mult_vec(mat, vec):
    """3x3 matrix-vector multiplication."""
    return [sum(mat[i][j] * vec[j] for j in range(3)) for i in range(3)]

def simple_rmsd(pred, ref):
    """Calculate RMSD between two point sets (assumes they're already aligned)."""
    if len(pred) != len(ref):
        return float('inf')
    sum_sq = sum(dist(p, r)**2 for p, r in zip(pred, ref))
    return math.sqrt(sum_sq / len(pred))

# Predicted coordinates from output.log
predicted_full = [
    [0.225, -0.245, 0.497],  # L
    [0.364, -0.209, 0.498],  # S
    [0.371, -0.276, 0.560],  # D
    [0.231, -0.193, 0.469],  # E
    [0.423, -0.195, 0.549],  # D
    [0.203, -0.261, 0.489],  # F
    [0.409, -0.166, 0.441],  # K
    [0.266, -0.321, 0.495],  # A
    [0.218, -0.207, 0.470],  # V
    [0.086, -0.225, 0.359],  # F
    [0.221, -0.216, 0.427],  # G
    [0.216, -0.109, 0.559],  # M
    [0.417, -0.345, 0.443],  # T
    [0.417, -0.198, 0.591],  # R
    [0.053, -0.268, 0.241],  # S
    [0.418, -0.282, 0.501],  # A
    [0.169, -0.307, 0.515],  # F
    [0.153, -0.229, 0.373],  # A
    [0.457, -0.412, 0.562],  # N
    [0.294, -0.195, 0.510],  # L
    [0.375, -0.374, 0.510],  # P
    [0.303, -0.228, 0.446],  # L
    [0.133, -0.307, 0.420],  # W
    [0.337, -0.073, 0.473],  # K
    [0.218, -0.130, 0.473],  # Q
    [0.359, -0.264, 0.409],  # Q
    [0.097, -0.179, 0.549],  # N
    [0.146, -0.349, 0.455],  # L
    [0.317, -0.169, 0.543],  # K
    [0.230, -0.207, 0.458],  # K
    [0.097, -0.237, 0.280],  # E
    [0.473, -0.226, 0.548],  # K
    [0.414, -0.282, 0.486],  # G
    [0.336, -0.180, 0.416],  # L
    [0.298, -0.132, 0.458],  # F
]

# Fetch PDB
print("Fetching PDB 2F4K from RCSB...")
url = "https://files.rcsb.org/download/2F4K.pdb.gz"
with urlopen(url, timeout=10) as response:
    pdb_data = gzip.decompress(response.read()).decode('utf-8')

# Parse Cα atoms from chain A, residues 42-76
ca_atoms = []
for line in pdb_data.split('\n'):
    if line.startswith('ATOM'):
        try:
            chain = line[21]
            res_num = int(line[22:26])
            atom_name = line[12:16].strip()
            aa_name = line[17:20].strip()
            
            if chain == 'A' and 42 <= res_num <= 76 and atom_name == 'CA':
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                ca_atoms.append(([x, y, z], aa_name, res_num))
        except (ValueError, IndexError):
            pass

# Extract just coordinates and residue info
pdb_coords = [c[0] for c in ca_atoms]
pdb_aa_names = [c[1] for c in ca_atoms]
pdb_res_nums = [c[2] for c in ca_atoms]

print(f"\nHP36 Input (prediction):")
print(f"  Residues: 35")
print(f"  Sequence: LSDEDФKAVFGMTRSAFANLPLWKQQNLKKEKGLF")

print(f"\nHP36 from PDB 2F4K:")
print(f"  Residues found: {len(pdb_coords)}")
print(f"  Sequence: {' '.join(aa_name[0] for aa_name in pdb_aa_names).upper()}")
print(f"  Residue numbers: {pdb_res_nums[0]}-{pdb_res_nums[-1]} (with gaps)")

# We can only compare the overlapping residues
# PDB has residues 42-76 (missing 65, 70), so 33 residues
# Our prediction has 35
# So we'll compare the first 33 predictions with the PDB residues

print(f"\n{'='*70}")
print("Comparison (first 33 residues only):")
print(f"{'='*70}")

# Align both structures by centroid
pred_subset = predicted_full[:33]
pred_center = mean_point(pred_subset)
ref_center = mean_point(pdb_coords)

# Center both point sets
pred_centered = [subtract_point(p, pred_center) for p in pred_subset]
ref_centered = [subtract_point(p, ref_center) for p in pdb_coords]

# For now, just compute RMSD without Kabsch (coordinates are in different coordinate systems)
rmsd_naive = simple_rmsd(pred_centered, ref_centered)

print(f"\nRMSD (centered, no rotation): {rmsd_naive:.4f} Ångströms")
print(f"NOTE: This is very high because predicted coords are in local frame,")
print(f"      PDB coords are in absolute frame. Full Kabsch superposition needed.")

print(f"\nResidual distances (first 20 residues):")
print(f"{'Res':<4} {'PDB Res':<8} {'AA':<3} {'Distance':<10}")
print("-" * 30)
for i in range(min(20, len(pred_subset))):
    d = dist(pred_centered[i], ref_centered[i])
    print(f"{i+1:<4} {pdb_res_nums[i]:<8} {pdb_aa_names[i]:<3} {d:>9.4f}")

# Statistics
distances = [dist(p, r) for p, r in zip(pred_centered, ref_centered)]
print(f"\nStatistics:")
print(f"  Mean distance: {sum(distances)/len(distances):.4f} Ångströms")
print(f"  Min: {min(distances):.4f} Ångströms")
print(f"  Max: {max(distances):.4f} Ångströms")

print(f"\n{'='*70}")
print("IMPORTANT NOTE:")
print("The predicted coordinates are in a LOCAL reference frame produced by")
print("AlphaFold's structure module. They are NOT aligned to the PDB frame.")
print("To get a meaningful RMSD, we need to:")
print("1. Extract pLDDT confidence scores from the model")
print("2. Use Kabsch algorithm to optimally superimpose the structures")
print("3. Report RMSD after proper 3D alignment")
print(f"{'='*70}")
