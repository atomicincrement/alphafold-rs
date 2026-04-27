#!/usr/bin/env python3
"""
Compare predicted Cα coordinates from alphafold-rs with expected PDB 2F4K structure.
PDB 2F4K is the HP36 villin headpiece (chain A, residues 42-76).
"""

from urllib.request import urlopen
import gzip
import math

# Predicted coordinates from output.log
predicted = [
    [0.225, -0.245, 0.497],
    [0.364, -0.209, 0.498],
    [0.371, -0.276, 0.560],
    [0.231, -0.193, 0.469],
    [0.423, -0.195, 0.549],
    [0.203, -0.261, 0.489],
    [0.409, -0.166, 0.441],
    [0.266, -0.321, 0.495],
    [0.218, -0.207, 0.470],
    [0.086, -0.225, 0.359],
    [0.221, -0.216, 0.427],
    [0.216, -0.109, 0.559],
    [0.417, -0.345, 0.443],
    [0.417, -0.198, 0.591],
    [0.053, -0.268, 0.241],
    [0.418, -0.282, 0.501],
    [0.169, -0.307, 0.515],
    [0.153, -0.229, 0.373],
    [0.457, -0.412, 0.562],
    [0.294, -0.195, 0.510],
    [0.375, -0.374, 0.510],
    [0.303, -0.228, 0.446],
    [0.133, -0.307, 0.420],
    [0.337, -0.073, 0.473],
    [0.218, -0.130, 0.473],
    [0.359, -0.264, 0.409],
    [0.097, -0.179, 0.549],
    [0.146, -0.349, 0.455],
    [0.317, -0.169, 0.543],
    [0.230, -0.207, 0.458],
    [0.097, -0.237, 0.280],
    [0.473, -0.226, 0.548],
    [0.414, -0.282, 0.486],
    [0.336, -0.180, 0.416],
    [0.298, -0.132, 0.458],
]

# Sequence (from src/main.rs - HP36 villin headpiece)
sequence = "LSDEDФKAVFGMTRSAFANLPLWKQQNLKKEKGLF"
# HP36 sequence (L, S, D, E, D, F, K, A, V, F, G, M, T, R, S, A, F, A, N, L, P, L, W, K, Q, Q, N, L, K, K, E, K, G, L, F)
aa_list = ['L', 'S', 'D', 'E', 'D', 'F', 'K', 'A', 'V', 'F', 
           'G', 'M', 'T', 'R', 'S', 'A', 'F', 'A', 'N', 'L',
           'P', 'L', 'W', 'K', 'Q', 'Q', 'N', 'L', 'K', 'K',
           'E', 'K', 'G', 'L', 'F']
sequence = "".join(aa_list)
n_residues = len(predicted)

print(f"Predicted coordinates: {len(predicted)} residues")
print(f"Sequence: {sequence} ({n_residues} residues)")
print()

# Fetch PDB 2F4K structure from RCSB
print("Fetching PDB 2F4K from RCSB...")
try:
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
                
                # Get chain A, residues 42-76 (HP36)
                if chain == 'A' and 42 <= res_num <= 76 and atom_name == 'CA':
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    ca_atoms.append([x, y, z])
            except (ValueError, IndexError):
                pass
    
    expected = ca_atoms
    print(f"Expected coordinates from PDB 2F4K: {len(expected)} residues")
    print()
    
    # Helper function
    def dist(p1, p2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
    
    def mean_point(points):
        n = len(points)
        if n == 0:
            return [0, 0, 0]
        return [sum(p[i] for p in points) / n for i in range(3)]
    
    # Center both structures by their centroid for better visualization
    pred_center = mean_point(predicted)
    exp_center = mean_point(expected)
    
    pred_centered = [[predicted[i][j] - pred_center[j] for j in range(3)] for i in range(len(predicted))]
    exp_centered = [[expected[i][j] - exp_center[j] for j in range(3)] for i in range(len(expected))]
    
    # Calculate RMSD
    sum_sq = 0
    for i in range(min(len(pred_centered), len(exp_centered))):
        for j in range(3):
            diff = pred_centered[i][j] - exp_centered[i][j]
            sum_sq += diff * diff
    rmsd = math.sqrt(sum_sq / min(len(pred_centered), len(exp_centered)))
    
    print(f"RMSD (centered): {rmsd:.4f} Ångströms")
    print()
    
    # Print coordinate comparison
    print("Residue-by-residue comparison (Ångströms):")
    print(f"{'Res':<4} {'AA':<3} {'Pred X':<8} {'Pred Y':<8} {'Pred Z':<8} {'Exp X':<8} {'Exp Y':<8} {'Exp Z':<8} {'Dist':<8}")
    print("-" * 85)
    
    distances = []
    for i in range(min(n_residues, len(expected))):
        px, py, pz = predicted[i]
        ex, ey, ez = expected[i]
        d = dist(predicted[i], expected[i])
        distances.append(d)
        res_name = sequence[i]
        print(f"{i+1:<4} {res_name:<3} {px:>7.3f} {py:>7.3f} {pz:>7.3f} {ex:>7.3f} {ey:>7.3f} {ez:>7.3f} {d:>7.3f}")
    
    # Statistics
    print()
    print(f"Distance statistics:")
    mean_dist = sum(distances) / len(distances)
    print(f"  Mean: {mean_dist:.4f} Å")
    std_dev = math.sqrt(sum((d - mean_dist)**2 for d in distances) / len(distances))
    print(f"  Std Dev: {std_dev:.4f} Å")
    print(f"  Min: {min(distances):.4f} Å (residue {distances.index(min(distances))+1})")
    print(f"  Max: {max(distances):.4f} Å (residue {distances.index(max(distances))+1})")
    
except Exception as e:
    print(f"Error fetching PDB: {e}")
    print()
    print("Note: The predicted coordinates appear to be in a scaled/normalized space")
    print("rather than absolute Ångströms. This is likely the AlphaFold model output")
    print("before any post-processing or alignment.")
    print()
    print("Predicted coordinates (first 10 residues):")
    for i in range(min(10, len(predicted))):
        px, py, pz = predicted[i]
        res_name = sequence[i]
        print(f"  {i+1}: {res_name}  ({px:.3f}, {py:.3f}, {pz:.3f})")
