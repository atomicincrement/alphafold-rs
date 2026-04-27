#!/usr/bin/env python3
"""Inspect PDB 2F4K to understand the structure and residue numbering."""

from urllib.request import urlopen
import gzip

print("Fetching PDB 2F4K...")
url = "https://files.rcsb.org/download/2F4K.pdb.gz"
with urlopen(url, timeout=10) as response:
    pdb_data = gzip.decompress(response.read()).decode('utf-8')

# Find all CA atoms in chain A
ca_residues = []
for line in pdb_data.split('\n'):
    if line.startswith('ATOM'):
        try:
            chain = line[21]
            res_num = int(line[22:26])
            atom_name = line[12:16].strip()
            if chain == 'A' and atom_name == 'CA':
                aa_name = line[17:20].strip()
                ca_residues.append((res_num, aa_name))
        except (ValueError, IndexError):
            pass

print(f"\nChain A has {len(ca_residues)} Cα atoms:")
print("Res#  AA")
for res_num, aa_name in ca_residues:
    print(f"{res_num:4d}  {aa_name}")

# The HP36 fragment should be residues 42-76 according to the plan
print(f"\nLooking for HP36 (residues 42-76):")
hp36_residues = [(num, aa) for num, aa in ca_residues if 42 <= num <= 76]
print(f"Found {len(hp36_residues)} residues in range 42-76")
print("Res#  AA")
for res_num, aa_name in hp36_residues:
    print(f"{res_num:4d}  {aa_name}")

# Check if there are any gaps
if len(hp36_residues) > 0:
    min_res = min(r[0] for r in hp36_residues)
    max_res = max(r[0] for r in hp36_residues)
    print(f"\nRange: {min_res}-{max_res}, Count: {len(hp36_residues)}")
    expected_count = max_res - min_res + 1
    print(f"Expected count: {expected_count}, Actual count: {len(hp36_residues)}")
    if len(hp36_residues) < expected_count:
        print("WARNING: Missing residues detected!")
