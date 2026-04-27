#!/usr/bin/env python3
"""
Extract Cα coordinates from a PDB file and save as JSON.
Downloads PDB 2F4K (HP36 villin headpiece) from RCSB if not present.
"""

import json
import sys
import urllib.request
from pathlib import Path


def download_pdb(pdb_id: str, output_file: str = None) -> str:
    """Download PDB file from RCSB."""
    if output_file is None:
        output_file = f"{pdb_id.lower()}.pdb"
    
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"Downloading PDB {pdb_id} from {url}...")
    
    try:
        urllib.request.urlretrieve(url, output_file)
        print(f"Saved to {output_file}")
        return output_file
    except Exception as e:
        print(f"Error downloading PDB: {e}")
        sys.exit(1)


def extract_ca_coords(pdb_file: str) -> list[list[float]]:
    """Extract Cα coordinates from PDB file."""
    coords = []
    
    with open(pdb_file, 'r') as f:
        for line in f:
            # PDB format: ATOM records
            if not line.startswith("ATOM  "):
                continue
            
            # Check if this is a Cα atom
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            
            # Extract coordinates (columns 30-38, 38-46, 46-54)
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
            except ValueError:
                print(f"Warning: Could not parse coordinates from line: {line}")
                continue
    
    return coords


def main():
    pdb_id = "2F4K"  # HP36 villin headpiece
    pdb_file = f"{pdb_id.lower()}.pdb"
    json_file = "coords.json"
    
    # Download if not present
    if not Path(pdb_file).exists():
        download_pdb(pdb_id, pdb_file)
    else:
        print(f"Using existing {pdb_file}")
    
    # Extract coordinates
    print(f"Extracting Cα coordinates from {pdb_file}...")
    coords = extract_ca_coords(pdb_file)
    
    if not coords:
        print("Error: No Cα coordinates found!")
        sys.exit(1)
    
    print(f"Found {len(coords)} residues")
    print(f"First residue: {coords[0]}")
    print(f"Last residue: {coords[-1]}")
    
    # Save to JSON
    with open(json_file, 'w') as f:
        json.dump(coords, f, indent=2)
    
    print(f"\nSaved {len(coords)} coordinates to {json_file}")
    print(f"\nUsage: ./target/debug/alphafold-rs --load-coords {json_file} --fasta <sequence.fasta>")


if __name__ == "__main__":
    main()
