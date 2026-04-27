# Verification Steps

## 1. Build Release Binary
```bash
cargo build --release
```

## 2. Generate Rust Implementation Output
Save Evoformer intermediate outputs for comparison:
```bash
./target/release/alphafold-rs generate sequence.fasta rust_coords.json \
    --save-evoformer rust_evoformer.json
```

Outputs:
- `rust_coords.json`: Predicted Cα coordinates from Rust implementation
- `rust_evoformer.json`: Evoformer statistics (shape, mean, std, min, max)

## 3. Install Reference AlphaFold2 Dependencies
Choose one approach:

**Option A: LocalColabFold (Recommended)**
```bash
pip install localcolabfold
```

**Option B: Full installation**
```bash
pip install dm-tree tensorflow jax dm-haiku absl-py
cd alphafold-reference
pip install -e .
```

## 4. Generate Reference Implementation Output
```bash
python alphafold-reference/run_alphafold.py \
    --fasta_paths=sequence.fasta \
    --output_dir=reference_output/ \
    --model_preset=monomer \
    --max_template_date=2024-01-01
```

## 5. Extract Reference Coordinates
Extract Cα coordinates from reference output (typically in `reference_output/ranked_0.pdb`):
```bash
grep "^ATOM.*CA " reference_output/ranked_0.pdb | awk '{print $7, $8, $9}' > reference_coords.json
```

Or use the `validate.py` script to extract them programmatically.

## 6. Compare Results

Use the validation script for structure comparison:
```bash
python compare_outputs.py
```

This will:
- Extract CA coordinates from experimental structure (PDB 2F4K)
- Align predicted vs experimental using Kabsch algorithm
- Calculate RMSD, per-residue distances, backbone geometry
- Generate validation report

## 7. Status and Findings

**Current Issue Identified:**
- Predicted structures deviate significantly from experimental (RMSD ~9 Å)
- Backbone distances are overstretched: predicted 5.05 Å vs expected 3.91 Å
- **Root cause**: Structure module coordinate generation (not scaling)
- Next: Debug structure_module.rs rigid frame prediction and coordinate transformation

## Expected Results

### Evoformer Outputs
- `single` shape: [L, 384]
- `pair` shape: [L, L, 128]
- Statistics should be within ~10% between implementations

### Coordinate Comparison
- RMSD: < 1.0 Å (good agreement)
- RMSD < 0.5 Å: excellent agreement
- 95% of residues within 0.5 Å: expected

### Visualization
```bash
./target/release/alphafold-rs visualise rust_coords.json reference_coords.json
```

Displays:
- Predicted structure (rainbow colored)
- Reference structure (grey, semi-transparent)
- Can rotate with mouse, zoom with scroll wheel

## Troubleshooting

**Reference binary missing/old:**
```bash
rm -rf alphafold-reference
git submodule update --init --recursive
```

**Dependencies conflicts:**
```bash
pip uninstall tensorflow jax && pip install localcolabfold
```

**Memory issues during reference run:**
Add `--model_preset=monomer_ptm` (uses more memory) or use a shorter sequence for testing.
