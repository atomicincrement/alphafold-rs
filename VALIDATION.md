# Validation Against Reference AlphaFold2

This repository includes the official [DeepMind AlphaFold2](https://github.com/deepmind/alphafold) implementation as a git submodule for validation purposes.

## Setup

### 1. Initialize the submodule
```bash
git submodule update --init --recursive
```

### 2. Install reference AlphaFold2 dependencies

The reference implementation requires several heavy dependencies (TensorFlow, JAX, etc.). We recommend using one of these approaches:

**Option A: LocalColabFold (Easiest)**
```bash
pip install localcolabfold
```
This provides a lightweight environment with pre-compiled dependencies.

**Option B: Full installation**
```bash
pip install dm-tree tensorflow jax dm-haiku absl-py
cd alphafold-reference
pip install -e .
```

## Comparing Outputs

### Step 1: Generate from our Rust implementation

Save intermediate Evoformer outputs:
```bash
./target/debug/alphafold-rs generate sequence.fasta rust_coords.json --save-evoformer rust_evoformer.json
```

This creates:
- `rust_coords.json`: Predicted Cα coordinates
- `rust_evoformer.json`: Evoformer intermediate statistics (shape, mean, std, min, max)

### Step 2: Generate from reference implementation

```bash
python alphafold-reference/run_alphafold.py \
    --fasta_paths=sequence.fasta \
    --output_dir=reference_output/ \
    --model_preset=monomer \
    --max_template_date=2024-01-01
```

This creates structure predictions in `reference_output/`

### Step 3: Compare results

Use the validation script:
```bash
python validate.py
```

This will:
1. Extract coordinates from both outputs
2. Calculate RMSD and per-residue distances
3. Compare Evoformer intermediate representations
4. Generate a comparison report

## What to Compare

### 1. **Cα Coordinates** (Direct output)
- Expected: RMSD < 1.0 Å for small structures
- Our implementation handles coordinate scaling (27× for physical Ångströms)
- The reference may use different coordinate systems (local vs. physical)

### 2. **Evoformer Outputs** (Intermediate)
- `single` representation shape: [L, 384]
- `pair` representation shape: [L, L, 128]
- Check that statistics (mean, std) are reasonably similar
- Small differences are expected due to:
  - Implementation details (numerical precision)
  - Different libraries (TensorFlow vs. manual numpy operations)
  - Random initialization in some layers

### 3. **Recycling Consistency**
- Our implementation does 3 recycles (as per AlphaFold2 defaults)
- Reference should do same by default
- Monitor how Evoformer outputs change across recycles

## Expected Differences

1. **Coordinate precision**: Floating-point differences accumulate through deep networks. RMSD < 0.5 Å is excellent agreement.

2. **Feature statistics**: The mean/std of Evoformer outputs may differ slightly due to:
   - Layer ordering variations
   - Batch normalization implementation details
   - Numerical libraries (NumPy vs. TensorFlow vs. JAX)

3. **Reference structures**: When comparing to experimental (PDB) coordinates:
   - AlphaFold should predict similar backbone geometry
   - Coordinate transformation/centering may differ
   - Use alignment tools (ProFit, PyMOL) for structural comparison

## Validation Checklist

- [ ] Both implementations use same FASTA input
- [ ] Both run 3 recycles of Evoformer
- [ ] Both use monomer model preset
- [ ] Coordinate scaling is understood
- [ ] Evoformer output shapes match
- [ ] Statistics (mean/std) are within ~10%
- [ ] Cα RMSD is < 1.0 Å
- [ ] Structure looks reasonable visually

## Files

- `validate.py`: Main validation script
- `alphafold-reference/`: Official AlphaFold2 (git submodule)
- `sequence.fasta`: Test sequence (HP36 villin)
- `coords.json`: Reference coordinates from PDB 2F4K

## Example Output

```
AlphaFold-rs Validation Results
================================
Sequence: LSDEDФKAVFGMTRSAFANLPLWKQQNLKKEKGLF (35 residues)

Evoformer Outputs:
  Single: [35, 384] - mean=0.042, std=0.89
  Pair:   [35, 35, 128] - mean=0.15, std=0.42

Coordinate Comparison (vs reference):
  RMSD: 0.32 Å
  MAE:  0.24 Å
  95% within 0.5 Å
```
