# Create a rust version of Alphafold as a learning excercise.

See

https://github.com/google-deepmind/alphafold

The goal here is not to replicate Alphafold but to create a dependency-free demonstrator
of the technology that can be understood by the lay person such as myself.

We will translate just enough of the project from Python only using ndarray for linear algebra.

## Steps

### 1. Fetch model data

AlphaFold2 weights are large (~100 GB for the full model). We will use the
AlphaFold-Multimer "model_1" params which are available from the EBI at a more
manageable size, or alternatively use a distilled/reduced checkpoint if one
exists in the community (e.g. on Hugging Face).

- [ ] Research whether a small/distilled AlphaFold checkpoint exists that is
      practical to download for a demo (target: < 500 MB).
- [ ] Add `reqwest` (blocking, TLS) and `indicatif` to `Cargo.toml` for
      streaming downloads with a progress bar.
- [ ] Write a `fetch` module (`src/fetch.rs`) that:
  - Checks for a local cache in `~/.cache/alphafold-rs/` before downloading.
  - Streams the file to disk in chunks so memory usage stays flat.
  - Verifies the SHA-256 hash of the downloaded file against a known good value.
- [ ] Expose a CLI flag `--model-path` so a user can supply a local file and
      skip the download entirely.

> Fallback: if no suitably small pre-trained checkpoint exists, jump straight
> to Step 4 (train on a single PDB) and return here once a checkpoint can be
> exported from that training run.

---

### 2. Decode model data into ndarray matrices

AlphaFold checkpoints are stored as NumPy `.npz` archives (zip files containing
`.npy` arrays) or as JAX/Haiku parameter trees serialised to MessagePack.

- [ ] Identify the exact serialisation format of the chosen checkpoint.
- [ ] Add `ndarray` and either `npy` or `rmp-serde` / `zip` to `Cargo.toml`
      depending on format.
- [ ] Write a `params` module (`src/params.rs`) that:
  - Opens the archive and iterates over named arrays.
  - Deserialises each array header (dtype, shape, fortran-order flag) per the
    `.npy` v1.0 spec.
  - Returns a `HashMap<String, ndarray::ArrayD<f32>>` of all parameter tensors.
- [ ] Print a summary table of tensor names, shapes and sizes so we can verify
      the load is correct.
- [ ] Write a unit test that round-trips a small synthetic `.npy` file through
      the loader and checks values.

---

### 3. Forward inference — predict a known structure

AlphaFold's full pipeline is complex (MSA, Evoformer, Structure Module). For
the demonstrator we will implement a stripped-down version that is still
scientifically meaningful.

- [ ] **Input pipeline** (`src/input.rs`):
  - Parse a single-sequence FASTA file (no MSA required — use a single-sequence
    "mock MSA" as AlphaFold supports this).
  - One-hot encode the amino-acid sequence (20 standard residues + gap).
  - Build the pair representation initialisation (relative position encoding).

- [ ] **Evoformer lite** (`src/evoformer.rs`):
  - Implement a single Evoformer block: row-wise gated self-attention over the
    pair representation, transition feed-forward, outer-product mean update.
  - Use `ndarray`'s `dot` / `einsum`-style loops — no GPU, no autograd.
  - Load the corresponding weight tensors from the params map by name.

- [ ] **Structure Module** (`src/structure_module.rs`):
  - Implement the invariant point attention (IPA) layer and backbone update
    (applying rigid-body frames to predict Cα positions).
  - Output a `Vec<[f32; 3]>` of predicted Cα coordinates per residue.

- [ ] **Validation**:
  - Choose a short, well-known peptide (e.g. the 13-residue villin headpiece
    fragment, PDB: 2F4K) as the reference.
  - Compute the GDT-TS or RMSD between predicted and experimental Cα positions
    after optimal superposition.
  - Print the RMSD to stdout so progress can be tracked without a visualiser.

---

### 4. 3D visualiser using Bevy

- [ ] Add `bevy` (default features or a minimal subset: `render`, `pbr`,
      `winit`) to `Cargo.toml` behind a feature flag `--features visualise` so
      the core inference binary stays lightweight.
- [ ] Write a `visualise` module (`src/visualise.rs`) that:
  - Spawns a Bevy app with an orbit camera (mouse drag to rotate, scroll to
    zoom).
  - Renders each Cα atom as a small sphere mesh.
  - Connects adjacent Cα atoms with cylinder "bond" meshes to form the backbone
    chain.
  - Colours residues by a rainbow gradient (N-terminus blue → C-terminus red).
- [ ] Overlay the experimental structure (loaded from the PDB file) as a
      semi-transparent grey chain for visual comparison.
- [ ] Add a simple UI panel (using `bevy_egui` or Bevy's built-in text) showing
      the sequence, RMSD, and per-residue confidence (pLDDT proxy).

---

### Fallback: train on a single PDB file

If no suitable pre-trained checkpoint can be sourced:

- [ ] Parse the target PDB file (`src/pdb.rs`) to extract the amino-acid
      sequence and the ground-truth Cα coordinates.
- [ ] Implement a minimal supervised training loop in `src/train.rs`:
  - Random initialise all Evoformer + Structure Module weights.
  - Mean-squared-error loss on predicted vs. true Cα positions.
  - SGD or Adam optimiser implemented from scratch with `ndarray`.
  - Save a checkpoint to `.npz` after every 100 steps.
- [ ] Run until RMSD < 2 Å on the training example (overfitting is acceptable —
      the goal is to verify the forward pass is correct).
- [ ] Once a checkpoint is saved, re-enter Step 2 to load it and proceed with
      the visualiser.

