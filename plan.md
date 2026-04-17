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

- [x] Research whether a small/distilled AlphaFold checkpoint exists that is
      practical to download for a demo (target: < 500 MB).
      **Finding:** No sub-500 MB AlphaFold checkpoint exists publicly. ESMFold
      (Meta, ~690 MB) uses a different architecture. The smallest official
      AlphaFold option is the EBI model_1_ptm params (~3.5 GB). The `--model-path`
      flag is therefore the primary entry point for running the demo; the
      downloader defaults to the EBI URL but warns about file size.
- [x] Add `reqwest` (blocking, TLS) and `indicatif` to `Cargo.toml` for
      streaming downloads with a progress bar.
- [x] Write a `fetch` module (`src/fetch.rs`) that:
  - Checks for a local cache in `~/.cache/alphafold-rs/` before downloading.
  - Streams the file to disk in chunks so memory usage stays flat.
  - Verifies the SHA-256 hash of the downloaded file against a known good value.
- [x] Expose a CLI flag `--model-path` so a user can supply a local file and
      skip the download entirely.

> Fallback: if no suitably small pre-trained checkpoint exists, jump straight
> to Step 4 (train on a single PDB) and return here once a checkpoint can be
> exported from that training run.

---

### 2. Decode model data into ndarray matrices

AlphaFold checkpoints are stored as NumPy `.npz` archives (zip files containing
`.npy` arrays) or as JAX/Haiku parameter trees serialised to MessagePack.

- [x] Identify the exact serialisation format of the chosen checkpoint.
      **Finding:** The EBI colab params tarball is a `.tar` of `.npz` files;
      each `.npz` is a zip archive of named `.npy` arrays (v1 or v2 format).
- [x] Add `ndarray`, `zip`, and `tar` to `Cargo.toml`.
- [x] Write a `params` module (`src/params.rs`) that:
  - Opens the archive and iterates over named arrays.
  - Deserialises each array header (dtype, shape, fortran-order flag) per the
    `.npy` v1.0/v2.0 spec; supports f16, f32, f64, i16, i32, i64, u8, u32, u64.
  - Returns a `HashMap<String, ndarray::ArrayD<f32>>` of all parameter tensors.
- [x] Print a summary table of tensor names, shapes and sizes so we can verify
      the load is correct.
- [x] Write a unit test that round-trips a small synthetic `.npy` file through
      the loader and checks values (5 tests: 1-D, 2-D, dtype cast, f16, Fortran order).

---

### 3. Forward inference — predict a known structure

The real params are loaded and fully understood. All 455 tensors sit under the
prefix `alphafold/alphafold_iteration/`. The concrete dimensions are:

| Representation | Dim | Source tensor |
|---|---|---|
| MSA / single (Evoformer) | 256 | `evoformer/preprocess_1d//weights [21, 256]` |
| Pair | 128 | `evoformer/pair_activiations//weights [65, 128]` |
| Extra-MSA | 64 | `evoformer/extra_msa_activations//weights [25, 64]` |
| Single (Structure Module) | 384 | `evoformer/single_activations//weights [256, 384]` |
| Evoformer blocks (stacked) | 48 | first axis of all `evoformer_iteration/…` tensors |
| Extra-MSA blocks (stacked) | 4 | first axis of all `extra_msa_stack/…` tensors |
| IPA heads | 12 | `fold_iteration/invariant_point_attention/q_scalar//weights [384, 192]` (192=12×16) |
| IPA scalar head dim | 16 | (from above) |
| IPA point heads (q/k) | 12 | `q_point_local//weights [384, 144]` (144=12×4×3) |
| IPA point heads (v) | 24 | `v_point_projection//weights [384, 12, 24]` |
| Relative position buckets | 73 | `~_relative_encoding/position_activations//weights [73, 128]` |
| Recycling bins (Cα dist) | 15 | `prev_pos_linear//weights [15, 128]` |

- [x] **Input pipeline** (`src/input.rs`):
  - Parse a single-sequence FASTA file. A "single-sequence mock MSA" means the
    MSA has depth 1 and the extra-MSA is also a single-row mock.
  - Amino-acid alphabet: 21 tokens (20 standard + unknown/gap).
  - Embed the sequence:
    - Single: `preprocess_1d//weights [21, 256]` + bias → shape `[L, 256]`
    - Pair init from left/right residue identity:
      `left_single//weights [21, 128]` + `right_single//weights [21, 128]`
      → outer sum → `[L, L, 128]`
    - Add relative positional encoding:
      `~_relative_encoding/position_activations//weights [73, 128]`:
      bucket `clip(i−j + 32, 0, 64)` → one-hot 73 → linear → `[L, L, 128]`
    - Add `pair_activiations//weights [65, 128]` (unit-position from `[L, L, 65]`
      features — set to zeros for single-sequence mode).

- [ ] **Evoformer** (`src/evoformer.rs`) — run all 48 blocks using the stacked
  weights (slice `weights[b, ..]` for block `b`):
  - **MSA row attention with pair bias** — gated multi-head attention over
    residue axis; pair repr adds a learned bias via
    `msa_row_attention_with_pair_bias//feat_2d_weights [48, 128, 8]`.
    Weights: `query_w [48, 256, 8, 32]`, `key_w`, `value_w`, `gating_w/b`,
    `output_w [48, 8, 32, 256]`, `output_b`. LayerNorm: `query_norm//scale/offset`.
  - **MSA column attention** — same weight shape, attends over MSA depth axis.
    In single-sequence mode this is a no-op (depth=1).
  - **MSA transition** — two-layer MLP, hidden 1024:
    `transition1//weights [48, 256, 1024]`, `transition2//weights [48, 1024, 256]`.
  - **Outer-product mean** — outer product of MSA rows projected to pair repr:
    `left_projection//weights [48, 256, 32]`, `right_projection [48, 256, 32]`,
    `output_w [48, 32, 32, 128]`. In single-seq mode the mean is just the one row.
  - **Triangle multiplication outgoing/incoming** — each has
    `left_projection/right_projection/gate/output_projection` weights, all
    shape `[48, 128, 128]`, plus `projection [48, 128, 256]` for gating.
  - **Triangle attention starting/ending** — gated attention on pair with
    `query_w [48, 128, 4, 32]`, `feat_2d_weights [48, 128, 4]`.
  - **Pair transition** — hidden 512:
    `transition1//weights [48, 128, 512]`, `transition2//weights [48, 512, 128]`.
  - After 48 blocks project single repr: `single_activations//weights [256, 384]`
    → `[L, 384]` for the Structure Module.

- [ ] **Recycling** (3 passes, reusing same weights):
  - Normalise previous single/pair: `prev_msa_first_row_norm` (γ/β [256]),
    `prev_pair_norm` (γ/β [128]).
  - Encode previous Cα distances into pair repr via
    `prev_pos_linear//weights [15, 128]` (15 Å-bins, one-hot encoded).
  - Add these to the embeddings at the start of each recycle pass.

- [ ] **Structure Module** (`src/structure_module.rs`) — 8 shared-weight
  iterations using `fold_iteration/…`:
  - Normalise inputs: `single_layer_norm` (γ/β [384]), `pair_layer_norm` (γ/β [128]).
  - Project single: `initial_projection//weights [384, 384]` (applied once before
    iterations).
  - **IPA** per iteration:
    - Scalar q/k/v: `q_scalar//weights [384, 192]` (12 heads × 16 dim),
      `kv_scalar//weights [384, 384]`, `kv_scalar//bias [384]`.
    - Point q: `q_point_local//weights [384, 144]` (12 heads × 4 points × 3 xyz)
      applied in local frame then transformed to global by current backbone frame.
    - Point k/v: `kv_point_local//weights [384, 432]` (432=12×(4+8)×3).
    - Pair bias: `attention_2d//weights [128, 12]`.
    - Concat [scalar_out, point_out, pair_out] → `output_projection [2112, 384]`.
    - Trainable softplus point weights: `trainable_point_weights [12]`.
  - **Backbone update**:
    `affine_update//weights [384, 6]` predicts 6-vector (3 axis-angle + 3
    translation) that is composed with the current rigid frame via quaternion
    multiplication. `quat_rigid/rigid//weights [384, 6]` is the equivalent in
    the quaternion parameterisation (use whichever is loaded; both are present).
  - **Sidechain / angle prediction** (needed to get full atom positions, but
    Cα can be extracted directly from the backbone frame translation):
    `rigid_sidechain/unnormalized_angles//weights [128, 14]` — 14 = 7 torsion
    angles × (sin, cos). For our Cα-only RMSD metric we can skip sidechain.
  - After 8 iterations extract Cα from each frame's translation vector →
    `Vec<[f32; 3]>`.

- [ ] **pLDDT confidence head** (optional but cheap):
  - `predicted_lddt_head/act_0//weights [384, 128]`,
    `act_1//weights [128, 128]`, `logits//weights [128, 50]` → softmax over 50
    bins, expected value = per-residue pLDDT. Print alongside RMSD.

- [ ] **Validation**:
  - Target: villin headpiece fragment HP36 (PDB: 2F4K, chain A, residues 42–76,
    36 residues). FASTA: `LSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF`.
  - Download the PDB, extract experimental Cα coordinates.
  - After inference compute RMSD with Kabsch optimal superposition (pure ndarray,
    SVD via power-iteration or a simple 3×3 closed-form solver).
  - Target: RMSD < 3 Å is a meaningful result for a 36-residue helix bundle.
  - Print: `RMSD = X.XX Å  mean-pLDDT = YY.Y`.


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

