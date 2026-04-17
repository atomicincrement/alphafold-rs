//! Evoformer stack: 48 blocks transforming MSA and pair representations.
//!
//! For single-sequence inference (MSA depth = 1) we skip MSA column attention,
//! which is a no-op when there is only one sequence.
//!
//! Architecture (per block b = 0..48):
//!   MSA:  row-attention-with-pair-bias → transition (MLP)
//!   Pair: outer-product-mean (from MSA first row) →
//!         triangle-multiplication-outgoing → triangle-multiplication-incoming →
//!         triangle-attention-starting → triangle-attention-ending →
//!         pair-transition (MLP)
//!
//! After 48 blocks the MSA first row is projected [256→384] to produce the
//! single representation fed to the structure module.

use anyhow::{anyhow, Result};
use ndarray::{s, Array1, Array2, Array3};
use std::collections::HashMap;

use crate::input::Inputs;
use crate::params::Tensor;

// ── tensor-key prefix ────────────────────────────────────────────────────────
const PFX: &str = "alphafold/alphafold_iteration/evoformer/";
const EVO: &str =
    "alphafold/alphafold_iteration/evoformer/evoformer_iteration/";
const XMSA: &str = "alphafold/alphafold_iteration/evoformer/extra_msa_stack/";

// ── output ───────────────────────────────────────────────────────────────────

/// The result of running the Evoformer neural network on a protein sequence.
///
/// # Fields
///
/// ## `single` — \[L, 384\]
/// A table with one row per amino acid and 384 numbers per row. Think of it as
/// a rich "character description" of each residue after the network has
/// considered the full sequence context. Each row summarises what that residue
/// "knows" about its neighbours, its chemical environment, and its likely role
/// in the folded structure. This feeds directly into the structure module to
/// predict backbone angles and positions.
///
/// Example: for HP36 (L=35), this is a 35×384 matrix. Row 0 describes leucine
/// at position 1, row 1 describes serine at position 2, and so on.
///
/// ## `pair` — \[L, L, 128\]
/// A square table — one cell for every pair of residues (i, j) — with 128
/// numbers describing the *relationship* between residue i and residue j. It
/// encodes things like "how likely are positions 5 and 22 to be in contact?"
/// or "what distance/orientation is expected between these two?". This is what
/// the triangle attention and triangle multiplication layers refine through all
/// 48 blocks.
///
/// Example: for L=35 this is 35×35×128 ≈ 156,800 numbers. The cell at
/// \[4, 21, :\] holds the 128-dimensional relationship vector between the 5th
/// and 22nd residues.
///
/// ## `msa_first_row` — \[L, 256\]
/// The final state of the MSA (Multiple Sequence Alignment) representation for
/// the query sequence itself (depth-1 row), before it was projected to 384
/// dims. Kept so it can be fed back into the recycling norms on the next
/// recycle pass, or used as an intermediate by downstream modules. At 256 dims
/// it is cheaper to store than `single` and retains the "pre-projection"
/// signal.
///
/// # Big picture
/// Imagine the network spending 48 rounds refining two whiteboards — one about
/// individual residues (`single`/`msa_first_row`) and one about pairs of
/// residues (`pair`). After 48 rounds (and 3 full recycles from scratch), these
/// whiteboards contain enough geometric information for the structure module to
/// translate them into 3-D atomic coordinates.
#[allow(dead_code)]
pub struct EvoformerOutput {
    pub single: Array2<f32>,       // [L, 384]
    pub pair: Array3<f32>,         // [L, L, 128]
    #[allow(dead_code)]
    pub msa_first_row: Array2<f32>, // [L, 256] — read by structure module
}

// ── tensor helpers ────────────────────────────────────────────────────────────

/// Extract the weight matrix for a single block from a stacked “all-blocks” tensor.
///
/// The checkpoint stores every block’s weights together in one big array shaped
/// `[num_blocks, in_dim, out_dim]`. This function picks out the slice for block
/// `b`, giving back a plain 2-D matrix `[in_dim, out_dim]` ready for
/// multiplication.
///
/// Example: the MSA-transition weights for block 5 live at
/// `weights[5, :, :]` inside a `[48, 256, 1024]` tensor; `mat2(..., 5)`
/// returns the `[256, 1024]` slice.
fn mat2(params: &HashMap<String, Tensor>, key: &str, b: usize) -> Result<Array2<f32>> {
    let t = params
        .get(key)
        .ok_or_else(|| anyhow!("missing tensor: {}", key))?;
    // shape [blocks, in_dim, out_dim]
    let s = t.data.shape();
    if s.len() != 3 {
        return Err(anyhow!("expected rank-3 for {}, got {}", key, s.len()));
    }
    Ok(t.data
        .slice(s![b, .., ..])
        .to_owned()
        .into_dimensionality::<ndarray::Ix2>()?)
}

/// Extract and flatten an attention weight tensor for block `b`.
///
/// Attention weights are stored per-block as `[blocks, in_dim, heads, head_dim]`.
/// This pulls out block `b` and merges the head and head-dimension axes so the
/// result is a plain `[in_dim, heads * head_dim]` matrix that can be used in a
/// single matrix multiply to project all heads at once.
///
/// Example: query weights live in `[48, 256, 8, 32]`; `mat_attn(..., 3)`
/// returns a `[256, 256]` matrix (8 heads × 32 dims each).
fn mat_attn(
    params: &HashMap<String, Tensor>,
    key: &str,
    b: usize,
) -> Result<Array2<f32>> {
    let t = params
        .get(key)
        .ok_or_else(|| anyhow!("missing tensor: {}", key))?;
    let s = t.data.shape();
    // [blocks, in, heads, head_dim]
    let in_dim = s[1];
    let heads = s[2];
    let head_dim = s[3];
    let slice = t.data.slice(s![b, .., .., ..]).to_owned();
    Ok(slice
        .into_shape_with_order((in_dim, heads * head_dim))?)
}

/// Extract and flatten the attention gating bias for block `b`.
///
/// Stored as `[blocks, heads, head_dim]`; returns a flat `[heads * head_dim]`
/// vector that is broadcast-added to the gating logits.
///
/// Example: a `[48, 8, 32]` tensor → a length-256 vector for block `b`.
fn bias_attn(params: &HashMap<String, Tensor>, key: &str, b: usize) -> Result<Array1<f32>> {
    let t = params
        .get(key)
        .ok_or_else(|| anyhow!("missing tensor: {}", key))?;
    let s = t.data.shape();
    let heads = s[1];
    let head_dim = s[2];
    let slice = t.data.slice(s![b, .., ..]).to_owned();
    Ok(slice.into_shape_with_order(heads * head_dim)?)
}

/// Extract and flatten the attention output projection for block `b`.
///
/// Stored as `[blocks, heads, head_dim, out_dim]`; returns
/// `[(heads * head_dim), out_dim]` so it can be applied with a single dot
/// product after concatenating all head outputs into one long vector.
///
/// Example: MSA row-attention output weights `[48, 8, 32, 256]` →
/// `[256, 256]` for block `b`.
fn mat_attn_out(
    params: &HashMap<String, Tensor>,
    key: &str,
    b: usize,
) -> Result<Array2<f32>> {
    let t = params
        .get(key)
        .ok_or_else(|| anyhow!("missing tensor: {}", key))?;
    let s = t.data.shape();
    let heads = s[1];
    let head_dim = s[2];
    let out_dim = s[3];
    let slice = t.data.slice(s![b, .., .., ..]).to_owned();
    Ok(slice.into_shape_with_order((heads * head_dim, out_dim))?)
}

/// Extract a 1-D bias or scale vector for block `b`.
///
/// Many tensors in the checkpoint are stacked as `[blocks, dim]`; this picks
/// out the `[dim]` slice for block `b`.
///
/// Example: MSA-transition layer-norm scale `[48, 256]` → a length-256 vector.
fn vec1(params: &HashMap<String, Tensor>, key: &str, b: usize) -> Result<Array1<f32>> {
    let t = params
        .get(key)
        .ok_or_else(|| anyhow!("missing tensor: {}", key))?;
    Ok(t.data.slice(s![b, ..]).to_owned().into_dimensionality::<ndarray::Ix1>()?)
}

/// Extract the pair-bias projection matrix for block `b`.
///
/// Used for `feat_2d_weights`, stored as `[blocks, pair_dim, heads]`; returns
/// the `[pair_dim, heads]` slice for block `b`. This matrix converts each
/// pair’s 128-dimensional description into a per-head scalar bias that
/// nudges attention towards or away from that pair.
///
/// Example: `[48, 128, 8]` → `[128, 8]` for block `b`.
fn mat_feat2d(
    params: &HashMap<String, Tensor>,
    key: &str,
    b: usize,
) -> Result<Array2<f32>> {
    let t = params
        .get(key)
        .ok_or_else(|| anyhow!("missing tensor: {}", key))?;
    Ok(t.data.slice(s![b, .., ..]).to_owned().into_dimensionality::<ndarray::Ix2>()?)
}

// ── math primitives ──────────────────────────────────────────────────────────

/// Squash any real number into the range (0, 1).
///
/// Used as a soft on/off switch for gating: values well below 0 approach 0
/// (gate closed), values well above 0 approach 1 (gate fully open).
/// `sigmoid(0) = 0.5`.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Pass positive values through unchanged; clamp negatives to zero.
///
/// The non-linearity used between the two linear layers of every MLP
/// (transition block). Without it, stacking linear layers would collapse to a
/// single linear map and the network could not learn non-linear patterns.
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// Normalise each row of a 2-D array independently, then rescale.
///
/// For each row: subtract its mean, divide by its standard deviation
/// (+ a tiny epsilon to avoid division by zero), multiply element-wise by
/// `scale`, and add `offset`. The learned `scale` and `offset` let the
/// network undo the normalisation if needed.
///
/// This keeps activations in a consistent numerical range across blocks,
/// which stabilises training and inference. Applied before every attention
/// and MLP sub-layer.
///
/// Example: a row `[10, 20, 30]` has mean 20 and std ≈ 8.16; after
/// normalisation it becomes roughly `[-1.22, 0.0, 1.22]`, then scaled
/// and shifted by the learned parameters.
fn layer_norm_rows(x: &Array2<f32>, scale: &Array1<f32>, offset: &Array1<f32>) -> Array2<f32> {
    let eps = 1e-5_f32;
    let mut out = x.clone();
    for mut row in out.rows_mut() {
        let n = row.len() as f32;
        let mean = row.sum() / n;
        let var = row.mapv(|v| (v - mean).powi(2)).sum() / n;
        let inv = (var + eps).sqrt().recip();
        for (i, v) in row.iter_mut().enumerate() {
            *v = (*v - mean) * inv * scale[i] + offset[i];
        }
    }
    out
}

/// Normalise each “channel” slice of a 3-D tensor along the last axis.
///
/// Identical in meaning to `layer_norm_rows` but operates on a tensor shaped
/// `[A, B, C]`, treating every `(i, j)` position as an independent row of
/// length `C`. Used whenever pair representations (shape `[L, L, C]`) need
/// normalising.
fn layer_norm_3d(x: &Array3<f32>, scale: &Array1<f32>, offset: &Array1<f32>) -> Array3<f32> {
    let (a, b, c) = x.dim();
    let eps = 1e-5_f32;
    let mut out = x.clone();
    for i in 0..a {
        for j in 0..b {
            let mut row = out.slice_mut(s![i, j, ..]);
            let n = c as f32;
            let mean = row.sum() / n;
            let var = row.mapv(|v| (v - mean).powi(2)).sum() / n;
            let inv = (var + eps).sqrt().recip();
            for k in 0..c {
                row[k] = (row[k] - mean) * inv * scale[k] + offset[k];
            }
        }
    }
    out
}

/// Apply a learned linear (fully-connected) layer to a 2-D input.
///
/// Multiplies every row of `x` by the weight matrix `w` and adds the bias
/// vector. This is the fundamental building block of every MLP and projection
/// in the network.
///
/// `x`: `[M, K]`, `w`: `[K, N]`, `bias`: `[N]` → output `[M, N]`.
///
/// Example: projecting 35 residues from 256 dims to 1024 dims:
/// `x` is `[35, 256]`, `w` is `[256, 1024]`, result is `[35, 1024]`.
fn linear(x: &Array2<f32>, w: &Array2<f32>, bias: &Array1<f32>) -> Array2<f32> {
    let mut y = x.dot(w);
    for mut row in y.rows_mut() {
        row += bias;
    }
    y
}

/// Apply a learned linear layer to a 3-D input, treating the first two axes as a batch.
///
/// Internally reshapes `[A, B, K]` to `[A*B, K]`, calls the ordinary matrix
/// multiply, then reshapes back to `[A, B, N]`. Used to apply the same
/// projection to every `(i, j)` cell of the pair representation without
/// writing explicit loops.
///
/// Example: projecting pair features `[35, 35, 128]` to `[35, 35, 512]`
/// in the pair-transition MLP.
fn linear3(x: &Array3<f32>, w: &Array2<f32>, bias: &Array1<f32>) -> Array3<f32> {
    let (a, b, k) = x.dim();
    let n = w.ncols();
    let xr = x.to_shape((a * b, k)).unwrap().to_owned();
    let yr = xr.dot(w);
    let mut y = yr.into_shape_with_order((a, b, n)).unwrap();
    for i in 0..a {
        for j in 0..b {
            let mut row = y.slice_mut(s![i, j, ..]);
            row += bias;
        }
    }
    y
}

/// Let each residue gather information from all other residues along the sequence,
/// guided by what the pair representation already knows about their relationship.
///
/// This is multi-head self-attention applied to a single row of the MSA (one
/// sequence). The key addition over plain self-attention is a *pair bias*:
/// before computing which positions attend to which, the pair table is projected
/// to a per-head scalar and added to the attention logits, so the network can
/// say “residues 3 and 17 are probably in contact — pay extra attention to that
/// pair.”
///
/// Steps:
/// 1. Layer-norm the MSA row and the pair table.
/// 2. Project each position into query/key/value/gating vectors (one set per head).
/// 3. Compute attention scores (`Q · Kᵀ / √D`), add pair bias, softmax.
/// 4. Weighted sum of values; multiply by a sigmoid gate.
/// 5. Project back to MSA dimension and add as a residual.
///
/// Result: updated MSA, same shape `[S, L, C_m]`, where each position now
/// “knows” what its neighbours look like.
fn msa_row_attention(
    msa: &Array3<f32>,   // [S, L, 256]
    pair: &Array3<f32>,  // [L, L, 128]
    params: &HashMap<String, Tensor>,
    pfx: &str,           // points to "msa_row_attention_with_pair_bias/"
    b: usize,
) -> Result<Array3<f32>> {
    let (_s, l, _cm) = msa.dim();

    let scale_key = format!("{pfx}query_norm//scale");
    let offset_key = format!("{pfx}query_norm//offset");
    let qn_scale = vec1(params, &scale_key, b)?;
    let qn_offset = vec1(params, &offset_key, b)?;

    let feat_norm_scale = vec1(params, &format!("{pfx}feat_2d_norm//scale"), b)?;
    let feat_norm_offset = vec1(params, &format!("{pfx}feat_2d_norm//offset"), b)?;

    let n_heads = {
        let t = params
            .get(&format!("{pfx}attention//query_w"))
            .ok_or_else(|| anyhow!("missing query_w"))?;
        t.data.shape()[2]
    };
    let head_dim = {
        let t = params
            .get(&format!("{pfx}attention//query_w"))
            .ok_or_else(|| anyhow!("missing query_w"))?;
        t.data.shape()[3]
    };

    // query/key/value/gating weights [in, heads*head_dim]
    let qw = mat_attn(params, &format!("{pfx}attention//query_w"), b)?;
    let kw = mat_attn(params, &format!("{pfx}attention//key_w"), b)?;
    let vw = mat_attn(params, &format!("{pfx}attention//value_w"), b)?;
    let gw = mat_attn(params, &format!("{pfx}attention//gating_w"), b)?;
    let gb = bias_attn(params, &format!("{pfx}attention//gating_b"), b)?;
    let ow = mat_attn_out(params, &format!("{pfx}attention//output_w"), b)?;
    let ob = vec1(params, &format!("{pfx}attention//output_b"), b)?;

    // feat_2d_weights [128, H_m]
    let fw = mat_feat2d(params, &format!("{pfx}feat_2d_weights"), b)?;

    let norm_pair = layer_norm_3d(pair, &feat_norm_scale, &feat_norm_offset);
    let pair_bias_flat = norm_pair
        .to_shape((l * l, norm_pair.shape()[2]))
        .unwrap()
        .to_owned();
    // [L*L, H_m]
    let pair_bias_lh = pair_bias_flat.dot(&fw);
    // reshape to [L, L, H_m] then use as [q_len, kv_len, H_m]
    let pair_bias = pair_bias_lh
        .into_shape_with_order((l, l, n_heads))
        .unwrap();

    let scale = (head_dim as f32).sqrt().recip();
    let mut out = Array3::<f32>::zeros(msa.dim());

    let s_dim = msa.shape()[0];
    for si in 0..s_dim {
        let row = msa.slice(s![si, .., ..]).to_owned(); // [L, C_m]
        // LayerNorm
        let normed = layer_norm_rows(&row, &qn_scale, &qn_offset); // [L, C_m]

        // Q, K, V, G  each [L, H*D]
        let q = normed.dot(&qw); // [L, H*D]
        let k = normed.dot(&kw);
        let v = normed.dot(&vw);
        let g_pre = normed.dot(&gw); // [L, H*D]
        let mut g = Array2::<f32>::zeros((l, n_heads * head_dim));
        for i in 0..l {
            for hd in 0..n_heads * head_dim {
                g[(i, hd)] = sigmoid(g_pre[(i, hd)] + gb[hd]);
            }
        }

        // Attention for each head: [L, L]
        let mut attn_out = Array2::<f32>::zeros((l, n_heads * head_dim));
        for h in 0..n_heads {
            let q_h = q.slice(s![.., h * head_dim..(h + 1) * head_dim]).to_owned(); // [L, D]
            let k_h = k.slice(s![.., h * head_dim..(h + 1) * head_dim]).to_owned();
            let v_h = v.slice(s![.., h * head_dim..(h + 1) * head_dim]).to_owned();

            // logits [L, L]
            let mut logits = q_h.dot(&k_h.t()) * scale;
            // add pair bias
            for qi in 0..l {
                for kj in 0..l {
                    logits[(qi, kj)] += pair_bias[(qi, kj, h)];
                }
            }
            // softmax over kj
            for qi in 0..l {
                let row_max = logits.slice(s![qi, ..]).fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut exp_sum = 0.0_f32;
                for kj in 0..l {
                    logits[(qi, kj)] = (logits[(qi, kj)] - row_max).exp();
                    exp_sum += logits[(qi, kj)];
                }
                for kj in 0..l {
                    logits[(qi, kj)] /= exp_sum;
                }
            }
            // weighted sum of V: [L, D]
            let ctx = logits.dot(&v_h);
            for qi in 0..l {
                for d in 0..head_dim {
                    attn_out[(qi, h * head_dim + d)] += ctx[(qi, d)];
                }
            }
        }
        // Apply gating
        for i in 0..l {
            for hd in 0..n_heads * head_dim {
                attn_out[(i, hd)] *= g[(i, hd)];
            }
        }
        // Output projection: [L, H*D] @ [H*D, C_m]
        let result = linear(&attn_out, &ow, &ob); // [L, C_m]
        for i in 0..l {
            for c in 0..result.ncols() {
                out[(si, i, c)] += result[(i, c)];
            }
        }
    }
    Ok(msa + &out)
}

/// Independently refine each position’s MSA description through a small two-layer network.
///
/// After attention has mixed information across positions, this MLP lets each
/// residue’s representation “digest” what it received without looking at
/// neighbours again. The hidden layer is 4× the input width (256 → 1024 → 256)
/// with a ReLU in between.
///
/// Operates identically on every `(sequence, position)` cell in the MSA, so
/// it is equivalent to running the same small network on each of the `S × L`
/// rows independently. The result is added back as a residual.
fn msa_transition(
    msa: &Array3<f32>,
    params: &HashMap<String, Tensor>,
    pfx: &str,
    b: usize,
) -> Result<Array3<f32>> {
    let scale = vec1(params, &format!("{pfx}input_layer_norm//scale"), b)?;
    let offset = vec1(params, &format!("{pfx}input_layer_norm//offset"), b)?;
    let w1 = mat2(params, &format!("{pfx}transition1//weights"), b)?;
    let b1 = vec1(params, &format!("{pfx}transition1//bias"), b)?;
    let w2 = mat2(params, &format!("{pfx}transition2//weights"), b)?;
    let b2 = vec1(params, &format!("{pfx}transition2//bias"), b)?;

    let (s, l, c) = msa.dim();
    let flat = msa.to_shape((s * l, c)).unwrap().to_owned();
    let normed = layer_norm_rows(&flat, &scale, &offset);
    let h = linear(&normed, &w1, &b1).mapv(relu);
    let out_flat = linear(&h, &w2, &b2);
    let out = out_flat.into_shape_with_order((s, l, c)).unwrap();
    Ok(msa + &out)
}

/// Bridge the MSA and pair representations: turn per-residue information into
/// pairwise information.
///
/// For every pair of positions `(i, j)`, compute the outer product of their
/// compressed MSA descriptions, then project the result into the pair
/// dimension. The “mean” in the name refers to averaging over MSA depth
/// (here depth=1, so no averaging needed).
///
/// Concretely:
/// 1. Layer-norm the MSA first row → `[L, 256]`.
/// 2. Project each position to 32 dims via separate left/right projections.
/// 3. For each pair `(i, j)`: form the 32×32 = 1024-element outer product
///    `left[i] ⊗ right[j]`, then project to 128 dims.
///
/// Example: if residue 5 “looks like a helix-former” and residue 22 “looks
/// like a beta-sheet-former”, the outer product encodes that combination and
/// the projection learns what it means for the pair (5, 22).
///
/// Returns the *update* `[L, L, 128]` which is added to the pair table.
fn outer_product_mean(
    msa: &Array3<f32>,   // [1, L, 256]
    params: &HashMap<String, Tensor>,
    pfx: &str,           // "outer_product_mean/"
    b: usize,
) -> Result<Array3<f32>> {
    let l = msa.shape()[1];
    let scale = vec1(params, &format!("{pfx}layer_norm_input//scale"), b)?;
    let offset = vec1(params, &format!("{pfx}layer_norm_input//offset"), b)?;

    let row = msa.slice(s![0, .., ..]).to_owned(); // [L, 256]
    let normed = layer_norm_rows(&row, &scale, &offset);

    let wl = mat2(params, &format!("{pfx}left_projection//weights"), b)?; // [256, 32]
    let bl = vec1(params, &format!("{pfx}left_projection//bias"), b)?;
    let wr = mat2(params, &format!("{pfx}right_projection//weights"), b)?;
    let br = vec1(params, &format!("{pfx}right_projection//bias"), b)?;

    let left = linear(&normed, &wl, &bl);  // [L, 32]
    let right = linear(&normed, &wr, &br); // [L, 32]

    let c = left.ncols(); // 32
    // outer product: [L, L, c, c]
    // output_w [32, 32, 128]  →  reshape [c*c, 128]
    let ow_tensor = params
        .get(&format!("{pfx}output_w"))
        .ok_or_else(|| anyhow!("missing {pfx}output_w"))?;
    let ow_block = ow_tensor.data.slice(s![b, .., .., ..]).to_owned(); // [32, 32, 128]
    let out_dim = ow_block.shape()[2];
    let ow = ow_block
        .into_shape_with_order((c * c, out_dim))
        .unwrap();
    let ob = vec1(params, &format!("{pfx}output_b"), b)?;

    let mut out = Array3::<f32>::zeros((l, l, out_dim));
    for i in 0..l {
        for j in 0..l {
            // outer product at (i,j): left[i,:] ⊗ right[j,:]  → [c*c]
            let mut op = Array1::<f32>::zeros(c * c);
            for p in 0..c {
                for q in 0..c {
                    op[p * c + q] = left[(i, p)] * right[(j, q)];
                }
            }
            let v = op.dot(&ow) + &ob;
            out.slice_mut(s![i, j, ..]).assign(&v);
        }
    }
    Ok(out)
}

/// Update each pair `(i, j)` by combining information from all “intermediary”
/// residues `k` that connect them.
///
/// The key insight is geometric: if we know the relationship between i and k,
/// and between j and k, we can infer something about i and j (like the third
/// side of a triangle). There are two variants:
///
/// - **Outgoing** (`outgoing = true`): `result[i,j] = Σ_k  left[i,k] ⊙ right[j,k]`
///   — “everything k is ‘downstream’ of i AND ‘downstream’ of j”.
/// - **Incoming** (`outgoing = false`): `result[i,j] = Σ_k  left[k,j] ⊙ right[k,i]`
///   — “everything k is ‘upstream’ of both i and j”.
///
/// Both projections are gated (element-wise sigmoid) before the summation,
/// and the result is gated again before being added as a residual to the pair
/// table.
///
/// Example (outgoing): to update the relationship between residues 3 and 14,
/// sum over all residues k the (gated) product of `pair[3,k]` and `pair[14,k]`.
fn triangle_multiplication(
    pair: &Array3<f32>,  // [L, L, 128]
    params: &HashMap<String, Tensor>,
    pfx: &str,
    b: usize,
    outgoing: bool,
) -> Result<Array3<f32>> {
    let (l, _, c) = pair.dim();

    let ln_scale = vec1(params, &format!("{pfx}layer_norm_input//scale"), b)?;
    let ln_offset = vec1(params, &format!("{pfx}layer_norm_input//offset"), b)?;

    // flatten [L, L, C] → [L*L, C]
    let flat = pair.to_shape((l * l, c)).unwrap().to_owned();
    let normed_flat = layer_norm_rows(&flat, &ln_scale, &ln_offset);
    let normed = normed_flat.into_shape_with_order((l, l, c)).unwrap();

    let lp_w = mat2(params, &format!("{pfx}left_projection//weights"), b)?;  // [128, 128]
    let lp_b = vec1(params, &format!("{pfx}left_projection//bias"), b)?;
    let lg_w = mat2(params, &format!("{pfx}left_gate//weights"), b)?;
    let lg_b = vec1(params, &format!("{pfx}left_gate//bias"), b)?;
    let rp_w = mat2(params, &format!("{pfx}right_projection//weights"), b)?;
    let rp_b = vec1(params, &format!("{pfx}right_projection//bias"), b)?;
    let rg_w = mat2(params, &format!("{pfx}right_gate//weights"), b)?;
    let rg_b = vec1(params, &format!("{pfx}right_gate//bias"), b)?;

    // [L, L, 128]
    let lp_flat = linear3(&normed, &lp_w, &lp_b);
    let lg_flat = linear3(&normed, &lg_w, &lg_b);
    let rp_flat = linear3(&normed, &rp_w, &rp_b);
    let rg_flat = linear3(&normed, &rg_w, &rg_b);

    // gated projections
    let left = Array3::from_shape_fn((l, l, c), |(i, j, d)| {
        lp_flat[(i, j, d)] * sigmoid(lg_flat[(i, j, d)])
    });
    let right = Array3::from_shape_fn((l, l, c), |(i, j, d)| {
        rp_flat[(i, j, d)] * sigmoid(rg_flat[(i, j, d)])
    });

    // Einsum
    // outgoing: act[i,j,d] = sum_k left[i,k,d] * right[j,k,d]   (ikc,jkc->ijc)
    // incoming: act[i,j,d] = sum_k left[k,j,d] * right[k,i,d]   (kjc,kic->ijc)
    let mut act = Array3::<f32>::zeros((l, l, c));
    if outgoing {
        for i in 0..l {
            for j in 0..l {
                for d in 0..c {
                    let mut s = 0.0_f32;
                    for k in 0..l {
                        s += left[(i, k, d)] * right[(j, k, d)];
                    }
                    act[(i, j, d)] = s;
                }
            }
        }
    } else {
        for i in 0..l {
            for j in 0..l {
                for d in 0..c {
                    let mut s = 0.0_f32;
                    for k in 0..l {
                        s += left[(k, j, d)] * right[(k, i, d)];
                    }
                    act[(i, j, d)] = s;
                }
            }
        }
    }

    // center norm
    let cn_scale = vec1(params, &format!("{pfx}center_norm//scale"), b)?;
    let cn_offset = vec1(params, &format!("{pfx}center_norm//offset"), b)?;
    let act_flat = act.to_shape((l * l, c)).unwrap().to_owned();
    let act_normed_flat = layer_norm_rows(&act_flat, &cn_scale, &cn_offset);
    let act_normed = act_normed_flat.into_shape_with_order((l, l, c)).unwrap();

    // output projection: [L, L, 128] @ [128, 128]
    let op_w = mat2(params, &format!("{pfx}output_projection//weights"), b)?;
    let op_b = vec1(params, &format!("{pfx}output_projection//bias"), b)?;
    let out = linear3(&act_normed, &op_w, &op_b);

    // gating linear: sigmoid(normed @ gating_linear) * out
    let gl_w = mat2(params, &format!("{pfx}gating_linear//weights"), b)?;
    let gl_b = vec1(params, &format!("{pfx}gating_linear//bias"), b)?;
    let gate_logit = linear3(&normed, &gl_w, &gl_b);
    let gated = Array3::from_shape_fn((l, l, c), |(i, j, d)| {
        out[(i, j, d)] * sigmoid(gate_logit[(i, j, d)])
    });

    Ok(pair + &gated)
}

/// Update the pair table by running attention *within* each row or column of
/// the pair matrix, biased by the pair values themselves.
///
/// Two variants:
/// - **Starting node** (`starting = true`): for each residue `i`, attend
///   across all `j` within the row `pair[i, :, :]`. Residue `i` asks:
///   “which other residues j should I update my relationship with, based on
///   how all my pair relationships look?”
/// - **Ending node** (`starting = false`): for each residue `j`, attend
///   across all `i` within the column `pair[:, j, :]`. Symmetric complement.
///
/// The “triangle” bias: the pair values for the query row/column are projected
/// to per-head scalars and added to the attention logits, so the network can
/// focus attention on pairs that already have strong signals.
///
/// Example (starting): when updating `pair[3, 14]`, residue 3 attends over
/// all its rows (`pair[3, 0]`, `pair[3, 1]`, …) and aggregates what it
/// learns, biased by the pair values in that row.
fn triangle_attention(
    pair: &Array3<f32>,
    params: &HashMap<String, Tensor>,
    pfx: &str,
    b: usize,
    starting: bool,
) -> Result<Array3<f32>> {
    let (l, _, c) = pair.dim();

    let qn_scale = vec1(params, &format!("{pfx}query_norm//scale"), b)?;
    let qn_offset = vec1(params, &format!("{pfx}query_norm//offset"), b)?;

    let n_heads = {
        let t = params
            .get(&format!("{pfx}attention//query_w"))
            .ok_or_else(|| anyhow!("missing {}", pfx))?;
        t.data.shape()[2]
    };
    let head_dim = {
        let t = params
            .get(&format!("{pfx}attention//query_w"))
            .ok_or_else(|| anyhow!("missing {}", pfx))?;
        t.data.shape()[3]
    };
    let scale = (head_dim as f32).sqrt().recip();

    // weights [c, n_heads * head_dim]
    let qw = mat_attn(params, &format!("{pfx}attention//query_w"), b)?;
    let kw = mat_attn(params, &format!("{pfx}attention//key_w"), b)?;
    let vw = mat_attn(params, &format!("{pfx}attention//value_w"), b)?;
    let gw = mat_attn(params, &format!("{pfx}attention//gating_w"), b)?;
    let gbv = bias_attn(params, &format!("{pfx}attention//gating_b"), b)?;
    // output_w [heads, head_dim, c] → [(H*D), c]
    let ow = mat_attn_out(params, &format!("{pfx}attention//output_w"), b)?;
    let ob = vec1(params, &format!("{pfx}attention//output_b"), b)?;
    // feat_2d_weights [c_z, n_heads]
    let fw = mat_feat2d(params, &format!("{pfx}feat_2d_weights"), b)?;

    // layer norm the whole pair [L, L, c]
    let pair_flat = pair.to_shape((l * l, c)).unwrap().to_owned();
    let normed_flat = layer_norm_rows(&pair_flat, &qn_scale, &qn_offset);
    let normed = normed_flat.into_shape_with_order((l, l, c)).unwrap();

    let mut delta = Array3::<f32>::zeros((l, l, c));

    // For starting_node: outer index is i (query row), inner seq length is l over j.
    // For ending_node:   outer index is j (key col), inner seq length is l over i.
    let outer = l;
    for idx in 0..outer {
        // Extract the (l, c) slice for this row/col
        let seq: Array2<f32> = if starting {
            normed.slice(s![idx, .., ..]).to_owned() // row idx, all j
        } else {
            normed.slice(s![.., idx, ..]).to_owned() // all i, col idx
        };

        // bias from the *same* slice: [l, n_heads] projected
        let bias_seq = seq.dot(&fw); // [l, n_heads]

        let q_seq = seq.dot(&qw); // [l, H*D]
        let k_seq = seq.dot(&kw);
        let v_seq = seq.dot(&vw);
        let g_logit = seq.dot(&gw); // [l, H*D]

        for h in 0..n_heads {
            let q_h = q_seq.slice(s![.., h * head_dim..(h + 1) * head_dim]).to_owned(); // [l, D]
            let k_h = k_seq.slice(s![.., h * head_dim..(h + 1) * head_dim]).to_owned();
            let v_h = v_seq.slice(s![.., h * head_dim..(h + 1) * head_dim]).to_owned();

            // logits [l, l]
            let mut logits = q_h.dot(&k_h.t()) * scale;
            for qi2 in 0..l {
                for kj2 in 0..l {
                    logits[(qi2, kj2)] += bias_seq[(kj2, h)];
                }
            }
            // softmax over kj2
            for qi2 in 0..l {
                let mx = logits.slice(s![qi2, ..]).fold(f32::NEG_INFINITY, |a, &v| a.max(v));
                let mut es = 0.0_f32;
                for kj2 in 0..l {
                    logits[(qi2, kj2)] = (logits[(qi2, kj2)] - mx).exp();
                    es += logits[(qi2, kj2)];
                }
                for kj2 in 0..l {
                    logits[(qi2, kj2)] /= es;
                }
            }
            // ctx [l, D]
            let ctx = logits.dot(&v_h);
            // apply gate + accumulate into delta
            for qi2 in 0..l {
                for d in 0..head_dim {
                    let hd_idx = h * head_dim + d;
                    let g = sigmoid(g_logit[(qi2, hd_idx)] + gbv[hd_idx]);
                    let val = ctx[(qi2, d)] * g;
                    if starting {
                        // qi2 is j
                        delta[(idx, qi2, hd_idx)] += val;
                    } else {
                        // qi2 is i
                        delta[(qi2, idx, hd_idx)] += val;
                    }
                }
            }
        }
    }

    // project combined [l*l, H*D] → [l*l, c]
    let delta_flat = delta.to_shape((l * l, n_heads * head_dim)).unwrap().to_owned();
    let out_flat = linear(&delta_flat, &ow, &ob);
    let out = out_flat.into_shape_with_order((l, l, c)).unwrap();
    Ok(pair + &out)
}

/// Independently refine each pair’s description through a small two-layer network.
///
/// The pair-table analogue of `msa_transition`: after the triangle operations
/// have mixed information across the pair table, this MLP lets each cell
/// `(i, j)` process its new state without looking at neighbours. Hidden layer
/// is 4× wider (128 → 512 → 128) with ReLU. Result added as a residual.
fn pair_transition(
    pair: &Array3<f32>,
    params: &HashMap<String, Tensor>,
    pfx: &str,
    b: usize,
) -> Result<Array3<f32>> {
    let (l, _, c) = pair.dim();
    let scale = vec1(params, &format!("{pfx}input_layer_norm//scale"), b)?;
    let offset = vec1(params, &format!("{pfx}input_layer_norm//offset"), b)?;
    let w1 = mat2(params, &format!("{pfx}transition1//weights"), b)?;
    let b1 = vec1(params, &format!("{pfx}transition1//bias"), b)?;
    let w2 = mat2(params, &format!("{pfx}transition2//weights"), b)?;
    let b2 = vec1(params, &format!("{pfx}transition2//bias"), b)?;

    let flat = pair.to_shape((l * l, c)).unwrap().to_owned();
    let normed = layer_norm_rows(&flat, &scale, &offset);
    let h = linear(&normed, &w1, &b1).mapv(relu);
    let out_flat = linear(&h, &w2, &b2);
    let out = out_flat.into_shape_with_order((l, l, c)).unwrap();
    Ok(pair + &out)
}

/// Run one complete Evoformer block, updating both the MSA and pair representations.
///
/// Each block applies these sub-layers in order:
/// 1. **MSA row attention with pair bias** — residues exchange information
///    along the sequence direction, guided by the current pair table.
/// 2. **MSA transition** — each position digests what it received (MLP).
/// 3. **Outer-product mean** — per-residue MSA info is folded into the pair table.
/// 4. **Triangle multiplication outgoing** — pair table updated via shared intermediaries.
/// 5. **Triangle multiplication incoming** — symmetric complement.
/// 6. **Triangle attention starting** — row-wise attention within the pair table.
/// 7. **Triangle attention ending** — column-wise attention within the pair table.
/// 8. **Pair transition** — each pair cell digests its update (MLP).
///
/// Every sub-layer is a residual: its output is *added* to the input, so the
/// block can only refine, never erase, what it received.
///
/// Returns `(updated_msa, updated_pair)`, same shapes as inputs.
fn evoformer_block(
    msa: &Array3<f32>,   // [1, L, 256]
    pair: &Array3<f32>,  // [L, L, 128]
    params: &HashMap<String, Tensor>,
    b: usize,
) -> Result<(Array3<f32>, Array3<f32>)> {
    let ep = EVO;

    // MSA row-attention with pair bias
    let msa = msa_row_attention(
        msa,
        pair,
        params,
        &format!("{ep}msa_row_attention_with_pair_bias/"),
        b,
    )?;

    // MSA transition
    let msa = msa_transition(&msa, params, &format!("{ep}msa_transition/"), b)?;

    // Outer-product mean → pair update
    let opm = outer_product_mean(&msa, params, &format!("{ep}outer_product_mean/"), b)?;
    let pair = pair + &opm;

    // Triangle multiplication (outgoing)
    let pair = triangle_multiplication(
        &pair,
        params,
        &format!("{ep}triangle_multiplication_outgoing/"),
        b,
        true,
    )?;

    // Triangle multiplication (incoming)
    let pair = triangle_multiplication(
        &pair,
        params,
        &format!("{ep}triangle_multiplication_incoming/"),
        b,
        false,
    )?;

    // Triangle attention starting
    let pair = triangle_attention(
        &pair,
        params,
        &format!("{ep}triangle_attention_starting_node/"),
        b,
        true,
    )?;

    // Triangle attention ending
    let pair = triangle_attention(
        &pair,
        params,
        &format!("{ep}triangle_attention_ending_node/"),
        b,
        false,
    )?;

    // Pair transition
    let pair = pair_transition(&pair, params, &format!("{ep}pair_transition/"), b)?;

    Ok((msa, pair))
}

/// Run one block of the extra-MSA stack, which pre-conditions the pair table
/// before the main 48-block Evoformer stack.
///
/// The extra-MSA representation is a compact 64-dimensional summary of the
/// sequence (compared to the full 256-dim MSA). Four such blocks run first,
/// allowing the pair table to absorb low-cost sequence-level signals before
/// the expensive main stack begins.
///
/// The architecture is identical to `evoformer_block` but uses different
/// checkpoint weights (`extra_msa_stack/` prefix) and a narrower residue
/// dimension (64 instead of 256).
///
/// Returns `(updated_extra_msa, updated_pair)`.
fn extra_msa_block(
    extra: &Array3<f32>,  // [1, L, 64]
    pair: &Array3<f32>,   // [L, L, 128]
    params: &HashMap<String, Tensor>,
    b: usize,
) -> Result<(Array3<f32>, Array3<f32>)> {
    let ep = XMSA;

    let extra = msa_row_attention(
        extra,
        pair,
        params,
        &format!("{ep}msa_row_attention_with_pair_bias/"),
        b,
    )?;
    let extra = msa_transition(&extra, params, &format!("{ep}msa_transition/"), b)?;

    let opm = outer_product_mean(&extra, params, &format!("{ep}outer_product_mean/"), b)?;
    let pair = pair + &opm;

    let pair = triangle_multiplication(
        &pair,
        params,
        &format!("{ep}triangle_multiplication_outgoing/"),
        b,
        true,
    )?;
    let pair = triangle_multiplication(
        &pair,
        params,
        &format!("{ep}triangle_multiplication_incoming/"),
        b,
        false,
    )?;
    let pair = triangle_attention(
        &pair,
        params,
        &format!("{ep}triangle_attention_starting_node/"),
        b,
        true,
    )?;
    let pair = triangle_attention(
        &pair,
        params,
        &format!("{ep}triangle_attention_ending_node/"),
        b,
        false,
    )?;
    let pair = pair_transition(&pair, params, &format!("{ep}pair_transition/"), b)?;

    Ok((extra, pair))
}

/// Inject the previous recycle’s output into the current recycle’s starting state.
///
/// AlphaFold2 runs the full Evoformer stack multiple times (“recycles”). After
/// each pass, the final MSA first row and pair table are saved. At the *start*
/// of the next pass they are layer-normed and added to the freshly embedded
/// inputs, giving the network a memory of what it predicted last time.
///
/// - `prev_msa_row` (from the last recycle) is normed and added to the
///   first row of the current MSA embedding.
/// - `prev_pair` (from the last recycle) is normed and added to the
///   current pair embedding.
///
/// On the very first recycle both are zero, so this is a no-op and the
/// network starts from scratch. By recycle 3 the representations are already
/// close to converged and the stack only makes small corrections.
fn apply_recycling(
    msa: &Array3<f32>,    // [1, L, 256] — current init
    pair: &Array3<f32>,   // [L, L, 128] — current init
    prev_msa_row: &Array2<f32>,  // [L, 256]
    prev_pair: &Array3<f32>,     // [L, L, 128]
    params: &HashMap<String, Tensor>,
) -> Result<(Array3<f32>, Array3<f32>)> {
    let pfx = PFX;

    // Norm prev_msa_first_row and add to msa[0, :, :]
    let pm_scale = params
        .get(&format!("{pfx}prev_msa_first_row_norm//scale"))
        .ok_or_else(|| anyhow!("missing prev_msa_first_row_norm//scale"))?
        .data
        .slice(s![..])
        .to_owned()
        .into_dimensionality::<ndarray::Ix1>()?;
    let pm_offset = params
        .get(&format!("{pfx}prev_msa_first_row_norm//offset"))
        .ok_or_else(|| anyhow!("missing prev_msa_first_row_norm//offset"))?
        .data
        .slice(s![..])
        .to_owned()
        .into_dimensionality::<ndarray::Ix1>()?;
    let normed_pm = layer_norm_rows(prev_msa_row, &pm_scale, &pm_offset);

    let mut new_msa = msa.clone();
    for i in 0..msa.shape()[1] {
        for c in 0..msa.shape()[2] {
            new_msa[(0, i, c)] += normed_pm[(i, c)];
        }
    }

    // Norm prev_pair and add
    let pp_scale = params
        .get(&format!("{pfx}prev_pair_norm//scale"))
        .ok_or_else(|| anyhow!("missing prev_pair_norm//scale"))?
        .data
        .slice(s![..])
        .to_owned()
        .into_dimensionality::<ndarray::Ix1>()?;
    let pp_offset = params
        .get(&format!("{pfx}prev_pair_norm//offset"))
        .ok_or_else(|| anyhow!("missing prev_pair_norm//offset"))?
        .data
        .slice(s![..])
        .to_owned()
        .into_dimensionality::<ndarray::Ix1>()?;

    let l = prev_pair.shape()[0];
    let c_z = prev_pair.shape()[2];
    let prev_pair_flat = prev_pair.to_shape((l * l, c_z)).unwrap().to_owned();
    let normed_pp_flat = layer_norm_rows(&prev_pair_flat, &pp_scale, &pp_offset);
    let normed_pp = normed_pp_flat.into_shape_with_order((l, l, c_z)).unwrap();

    let new_pair = pair + &normed_pp;
    Ok((new_msa, new_pair))
}

// ── Main entry point ──────────────────────────────────────────────────────────

/// Run the full Evoformer pipeline and return per-residue and pairwise representations.
///
/// This is the top-level function that orchestrates everything:
///
/// 1. **4 extra-MSA blocks** — warm up the pair table cheaply.
/// 2. **48 main Evoformer blocks** — deeply refine both MSA and pair.
/// 3. **3 recycles** — repeat steps 1–2 three more times, each time
///    seeding the starting state with the previous pass’s output.
/// 4. **Final projection** — project the MSA first row from 256 → 384 dims
///    (with ReLU) to produce the `single` representation consumed by the
///    structure module.
///
/// Progress is printed to stderr (`recycle N/3`, `extra-msa block b`,
/// `evoformer block b`) so you can see it working even for small sequences.
///
/// Returns an [`EvoformerOutput`] containing:
/// - `single`        — `[L, 384]` per-residue features for the structure module.
/// - `pair`          — `[L, L, 128]` pairwise features for the structure module.
/// - `msa_first_row` — `[L, 256]` pre-projection MSA row, retained for inspection.
pub fn run(inputs: &Inputs, params: &HashMap<String, Tensor>) -> Result<EvoformerOutput> {
    let l = inputs.len;

    let mut prev_msa_row = Array2::<f32>::zeros((l, 256));
    let mut prev_pair = Array3::<f32>::zeros((l, l, 128));

    let mut final_msa: Array3<f32> = Array3::zeros((1, l, 256));
    let mut final_pair: Array3<f32> = Array3::zeros((l, l, 128));

    let n_recycles = 3_usize;

    for recycle in 0..=n_recycles {
        let is_last = recycle == n_recycles;
        eprintln!("  recycle {}/{}", recycle, n_recycles);

        // Start from embedded inputs
        let (mut msa, mut pair) = if recycle == 0 {
            (inputs.msa.clone(), inputs.pair.clone())
        } else {
            // Apply recycling norms on top of the fresh embeddings
            apply_recycling(
                &inputs.msa,
                &inputs.pair,
                &prev_msa_row,
                &prev_pair,
                params,
            )?
        };

        // Extra-MSA stack (4 blocks)
        let mut extra = inputs.extra_msa.clone(); // [1, L, 64]
        for b in 0..4 {
            eprintln!("    extra-msa block {b}");
            let (e2, p2) = extra_msa_block(&extra, &pair, params, b)?;
            extra = e2;
            pair = p2;
        }

        // Main Evoformer stack (48 blocks)
        for b in 0..48 {
            eprintln!("    evoformer block {b}");
            let (m2, p2) = evoformer_block(&msa, &pair, params, b)?;
            msa = m2;
            pair = p2;
        }

        if is_last {
            final_msa = msa;
            final_pair = pair;
        } else {
            prev_msa_row = msa.slice(s![0, .., ..]).to_owned();
            prev_pair = pair;
        }
    }

    // Project MSA first row [L, 256] → [L, 384]
    let single_w = params
        .get(&format!("{PFX}single_activations//weights"))
        .ok_or_else(|| anyhow!("missing single_activations//weights"))?
        .data
        .to_shape((256, 384))
        .unwrap()
        .to_owned()
        .into_dimensionality::<ndarray::Ix2>()?;
    let single_b = params
        .get(&format!("{PFX}single_activations//bias"))
        .ok_or_else(|| anyhow!("missing single_activations//bias"))?
        .data
        .slice(s![..])
        .to_owned()
        .into_dimensionality::<ndarray::Ix1>()?;

    let msa_row = final_msa.slice(s![0, .., ..]).to_owned();
    let single = linear(&msa_row, &single_w, &single_b).mapv(relu);

    Ok(EvoformerOutput {
        single,
        pair: final_pair,
        msa_first_row: msa_row,
    })
}
