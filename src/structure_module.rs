//! Structure Module: converts Evoformer representations into 3-D backbone frames.
//!
//! The structure module runs 8 iterations of a "fold_iteration" block that uses
//! the same shared weights every time.  Each iteration refines a rigid-body
//! frame (rotation + translation) for every residue.  At the end the translation
//! of each frame is the predicted Cα position.
//!
//! # Algorithm overview (per iteration)
//! ```text
//! 1.  LayerNorm(single)                         ← attention_layer_norm
//! 2.  single += IPA(single, pair, frames)       ← invariant point attention
//! 3.  single  = LayerNorm(single)               ← attention_layer_norm (reused)
//! 4.  single += transition_MLP(single)          ← 3× [384→384] ReLU layers
//! 5.  single  = LayerNorm(single)               ← transition_layer_norm
//! 6.  frames  = update_frames(frames, single)   ← affine_update linear [384→6]
//! ```
//! After 8 iterations, the translation vectors of the frames are the Cα coords.
//!
//! # IPA output size
//! 12 heads × (scalar-v 16 + point-v 3×8 + point-norm 8 + pair 128) = 2112
//! → projected to 384 by output_projection.

use anyhow::{anyhow, Result};
use ndarray::{s, Array1, Array2, Array3};
use std::collections::HashMap;

use crate::evoformer::EvoformerOutput;
use crate::params::Tensor;

// ── tensor key prefixes ───────────────────────────────────────────────────────
const SM: &str = "alphafold/alphafold_iteration/structure_module/";
const FI: &str = "alphafold/alphafold_iteration/structure_module/fold_iteration/";
const IPA: &str =
    "alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/";

// ── IPA hyper-parameters (read from checkpoint shapes) ───────────────────────
const N_HEADS: usize = 12;
const SCALAR_DIM: usize = 16;  // per head; hidden 12×16=192 for q, 12×32=384 for kv
const N_PTS_QK: usize = 4;     // 3-D query/key points per head
const N_PTS_V: usize = 8;      // 3-D value points per head
const PAIR_DIM: usize = 128;
const NUM_ITER: usize = 8;

// ── output ────────────────────────────────────────────────────────────────────

/// Predicted 3-D backbone co-ordinates produced by the structure module.
///
/// Each element of `ca_coords` is the (x, y, z) position in Ångström units of
/// the Cα atom of that residue.  Element 0 is the first residue in the input
/// FASTA sequence, element 1 is the second, and so on.
///
/// For HP36 (L=35) this is a Vec of 35 `[f32; 3]` vectors.  You can feed these
/// directly into a RMSD calculation or into the Bevy visualiser.
///
/// The `single` field contains the final per-residue embedding after the last
/// fold iteration — it is used by auxiliary heads (e.g. pLDDT) but is not
/// needed for Cα extraction.
pub struct StructureOutput {
    /// Predicted Cα positions, one [x,y,z] per residue (Ångströms).
    pub ca_coords: Vec<[f32; 3]>,
    /// Final per-residue embedding [L, 384] — used by pLDDT head.
    pub single: Array2<f32>,
}

// ── rigid-body frame arithmetic ───────────────────────────────────────────────

/// A rigid-body frame: rotation matrix R (row-major, row = output axis) plus
/// translation vector t (the position of the frame origin in global space).
///
/// `apply(p)` maps a point from local frame co-ordinates to global space:
/// `global = R @ p + t`.
#[derive(Clone)]
struct Frame {
    /// 3×3 rotation matrix stored as three row vectors.
    r: [[f32; 3]; 3],
    /// Translation (= Cα position in global space after convergence).
    t: [f32; 3],
}

impl Frame {
    /// Identity frame: no rotation, origin at zero.
    fn identity() -> Self {
        Frame {
            r: [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
            t: [0., 0., 0.],
        }
    }

    /// Rotate a 3-D point from this frame's local space to global space.
    ///
    /// Uses R @ p (rotation only, no translation).  Used for rotating direction
    /// vectors such as the 3-D query/key points in IPA.
    fn rotate(&self, p: [f32; 3]) -> [f32; 3] {
        [
            self.r[0][0] * p[0] + self.r[0][1] * p[1] + self.r[0][2] * p[2],
            self.r[1][0] * p[0] + self.r[1][1] * p[1] + self.r[1][2] * p[2],
            self.r[2][0] * p[0] + self.r[2][1] * p[1] + self.r[2][2] * p[2],
        ]
    }

    /// Map a point from local frame space to global space: global = R @ p + t.
    fn apply(&self, p: [f32; 3]) -> [f32; 3] {
        let rp = self.rotate(p);
        [rp[0] + self.t[0], rp[1] + self.t[1], rp[2] + self.t[2]]
    }

    /// Map a point from global space back to this frame's local space:
    /// local = R^T @ (global − t).
    fn invert_apply(&self, global: [f32; 3]) -> [f32; 3] {
        let d = [
            global[0] - self.t[0],
            global[1] - self.t[1],
            global[2] - self.t[2],
        ];
        // R^T @ d  (transpose of row matrix = column matrix, i.e. dot with cols)
        [
            self.r[0][0] * d[0] + self.r[1][0] * d[1] + self.r[2][0] * d[2],
            self.r[0][1] * d[0] + self.r[1][1] * d[1] + self.r[2][1] * d[2],
            self.r[0][2] * d[0] + self.r[1][2] * d[1] + self.r[2][2] * d[2],
        ]
    }

    /// Compose two frames: self ∘ rhs = (self.R @ rhs.R, self.R @ rhs.t + self.t).
    ///
    /// This is used to apply a small incremental update (rhs) expressed in the
    /// current frame's local co-ordinates to produce the new global frame.
    fn compose(&self, rhs: &Frame) -> Frame {
        // R_new = self.R @ rhs.R
        let mut r_new = [[0f32; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    r_new[i][j] += self.r[i][k] * rhs.r[k][j];
                }
            }
        }
        Frame { r: r_new, t: self.apply(rhs.t) }
    }
}

/// Build a rotation matrix from a unit quaternion [w, x, y, z].
///
/// Quaternion convention: w is the scalar part, (x, y, z) the vector part.
/// The result is a right-hand-rule rotation matrix (same convention as the
/// original AlphaFold2 JAX code).
fn quat_to_rot(q: [f32; 4]) -> [[f32; 3]; 3] {
    let [w, x, y, z] = q;
    [
        [
            1. - 2. * (y * y + z * z),
            2. * (x * y - w * z),
            2. * (x * z + w * y),
        ],
        [
            2. * (x * y + w * z),
            1. - 2. * (x * x + z * z),
            2. * (y * z - w * x),
        ],
        [
            2. * (x * z - w * y),
            2. * (y * z + w * x),
            1. - 2. * (x * x + y * y),
        ],
    ]
}

/// Update a set of backbone frames given a [L, 6] matrix of incremental updates.
///
/// Each row of `delta` is [b, c, d, x, y, z].  The quaternion update is built
/// as [1, b, c, d] (unit scalar part) and normalised, giving a rotation.  The
/// translation [x, y, z] is expressed in the **local** frame co-ordinates of
/// the residue being updated.  The new frame is the composition
/// `old_frame ∘ (R_delta, t_delta)`.
fn update_frames(frames: &mut Vec<Frame>, delta: &Array2<f32>) {
    let l = frames.len();
    for i in 0..l {
        let b = delta[[i, 0]];
        let c = delta[[i, 1]];
        let d = delta[[i, 2]];
        let x = delta[[i, 3]];
        let y = delta[[i, 4]];
        let z = delta[[i, 5]];

        // Normalise quaternion [1, b, c, d]
        let norm = (1.0_f32 + b * b + c * c + d * d).sqrt();
        let q = [1. / norm, b / norm, c / norm, d / norm];
        let r = quat_to_rot(q);
        let delta_frame = Frame { r, t: [x, y, z] };
        frames[i] = frames[i].compose(&delta_frame);
    }
}

// ── shared linear helpers ─────────────────────────────────────────────────────

/// Fetch a weight matrix from `params` as a 2-D f32 array.
///
/// Panics if the key is missing or the tensor is not 2-D.  The weight matrix
/// is read with shape [in_dim, out_dim] matching the checkpoint convention.
fn w2(params: &HashMap<String, Tensor>, key: &str) -> Result<Array2<f32>> {
    params
        .get(key)
        .ok_or_else(|| anyhow!("missing tensor: {key}"))?
        .data
        .clone()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e| anyhow!("{key}: {e}"))
}

/// Fetch a bias / 1-D weight from `params`.
fn w1(params: &HashMap<String, Tensor>, key: &str) -> Result<Array1<f32>> {
    params
        .get(key)
        .ok_or_else(|| anyhow!("missing tensor: {key}"))?
        .data
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .map_err(|e| anyhow!("{key}: {e}"))
}

/// Apply a linear layer: out[i] = x[i] @ W + b  (row-wise).
///
/// * `x`  — shape \[L, in_dim\]
/// * `W`  — shape \[in_dim, out_dim\]
/// * `b`  — shape \[out_dim\]
///
/// Returns shape \[L, out_dim\].
fn linear(x: &Array2<f32>, w: &Array2<f32>, b: &Array1<f32>) -> Array2<f32> {
    let mut out = x.dot(w);
    out.rows_mut().into_iter().for_each(|mut row| row += b);
    out
}

/// Apply ReLU element-wise to a 2-D array (sets negative values to 0).
fn relu2(x: Array2<f32>) -> Array2<f32> {
    x.mapv(|v| v.max(0.0))
}

/// Layer-normalise along the last axis (rows), in-place.  Each row is shifted
/// to zero mean and scaled to unit variance, then affinely transformed by the
/// learned scale (gamma) and offset (beta) vectors.
///
/// `eps = 1e-5` following the JAX/Haiku default.
fn layer_norm(x: &Array2<f32>, scale: &Array1<f32>, offset: &Array1<f32>) -> Array2<f32> {
    let (rows, cols) = (x.nrows(), x.ncols());
    let mut out = Array2::<f32>::zeros((rows, cols));
    for i in 0..rows {
        let row = x.row(i);
        let mean = row.sum() / cols as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / cols as f32;
        let std = (var + 1e-5).sqrt();
        for j in 0..cols {
            out[[i, j]] = (row[j] - mean) / std * scale[j] + offset[j];
        }
    }
    out
}

// ── Invariant Point Attention (IPA) ───────────────────────────────────────────

/// Compute Invariant Point Attention for one fold iteration.
///
/// IPA is the key operation that lets the network reason about 3-D structure
/// while remaining equivariant to rotations and translations.  Queries, keys
/// and values consist of *both* ordinary scalar features and 3-D point
/// positions anchored to each residue's backbone frame.  When the frames
/// rotate/translate, the point features rotate/translate consistently, so the
/// attention scores are invariant to global rigid-body motions.
///
/// # Inputs
/// * `s`      — per-residue single representation \[L, 384\] (layer-normed)
/// * `z`      — pairwise representation \[L, L, 128\]
/// * `frames` — current backbone frames for each residue (length L)
/// * `params` — full parameter map
///
/// # Output
/// Shape \[L, 384\]: updated single representation after IPA projection.
///
/// # Sizes
/// | Tensor | Shape | Meaning |
/// |--------|-------|---------|
/// | q_scalar | \[L, 12, 16\] | 12 attention heads, 16-dim scalar query |
/// | k_scalar | \[L, 12, 16\] | scalar key |
/// | v_scalar | \[L, 12, 16\] | scalar value |
/// | q_pts (global) | \[L, 12, 4, 3\] | 4 query points per head in global space |
/// | k_pts (global) | \[L, 12, 4, 3\] | key points |
/// | v_pts (global) | \[L, 12, 8, 3\] | 8 value points per head |
/// | pair bias | \[L, L, 12\] | per-head bias from pair rep |
fn ipa(
    s: &Array2<f32>,
    z: &Array3<f32>,
    frames: &[Frame],
    params: &HashMap<String, Tensor>,
) -> Result<Array2<f32>> {
    let l = s.nrows();

    // ── scalar projections ──────────────────────────────────────────────────
    // q_scalar: [384, 192] → after linear: [L, 192] → reshape [L, 12, 16]
    let wq_s = w2(params, &format!("{IPA}q_scalar//weights"))?;
    let bq_s = w1(params, &format!("{IPA}q_scalar//bias"))?;
    let q_s_flat = linear(s, &wq_s, &bq_s); // [L, 192]

    // kv_scalar: [384, 384]  (k and v concatenated, 12*16 each = 192 per part)
    let wkv_s = w2(params, &format!("{IPA}kv_scalar//weights"))?;
    let bkv_s = w1(params, &format!("{IPA}kv_scalar//bias"))?;
    let kv_s_flat = linear(s, &wkv_s, &bkv_s); // [L, 384]

    // ── point projections (local frame) ────────────────────────────────────
    // q_point_local: [384, 144]  → 144 = 12*4*3
    let wq_p = w2(params, &format!("{IPA}q_point_local//weights"))?;
    let bq_p = w1(params, &format!("{IPA}q_point_local//bias"))?;
    let q_p_flat = linear(s, &wq_p, &bq_p); // [L, 144]

    // kv_point_local: [384, 432]  → 432 = 12*(4+8)*3
    let wkv_p = w2(params, &format!("{IPA}kv_point_local//weights"))?;
    let bkv_p = w1(params, &format!("{IPA}kv_point_local//bias"))?;
    let kv_p_flat = linear(s, &wkv_p, &bkv_p); // [L, 432]

    // ── trainable per-head point weights (softplus-activated) ──────────────
    // Controls how much the point distances contribute vs scalar attention.
    let raw_pw = w1(params, &format!("{IPA}trainable_point_weights"))?; // [12]
    // softplus: log(1 + exp(x))
    let gamma: Vec<f32> = raw_pw
        .iter()
        .map(|&v| (1.0_f32 + v.exp()).ln())
        .collect();

    // Scale: (w_L/2)^0.5 where w_L = num_point_qk = 4
    let point_scale = (N_PTS_QK as f32 / 2.0).sqrt();

    // ── pair bias: [L, L, 12] ──────────────────────────────────────────────
    let w2d = w2(params, &format!("{IPA}attention_2d//weights"))?; // [128, 12]
    let b2d = w1(params, &format!("{IPA}attention_2d//bias"))?;    // [12]
    // Compute pair bias: for each (i,j), z[i,j,:] @ w2d + b2d → [12]
    let mut pair_bias = Array3::<f32>::zeros((l, l, N_HEADS));
    for i in 0..l {
        for j in 0..l {
            let zij = z.slice(s![i, j, ..]);
            for h in 0..N_HEADS {
                let mut v = b2d[h];
                for k in 0..PAIR_DIM {
                    v += zij[k] * w2d[[k, h]];
                }
                pair_bias[[i, j, h]] = v;
            }
        }
    }

    // ── transform points to global space ────────────────────────────────────
    // q_pts[i, h, p, xyz] = frames[i].apply(q_p_flat[i, (h*N_PTS_QK+p)*3 .. +3])
    let mut q_pts = vec![[[[0f32; 3]; N_PTS_QK]; N_HEADS]; l];
    let mut k_pts = vec![[[[0f32; 3]; N_PTS_QK]; N_HEADS]; l];
    let mut v_pts = vec![[[[0f32; 3]; N_PTS_V]; N_HEADS]; l];

    for i in 0..l {
        for h in 0..N_HEADS {
            for p in 0..N_PTS_QK {
                let base = (h * N_PTS_QK + p) * 3;
                let local = [q_p_flat[[i, base]], q_p_flat[[i, base + 1]], q_p_flat[[i, base + 2]]];
                q_pts[i][h][p] = frames[i].apply(local);
            }
            for p in 0..N_PTS_QK {
                // k points come from the first N_PTS_QK*3 of kv_point_local
                let base = (h * (N_PTS_QK + N_PTS_V) + p) * 3;
                let local = [kv_p_flat[[i, base]], kv_p_flat[[i, base + 1]], kv_p_flat[[i, base + 2]]];
                k_pts[i][h][p] = frames[i].apply(local);
            }
            for p in 0..N_PTS_V {
                // v points come from the next N_PTS_V*3 of kv_point_local
                let base = (h * (N_PTS_QK + N_PTS_V) + N_PTS_QK + p) * 3;
                let local = [kv_p_flat[[i, base]], kv_p_flat[[i, base + 1]], kv_p_flat[[i, base + 2]]];
                v_pts[i][h][p] = frames[i].apply(local);
            }
        }
    }

    // ── compute attention weights [L, L, N_HEADS] ──────────────────────────
    // attn[i,j,h] = scalar_logit(i,j,h) + point_logit(i,j,h) + pair_bias(i,j,h)
    // then softmax over j dimension.
    let scalar_scale = 1.0 / (SCALAR_DIM as f32).sqrt();
    let mut attn = Array3::<f32>::zeros((l, l, N_HEADS));

    for i in 0..l {
        for j in 0..l {
            for h in 0..N_HEADS {
                // scalar: q_s[i,h,:] · k_s[j,h,:]
                let qi_base = h * SCALAR_DIM;
                let mut scalar_dot = 0f32;
                for d in 0..SCALAR_DIM {
                    scalar_dot += q_s_flat[[i, qi_base + d]] * kv_s_flat[[j, qi_base + d]];
                }
                let scalar_logit = scalar_scale * scalar_dot;

                // point: -0.5 * gamma[h] * point_scale * Σ_p ||q_pt - k_pt||^2
                let mut sq_dist = 0f32;
                for p in 0..N_PTS_QK {
                    for c in 0..3 {
                        let diff = q_pts[i][h][p][c] - k_pts[j][h][p][c];
                        sq_dist += diff * diff;
                    }
                }
                let point_logit = -0.5 * gamma[h] * point_scale * sq_dist;

                attn[[i, j, h]] = scalar_logit + point_logit + pair_bias[[i, j, h]];
            }
        }
    }

    // Softmax over j for each (i, h)
    for i in 0..l {
        for h in 0..N_HEADS {
            let max_v = (0..l).map(|j| attn[[i, j, h]]).fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            for j in 0..l { let e = (attn[[i, j, h]] - max_v).exp(); attn[[i, j, h]] = e; sum += e; }
            for j in 0..l { attn[[i, j, h]] /= sum; }
        }
    }

    // ── aggregate: scalar values ────────────────────────────────────────────
    // v_scalar = kv_s_flat[j, N_HEADS*SCALAR_DIM .. 2*N_HEADS*SCALAR_DIM]
    // out_scalar[i, h, d] = Σ_j attn[i,j,h] * v_s[j, h, d]
    let kv_offset = N_HEADS * SCALAR_DIM; // start of v in kv_scalar
    let mut out_scalar = Array2::<f32>::zeros((l, N_HEADS * SCALAR_DIM));
    for i in 0..l {
        for h in 0..N_HEADS {
            for d in 0..SCALAR_DIM {
                let col = h * SCALAR_DIM + d;
                let mut acc = 0f32;
                for j in 0..l {
                    acc += attn[[i, j, h]] * kv_s_flat[[j, kv_offset + col]];
                }
                out_scalar[[i, h * SCALAR_DIM + d]] = acc;
            }
        }
    }

    // ── aggregate: point values → global then back to local ─────────────────
    // out_pts[i, h, p] = Σ_j attn[i,j,h] * v_pts[j, h, p]  (global)
    // then rotate to frame[i] local: frame[i].invert_apply(global_v_pt)
    // output includes both the 3 coordinates and the L2 norm of each point.
    let n_pts_out = N_HEADS * N_PTS_V;
    let mut out_pts_xyz = Array2::<f32>::zeros((l, n_pts_out * 3));
    let mut out_pts_norm = Array2::<f32>::zeros((l, n_pts_out));

    for i in 0..l {
        for h in 0..N_HEADS {
            for p in 0..N_PTS_V {
                // Weighted sum of value points in global space
                let mut gx = 0f32;
                let mut gy = 0f32;
                let mut gz = 0f32;
                for j in 0..l {
                    let a = attn[[i, j, h]];
                    gx += a * v_pts[j][h][p][0];
                    gy += a * v_pts[j][h][p][1];
                    gz += a * v_pts[j][h][p][2];
                }
                // Rotate to frame i local (no translation subtraction — norm is invariant anyway)
                let local = frames[i].invert_apply([gx, gy, gz]);
                let idx = h * N_PTS_V + p;
                out_pts_xyz[[i, idx * 3    ]] = local[0];
                out_pts_xyz[[i, idx * 3 + 1]] = local[1];
                out_pts_xyz[[i, idx * 3 + 2]] = local[2];
                out_pts_norm[[i, idx]] = (local[0]*local[0] + local[1]*local[1] + local[2]*local[2]).sqrt();
            }
        }
    }

    // ── aggregate: pair features → per-head weighted sum ───────────────────
    // out_pair[i, h, k] = Σ_j attn[i,j,h] * z[i,j,k]
    let mut out_pair = Array2::<f32>::zeros((l, N_HEADS * PAIR_DIM));
    for i in 0..l {
        for h in 0..N_HEADS {
            for k in 0..PAIR_DIM {
                let mut acc = 0f32;
                for j in 0..l {
                    acc += attn[[i, j, h]] * z[[i, j, k]];
                }
                out_pair[[i, h * PAIR_DIM + k]] = acc;
            }
        }
    }

    // ── concatenate all outputs and project ─────────────────────────────────
    // [L, 192 + 288 + 96 + 1536] = [L, 2112]  → [L, 384]
    let mut concat = Array2::<f32>::zeros((l, N_HEADS * SCALAR_DIM + n_pts_out * 3 + n_pts_out + N_HEADS * PAIR_DIM));
    let c1 = N_HEADS * SCALAR_DIM;
    let c2 = c1 + n_pts_out * 3;
    let c3 = c2 + n_pts_out;
    // let c4 = c3 + N_HEADS * PAIR_DIM;  // = 2112

    concat.slice_mut(s![.., 0..c1]).assign(&out_scalar);
    concat.slice_mut(s![.., c1..c2]).assign(&out_pts_xyz);
    concat.slice_mut(s![.., c2..c3]).assign(&out_pts_norm);
    concat.slice_mut(s![.., c3..]).assign(&out_pair);

    // Output projection [2112, 384]
    let wout = w2(params, &format!("{IPA}output_projection//weights"))?;
    let bout = w1(params, &format!("{IPA}output_projection//bias"))?;
    Ok(linear(&concat, &wout, &bout))
}

// ── structure module entry point ──────────────────────────────────────────────

/// Run the Structure Module to predict Cα co-ordinates from Evoformer output.
///
/// # What it does
/// Starts with a blank backbone (every residue at the origin with no rotation),
/// then runs 8 iterations of the fold-iteration block.  Each iteration uses
/// Invariant Point Attention (IPA) to let every residue consider the 3-D
/// positions of all other residues, then updates its backbone frame accordingly.
/// After 8 iterations the frame translation vectors are the predicted Cα
/// positions.
///
/// # Inputs
/// * `evo`    — output of the Evoformer (single \[L, 384\], pair \[L, L, 128\])
/// * `params` — full parameter map loaded from the checkpoint
///
/// # Output
/// `StructureOutput` containing the Cα co-ordinates and the final single rep.
///
/// # Example
/// For HP36 (L=35) you get a `Vec<[f32;3]>` of 35 positions in Ångströms.
/// Comparing these to the PDB 2F4K Cα atoms gives the RMSD quality metric.
pub fn run(
    evo: &EvoformerOutput,
    params: &HashMap<String, Tensor>,
) -> Result<StructureOutput> {
    let l = evo.single.nrows();

    // ── initial per-residue norm + projection ─────────────────────────────
    let norm_s  = w1(params, &format!("{SM}single_layer_norm//scale"))?;
    let norm_o  = w1(params, &format!("{SM}single_layer_norm//offset"))?;
    let wproj   = w2(params, &format!("{SM}initial_projection//weights"))?;
    let bproj   = w1(params, &format!("{SM}initial_projection//bias"))?;

    let normed = layer_norm(&evo.single, &norm_s, &norm_o);
    let mut single = relu2(linear(&normed, &wproj, &bproj)); // [L, 384]

    // Normalise the pair representation once (shared across iterations)
    let pnorm_s = w1(params, &format!("{SM}pair_layer_norm//scale"))?;
    let pnorm_o = w1(params, &format!("{SM}pair_layer_norm//offset"))?;
    let pair_normed = {
        // pair is [L, L, 128]; normalise each (i,j) slice of 128 independently.
        let mut p2 = evo.pair.clone();
        for i in 0..l {
            let tmp = p2
                .slice(s![i, .., ..])
                .to_owned();
            // treat each row j as a vector of 128 features and layer-norm it
            let normed_row = layer_norm(&tmp, &pnorm_s, &pnorm_o);
            p2.slice_mut(s![i, .., ..]).assign(&normed_row);
        }
        p2
    };

    // ── load fold-iteration weights ────────────────────────────────────────
    // These are shared: the same weights are used in all 8 iterations.
    let attn_ln_s = w1(params, &format!("{FI}attention_layer_norm//scale"))?;
    let attn_ln_o = w1(params, &format!("{FI}attention_layer_norm//offset"))?;

    let trans_ln_s = w1(params, &format!("{FI}transition_layer_norm//scale"))?;
    let trans_ln_o = w1(params, &format!("{FI}transition_layer_norm//offset"))?;

    // Transition MLP (3 layers)
    let wt0 = w2(params, &format!("{FI}transition//weights"))?;
    let bt0 = w1(params, &format!("{FI}transition//bias"))?;
    let wt1 = w2(params, &format!("{FI}transition_1//weights"))?;
    let bt1 = w1(params, &format!("{FI}transition_1//bias"))?;
    let wt2 = w2(params, &format!("{FI}transition_2//weights"))?;
    let bt2 = w1(params, &format!("{FI}transition_2//bias"))?;

    // Backbone update: [384, 6]
    // The checkpoint has both `affine_update` and `quat_rigid/rigid` — they are
    // the same layer under different names from different AlphaFold releases.
    // We prefer `affine_update` and fall back to `quat_rigid/rigid`.
    let (waf, baf) = if params.contains_key(&format!("{FI}affine_update//weights")) {
        (
            w2(params, &format!("{FI}affine_update//weights"))?,
            w1(params, &format!("{FI}affine_update//bias"))?,
        )
    } else {
        (
            w2(params, &format!("{FI}quat_rigid/rigid//weights"))?,
            w1(params, &format!("{FI}quat_rigid/rigid//bias"))?,
        )
    };

    // ── initialise backbone frames to identity ────────────────────────────
    let mut frames: Vec<Frame> = (0..l).map(|_| Frame::identity()).collect();

    // ── 8 fold iterations ─────────────────────────────────────────────────
    for _iter in 0..NUM_ITER {
        // 1. Layer-norm single before IPA
        let s_normed = layer_norm(&single, &attn_ln_s, &attn_ln_o);

        // 2. IPA
        let ipa_out = ipa(&s_normed, &pair_normed, &frames, params)?;

        // 3. Residual add + layer-norm (reuse attention_layer_norm weights)
        single = single + &ipa_out;
        single = layer_norm(&single, &attn_ln_s, &attn_ln_o);

        // 4. Transition MLP: three ReLU layers, then residual add
        let t0 = relu2(linear(&single, &wt0, &bt0));
        let t1 = relu2(linear(&t0,     &wt1, &bt1));
        let t2 =       linear(&t1,     &wt2, &bt2);
        single = single + &t2;

        // 5. Transition layer-norm
        single = layer_norm(&single, &trans_ln_s, &trans_ln_o);

        // 6. Backbone update: produce [L, 6] delta, update frames
        let delta = linear(&single, &waf, &baf); // [L, 6]
        update_frames(&mut frames, &delta);
    }

    // ── extract Cα positions from frame translations ──────────────────────
    let ca_coords: Vec<[f32; 3]> = frames.iter().map(|f| f.t).collect();

    Ok(StructureOutput { ca_coords, single })
}
