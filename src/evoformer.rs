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
#[allow(dead_code)]
pub struct EvoformerOutput {
    pub single: Array2<f32>,       // [L, 384]
    pub pair: Array3<f32>,         // [L, L, 128]
    #[allow(dead_code)]
    pub msa_first_row: Array2<f32>, // [L, 256] — read by structure module
}

// ── tensor helpers ────────────────────────────────────────────────────────────

/// Fetch a slice of the stacked tensor at block index `b`.
/// `w` has shape [num_blocks, ...]; returns view of [...].
/// Get the 2-D weight matrix for block `b` from a stacked [blocks, in, out] tensor.
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

/// Get the 4-D weight [in, h, d] → as [in, h*d] for block `b`.
/// Source shape [blocks, in_dim, heads, head_dim].
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

/// [blocks, heads, head_dim] → [heads*head_dim]
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

/// [blocks, heads, head_dim, out] → [(heads*head_dim), out]
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

/// [blocks, dim] → [dim]
fn vec1(params: &HashMap<String, Tensor>, key: &str, b: usize) -> Result<Array1<f32>> {
    let t = params
        .get(key)
        .ok_or_else(|| anyhow!("missing tensor: {}", key))?;
    Ok(t.data.slice(s![b, ..]).to_owned().into_dimensionality::<ndarray::Ix1>()?)
}

/// [blocks, h, d] → [h, d]  (for feat_2d_weights)
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

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// Layer norm over the last axis.
/// `x`: [*, C], `scale`: [C], `offset`: [C].
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

/// Layer norm over last axis for a 3-D tensor [A, B, C].
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

/// Linear: [M, K] @ [K, N] + [N] → [M, N].
fn linear(x: &Array2<f32>, w: &Array2<f32>, bias: &Array1<f32>) -> Array2<f32> {
    let mut y = x.dot(w);
    for mut row in y.rows_mut() {
        row += bias;
    }
    y
}

/// Same for 3-D input [A, B, K] → [A, B, N].
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

// ── MSA row attention with pair bias ─────────────────────────────────────────
//
// For MSA depth S=1, msa has shape [S, L, C_m].
// We attend over positions (j) for each sequence (i) independently.
// Pair bias: pair [L, L, C_z] → bias per head [L, L, H_m].

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

// ── MSA transition (MLP) ──────────────────────────────────────────────────────

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

// ── Outer-product mean → pair update ─────────────────────────────────────────
//
// For depth-1 MSA: outer product of the single row with itself.
// Result: [L, L, 128].

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

// ── Triangle multiplication ───────────────────────────────────────────────────

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

// ── Triangle attention ────────────────────────────────────────────────────────
//
// starting_node: for each i, attend over j using pair[i,j,:] as seq,
//   with pair bias from pair[i,:,:] projected.
// ending_node:   for each j, attend over i using pair[i,j,:],
//   with pair bias from pair[:,j,:] projected.

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

// ── Pair transition (MLP) ─────────────────────────────────────────────────────

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

// ── Single Evoformer block ────────────────────────────────────────────────────

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

// ── Extra-MSA stack (4 blocks, dim=64) ───────────────────────────────────────
// Runs on the extra_msa [1, L, 64] representation and updates pair.
// Structure is identical to evoformer with different weight prefixes and dims.

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

// ── Recycling helper ──────────────────────────────────────────────────────────
// Apply layer-norms to prev_msa_first_row and prev_pair, then add to the
// current pair representation before the main stack.

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

/// Run the full Evoformer stack (3 recycles × 48 blocks) and return
/// single [L,384] and pair [L,L,128] representations.
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
