//! Input pipeline: FASTA parsing → token encoding → initial single and pair
//! representations ready for the Evoformer.
//!
//! ## Single-sequence convention
//! The real model supports deep MSAs, but runs fine with depth-1 "mock MSAs".
//! We generate all embeddings that the Evoformer expects using only the query
//! sequence, setting MSA-only features (profile, deletion stats) to zero.
//!
//! ## Tensor key prefix
//! All weight lookups use the prefix
//! `alphafold/alphafold_iteration/evoformer/`
//! and fetch tensors from the `HashMap<String, params::Tensor>` returned by
//! `params::load`.

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use ndarray::{s, Array1, Array2, Array3, ArrayView1};

use crate::params::Tensor;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// AlphaFold amino-acid alphabet: 20 standard residues + unknown token.
/// Index in this slice == token id used throughout the model.
pub const RESTYPES: &[u8] = b"ARNDCQEGHILKMFPSTWYV";
/// Token id for any character not in RESTYPES (unknown / gap / X).
pub const UNK_TOKEN: usize = 20;
/// Total number of residue tokens (20 standard + UNK).
pub const NUM_AA: usize = 21;

/// Number of MSA cluster feature dimensions expected by `preprocess_msa`.
/// Layout: [one-hot AA (23)] + [has_deletion (1)] + [deletion_value (1)]
///       + [cluster profile (23)] + [deletion_mean (1)] = 49.
pub const MSA_FEAT_DIM: usize = 49;
/// One-hot size for MSA AA encoding (20 + gap + unknown + mask token).
#[allow(dead_code)]
pub const MSA_AA_DIM: usize = 23;

/// Extra-MSA feature dimensions expected by `extra_msa_activations`.
/// Layout: [one-hot AA (23)] + [has_deletion (1)] + [deletion_value (1)] = 25.
pub const EXTRA_MSA_FEAT_DIM: usize = 25;

/// Number of relative position buckets for the multimer relative encoding.
///  65 within-chain relative-position bins  (clamp(i−j+32, 0, 64))
///+  1 same_entity flag
///+  1 same_chain flag
///+  6 inter-chain relative chain-id bins   (clamp(c_i−c_j+2, 0, 5) one-hot)
/// = 73
pub const REL_POS_DIM: usize = 73;
pub const REL_POS_BINS: i32 = 32; // symmetric clip radius → bins [0, 64]
#[allow(dead_code)]
pub const CHAIN_REL_BINS: i32 = 2; // symmetric clip radius → bins [0, 4+1]=6? no: 5

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// All initial representations produced by the input pipeline, ready for the
/// Evoformer.
pub struct Inputs {
    /// Sequence length.
    pub len: usize,
    /// Per-residue token ids in [0, NUM_AA).  Shape: [L].
    #[allow(dead_code)]
    pub tokens: Array1<usize>,
    /// Initial single representation.  Shape: [L, 256].
    pub single: Array2<f32>,
    /// Initial MSA representation (depth=1).  Shape: [1, L, 256].
    pub msa: Array3<f32>,
    /// Initial extra-MSA representation (depth=1).  Shape: [1, L, 64].
    pub extra_msa: Array3<f32>,
    /// Initial pair representation.  Shape: [L, L, 128].
    pub pair: Array3<f32>,
}

// ---------------------------------------------------------------------------
// FASTA parser
// ---------------------------------------------------------------------------

/// Parse the first sequence from a FASTA file (or raw sequence string).
///
/// Lines starting with `>` are treated as headers and skipped.
/// All whitespace is stripped; the sequence is returned as uppercase bytes.
pub fn parse_fasta(text: &str) -> Result<Vec<u8>> {
    let mut seq: Vec<u8> = Vec::new();
    let mut found_header = false;

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('>') {
            if found_header && !seq.is_empty() {
                // Second sequence starts → stop.
                break;
            }
            found_header = true;
            continue;
        }
        seq.extend(line.bytes().map(|b| b.to_ascii_uppercase()));
    }

    // Allow bare sequences without a header line.
    if seq.is_empty() {
        bail!("No sequence found in FASTA input");
    }
    Ok(seq)
}

// ---------------------------------------------------------------------------
// Tokeniser
// ---------------------------------------------------------------------------

/// Convert a raw amino-acid byte sequence to token ids.
/// Unknown / non-standard residues map to `UNK_TOKEN`.
pub fn tokenise(seq: &[u8]) -> Array1<usize> {
    seq.iter()
        .map(|&aa| {
            RESTYPES
                .iter()
                .position(|&r| r == aa)
                .unwrap_or(UNK_TOKEN)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Embedding helpers
// ---------------------------------------------------------------------------

/// Dense one-hot vector of length `n_classes` for a single index.
fn one_hot(idx: usize, n_classes: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; n_classes];
    if idx < n_classes {
        v[idx] = 1.0;
    }
    v
}

/// Linear layer: `x @ W + b` where `x` has shape `[in_dim]`,
/// `W` has shape `[in_dim, out_dim]`, `b` has shape `[out_dim]`.
fn linear(
    x: ArrayView1<f32>,
    w: &Array2<f32>,
    b: &Array1<f32>,
) -> Array1<f32> {
    w.t().dot(&x) + b
}

/// Extract a 2-D view of a stored tensor, panicking with a clear message on
/// shape mismatch.
fn as_2d<'a>(t: &'a Tensor, key: &str) -> Array2<f32> {
    let d = t.data.shape();
    assert_eq!(
        d.len(),
        2,
        "Expected 2-D tensor for '{key}' but got shape {d:?}"
    );
    t.data
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap_or_else(|_| panic!("reshape failed for '{key}'"))
        .to_owned()
}

/// Extract a 1-D view of a stored tensor.
fn as_1d(t: &Tensor, key: &str) -> Array1<f32> {
    let d = t.data.shape();
    assert_eq!(
        d.len(),
        1,
        "Expected 1-D tensor for '{key}' but got shape {d:?}"
    );
    t.data
        .view()
        .into_dimensionality::<ndarray::Ix1>()
        .unwrap_or_else(|_| panic!("reshape failed for '{key}'"))
        .to_owned()
}

/// Convenience: look up a tensor and panic with its full key if absent.
fn get<'a>(params: &'a HashMap<String, Tensor>, key: &str) -> &'a Tensor {
    params
        .get(key)
        .unwrap_or_else(|| panic!("Missing param tensor '{key}'"))
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

const PREFIX: &str = "alphafold/alphafold_iteration/evoformer/";

/// Build all initial representations from a parsed, tokenised sequence.
///
/// `params` is the full map returned by `params::load`.
pub fn embed(tokens: &Array1<usize>, params: &HashMap<String, Tensor>) -> Result<Inputs> {
    let l = tokens.len();
    if l == 0 {
        bail!("Cannot embed an empty sequence");
    }

    // -----------------------------------------------------------------------
    // 1. Single representation  [L, 256]
    //    preprocess_1d: one-hot(token, 21) @ W[21,256] + b[256]
    // -----------------------------------------------------------------------
    let w_1d = as_2d(get(params, &format!("{PREFIX}preprocess_1d//weights")),
                     "preprocess_1d//weights");
    let b_1d = as_1d(get(params, &format!("{PREFIX}preprocess_1d//bias")),
                     "preprocess_1d//bias");
    let single_dim = b_1d.len(); // 256

    let mut single = Array2::<f32>::zeros((l, single_dim));
    for (i, &tok) in tokens.iter().enumerate() {
        let oh = Array1::from(one_hot(tok, NUM_AA));
        single.row_mut(i).assign(&linear(oh.view(), &w_1d, &b_1d));
    }

    // -----------------------------------------------------------------------
    // 2. MSA representation (depth=1)  [1, L, 256]
    //    preprocess_msa: 49-dim feature @ W[49,256] + b[256]
    //    In single-seq mode the 49-dim vector is:
    //      [one_hot_23(tok)] [0: has_deletion] [0: deletion_value]
    //      [one_hot_23(tok)] [0: deletion_mean]
    //    i.e. profile = one-hot of the residue itself (no real MSA).
    // -----------------------------------------------------------------------
    let w_msa = as_2d(get(params, &format!("{PREFIX}preprocess_msa//weights")),
                      "preprocess_msa//weights");
    let b_msa = as_1d(get(params, &format!("{PREFIX}preprocess_msa//bias")),
                      "preprocess_msa//bias");
    let msa_out_dim = b_msa.len(); // 256

    let mut msa = Array3::<f32>::zeros((1, l, msa_out_dim));
    for (j, &tok) in tokens.iter().enumerate() {
        let feat = build_msa_feat(tok);
        let emb = linear(feat.view(), &w_msa, &b_msa);
        msa.slice_mut(s![0, j, ..]).assign(&emb);
    }

    // -----------------------------------------------------------------------
    // 3. Extra-MSA representation (depth=1)  [1, L, 64]
    //    extra_msa_activations: 25-dim feature @ W[25,64] + b[64]
    //    25-dim = one_hot_23(tok) + [0: has_deletion] + [0: deletion_value]
    // -----------------------------------------------------------------------
    let w_extra = as_2d(
        get(params, &format!("{PREFIX}extra_msa_activations//weights")),
        "extra_msa_activations//weights",
    );
    let b_extra = as_1d(
        get(params, &format!("{PREFIX}extra_msa_activations//bias")),
        "extra_msa_activations//bias",
    );
    let extra_out_dim = b_extra.len(); // 64

    let mut extra_msa = Array3::<f32>::zeros((1, l, extra_out_dim));
    for (j, &tok) in tokens.iter().enumerate() {
        let feat = build_extra_msa_feat(tok);
        let emb = linear(feat.view(), &w_extra, &b_extra);
        extra_msa.slice_mut(s![0, j, ..]).assign(&emb);
    }

    // -----------------------------------------------------------------------
    // 4. Pair representation  [L, L, 128]
    //    pair[i,j] = left_single(i) + right_single(j) + relpos_encoding(i,j)
    //
    //    left_single / right_single:
    //      one_hot(tok, 21) @ W[21,128] + b[128]
    //
    //    Multimer relative encoding (73-dim → 128):
    //      relpos_73(i,j) @ W[73,128] + b[128]
    //      73 = 65 (within-chain relpos one-hot) + 1 (same_entity) +
    //           1 (same_chain) + 6 (inter-chain relpos one-hot)
    //    For single-sequence: all residues on chain 0, same entity.
    // -----------------------------------------------------------------------
    let w_left = as_2d(get(params, &format!("{PREFIX}left_single//weights")),
                       "left_single//weights");
    let b_left = as_1d(get(params, &format!("{PREFIX}left_single//bias")),
                       "left_single//bias");
    let w_right = as_2d(get(params, &format!("{PREFIX}right_single//weights")),
                        "right_single//weights");
    let b_right = as_1d(get(params, &format!("{PREFIX}right_single//bias")),
                        "right_single//bias");

    let w_rel = as_2d(
        get(
            params,
            &format!("{PREFIX}~_relative_encoding/position_activations//weights"),
        ),
        "~_relative_encoding/position_activations//weights",
    );
    let b_rel = as_1d(
        get(
            params,
            &format!("{PREFIX}~_relative_encoding/position_activations//bias"),
        ),
        "~_relative_encoding/position_activations//bias",
    );
    let pair_dim = b_left.len(); // 128

    // Pre-compute per-residue left and right embeddings.
    let mut left_embs = Array2::<f32>::zeros((l, pair_dim));
    let mut right_embs = Array2::<f32>::zeros((l, pair_dim));
    for (i, &tok) in tokens.iter().enumerate() {
        let oh = Array1::from(one_hot(tok, NUM_AA));
        left_embs.row_mut(i).assign(&linear(oh.view(), &w_left, &b_left));
        right_embs.row_mut(i).assign(&linear(oh.view(), &w_right, &b_right));
    }

    let mut pair = Array3::<f32>::zeros((l, l, pair_dim));
    for i in 0..l {
        for j in 0..l {
            // left + right outer-sum
            let mut v = left_embs.row(i).to_owned() + &right_embs.row(j);
            // multimer relative position encoding
            let rel_feat = build_relpos_feat(i as i32, j as i32);
            v += &linear(rel_feat.view(), &w_rel, &b_rel);
            pair.slice_mut(s![i, j, ..]).assign(&v);
        }
    }

    Ok(Inputs {
        len: l,
        tokens: tokens.clone(),
        single,
        msa,
        extra_msa,
        pair,
    })
}

/// Parse a FASTA string, tokenise, embed — all in one call.
pub fn encode_fasta(
    fasta: &str,
    params: &HashMap<String, Tensor>,
) -> Result<Inputs> {
    let seq = parse_fasta(fasta).context("parsing FASTA")?;
    let tokens = tokenise(&seq);
    embed(&tokens, params)
}

// ---------------------------------------------------------------------------
// Feature-vector builders
// ---------------------------------------------------------------------------

/// 49-dim MSA cluster feature for a single residue (single-sequence mock).
/// Layout matches AF2's `msa_feat` cluster feature:
///   [0..23]  = one-hot AA  (23 classes: 20 std + gap + unk + mask)
///   [23]     = has_deletion (0)
///   [24]     = deletion_value (0)
///   [25..48] = cluster profile = same one-hot (no real MSA)
///   [48]     = deletion_mean (0)
fn build_msa_feat(token: usize) -> Array1<f32> {
    let mut feat = Array1::<f32>::zeros(MSA_FEAT_DIM);
    // Tokens [0,20) map directly; 20 (UNK) maps to position 21 in the
    // 23-class one-hot (class 20 = gap, class 21 = UNK, class 22 = mask).
    let msa_tok = if token < 20 { token } else { 21 };
    feat[msa_tok] = 1.0;          // one-hot AA
    feat[25 + msa_tok] = 1.0;     // cluster profile  (same one-hot)
    // has_deletion [23], deletion_value [24], deletion_mean [48] stay 0.
    feat
}

/// 25-dim extra-MSA feature for a single residue.
/// Layout: [0..23] = one-hot AA (23), [23] = has_deletion, [24] = deletion_value.
fn build_extra_msa_feat(token: usize) -> Array1<f32> {
    let mut feat = Array1::<f32>::zeros(EXTRA_MSA_FEAT_DIM);
    let msa_tok = if token < 20 { token } else { 21 };
    feat[msa_tok] = 1.0;
    feat
}

/// 73-dim multimer relative position feature for residue pair (i, j).
///
/// Layout:
///   [0..65]  = one-hot relative position clamp(i−j + 32, 0, 64)
///   [65]     = same_entity  (always 1 for single-sequence input)
///   [66]     = same_chain   (always 1 for single-sequence input)
///   [67..73] = one-hot relative chain-id clamp(chain_i−chain_j + 2, 0, 5)
///              (always [1,0,0,0,0,0] for single-sequence input, i.e. rel=0)
fn build_relpos_feat(i: i32, j: i32) -> Array1<f32> {
    let mut feat = Array1::<f32>::zeros(REL_POS_DIM);
    // Within-chain relative position: 65 bins.
    let rel = (i - j + REL_POS_BINS).clamp(0, 2 * REL_POS_BINS) as usize;
    feat[rel] = 1.0;
    // same_entity and same_chain are always 1 for single-sequence.
    feat[65] = 1.0;
    feat[66] = 1.0;
    // Relative chain id: bin 2 (= chain 0 − 0 + 2 = 2 in [0,4], one-hot 6).
    let chain_rel = 2usize; // clamp(0 - 0 + 2, 0, 5)
    feat[67 + chain_rel] = 1.0;
    feat
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenise_standard() {
        let seq = b"ARNDCQEGHILKMFPSTWYV";
        let tokens = tokenise(seq);
        let expected: Vec<usize> = (0..20).collect();
        assert_eq!(tokens.to_vec(), expected);
    }

    #[test]
    fn tokenise_unk() {
        let tokens = tokenise(b"XZ-");
        assert!(tokens.iter().all(|&t| t == UNK_TOKEN));
    }

    #[test]
    fn parse_fasta_with_header() {
        let fa = ">sp|TEST|HP36\nLSDEDFK\nAVFGMTR\n";
        let seq = parse_fasta(fa).unwrap();
        assert_eq!(seq, b"LSDEDFKAVFGMTR");
    }

    #[test]
    fn parse_fasta_bare() {
        let fa = "ACDEFGH";
        let seq = parse_fasta(fa).unwrap();
        assert_eq!(seq, b"ACDEFGH");
    }

    #[test]
    fn parse_fasta_multi_only_first() {
        let fa = ">seq1\nAAAA\n>seq2\nCCCC\n";
        let seq = parse_fasta(fa).unwrap();
        assert_eq!(seq, b"AAAA");
    }

    #[test]
    fn parse_fasta_empty_fails() {
        assert!(parse_fasta("").is_err());
    }

    #[test]
    fn msa_feat_shape_and_sum() {
        let f = build_msa_feat(0); // Ala
        assert_eq!(f.len(), MSA_FEAT_DIM);
        // Two one-hot positions set (one-hot AA and cluster profile).
        assert_eq!(f.sum() as usize, 2);
    }

    #[test]
    fn extra_msa_feat_sum() {
        let f = build_extra_msa_feat(0);
        assert_eq!(f.len(), EXTRA_MSA_FEAT_DIM);
        assert_eq!(f.sum() as usize, 1);
    }

    #[test]
    fn relpos_feat_sum_and_shape() {
        let f = build_relpos_feat(0, 0);
        assert_eq!(f.len(), REL_POS_DIM);
        // rel_pos bin 32 + same_entity + same_chain + chain_rel bin 2
        assert_eq!(f.sum() as usize, 4);
        assert_eq!(f[32], 1.0); // i−j = 0 → bin 32
        assert_eq!(f[65], 1.0); // same_entity
        assert_eq!(f[66], 1.0); // same_chain
        assert_eq!(f[67 + 2], 1.0); // chain rel = 0 → bin 2
    }

    #[test]
    fn relpos_feat_clipping() {
        // Very negative offset should clamp to bin 0.
        let f = build_relpos_feat(0, 100);
        assert_eq!(f[0], 1.0);
        // Very positive offset should clamp to bin 64.
        let g = build_relpos_feat(100, 0);
        assert_eq!(g[64], 1.0);
    }
}
