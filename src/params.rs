//! Load AlphaFold model parameters from `.npz` or `.tar` (containing `.npz`)
//! archives into a map of named `ndarray::ArrayD<f32>` tensors.
//!
//! ## Format chain
//! ```text
//! params.tar
//!   └── params_model_1.npz   (zip archive)
//!         ├── alphafold/common/...  (no extension — .npy data)
//!         └── …
//! ```
//!
//! The `.npy` binary format v1.0/v2.0 spec:
//! <https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html>

use std::collections::HashMap;
use std::io::{Cursor, Read};
use std::path::Path;

use anyhow::{bail, Context, Result};
use ndarray::ArrayD;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// A parameter tensor together with its original on-disk dtype descriptor
/// (e.g. `"<f4"`, `">f2"`, `"<i4"`).
#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: ArrayD<f32>,
    /// NumPy dtype string as it appeared in the `.npy` header.
    pub dtype: String,
}

/// Load all parameter tensors from `path`.
///
/// `path` may be:
/// - A `.npz` file (zip archive of `.npy` entries), or
/// - A `.tar` file whose entries include one or more `.npz` files.
///
/// All arrays are cast to `f32` regardless of their on-disk dtype.
pub fn load(path: &Path) -> Result<HashMap<String, Tensor>> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    let bytes = std::fs::read(path)
        .with_context(|| format!("reading {}", path.display()))?;

    match ext.as_str() {
        "npz" => load_npz(&bytes),
        "tar" => load_tar(&bytes),
        _ => {
            // Try tar first, then npz.
            load_tar(&bytes).or_else(|_| load_npz(&bytes))
        }
    }
}

/// Pretty-print a summary table of all loaded tensors
/// (name, dtype, shape, elements).
pub fn print_summary(params: &HashMap<String, Tensor>) {
    let mut names: Vec<&String> = params.keys().collect();
    names.sort();

    let name_w = names.iter().map(|n| n.len()).max().unwrap_or(4).max(4);
    let dtype_w = params
        .values()
        .map(|t| t.dtype.len())
        .max()
        .unwrap_or(5)
        .max(5);
    let shape_w = 30usize;

    println!(
        "\n{:<name_w$}  {:<dtype_w$}  {:<shape_w$}  {:>12}",
        "Name", "Dtype", "Shape", "Elements"
    );
    println!("{}", "-".repeat(name_w + 2 + dtype_w + 2 + shape_w + 2 + 12));

    let mut total_elements: usize = 0;
    for name in &names {
        let tensor = &params[*name];
        let shape_str = format!("{:?}", tensor.data.shape());
        let elems = tensor.data.len();
        total_elements += elems;
        println!(
            "{:<name_w$}  {:<dtype_w$}  {:<shape_w$}  {:>12}",
            name, tensor.dtype, shape_str, elems
        );
    }

    println!("{}", "-".repeat(name_w + 2 + dtype_w + 2 + shape_w + 2 + 12));
    println!(
        "{:<name_w$}  {:<dtype_w$}  {:<shape_w$}  {:>12}",
        format!("{} tensors", names.len()),
        "",
        "",
        total_elements
    );
    println!();
}

// ---------------------------------------------------------------------------
// Tar → npz → npy
// ---------------------------------------------------------------------------

fn load_tar(bytes: &[u8]) -> Result<HashMap<String, Tensor>> {
    let mut archive = tar::Archive::new(Cursor::new(bytes));
    let mut params: HashMap<String, Tensor> = HashMap::new();
    let mut found = 0usize;

    for entry in archive.entries().context("iterating tar entries")? {
        let mut entry = entry.context("reading tar entry")?;
        let path = entry
            .path()
            .context("tar entry path")?
            .to_string_lossy()
            .to_string();

        if path.ends_with(".npz") {
            found += 1;
            let mut buf = Vec::new();
            entry.read_to_end(&mut buf).context("reading npz from tar")?;
            let subset = load_npz(&buf)
                .with_context(|| format!("parsing {path}"))?;
            params.extend(subset);
        }
    }

    if found == 0 {
        bail!("No .npz files found inside tar archive");
    }
    Ok(params)
}

fn load_npz(bytes: &[u8]) -> Result<HashMap<String, Tensor>> {
    let reader = Cursor::new(bytes);
    let mut zip = zip::ZipArchive::new(reader).context("opening npz as zip")?;
    let mut params = HashMap::new();

    for i in 0..zip.len() {
        let mut file = zip.by_index(i).with_context(|| format!("zip entry {i}"))?;
        let raw_name = file.name().to_owned();

        // Strip leading path components and trailing ".npy" extension.
        let tensor_name = raw_name
            .trim_end_matches(".npy")
            .replace('\\', "/");

        let mut buf = Vec::with_capacity(file.size() as usize);
        file.read_to_end(&mut buf)
            .with_context(|| format!("reading zip entry {raw_name}"))?;

        let tensor = parse_npy(&buf)
            .with_context(|| format!("parsing .npy for '{raw_name}'"))?;
        params.insert(tensor_name, tensor);
    }

    Ok(params)
}

// ---------------------------------------------------------------------------
// .npy parser
// ---------------------------------------------------------------------------

const MAGIC: &[u8] = b"\x93NUMPY";

fn parse_npy(data: &[u8]) -> Result<Tensor> {
    if data.len() < 10 || !data.starts_with(MAGIC) {
        bail!("not a .npy file (bad magic)");
    }

    let major = data[6];
    let _minor = data[7];

    let (header_len, data_start) = match major {
        1 => {
            let len = u16::from_le_bytes([data[8], data[9]]) as usize;
            (len, 10 + len)
        }
        2 | 3 => {
            if data.len() < 12 {
                bail!("truncated npy v2 header");
            }
            let len = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
            (len, 12 + len)
        }
        v => bail!("unsupported .npy major version {v}"),
    };

    let header_bytes = data
        .get(data_start - header_len..data_start)
        .context("header slice out of bounds")?;
    let header = std::str::from_utf8(header_bytes)
        .context("npy header is not UTF-8")?;

    let (dtype, fortran_order, shape) = parse_header(header)
        .with_context(|| format!("parsing header: {header:?}"))?;

    let body = data
        .get(data_start..)
        .context("data slice out of bounds")?;

    let data = cast_to_f32(body, &dtype, fortran_order, &shape)?;
    Ok(Tensor { data, dtype })
}

/// Parse the Python dict-literal header string.
/// Returns (dtype_str, fortran_order, shape).
fn parse_header(header: &str) -> Result<(String, bool, Vec<usize>)> {
    let descr = extract_str_value(header, "descr")
        .context("missing 'descr' in npy header")?;
    let fortran = extract_bool_value(header, "fortran_order")
        .context("missing 'fortran_order' in npy header")?;
    let shape = extract_shape(header)
        .context("missing 'shape' in npy header")?;
    Ok((descr, fortran, shape))
}

fn extract_str_value(header: &str, key: &str) -> Option<String> {
    // Matches  'key': 'value'  or  'key': "value"
    let search = format!("'{key}'");
    let start = header.find(&search)? + search.len();
    let rest = header[start..].trim_start_matches([' ', ':'].as_slice()).trim_start();
    let quote = rest.chars().next()?;
    if quote != '\'' && quote != '"' {
        return None;
    }
    let inner = &rest[1..];
    let end = inner.find(quote)?;
    Some(inner[..end].to_owned())
}

fn extract_bool_value(header: &str, key: &str) -> Option<bool> {
    let search = format!("'{key}'");
    let start = header.find(&search)? + search.len();
    let rest = header[start..].trim_start_matches([' ', ':'].as_slice()).trim_start();
    if rest.starts_with("True") {
        Some(true)
    } else if rest.starts_with("False") {
        Some(false)
    } else {
        None
    }
}

fn extract_shape(header: &str) -> Option<Vec<usize>> {
    let key = "'shape'";
    let start = header.find(key)? + key.len();
    let rest = header[start..].trim_start_matches([' ', ':'].as_slice()).trim_start();
    // rest now starts with '(' … ')'
    let lparen = rest.find('(')?;
    let rparen = rest.find(')')?;
    let inner = rest[lparen + 1..rparen].trim();
    if inner.is_empty() {
        return Some(vec![]); // scalar
    }
    let dims: Result<Vec<usize>, _> = inner
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>())
        .collect();
    dims.ok()
}

// ---------------------------------------------------------------------------
// Dtype → f32 casting
// ---------------------------------------------------------------------------

fn cast_to_f32(
    body: &[u8],
    dtype: &str,
    fortran_order: bool,
    shape: &[usize],
) -> Result<ArrayD<f32>> {
    // Strip endian prefix for matching: '<', '>', '=', '|'
    let (endian, kind_str) = if dtype.starts_with(['<', '>', '=', '|']) {
        (&dtype[..1], &dtype[1..])
    } else {
        ("<", dtype)
    };
    let big_endian = endian == ">";

    let n_elements: usize = shape.iter().product::<usize>().max(1);

    macro_rules! read_primitives {
        ($t:ty, $size:expr, $conv:expr) => {{
            let expected = n_elements * $size;
            if body.len() < expected {
                bail!(
                    "body too short: expected {} bytes, got {}",
                    expected,
                    body.len()
                );
            }
            let mut out = Vec::with_capacity(n_elements);
            for chunk in body[..expected].chunks_exact($size) {
                let arr: [u8; $size] = chunk.try_into().unwrap();
                let v: $t = if big_endian {
                    <$t>::from_be_bytes(arr)
                } else {
                    <$t>::from_le_bytes(arr)
                };
                out.push($conv(v));
            }
            out
        }};
    }

    let f32_data: Vec<f32> = match kind_str {
        "f4" => {
            read_primitives!(f32, 4, |v| v)
        }
        "f8" => {
            read_primitives!(f64, 8, |v: f64| v as f32)
        }
        "f2" => {
            // float16: manual decode
            let expected = n_elements * 2;
            if body.len() < expected {
                bail!("body too short for f16");
            }
            body[..expected]
                .chunks_exact(2)
                .map(|c| {
                    let bits = if big_endian {
                        u16::from_be_bytes([c[0], c[1]])
                    } else {
                        u16::from_le_bytes([c[0], c[1]])
                    };
                    f16_to_f32(bits)
                })
                .collect()
        }
        "i4" => {
            read_primitives!(i32, 4, |v: i32| v as f32)
        }
        "i2" => {
            read_primitives!(i16, 2, |v: i16| v as f32)
        }
        "i8" => {
            read_primitives!(i64, 8, |v: i64| v as f32)
        }
        "u1" | "b1" => body[..n_elements]
            .iter()
            .map(|&b| b as f32)
            .collect(),
        "u4" => {
            read_primitives!(u32, 4, |v: u32| v as f32)
        }
        "u8" => {
            read_primitives!(u64, 8, |v: u64| v as f32)
        }
        other => bail!("unsupported dtype '{dtype}' (kind '{other}')"),
    };

    // Build ndarray with the right shape.
    let shape_usize: Vec<usize> = if shape.is_empty() {
        vec![1]
    } else {
        shape.to_vec()
    };

    let array = if fortran_order {
        // Fortran (column-major) layout: the data is stored fastest-first in
        // the *first* axis, which is the opposite of ndarray's C order.  We
        // create the array with a reversed shape then transpose.
        let rev_shape: Vec<usize> = shape_usize.iter().copied().rev().collect();
        let a = ArrayD::from_shape_vec(rev_shape, f32_data)
            .context("building ndarray from Fortran-order data")?;
        // Reverse the axis order to get back to the original shape.
        let n = a.ndim();
        let axes: Vec<usize> = (0..n).rev().collect();
        a.permuted_axes(axes)
            .as_standard_layout()
            .to_owned()
    } else {
        ArrayD::from_shape_vec(shape_usize, f32_data)
            .context("building ndarray from C-order data")?
    };

    Ok(array)
}

/// Minimal IEEE 754 half-precision → single-precision conversion.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    let f32_bits = if exp == 0 {
        if mant == 0 {
            sign // ±0
        } else {
            // Subnormal
            let mut m = mant;
            let mut e = 127u32 - 14;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e) << 23) | ((m & 0x3FF) << 13)
        }
    } else if exp == 0x1F {
        sign | 0x7F80_0000 | (mant << 13) // inf / NaN
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(f32_bits)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal v1.0 .npy file from scratch and check round-trip.
    fn make_npy_v1(dtype: &str, shape: &[usize], data_bytes: Vec<u8>) -> Vec<u8> {
        let shape_str = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let shape_tuple = if shape.len() == 1 {
            format!("({shape_str},)")
        } else {
            format!("({shape_str})")
        };
        let header_str = format!(
            "{{'descr': '{dtype}', 'fortran_order': False, 'shape': {shape_tuple}, }}"
        );
        // Pad header to a multiple of 64 bytes (spec requires multiple of 64 for v1).
        let prefix = 10usize; // magic(6) + ver(2) + len(2)
        let header_len_needed = {
            let raw = header_str.len() + 1; // +1 for '\n' terminator
            // Round up to multiple of 64
            ((raw + prefix + 63) / 64) * 64 - prefix
        };
        let mut padded = header_str.into_bytes();
        while padded.len() < header_len_needed - 1 {
            padded.push(b' ');
        }
        padded.push(b'\n');

        let mut out = Vec::new();
        out.extend_from_slice(b"\x93NUMPY");
        out.push(1); // major
        out.push(0); // minor
        let hl = padded.len() as u16;
        out.extend_from_slice(&hl.to_le_bytes());
        out.extend_from_slice(&padded);
        out.extend_from_slice(&data_bytes);
        out
    }

    #[test]
    fn round_trip_f32_1d() {
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let npy = make_npy_v1("<f4", &[4], bytes);

        let tensor = parse_npy(&npy).unwrap();
        assert_eq!(tensor.dtype, "<f4");
        assert_eq!(tensor.data.shape(), &[4]);
        let got: Vec<f32> = tensor.data.iter().copied().collect();
        assert_eq!(got, values);
    }

    #[test]
    fn round_trip_f32_2d() {
        let values: Vec<f32> = (0..6).map(|i| i as f32 * 0.5).collect();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let npy = make_npy_v1("<f4", &[2, 3], bytes);

        let tensor = parse_npy(&npy).unwrap();
        assert_eq!(tensor.dtype, "<f4");
        assert_eq!(tensor.data.shape(), &[2, 3]);
        let got: Vec<f32> = tensor.data.iter().copied().collect();
        assert_eq!(got, values);
    }

    #[test]
    fn round_trip_f64_cast_to_f32() {
        let values: Vec<f64> = vec![1.5, 2.5];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let npy = make_npy_v1("<f8", &[2], bytes);

        let tensor = parse_npy(&npy).unwrap();
        assert_eq!(tensor.dtype, "<f8");
        assert_eq!(tensor.data[[0]], 1.5f32);
        assert_eq!(tensor.data[[1]], 2.5f32);
    }

    #[test]
    fn round_trip_f16() {
        // Encode 1.0 and 2.0 in float16.
        // 1.0_f16 = 0x3C00, 2.0_f16 = 0x4000
        let bytes: Vec<u8> = vec![0x00, 0x3C, 0x00, 0x40];
        let npy = make_npy_v1("<f2", &[2], bytes);

        let tensor = parse_npy(&npy).unwrap();
        assert_eq!(tensor.dtype, "<f2");
        assert!((tensor.data[[0]] - 1.0f32).abs() < 1e-3);
        assert!((tensor.data[[1]] - 2.0f32).abs() < 1e-3);
    }

    #[test]
    fn round_trip_fortran_order() {
        // 2×3 matrix in Fortran order:
        // storage: col0=[1,4], col1=[2,5], col2=[3,6]
        // logical: [[1,2,3],[4,5,6]]
        let storage: Vec<f32> = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let bytes: Vec<u8> = storage.iter().flat_map(|v| v.to_le_bytes()).collect();

        let shape_str = "(2, 3)";
        let header_str = format!(
            "{{'descr': '<f4', 'fortran_order': True, 'shape': {shape_str}, }}"
        );
        let prefix = 10usize;
        let header_len_needed = {
            let raw = header_str.len() + 1;
            ((raw + prefix + 63) / 64) * 64 - prefix
        };
        let mut padded = header_str.into_bytes();
        while padded.len() < header_len_needed - 1 {
            padded.push(b' ');
        }
        padded.push(b'\n');

        let mut npy = Vec::new();
        npy.extend_from_slice(b"\x93NUMPY");
        npy.push(1);
        npy.push(0);
        let hl = padded.len() as u16;
        npy.extend_from_slice(&hl.to_le_bytes());
        npy.extend_from_slice(&padded);
        npy.extend_from_slice(&bytes);

        let tensor = parse_npy(&npy).unwrap();
        assert_eq!(tensor.dtype, "<f4");
        assert_eq!(tensor.data.shape(), &[2, 3]);
        // Row 0 should be [1, 2, 3], row 1 should be [4, 5, 6].
        assert_eq!(tensor.data[[0, 0]], 1.0);
        assert_eq!(tensor.data[[0, 1]], 2.0);
        assert_eq!(tensor.data[[0, 2]], 3.0);
        assert_eq!(tensor.data[[1, 0]], 4.0);
        assert_eq!(tensor.data[[1, 1]], 5.0);
        assert_eq!(tensor.data[[1, 2]], 6.0);
    }

    #[test]
    fn bad_magic_rejected() {
        let result = parse_npy(b"NOTNUMPY\x00\x00");
        assert!(result.is_err());
    }
}
