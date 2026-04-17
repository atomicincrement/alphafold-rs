//! Download AlphaFold model parameters with caching and integrity checking.
//!
//! The default target is the EBI-hosted AlphaFold model_1_ptm params (~3.5 GB).
//! Supply `--model-path` on the command line to use a local file and skip the
//! network entirely.

use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use sha2::{Digest, Sha256};

/// Default download URL — AlphaFold2 model_1_ptm params (CC BY 4.0).
pub const DEFAULT_URL: &str =
    "https://storage.googleapis.com/alphafold/alphafold_params_colab_2022-12-06.tar";

/// Known SHA-256 of the colab params tarball (hex, lowercase).
/// Set to empty string to skip verification (useful while iterating).
pub const EXPECTED_SHA256: &str =
    ""; // fill in after first successful download

/// Returns the default local cache path: `~/.cache/alphafold-rs/params.npz`.
pub fn default_cache_path() -> Result<PathBuf> {
    let home = std::env::var("HOME").context("HOME env var not set")?;
    let dir = PathBuf::from(home).join(".cache").join("alphafold-rs");
    fs::create_dir_all(&dir)
        .with_context(|| format!("creating cache dir {}", dir.display()))?;
    Ok(dir.join("params.tar"))
}

/// Ensure the model params file exists locally, downloading if necessary.
///
/// - If `path` is `Some`, that path is used as-is (no download).
/// - If `path` is `None`, the default cache location is checked first; if
///   absent the file is streamed from [`DEFAULT_URL`].
///
/// Returns the path to the local params file.
pub fn ensure_model(path: Option<&Path>) -> Result<PathBuf> {
    if let Some(p) = path {
        if !p.exists() {
            bail!("--model-path '{}' does not exist", p.display());
        }
        println!("Using local model at {}", p.display());
        return Ok(p.to_path_buf());
    }

    let cache = default_cache_path()?;
    if cache.exists() {
        println!("Found cached model at {}", cache.display());
        maybe_verify(&cache)?;
        return Ok(cache);
    }

    eprintln!(
        "WARNING: No small AlphaFold checkpoint is publicly available (<500 MB).\n\
         The default download is ~3.5 GB. Consider using --model-path with a\n\
         locally available file to skip this download.\n\
         Downloading from:\n  {DEFAULT_URL}\n"
    );

    download(DEFAULT_URL, &cache)?;
    maybe_verify(&cache)?;
    Ok(cache)
}

/// Stream `url` to `dest`, showing a progress bar.
fn download(url: &str, dest: &Path) -> Result<()> {
    let client = reqwest::blocking::Client::builder()
        .use_rustls_tls()
        .build()
        .context("building HTTP client")?;

    let response = client
        .get(url)
        .send()
        .with_context(|| format!("GET {url}"))?
        .error_for_status()
        .with_context(|| format!("HTTP error for {url}"))?;

    let total = response.content_length();

    let pb = ProgressBar::new(total.unwrap_or(0));
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] \
             {bytes}/{total_bytes} ({eta})",
        )
        .unwrap()
        .progress_chars("#>-"),
    );

    let tmp = dest.with_extension("tmp");
    {
        let mut file = fs::File::create(&tmp)
            .with_context(|| format!("creating {}", tmp.display()))?;

        let mut reader = pb.wrap_read(response);
        let mut buf = vec![0u8; 1024 * 1024]; // 1 MiB chunks
        loop {
            let n = reader.read(&mut buf).context("reading response body")?;
            if n == 0 {
                break;
            }
            file.write_all(&buf[..n]).context("writing to temp file")?;
        }
    }

    pb.finish_with_message("download complete");
    fs::rename(&tmp, dest)
        .with_context(|| format!("renaming {} → {}", tmp.display(), dest.display()))?;
    println!("Saved to {}", dest.display());
    Ok(())
}

/// Verify the SHA-256 of `path` if [`EXPECTED_SHA256`] is non-empty.
fn maybe_verify(path: &Path) -> Result<()> {
    if EXPECTED_SHA256.is_empty() {
        return Ok(());
    }

    print!("Verifying SHA-256… ");
    io::stdout().flush().ok();

    let mut file = fs::File::open(path)
        .with_context(|| format!("opening {} for verification", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buf).context("reading file for hash")?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let got = hex::encode(hasher.finalize());

    if got != EXPECTED_SHA256 {
        bail!(
            "SHA-256 mismatch for {}:\n  expected {EXPECTED_SHA256}\n  got      {got}",
            path.display()
        );
    }
    println!("OK");
    Ok(())
}
