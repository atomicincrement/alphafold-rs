mod evoformer;
mod fetch;
mod input;
mod params;

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

/// AlphaFold-rs: a dependency-light AlphaFold2 demonstrator in Rust.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to a local AlphaFold params file (.tar / .npz).
    /// When supplied, the network download is skipped entirely.
    #[arg(long, value_name = "FILE")]
    model_path: Option<PathBuf>,

    /// FASTA file (or string) to embed. If not supplied, a short demo
    /// sequence (HP36 villin headpiece) is used.
    #[arg(long, value_name = "FASTA")]
    fasta: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let model = fetch::ensure_model(args.model_path.as_deref())?;
    println!("Model params ready at: {}", model.display());

    println!("Loading parameter tensors…");
    let tensors = params::load(&model)?;
    println!("Loaded {} tensors.", tensors.len());
    params::print_summary(&tensors);

    // -------------------------------------------------------------------
    // Input embedding
    // -------------------------------------------------------------------
    let fasta_text = match &args.fasta {
        Some(path) => std::fs::read_to_string(path)?,
        None => {
            println!("No --fasta supplied; using HP36 villin headpiece demo.");
            ">HP36\nLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF\n".to_owned()
        }
    };

    println!("Embedding sequence…");
    let inputs = input::encode_fasta(&fasta_text, &tensors)?;
    println!(
        "Sequence length: {L}\n\
         single : {L}×{s}\n\
         pair   : {L}×{L}×{p}\n\
         msa    : 1×{L}×{m}\n\
         extra  : 1×{L}×{e}",
        L = inputs.len,
        s = inputs.single.shape()[1],
        p = inputs.pair.shape()[2],
        m = inputs.msa.shape()[2],
        e = inputs.extra_msa.shape()[2],
    );

    // -------------------------------------------------------------------
    // Evoformer stack
    // -------------------------------------------------------------------
    println!("Running Evoformer (3 recycles × 48 blocks)…");
    let evo = evoformer::run(&inputs, &tensors)?;
    println!(
        "Evoformer done.\n\
         single : {}×{}\n\
         pair   : {}×{}×{}",
        evo.single.shape()[0],
        evo.single.shape()[1],
        evo.pair.shape()[0],
        evo.pair.shape()[1],
        evo.pair.shape()[2],
    );

    Ok(())
}

