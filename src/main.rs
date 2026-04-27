mod evoformer;
mod fetch;
mod input;
mod params;
mod structure_module;
mod visualise;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

/// AlphaFold-rs: a dependency-light AlphaFold2 demonstrator in Rust.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Commands,

    /// Path to a local AlphaFold params file (.tar / .npz).
    /// When supplied, the network download is skipped entirely.
    #[arg(long, value_name = "FILE", global = true)]
    model_path: Option<PathBuf>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Generate predicted structure coordinates from a FASTA sequence
    Generate {
        /// Input FASTA file with protein sequence
        fasta: PathBuf,

        /// Output JSON file for coordinates
        output: PathBuf,
    },

    /// Visualize one or more coordinate files
    Visualise {
        /// Coordinate JSON files to visualize (first is predicted, rest are references)
        coords: Vec<PathBuf>,
    },
}

fn main() -> Result<()> {
    let args = Args::parse();

    match args.command {
        Commands::Generate { fasta, output } => {
            cmd_generate(&fasta, &output, args.model_path)?;
        }
        Commands::Visualise { coords } => {
            cmd_visualise(&coords)?;
        }
    }

    Ok(())
}

/// Generate predicted Cα coordinates from FASTA and save to JSON
fn cmd_generate(fasta_path: &PathBuf, output_path: &PathBuf, model_path: Option<PathBuf>) -> Result<()> {
    println!("Reading FASTA from: {}", fasta_path.display());
    let fasta_text = std::fs::read_to_string(fasta_path)?;

    let sequence = fasta_text
        .lines()
        .filter(|line| !line.starts_with('>'))
        .collect::<String>();

    println!("Sequence: {}", sequence);
    println!("Length: {} residues", sequence.len());

    let model = fetch::ensure_model(model_path.as_deref())?;
    println!("Model params ready at: {}", model.display());

    println!("Loading parameter tensors…");
    let tensors = params::load(&model)?;
    println!("Loaded {} tensors.", tensors.len());
    params::print_summary(&tensors);

    params::print_matching(&tensors, "msa_row_attention");

    // -------------------------------------------------------------------
    // Input embedding
    // -------------------------------------------------------------------
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

    // -------------------------------------------------------------------
    // Structure Module
    // -------------------------------------------------------------------
    println!("Running Structure Module (8 fold iterations)…");
    let struc = structure_module::run(&evo, &tensors)?;
    println!(
        "Structure Module done.  Final single rep: {}×{}\nPredicted Cα positions (Å):",
        struc.single.shape()[0],
        struc.single.shape()[1],
    );
    for (i, xyz) in struc.ca_coords.iter().enumerate() {
        println!("  {:>3}  {:8.3}  {:8.3}  {:8.3}", i + 1, xyz[0], xyz[1], xyz[2]);
    }

    // Save to JSON
    let json = serde_json::to_string_pretty(&struc.ca_coords)?;
    std::fs::write(&output_path, json)?;
    println!("\nSaved {} coordinates to: {}", struc.ca_coords.len(), output_path.display());

    Ok(())
}

/// Visualize one or more coordinate files
fn cmd_visualise(coord_paths: &[PathBuf]) -> Result<()> {
    if coord_paths.is_empty() {
        anyhow::bail!("At least one coordinate file required for visualise command");
    }

    println!("Loading {} coordinate file(s)…", coord_paths.len());

    // Load predicted coordinates (first file)
    let predicted_json = std::fs::read_to_string(&coord_paths[0])?;
    let predicted_coords: Vec<[f32; 3]> = serde_json::from_str(&predicted_json)?;
    println!("Predicted: {} residues from {}", predicted_coords.len(), coord_paths[0].display());

    // Load reference coordinates (remaining files)
    let mut reference_coords = None;
    if coord_paths.len() > 1 {
        let ref_json = std::fs::read_to_string(&coord_paths[1])?;
        let ref_coords: Vec<[f32; 3]> = serde_json::from_str(&ref_json)?;
        println!("Reference: {} residues from {}", ref_coords.len(), coord_paths[1].display());
        reference_coords = Some(ref_coords);

        // Show any additional coordinate files
        for (i, path) in coord_paths[2..].iter().enumerate() {
            let coords_json = std::fs::read_to_string(path)?;
            let coords: Vec<[f32; 3]> = serde_json::from_str(&coords_json)?;
            println!(
                "Extra {}: {} residues from {}",
                i + 1,
                coords.len(),
                path.display()
            );
        }
    }

    // Create visualizer config
    let config = visualise::VisualizerConfig {
        predicted_coords,
        reference_coords,
        sequence: String::new(), // No sequence info
        plddt_scores: None,
        rmsd: None,
    };

    println!("\nLaunching 3D visualiser…");
    visualise::visualize(config);

    Ok(())
}

