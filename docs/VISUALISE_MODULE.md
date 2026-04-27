# Visualize Module Documentation

## Overview

The `visualise` module provides an interactive 3D visualization of predicted protein structures using Bevy 3D engine.

## Features

- **Interactive 3D Viewer**: Orbit camera with mouse controls
- **Structure Rendering**: 
  - Cα atoms as colored spheres
  - Backbone bonds as cylinders connecting adjacent atoms
- **Rainbow Coloring**: N-terminus (blue) → C-terminus (red) gradient
- **Reference Overlay**: Semi-transparent grey structure for experimental comparison
- **Confidence Display**: Optional per-residue pLDDT scores

## Usage

### Compilation

To enable the visualizer, compile with the `visualise` feature:

```bash
cargo build --release --features visualise
```

Without the feature flag, the module provides a no-op stub that prints status to stderr.

### Code Example

```rust
use alphafold_rs::visualise::VisualizerConfig;

let config = VisualizerConfig {
    predicted_coords: vec![
        [0.225, -0.245, 0.497],
        [0.364, -0.209, 0.498],
        // ... more Cα coordinates
    ],
    reference_coords: Some(vec![
        [14.234, -1.548, 57.370],
        [13.573, -5.175, 56.555],
        // ... PDB coordinates
    ]),
    sequence: "LSDEDФKAVFGMTRSAFANLPLWKQQNLKKEKGLF".to_string(),
    plddt_scores: Some(vec![85.0, 88.0, 82.0, /* ... */]),
    rmsd: Some(2.15),
};

visualise::visualize(config);
```

## Interactive Controls

| Control | Action |
|---------|--------|
| **Mouse Drag** | Orbit camera around structure |
| **Mouse Scroll** | Zoom in/out |
| **ESC** | Close window |

## Visual Elements

### Predicted Structure (Colored)
- **Spheres**: Cα atom positions, colored by gradient
  - Radius: 0.3 Å
  - Color: Rainbow (blue to red)
  - Glow: Slight emissive to aid visibility

- **Cylinders**: Backbone bonds connecting adjacent Cα atoms
  - Radius: 0.1 Å
  - Color: Matches residue color gradient

### Reference Structure (Grey, Semi-transparent)
- **Spheres**: Reference Cα positions (if provided)
  - Radius: 0.25 Å
  - Color: Grey (60% opacity)

- **Cylinders**: Reference backbone bonds
  - Radius: 0.08 Å
  - Color: Dark grey (30% opacity)

### Lighting
- **Directional Light**: Simulates sun with angled shadows
- **Ambient Light**: Uniform illumination (50% brightness)
- **Camera**: Starts at 10 Å from origin, can be zoomed

## Structure

### `bevy_app` Module (When `visualise` Feature Enabled)

Contains the full 3D visualization implementation using Bevy:

- `VisualizerConfig`: Holds structure data
- `OrbitCamera`: Camera controller with mouse interaction
- `PredictedAtom`, `ReferenceAtom`: Component markers
- `BackboneBond`: Component for backbone connections
- `rainbow_color()`: Generates gradient colors by position
- `setup_scene()`: Spawns all 3D meshes
- `update_camera()`: Handles camera movement
- `run()`: Launches the Bevy application

### `bevy_app` Module (When `visualise` Feature Disabled)

Provides a minimal stub that prints status information to stderr:

```
Visualizer not available. Compile with --features visualise to enable.
Predicted 35 residues: LSDEDФKAVFGMTRSAFANLPLWKQQNLKKEKGLF
RMSD: 2.15 Å
```

## Dependencies

- **Bevy 0.13**: 3D rendering engine (optional)
  - Automatically excluded when `visualise` feature is disabled

## Performance Notes

- Bevy compilation takes ~1-2 minutes first time
- Runtime performance is interactive on most modern hardware
- Memory usage: ~100-200 MB for typical protein structures
- Frame rate: 60+ FPS typical

## Future Enhancements

Potential improvements not yet implemented:

1. **UI Panel**: In-app text display of sequence and metrics
2. **Per-Residue Tooltips**: Hover information on atoms
3. **Confidence Coloring**: Color by pLDDT instead of position
4. **Animation**: Playback of refinement steps
5. **Measurement Tools**: Distances between atoms
6. **Export**: Screenshot and 3D model export (OBJ/GLTF)
7. **Multiple Models**: Compare several predictions

## Troubleshooting

### "Visualizer not available" message
- Recompile with: `cargo build --features visualise`
- Bevy requires graphics libraries (OpenGL/Vulkan)

### Performance issues
- Reduce mesh detail (lower subdivisions on spheres)
- Disable reference structure overlay
- Run in release mode: `--release`

### Window doesn't appear
- Check Bevy's system requirements
- Ensure graphics drivers are up to date
- Try running with `RUST_LOG=bevy_render=debug` for diagnostics

## See Also

- Bevy Documentation: https://bevyengine.org/learn/
- AlphaFold2 Paper: https://www.nature.com/articles/s41586-021-03819-2
- PDB Format: https://www.wwpdb.org/documentation/file-format
