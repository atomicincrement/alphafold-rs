//! 3D structure visualization using Bevy.
//!
//! This module provides utilities to visualize predicted protein structures
//! alongside experimental references using an interactive 3D viewer.

pub mod bevy_app {
    use bevy::prelude::*;

    /// Configuration for the visualizer.
    #[derive(Debug, Clone)]
    pub struct VisualizerConfig {
        /// Predicted Cα positions (in local AlphaFold frame).
        pub predicted_coords: Vec<[f32; 3]>,
        /// Optional experimental reference coordinates (from PDB).
        pub reference_coords: Option<Vec<[f32; 3]>>,
        /// Amino acid sequence.
        pub sequence: String,
        /// Per-residue confidence scores (pLDDT), 0-100.
        pub plddt_scores: Option<Vec<f32>>,
        /// RMSD value to display (if computed).
        pub rmsd: Option<f32>,
    }

    /// RGB color triple.
    #[derive(Debug, Clone, Copy)]
    pub struct Color {
        pub r: f32,
        pub g: f32,
        pub b: f32,
    }

    impl Color {
        pub fn to_linear_rgb(self) -> [f32; 3] {
            [self.r, self.g, self.b]
        }
    }

    /// Rainbow gradient: blue (N-terminus) → red (C-terminus).
    pub fn rainbow_color(fraction: f32) -> Color {
        let f = fraction.clamp(0.0, 1.0);
        // Blue → Cyan → Green → Yellow → Red
        if f < 0.25 {
            // Blue → Cyan
            let t = f / 0.25;
            Color {
                r: 0.0,
                g: t,
                b: 1.0,
            }
        } else if f < 0.5 {
            // Cyan → Green
            let t = (f - 0.25) / 0.25;
            Color {
                r: 0.0,
                g: 1.0,
                b: 1.0 - t,
            }
        } else if f < 0.75 {
            // Green → Yellow
            let t = (f - 0.5) / 0.25;
            Color {
                r: t,
                g: 1.0,
                b: 0.0,
            }
        } else {
            // Yellow → Red
            let t = (f - 0.75) / 0.25;
            Color {
                r: 1.0,
                g: 1.0 - t,
                b: 0.0,
            }
        }
    }

    /// Camera controller for orbit interaction.
    #[derive(Component, Clone)]
    pub struct OrbitCamera {
        pub distance: f32,
        pub yaw: f32,
        pub pitch: f32,
    }

    impl Default for OrbitCamera {
        fn default() -> Self {
            Self {
                distance: 10.0,
                yaw: 0.0,
                pitch: 0.3,
            }
        }
    }

    /// Marker for predicted structure (blue-tinted).
    #[derive(Component)]
    pub struct PredictedAtom {
        pub residue_idx: usize,
    }

    /// Marker for reference structure (grey-tinted).
    #[derive(Component)]
    pub struct ReferenceAtom {
        pub residue_idx: usize,
    }

    /// Marker for backbone bond.
    #[derive(Component)]
    pub struct BackboneBond {
        pub residue_idx: usize,
    }

    /// Resource to hold configuration and state.
    #[derive(Resource)]
    pub struct VisualizerState {
        pub config: VisualizerConfig,
        pub show_reference: bool,
        pub show_predicted: bool,
        pub camera_orbit: OrbitCamera,
    }

    /// Resource to hold the structure centroid for camera positioning.
    #[derive(Resource, Clone, Copy)]
    pub struct StructureCentroid(pub Vec3);

    /// Create a proper UV sphere mesh.
    fn create_sphere_mesh(radius: f32) -> Mesh {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let segments = 32;  // Horizontal segments
        let rings = 16;     // Vertical rings

        // Create vertices
        for ring in 0..=rings {
            let theta = std::f32::consts::PI * ring as f32 / rings as f32;
            let sin_theta = theta.sin();
            let cos_theta = theta.cos();

            for seg in 0..=segments {
                let phi = 2.0 * std::f32::consts::PI * seg as f32 / segments as f32;
                let sin_phi = phi.sin();
                let cos_phi = phi.cos();

                let x = radius * sin_theta * cos_phi;
                let y = radius * cos_theta;
                let z = radius * sin_theta * sin_phi;

                vertices.push([x, y, z]);
            }
        }

        // Create indices
        for ring in 0..rings {
            for seg in 0..segments {
                let a = ring * (segments + 1) + seg;
                let b = a + 1;
                let c = a + (segments + 1);
                let d = c + 1;

                indices.push(a as u32);
                indices.push(c as u32);
                indices.push(b as u32);

                indices.push(b as u32);
                indices.push(c as u32);
                indices.push(d as u32);
            }
        }

        let mut mesh = Mesh::new(
            bevy::render::mesh::PrimitiveTopology::TriangleList,
            bevy::render::render_asset::RenderAssetUsages::default(),
        );
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_POSITION,
            vertices,
        );
        mesh.insert_indices(bevy::render::mesh::Indices::U32(indices));
        mesh
    }

    /// Create a cylinder mesh for backbone bonds.
    fn create_cylinder_mesh(radius: f32, height: f32) -> Mesh {
        // Create a simple cylinder mesh
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let segments = 8;
        let half_height = height / 2.0;

        // Create top and bottom circles
        for i in 0..segments {
            let angle = 2.0 * std::f32::consts::PI * i as f32 / segments as f32;
            let x = radius * angle.cos();
            let z = radius * angle.sin();

            // Top circle
            vertices.push([x, half_height, z]);
            // Bottom circle
            vertices.push([x, -half_height, z]);
        }

        // Create top and bottom center vertices
        let top_center = vertices.len() as u32;
        vertices.push([0.0, half_height, 0.0]);
        let bottom_center = vertices.len() as u32;
        vertices.push([0.0, -half_height, 0.0]);

        // Create indices for side faces
        for i in 0..segments {
            let next = (i + 1) % segments;
            let top1 = (i * 2) as u32;
            let bottom1 = (i * 2 + 1) as u32;
            let top2 = (next * 2) as u32;
            let bottom2 = (next * 2 + 1) as u32;

            // Side face
            indices.push(top1);
            indices.push(bottom1);
            indices.push(top2);
            indices.push(top2);
            indices.push(bottom1);
            indices.push(bottom2);
        }

        // Create indices for top cap
        for i in 0..segments {
            let next = (i + 1) % segments;
            indices.push(top_center);
            indices.push((next * 2) as u32);
            indices.push((i * 2) as u32);
        }

        // Create indices for bottom cap
        for i in 0..segments {
            let next = (i + 1) % segments;
            indices.push(bottom_center);
            indices.push((i * 2 + 1) as u32);
            indices.push((next * 2 + 1) as u32);
        }

        let mut mesh = Mesh::new(
            bevy::render::mesh::PrimitiveTopology::TriangleList,
            bevy::render::render_asset::RenderAssetUsages::default(),
        );
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_POSITION,
            vertices
                .iter()
                .map(|v| [v[0], v[1], v[2]])
                .collect::<Vec<_>>(),
        );
        mesh.insert_indices(bevy::render::mesh::Indices::U32(indices));
        mesh
    }

    /// Spawn the main 3D scene with predicted and optional reference structures.
    pub fn setup_scene(
        mut commands: Commands,
        mut meshes: ResMut<Assets<Mesh>>,
        mut materials: ResMut<Assets<StandardMaterial>>,
        state: Res<VisualizerState>,
    ) {
        let cfg = &state.config;
        let n_residues = cfg.predicted_coords.len();

        // Calculate centroid of predicted structure
        let mut centroid = Vec3::ZERO;
        for &coord in &cfg.predicted_coords {
            centroid += Vec3::new(coord[0], coord[1], coord[2]);
        }
        if n_residues > 0 {
            centroid /= n_residues as f32;
        }

        // Spawn predicted structure atoms (small spheres).
        if state.show_predicted {
            for (i, &coord) in cfg.predicted_coords.iter().enumerate() {
                let fraction = i as f32 / (n_residues as f32).max(1.0);
                let color = rainbow_color(fraction);

                let sphere_mesh = meshes.add(create_sphere_mesh(0.18));

                let material = materials.add(StandardMaterial {
                    base_color: bevy::render::color::Color::rgb(
                        color.r,
                        color.g,
                        color.b,
                    ),
                    emissive: bevy::render::color::Color::rgb(
                        color.r * 2.0,
                        color.g * 2.0,
                        color.b * 2.0,
                    ),
                    metallic: 0.2,
                    perceptual_roughness: 0.2,
                    ..default()
                });

                commands
                    .spawn(PbrBundle {
                        mesh: sphere_mesh,
                        material,
                        transform: Transform::from_xyz(coord[0], coord[1], coord[2]),
                        ..default()
                    })
                    .insert(PredictedAtom {
                        residue_idx: i,
                    });
            }

            // Spawn backbone bonds (cylinders connecting adjacent atoms).
            for i in 0..(n_residues - 1) {
                let start = cfg.predicted_coords[i];
                let end = cfg.predicted_coords[i + 1];

                let mid = [
                    (start[0] + end[0]) / 2.0,
                    (start[1] + end[1]) / 2.0,
                    (start[2] + end[2]) / 2.0,
                ];

                let dx = end[0] - start[0];
                let dy = end[1] - start[1];
                let dz = end[2] - start[2];
                let length = (dx * dx + dy * dy + dz * dz).sqrt();

                let fraction = i as f32 / (n_residues as f32).max(1.0);
                let color = rainbow_color(fraction);

                let cylinder_mesh = meshes.add(create_cylinder_mesh(0.04, length));

                let material = materials.add(StandardMaterial {
                    base_color: bevy::render::color::Color::rgb(
                        color.r,
                        color.g,
                        color.b,
                    ),
                    emissive: bevy::render::color::Color::rgb(
                        color.r * 2.0,
                        color.g * 2.0,
                        color.b * 2.0,
                    ),
                    metallic: 0.2,
                    perceptual_roughness: 0.2,
                    ..default()
                });

                // Position cylinder at MIDPOINT so it's centered between atoms
                let mut transform = Transform::from_xyz(mid[0], mid[1], mid[2]);
                
                // Calculate direction from start to end and create proper rotation
                let direction = Vec3::new(dx, dy, dz);
                if direction.length() > 0.0001 {
                    // Rotate so cylinder points along the direction vector
                    let forward = direction.normalize();
                    transform.rotation = Quat::from_rotation_arc(Vec3::Y, forward);
                }

                commands
                    .spawn(PbrBundle {
                        mesh: cylinder_mesh,
                        material,
                        transform,
                        ..default()
                    })
                    .insert(BackboneBond { residue_idx: i });
            }
        }

        // Spawn reference structure (if provided) as semi-transparent grey.
        if state.show_reference {
            if let Some(ref_coords) = &cfg.reference_coords {
                for (i, &coord) in ref_coords.iter().enumerate() {
                    let sphere_mesh = meshes.add(create_sphere_mesh(0.15));

                    let material = materials.add(StandardMaterial {
                        base_color: bevy::render::color::Color::rgba(0.6, 0.6, 0.6, 0.5),
                        emissive: bevy::render::color::Color::rgb(0.4, 0.4, 0.4),
                        metallic: 0.2,
                        perceptual_roughness: 0.2,
                        ..default()
                    });

                    commands
                        .spawn(PbrBundle {
                            mesh: sphere_mesh,
                            material,
                            transform: Transform::from_xyz(coord[0], coord[1], coord[2]),
                            ..default()
                        })
                        .insert(ReferenceAtom { residue_idx: i });
                }

                // Reference backbone bonds.
                for i in 0..(ref_coords.len() - 1) {
                    let start = ref_coords[i];
                    let end = ref_coords[i + 1];

                    let mid = [
                        (start[0] + end[0]) / 2.0,
                        (start[1] + end[1]) / 2.0,
                        (start[2] + end[2]) / 2.0,
                    ];

                    let dx = end[0] - start[0];
                    let dy = end[1] - start[1];
                    let dz = end[2] - start[2];
                    let length = (dx * dx + dy * dy + dz * dz).sqrt();

                    let cylinder_mesh = meshes.add(create_cylinder_mesh(0.032, length));

                    let material = materials.add(StandardMaterial {
                        base_color: bevy::render::color::Color::rgba(0.5, 0.5, 0.5, 0.5),
                        emissive: bevy::render::color::Color::rgb(0.4, 0.4, 0.4),
                        metallic: 0.2,
                        perceptual_roughness: 0.2,
                        ..default()
                    });

                    // Position at MIDPOINT - cylinder mesh is centered, so this centers the bond properly
                    let mut transform = Transform::from_xyz(mid[0], mid[1], mid[2]);
                    // Rotate cylinder to point along the bond direction
                    let direction = Vec3::new(dx, dy, dz);
                    if direction.length() > 0.0001 {
                        let forward = direction.normalize();
                        transform.rotation = Quat::from_rotation_arc(Vec3::Y, forward);
                    }

                    commands.spawn(PbrBundle {
                        mesh: cylinder_mesh,
                        material,
                        transform,
                        ..default()
                    });
                }
            }
        }

        // Spawn camera.
        let orbit = state.camera_orbit.clone();
        let cam_pos = Vec3::new(
            orbit.distance * orbit.yaw.cos() * orbit.pitch.cos(),
            orbit.distance * orbit.pitch.sin(),
            orbit.distance * orbit.yaw.sin() * orbit.pitch.cos(),
        ) + centroid;

        commands
            .spawn(Camera3dBundle {
                transform: Transform::from_translation(cam_pos)
                    .looking_at(centroid, Vec3::Y),
                ..default()
            })
            .insert(orbit);

        // Store centroid as a resource for other systems to use
        commands.insert_resource(StructureCentroid(centroid));

        // Spawn lights - significantly increased for better visibility
        commands.spawn(DirectionalLightBundle {
            directional_light: DirectionalLight {
                illuminance: 100000.0,
                shadows_enabled: false,
                ..default()
            },
            transform: Transform::from_rotation(Quat::from_euler(
                EulerRot::ZYX,
                0.0,
                std::f32::consts::PI / 4.0,
                -std::f32::consts::PI / 4.0,
            )),
            ..default()
        });

        // Add multiple point lights for better coverage
        for (px, py, pz) in &[(30.0, 30.0, 30.0), (-30.0, 20.0, 30.0), (20.0, -20.0, 30.0)] {
            commands.spawn(PointLightBundle {
                point_light: PointLight {
                    intensity: 150000.0,
                    range: 300.0,
                    shadows_enabled: false,
                    ..default()
                },
                transform: Transform::from_xyz(*px, *py, *pz),
                ..default()
            });
        }

        commands.insert_resource(AmbientLight {
            brightness: 0.5,
            ..default()
        });
    }

    /// Update orbit camera based on mouse input.
    pub fn update_camera(
        mut query: Query<(&mut Transform, &mut OrbitCamera)>,
        mut mouse_motion: EventReader<bevy::input::mouse::MouseMotion>,
        mut scroll: EventReader<bevy::input::mouse::MouseWheel>,
        centroid_res: Res<StructureCentroid>,
    ) {
        let centroid = centroid_res.0;
        for motion in mouse_motion.read() {
            for (mut transform, mut orbit) in query.iter_mut() {
                orbit.yaw -= motion.delta.x * 0.01;
                orbit.pitch = (orbit.pitch + motion.delta.y * 0.01)
                    .clamp(-std::f32::consts::PI / 2.0, std::f32::consts::PI / 2.0);

                let cam_pos = Vec3::new(
                    orbit.distance * orbit.yaw.cos() * orbit.pitch.cos(),
                    orbit.distance * orbit.pitch.sin(),
                    orbit.distance * orbit.yaw.sin() * orbit.pitch.cos(),
                ) + centroid;

                transform.translation = cam_pos;
                transform.look_at(centroid, Vec3::Y);
            }
        }

        for scroll_event in scroll.read() {
            for (mut transform, mut orbit) in query.iter_mut() {
                orbit.distance *= 1.0 - scroll_event.y * 0.1;
                orbit.distance = orbit.distance.clamp(1.0, 50.0);

                let cam_pos = Vec3::new(
                    orbit.distance * orbit.yaw.cos() * orbit.pitch.cos(),
                    orbit.distance * orbit.pitch.sin(),
                    orbit.distance * orbit.yaw.sin() * orbit.pitch.cos(),
                ) + centroid;

                transform.translation = cam_pos;
                transform.look_at(centroid, Vec3::Y);
            }
        }
    }

    /// Handle ESC key to exit the visualizer.
    pub fn handle_esc_exit(
        keyboard_input: Res<bevy::input::ButtonInput<bevy::input::keyboard::KeyCode>>,
        mut exit_events: EventWriter<bevy::app::AppExit>,
    ) {
        if keyboard_input.just_pressed(bevy::input::keyboard::KeyCode::Escape) {
            exit_events.send(bevy::app::AppExit);
        }
    }

    /// Spawn the Bevy app and run the visualizer.
    pub fn run(config: VisualizerConfig) {
        let state = VisualizerState {
            config,
            show_reference: true,
            show_predicted: true,
            camera_orbit: OrbitCamera::default(),
        };

        App::new()
            .add_plugins(DefaultPlugins)
            .insert_resource(state)
            .add_systems(Startup, setup_scene)
            .add_systems(Update, (update_camera, handle_esc_exit))
            .run();
    }
}

/// Export the public API.
pub use bevy_app::VisualizerConfig;

/// Public function to create and run a visualizer.
#[allow(dead_code)]
pub fn visualize(config: VisualizerConfig) {
    bevy_app::run(config);
}
