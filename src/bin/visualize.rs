use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotBounds, PlotPoints, Points, Polygon};
use rna::neuron::history::{HistoryFile, SnapshotKind, TrainingSnapshot};
use rna::viz::{boundary_line, class1_polygon, regression_line, WEIGHT_ZERO_THRESHOLD};

// ── Rendering mode ────────────────────────────────────────────────────────────

/// Selects the plot layout based on the number of inputs in the loaded dataset.
#[derive(Copy, Clone)]
enum Mode {
    /// Two-input dataset: scatter + decision boundary + class-1 half-plane.
    Classification2D,
    /// One-input dataset: scatter (x, d) + regression line y = w1·x + bias.
    Regression1D,
}

impl Mode {
    /// Label shown in the side panel when no geometry can be drawn yet.
    fn degenerate_label(self) -> &'static str {
        match self {
            Mode::Classification2D => "No boundary yet\n(weights ≈ 0)",
            Mode::Regression1D => "No regression line yet\n(w1 ≈ 0)",
        }
    }

    /// Warning shown below the plot when no geometry can be drawn yet.
    fn degenerate_warning(self) -> &'static str {
        match self {
            Mode::Classification2D => "⚠ No boundary yet (weights ≈ 0)",
            Mode::Regression1D => "⚠ No regression line yet (w1 ≈ 0)",
        }
    }
}

// ── App state ─────────────────────────────────────────────────────────────────

/// Generic training-history visualiser.
///
/// Reads a `history.json` produced by any of the neuron binaries (perceptron,
/// gradient, …).  The file is deserialised as `HistoryFile<f64>`, which is
/// compatible with both integer-labelled (Perceptron) and float-labelled
/// (Gradient / ADALINE) history files: `serde_json` transparently promotes
/// JSON integers to `f64` when the target type is `f64`.
struct TrainingViz {
    history: HistoryFile<f64>,
    mode: Mode,
    /// Precomputed scatter data — depends only on `history.dataset` and
    /// `history.class_threshold`, both fixed after construction, so building
    /// them once here avoids per-frame allocation.
    ///
    /// - `Classification2D`: `class0_pts` / `class1_pts` hold the two class
    ///   splits; `scatter_pts` is empty.
    /// - `Regression1D`: `scatter_pts` holds all (x, d) pairs; both class
    ///   splits are empty.
    class0_pts: Vec<[f64; 2]>,
    class1_pts: Vec<[f64; 2]>,
    scatter_pts: Vec<[f64; 2]>,
    current_idx: usize,
    playing: bool,
    speed: f64,        // snapshots per second
    last_advance: f64, // ctx.time at the last automatic step
    /// When `true`, the next frame's plot closure calls `set_plot_bounds` once,
    /// switching `egui_plot` into manual-bounds mode and clearing the flag.
    fit_requested: bool,
    /// Bounds computed from the dataset at startup, reused for "Fit to data".
    fit_bounds: PlotBounds,
}

impl TrainingViz {
    fn new(history: HistoryFile<f64>, mode: Mode) -> Self {
        const MARGIN: f64 = 0.5;
        let (mut x_min, mut x_max, mut y_min, mut y_max) = (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        );
        // A single pass covers both modes: x always comes from feature 0; y
        // comes from feature 1 (Classification2D) or the target label d
        // (Regression1D).
        for (inputs, d) in &history.dataset {
            if let Some(&x) = inputs.first() {
                x_min = x_min.min(x);
                x_max = x_max.max(x);
            }
            let y = match mode {
                Mode::Classification2D => inputs.get(1).copied(),
                Mode::Regression1D => Some(*d),
            };
            if let Some(y) = y {
                y_min = y_min.min(y);
                y_max = y_max.max(y);
            }
        }
        let fit_bounds = PlotBounds::from_min_max(
            [x_min - MARGIN, y_min - MARGIN],
            [x_max + MARGIN, y_max + MARGIN],
        );

        // color_threshold: splits dataset labels into class-0 / class-1 for
        // scatter colouring.  Computed once here — see boundary_threshold in
        // update() for the separate net-input level that positions the geometry.
        let (class0_pts, class1_pts, scatter_pts) = match mode {
            Mode::Classification2D => {
                let t = history.class_threshold;
                let c0 = history
                    .dataset
                    .iter()
                    .filter(|(_, l)| *l < t)
                    .map(|(pts, _)| [pts[0], pts[1]])
                    .collect();
                let c1 = history
                    .dataset
                    .iter()
                    .filter(|(_, l)| *l >= t)
                    .map(|(pts, _)| [pts[0], pts[1]])
                    .collect();
                (c0, c1, vec![])
            }
            Mode::Regression1D => {
                let sc = history
                    .dataset
                    .iter()
                    .filter_map(|(inputs, d)| inputs.first().map(|&x| [x, *d]))
                    .collect();
                (vec![], vec![], sc)
            }
        };

        Self {
            history,
            mode,
            class0_pts,
            class1_pts,
            scatter_pts,
            current_idx: 0,
            playing: false,
            speed: 5.0,
            last_advance: 0.0,
            fit_requested: true, // apply fit_bounds on the very first frame
            fit_bounds,
        }
    }

    fn snap(&self) -> &TrainingSnapshot<f64> {
        &self.history.snapshots[self.current_idx]
    }

    fn n_snaps(&self) -> usize {
        self.history.snapshots.len()
    }
}

// ── egui app ──────────────────────────────────────────────────────────────────

impl eframe::App for TrainingViz {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let n = self.n_snaps();

        // Automatic playback advance
        if self.playing {
            let now = ctx.input(|i| i.time);
            if now - self.last_advance >= 1.0 / self.speed {
                self.last_advance = now;
                if self.current_idx + 1 < n {
                    self.current_idx += 1;
                } else {
                    self.playing = false;
                }
            }
            ctx.request_repaint();
        }

        // clone() releases the borrow on self before the egui closures below
        // take &mut self to update playback state.
        let snap = self.snap().clone();
        let w1 = snap.weights.first().copied().unwrap_or(0.0);
        let w2 = snap.weights.get(1).copied().unwrap_or(0.0);
        let bias = snap.bias;
        // Degenerate check: in regression mode only w1 matters (w2 is absent);
        // in classification both weights must be near-zero before the boundary
        // is suppressed.
        let degenerate = match self.mode {
            Mode::Classification2D => {
                w1.abs() < WEIGHT_ZERO_THRESHOLD && w2.abs() < WEIGHT_ZERO_THRESHOLD
            }
            Mode::Regression1D => w1.abs() < WEIGHT_ZERO_THRESHOLD,
        };

        // ── side panel ───────────────────────────────────────────────────────
        egui::SidePanel::right("info_panel")
            .min_width(230.0)
            .show(ctx, |ui| {
                ui.heading("Snapshot info");
                ui.separator();

                ui.label(format!("#{} / {}", self.current_idx, n.saturating_sub(1)));
                ui.label(format!("Epoch: {}", snap.epoch));
                ui.label(format!("Sample in epoch: {}", snap.sample_index));

                ui.separator();
                ui.label("Weights:");
                for (i, w) in snap.weights.iter().enumerate() {
                    ui.label(format!("  w{} = {:.6}", i + 1, w));
                }
                ui.label(format!("Bias: {:.6}", snap.bias));

                ui.separator();
                match snap.kind {
                    // ── per-sample weight update (Perceptron, ADALINE) ────────
                    SnapshotKind::Update => {
                        if let (Some(inp), Some(exp), Some(pred), Some(err)) =
                            (&snap.input, snap.expected, snap.predicted, snap.error)
                        {
                            ui.label(format!(
                                "Input: [{}]",
                                inp.iter()
                                    .map(|x| format!("{x:.1}"))
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            ));
                            ui.label(format!("Expected:  {exp}"));
                            ui.label(format!("Predicted: {pred}"));
                            ui.label(format!("Error:     {err}"));
                            ui.label(format!("Cumul. error: {}", snap.total_error_so_far));
                        } else {
                            // Backward compat: old history files lack `kind` and
                            // default all entries to `Update`.  Detect sentinels
                            // by position and missing payload.
                            if self.current_idx == 0 {
                                ui.label("(pre-training — no update yet)");
                            } else {
                                ui.label("(converged — zero error)");
                                ui.label(format!("Cumul. error: {}", snap.total_error_so_far));
                            }
                        }
                    }
                    // ── pre-training sentinel ─────────────────────────────────
                    SnapshotKind::PreTraining => {
                        ui.label("(pre-training — no update yet)");
                    }
                    // ── end-of-epoch (Gradient descent) ───────────────────────
                    SnapshotKind::EpochEnd => {
                        ui.label(format!(
                            "Mean squared error: {:.6}",
                            snap.total_error_so_far
                        ));
                    }
                    // ── convergence sentinel ──────────────────────────────────
                    SnapshotKind::Converged => {
                        ui.label("(converged — zero error)");
                    }
                }

                if degenerate {
                    ui.separator();
                    ui.colored_label(
                        egui::Color32::from_rgb(180, 60, 60),
                        self.mode.degenerate_label(),
                    );
                }
            });

        // ── central panel ────────────────────────────────────────────────────
        egui::CentralPanel::default().show(ctx, |ui| {
            // Playback controls row
            ui.horizontal(|ui| {
                if ui.button("⏮ Reset").clicked() {
                    self.current_idx = 0;
                    self.playing = false;
                }
                if ui.button("◀ Prev").clicked() {
                    self.current_idx = self.current_idx.saturating_sub(1);
                    self.playing = false;
                }

                let play_lbl = if self.playing {
                    "⏸ Pause"
                } else {
                    "▶ Play"
                };
                if ui.button(play_lbl).clicked() {
                    self.playing = !self.playing;
                    self.last_advance = ctx.input(|i| i.time);
                }

                if ui.button("Next ▶").clicked() {
                    if self.current_idx + 1 < n {
                        self.current_idx += 1;
                    }
                    self.playing = false;
                }

                ui.separator();
                ui.label("Speed:");
                egui::ComboBox::from_id_salt("speed_combo")
                    .selected_text(format!("{} snap/s", self.speed as usize))
                    .show_ui(ui, |ui| {
                        for &s in &[1.0_f64, 2.0, 5.0, 10.0, 30.0] {
                            ui.selectable_value(
                                &mut self.speed,
                                s,
                                format!("{} snap/s", s as usize),
                            );
                        }
                    });

                ui.separator();
                if ui.button("Fit to data").clicked() {
                    self.fit_requested = true;
                }
            });

            // Snapshot scrub slider
            ui.add(
                egui::Slider::new(&mut self.current_idx, 0..=n.saturating_sub(1)).text("Snapshot"),
            );

            // Stable ID — never changes between frames.  egui_plot persists the
            // pan/zoom transform in Memory keyed by this ID; changing it would
            // discard the stored transform and snap back to default bounds.
            Plot::new("training_plot")
                .legend(Legend::default())
                .show(ui, |plot_ui| {
                    // One-shot fit: call set_plot_bounds exactly once (on startup or
                    // when the user clicks "Fit to data").  set_plot_bounds switches
                    // egui_plot from auto-bounds mode into manual-bounds mode, so no
                    // auto-fit runs on subsequent frames — bounds stay wherever the
                    // user leaves them until the next fit request.
                    if self.fit_requested {
                        plot_ui.set_plot_bounds(self.fit_bounds);
                        self.fit_requested = false;
                    }

                    // Read the current (now stable) viewport for geometry clipping.
                    // vx_min/vx_max are used by both modes; vy_min/vy_max are only
                    // needed for the 2-D boundary so they are read inside that arm.
                    let b = plot_ui.plot_bounds();
                    let (vx_min, vx_max) = (b.min()[0], b.max()[0]);

                    match self.mode {
                        // ── 2-input classification ────────────────────────────
                        Mode::Classification2D => {
                            let (vy_min, vy_max) = (b.min()[1], b.max()[1]);

                            // boundary_threshold: net-input level at which the neuron fires;
                            //   positions the boundary line and half-plane shading independently
                            //   of label scale.  See also: color_threshold, used in new() to
                            //   precompute the class-split scatter points stored in class0_pts /
                            //   class1_pts.
                            let boundary_threshold = self.history.boundary_threshold;

                            // Shaded class-1 half-plane
                            if !degenerate {
                                let poly = class1_polygon(
                                    w1,
                                    w2,
                                    bias,
                                    boundary_threshold,
                                    vx_min,
                                    vx_max,
                                    vy_min,
                                    vy_max,
                                );
                                if !poly.is_empty() {
                                    plot_ui.polygon(
                                        Polygon::new(poly)
                                            .name("Class 1 region")
                                            .fill_color(egui::Color32::from_rgba_unmultiplied(
                                                200, 80, 80, 25,
                                            )),
                                    );
                                }

                                // Decision boundary line spanning the full viewport width
                                if let Some(pts) = boundary_line(
                                    w1,
                                    w2,
                                    bias,
                                    boundary_threshold,
                                    vx_min,
                                    vx_max,
                                    vy_min,
                                    vy_max,
                                ) {
                                    plot_ui.line(
                                        Line::new(pts)
                                            .name("Decision boundary")
                                            .color(egui::Color32::from_rgb(125, 125, 125))
                                            .width(2.0),
                                    );
                                }
                            }

                            // Scatter: class 0 (blue circles)
                            plot_ui.points(
                                Points::new(PlotPoints::from(self.class0_pts.clone()))
                                    .name("Class 0")
                                    .color(egui::Color32::from_rgb(60, 100, 210))
                                    .radius(8.0),
                            );
                            // Scatter: class 1 (red circles)
                            plot_ui.points(
                                Points::new(PlotPoints::from(self.class1_pts.clone()))
                                    .name("Class 1")
                                    .color(egui::Color32::from_rgb(210, 60, 60))
                                    .radius(8.0),
                            );
                        }

                        // ── 1-input regression ────────────────────────────────
                        Mode::Regression1D => {
                            // Regression line y = w1·x + bias spanning the viewport x-range
                            if !degenerate {
                                let pts = regression_line(w1, bias, vx_min, vx_max);
                                plot_ui.line(
                                    Line::new(pts)
                                        .name("Regression line")
                                        .color(egui::Color32::from_rgb(200, 160, 60))
                                        .width(2.0),
                                );
                            }

                            // Scatter all (x, d) points in a neutral colour (no class split)
                            plot_ui.points(
                                Points::new(PlotPoints::from(self.scatter_pts.clone()))
                                    .name("Data")
                                    .color(egui::Color32::from_rgb(160, 160, 160))
                                    .radius(8.0),
                            );
                        }
                    }
                });

            if degenerate {
                ui.label(
                    egui::RichText::new(self.mode.degenerate_warning())
                        .color(egui::Color32::from_rgb(180, 60, 60)),
                );
            }
        });
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() -> eframe::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "history.json".to_string());

    let content = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Cannot read '{path}': {e}");
            std::process::exit(1);
        }
    };

    // Deserialise as HistoryFile<f64>.  serde_json promotes JSON integers to
    // f64, so Perceptron histories (integer labels 0 / 1) load without error.
    let history: HistoryFile<f64> = match serde_json::from_str(&content) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("Cannot parse '{path}': {e}");
            std::process::exit(1);
        }
    };

    let n_inputs = history
        .dataset
        .first()
        .map(|(pts, _)| pts.len())
        .unwrap_or(0);

    let mode = match n_inputs {
        1 => Mode::Regression1D,
        2 => Mode::Classification2D,
        _ => {
            eprintln!(
                "Visualiser supports 1-input regression and 2-input classification; \
                 this dataset has {n_inputs} input(s). Exiting."
            );
            std::process::exit(1);
        }
    };

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([960.0, 660.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Training Visualiser",
        options,
        Box::new(|_cc| Ok(Box::new(TrainingViz::new(history, mode)))),
    )
}
