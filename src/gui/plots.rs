use egui::Ui;
use egui_plot::{Legend, Line, MarkerShape, Plot, PlotImage, PlotPoint, PlotPoints, Points};

use crate::history::types::{
    ActivationName, EpochSnapshot, History, MlpEpochSnapshot, MlpHistory, SingleLayerHistory,
    forward_snapshot,
};

// ── Loss curve ────────────────────────────────────────────────────────────────

pub fn draw_loss_curve(ui: &mut Ui, history: &History, id_salt: &str, scale: f32) {
    let points: PlotPoints = history
        .snapshots
        .iter()
        .map(|s| [s.epoch as f64, s.loss])
        .collect();

    Plot::new(format!("loss_curve_{}", id_salt))
        .legend(Legend::default())
        .height(200.0 * scale)
        .show(ui, |plot_ui| {
            plot_ui.line(Line::new(points).name("loss"));
        });
}

pub fn draw_loss_curve_mlp(ui: &mut Ui, history: &MlpHistory, id_salt: &str, scale: f32) {
    let points: PlotPoints = history
        .snapshots
        .iter()
        .map(|s| [s.epoch as f64, s.loss])
        .collect();

    Plot::new(format!("loss_curve_mlp_{}", id_salt))
        .legend(Legend::default())
        .height(200.0 * scale)
        .show(ui, |plot_ui| {
            plot_ui.line(Line::new(points).name("MSE"));
        });
}

/// Multi-series loss: one line per output neuron + one thicker total-MSE line.
/// Falls back to the single-line version when there is only 1 output.
pub fn draw_loss_curve_mlp_multi(ui: &mut Ui, history: &MlpHistory, id_salt: &str, scale: f32) {
    let n_out = history.snapshots.first().map(|s| s.per_output_mse.len()).unwrap_or(0);
    if n_out <= 1 {
        return draw_loss_curve_mlp(ui, history, id_salt, scale);
    }

    Plot::new(format!("loss_curve_mlp_multi_{}", id_salt))
        .legend(Legend::default())
        .height(200.0 * scale)
        .show(ui, |plot_ui| {
            // Total MSE (bold)
            let total: PlotPoints = history
                .snapshots.iter().map(|s| [s.epoch as f64, s.loss]).collect();
            plot_ui.line(Line::new(total).name("MSE total").width(2.5));

            // Per-output lines
            for oi in 0..n_out {
                let pts: PlotPoints = history
                    .snapshots.iter()
                    .map(|s| [s.epoch as f64, s.per_output_mse.get(oi).copied().unwrap_or(0.0)])
                    .collect();
                plot_ui.line(Line::new(pts).name(format!("output {}", oi)));
            }
        });
}

pub fn draw_loss_curve_single_layer(ui: &mut Ui, history: &SingleLayerHistory, id_salt: &str, scale: f32) {
    Plot::new(format!("loss_sl_{}", id_salt))
        .legend(Legend::default())
        .height(200.0 * scale)
        .show(ui, |plot_ui| {
            for nh in &history.neuron_histories {
                let pts: PlotPoints = nh
                    .snapshots
                    .iter()
                    .map(|s| [s.epoch as f64, s.loss])
                    .collect();
                plot_ui.line(Line::new(pts).name(format!("neuron {}", nh.neuron_index)));
            }
        });
}

// ── Shared bounds helper ──────────────────────────────────────────────────────

fn data_bounds_2d(dataset: &[(Vec<f64>, f64)], pad: f64) -> [f64; 4] {
    let xs: Vec<f64> = dataset.iter().map(|(inp, _)| inp[0]).collect();
    let ys: Vec<f64> = dataset.iter().map(|(inp, _)| inp[1]).collect();
    let x_min = xs.iter().cloned().fold(f64::INFINITY, f64::min) - pad;
    let x_max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + pad;
    let y_min = ys.iter().cloned().fold(f64::INFINITY, f64::min) - pad;
    let y_max = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + pad;
    [x_min, x_max, y_min, y_max]
}

// ── 2D scatter + decision boundary (perceptron) ───────────────────────────────

/// Linear decision boundary for single-perceptron experiments.
/// Boundary line is clamped to dataset y range so auto-bounds never zooms out
/// to follow extreme slope values when scrubbing epochs.
pub fn draw_scatter_2d_boundary(
    ui: &mut Ui,
    dataset: &[(Vec<f64>, f64)],
    snapshot: &EpochSnapshot,
    id_salt: &str,
    scale: f32,
) {
    let pos_pts: PlotPoints = dataset
        .iter()
        .filter(|(_, label)| *label > 0.0)
        .map(|(inp, _)| [inp[0], inp[1]])
        .collect();
    let neg_pts: PlotPoints = dataset
        .iter()
        .filter(|(_, label)| *label <= 0.0)
        .map(|(inp, _)| [inp[0], inp[1]])
        .collect();

    let w = &snapshot.neuron.weights;
    let b = snapshot.neuron.bias;

    let [x_min, x_max, y_min, y_max] = data_bounds_2d(dataset, 1.5);

    let boundary: PlotPoints = if w.len() >= 2 && w[1].abs() > 1e-12 {
        (0..=200)
            .map(|i| {
                let x = x_min + (x_max - x_min) * i as f64 / 200.0;
                let y = (-(w[0] * x + b) / w[1]).clamp(y_min, y_max);
                [x, y]
            })
            .collect()
    } else {
        let xb = if w[0].abs() > 1e-12 { -b / w[0] } else { 0.0 };
        vec![[xb, y_min], [xb, y_max]].into_iter().collect()
    };

    Plot::new(format!("scatter_2d_{}", id_salt))
        .legend(Legend::default())
        .data_aspect(1.0)
        .height(280.0 * scale)
        .include_x(x_min)
        .include_x(x_max)
        .include_y(y_min)
        .include_y(y_max)
        .show(ui, |plot_ui| {
            plot_ui.points(
                Points::new(pos_pts)
                    .name("class +")
                    .color(egui::Color32::from_rgb(80, 180, 80))
                    .shape(MarkerShape::Circle)
                    .radius(5.0),
            );
            plot_ui.points(
                Points::new(neg_pts)
                    .name("class -")
                    .color(egui::Color32::from_rgb(200, 80, 80))
                    .shape(MarkerShape::Cross)
                    .radius(5.0),
            );
            plot_ui.line(
                Line::new(boundary)
                    .name("boundary")
                    .color(egui::Color32::from_rgb(60, 120, 200))
                    .width(2.0),
            );
        });
}

// ── 2D MLP decision boundary (texture background + scatter overlay) ───────────

/// Per-class background colors for the decision boundary grid (RGBA, alpha=90).
const CLASS_COLORS: &[(u8, u8, u8)] = &[
    (80,  120, 220), // 0 — blue
    (220, 130,  60), // 1 — orange
    (60,  200, 100), // 2 — green
    (160,  60, 220), // 3 — purple
    (60,  200, 200), // 4 — teal
];

fn class_color(class_idx: usize) -> (u8, u8, u8) {
    CLASS_COLORS[class_idx % CLASS_COLORS.len()]
}

/// Render the MLP decision boundary for a 2D-input experiment.
///
/// Handles both binary (1 output, threshold comparison) and multi-class
/// (n outputs, argmax) automatically.
///
/// `inputs` / `targets` — full multi-output dataset as returned by
/// `get_full_dataset`. Circle markers = correctly classified, X = wrong.
pub fn draw_mlp_boundary_2d(
    ui: &mut Ui,
    ctx: &egui::Context,
    inputs: &[Vec<f64>],
    targets: &[Vec<f64>],
    snapshot: &MlpEpochSnapshot,
    activation: &ActivationName,
    threshold: f64,
    id_salt: &str,
    scale: f32,
) {
    const GRID: usize = 64;

    // Bounds from input features only.
    let dataset_1d: Vec<(Vec<f64>, f64)> = inputs
        .iter()
        .zip(targets.iter())
        .map(|(i, t)| (i.clone(), t.first().copied().unwrap_or(0.0)))
        .collect();
    let [x_min, x_max, y_min, y_max] = data_bounds_2d(&dataset_1d, 0.5);

    let n_outputs = snapshot.layers.last().map(|l| l.neurons.len()).unwrap_or(1);
    let multi_class = n_outputs > 1;
    // Binary classification: truth ∈ {0,1} but n_outputs == 1 → need 2 buckets.
    // Multi-class: truth ∈ 0..n_outputs → n_outputs buckets.
    let n_groups = if multi_class { n_outputs } else { 2 };

    // ── Grid ─────────────────────────────────────────────────────────────────
    let mut pixels = vec![0u8; GRID * GRID * 4];
    for row in 0..GRID {
        let y = y_max - (y_max - y_min) * row as f64 / (GRID - 1) as f64;
        for col in 0..GRID {
            let x = x_min + (x_max - x_min) * col as f64 / (GRID - 1) as f64;
            let out = forward_snapshot(&snapshot.layers, &[x, y], activation);
            let class_idx = if multi_class {
                out.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            } else {
                if out.first().copied().unwrap_or(0.0) >= threshold { 1 } else { 0 }
            };
            let (r, g, b) = class_color(class_idx);
            let idx = (row * GRID + col) * 4;
            pixels[idx] = r; pixels[idx+1] = g; pixels[idx+2] = b; pixels[idx+3] = 90;
        }
    }

    let image = egui::ColorImage::from_rgba_unmultiplied([GRID, GRID], &pixels);
    let tex_name = format!("mlp_boundary_{}_{}", id_salt, snapshot.epoch);
    let texture = ctx.load_texture(&tex_name, image, egui::TextureOptions::LINEAR);

    // ── Scatter ───────────────────────────────────────────────────────────────
    // Group by (true_class, correct?) for coloring/shaping.
    let mut by_class_correct: Vec<Vec<[f64; 2]>> = vec![Vec::new(); n_groups];
    let mut by_class_wrong:   Vec<Vec<[f64; 2]>> = vec![Vec::new(); n_groups];

    for (inp, tgt) in inputs.iter().zip(targets.iter()) {
        if inp.len() < 2 { continue; }
        let out = forward_snapshot(&snapshot.layers, inp, activation);
        let pred = if multi_class {
            out.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i,_)|i).unwrap_or(0)
        } else {
            if out.first().copied().unwrap_or(0.0) >= threshold { 1 } else { 0 }
        };
        let truth = if multi_class {
            tgt.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i,_)|i).unwrap_or(0)
        } else {
            if tgt.first().copied().unwrap_or(0.0) > 0.0 { 1 } else { 0 }
        };
        let pt = [inp[0], inp[1]];
        if pred == truth { by_class_correct[truth].push(pt); }
        else             { by_class_wrong[truth].push(pt); }
    }

    let center_x = (x_min + x_max) / 2.0;
    let center_y = (y_min + y_max) / 2.0;

    Plot::new(format!("mlp_boundary_plot_{}", id_salt))
        .legend(Legend::default())
        .data_aspect(1.0)
        .height(300.0 * scale)
        .include_x(x_min).include_x(x_max)
        .include_y(y_min).include_y(y_max)
        .show(ui, |plot_ui| {
            plot_ui.image(
                PlotImage::new(
                    &texture,
                    PlotPoint::new(center_x, center_y),
                    egui::Vec2::new((x_max - x_min) as f32, (y_max - y_min) as f32),
                ).name("decision regions"),
            );
            for ci in 0..n_groups {
                let (r, g, b) = class_color(ci);
                let col = egui::Color32::from_rgb(r, g, b);
                if !by_class_correct[ci].is_empty() {
                    plot_ui.points(
                        Points::new(PlotPoints::new(by_class_correct[ci].clone()))
                            .name(format!("class {} ✓", ci))
                            .color(col).shape(MarkerShape::Circle).radius(6.0),
                    );
                }
                if !by_class_wrong[ci].is_empty() {
                    plot_ui.points(
                        Points::new(PlotPoints::new(by_class_wrong[ci].clone()))
                            .name(format!("class {} ✗", ci))
                            .color(col).shape(MarkerShape::Cross).radius(7.0),
                    );
                }
            }
        });
}

/// 1D MLP regression: scatter of (x, y) data plus predicted output line.
pub fn draw_mlp_regression_1d(
    ui: &mut Ui,
    inputs: &[Vec<f64>],
    targets: &[Vec<f64>],
    snapshot: &MlpEpochSnapshot,
    activation: &ActivationName,
    id_salt: &str,
    scale: f32,
) {
    let scatter: PlotPoints = inputs
        .iter()
        .zip(targets.iter())
        .map(|(i, t)| [i[0], t.first().copied().unwrap_or(0.0)])
        .collect();

    let xs: Vec<f64> = inputs.iter().map(|i| i[0]).collect();
    let ys: Vec<f64> = targets.iter().filter_map(|t| t.first().copied()).collect();
    let x_min = xs.iter().cloned().fold(f64::INFINITY, f64::min) - 1.0;
    let x_max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 1.0;
    let y_min = ys.iter().cloned().fold(f64::INFINITY, f64::min) - 1.0;
    let y_max = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 1.0;

    let line: PlotPoints = (0..=100)
        .map(|i| {
            let x = x_min + (x_max - x_min) * i as f64 / 100.0;
            let y = forward_snapshot(&snapshot.layers, &[x], activation)
                .first().copied().unwrap_or(0.0)
                .clamp(y_min, y_max);
            [x, y]
        })
        .collect();

    Plot::new(format!("mlp_reg1d_{}", id_salt))
        .legend(Legend::default())
        .height(280.0 * scale)
        .include_x(x_min).include_x(x_max)
        .include_y(y_min).include_y(y_max)
        .show(ui, |plot_ui| {
            plot_ui.points(
                Points::new(scatter)
                    .name("data")
                    .color(egui::Color32::from_rgb(80, 160, 220))
                    .shape(MarkerShape::Circle).radius(4.0),
            );
            plot_ui.line(
                Line::new(line)
                    .name("MLP output")
                    .color(egui::Color32::from_rgb(220, 120, 50))
                    .width(2.0),
            );
        });
}

// ── 1-D regression line ───────────────────────────────────────────────────────

pub fn draw_regression_1d(
    ui: &mut Ui,
    dataset: &[(Vec<f64>, f64)],
    snapshot: &EpochSnapshot,
    id_salt: &str,
    scale: f32,
) {
    let scatter: PlotPoints = dataset
        .iter()
        .map(|(inp, target)| [inp[0], *target])
        .collect();

    let xs: Vec<f64> = dataset.iter().map(|(inp, _)| inp[0]).collect();
    let ys: Vec<f64> = dataset.iter().map(|(_, t)| *t).collect();
    let x_min = xs.iter().cloned().fold(f64::INFINITY, f64::min) - 1.0;
    let x_max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 1.0;
    let y_min = ys.iter().cloned().fold(f64::INFINITY, f64::min) - 1.0;
    let y_max = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 1.0;

    let w0 = snapshot.neuron.weights[0];
    let b = snapshot.neuron.bias;

    let line: PlotPoints = (0..=100)
        .map(|i| {
            let x = x_min + (x_max - x_min) * i as f64 / 100.0;
            let y = (w0 * x + b).clamp(y_min, y_max);
            [x, y]
        })
        .collect();

    Plot::new(format!("regression_1d_{}", id_salt))
        .legend(Legend::default())
        .height(280.0 * scale)
        .include_x(x_min)
        .include_x(x_max)
        .include_y(y_min)
        .include_y(y_max)
        .show(ui, |plot_ui| {
            plot_ui.points(
                Points::new(scatter)
                    .name("data")
                    .color(egui::Color32::from_rgb(80, 160, 220))
                    .shape(MarkerShape::Circle)
                    .radius(4.0),
            );
            plot_ui.line(
                Line::new(line)
                    .name("regression")
                    .color(egui::Color32::from_rgb(220, 120, 50))
                    .width(2.0),
            );
        });
}

// ── Confusion matrix ──────────────────────────────────────────────────────────

/// Draw an n×n confusion matrix as a colored text grid.
/// Diagonal cells (correct) shown in green, off-diagonal errors in red, zeros in dark gray.
pub fn draw_confusion_matrix(
    ui: &mut Ui,
    true_classes: &[usize],
    pred_classes: &[usize],
    n_classes: usize,
    id_salt: &str,
) {
    if n_classes == 0 || true_classes.is_empty() {
        return;
    }

    let mut matrix = vec![vec![0usize; n_classes]; n_classes];
    for (&t, &p) in true_classes.iter().zip(pred_classes.iter()) {
        if t < n_classes && p < n_classes {
            matrix[t][p] += 1;
        }
    }
    let total = true_classes.len();
    let correct: usize = (0..n_classes).map(|i| matrix[i][i]).sum();
    let accuracy = correct as f64 / total as f64 * 100.0;

    ui.label(format!(
        "Accuracy: {}/{} ({:.1}%)",
        correct, total, accuracy
    ));

    egui::Grid::new(format!("conf_mat_{}", id_salt))
        .num_columns(n_classes + 1)
        .min_col_width(32.0)
        .spacing([4.0, 2.0])
        .show(ui, |ui| {
            ui.label(egui::RichText::new("T \\ P").small().strong());
            for j in 0..n_classes {
                ui.label(egui::RichText::new(format!("P{}", j)).small().strong());
            }
            ui.end_row();

            for i in 0..n_classes {
                ui.label(egui::RichText::new(format!("T{}", i)).small().strong());
                for j in 0..n_classes {
                    let count = matrix[i][j];
                    let text = egui::RichText::new(format!("{:4}", count)).small().monospace();
                    let text = if i == j && count > 0 {
                        text.color(egui::Color32::from_rgb(80, 220, 80))
                    } else if count > 0 {
                        text.color(egui::Color32::from_rgb(220, 80, 80))
                    } else {
                        text.color(egui::Color32::DARK_GRAY)
                    };
                    ui.label(text);
                }
                ui.end_row();
            }
        });
}

// ── Weight sparklines ─────────────────────────────────────────────────────────

/// Threshold above which the per-line legend is hidden to avoid overflow.
const LEGEND_LINE_LIMIT: usize = 8;

pub fn draw_weight_sparklines_mlp(
    ui: &mut Ui,
    history: &MlpHistory,
    layer_idx: usize,
    neuron_idx: usize,
    id_salt: &str,
    scale: f32,
) {
    let ok = history
        .snapshots
        .first()
        .and_then(|s| s.layers.get(layer_idx))
        .map(|l| l.neurons.len() > neuron_idx)
        .unwrap_or(false);
    if !ok {
        ui.label("(no data for this layer/neuron)");
        return;
    }

    let n_weights = history.snapshots[0].layers[layer_idx].neurons[neuron_idx].weights.len();
    // More weights → taller plot so curves don't crowd each other.
    let plot_height = (160.0 + (n_weights.saturating_sub(4) as f32) * 4.0).min(320.0) * scale;
    let show_legend = n_weights <= LEGEND_LINE_LIMIT;

    if !show_legend {
        ui.label(format!(
            "({} weights — legend hidden to prevent overflow; colors cycle w0..w{})",
            n_weights,
            n_weights - 1
        ));
    }

    let mut p = Plot::new(format!("sparkline_mlp_l{}_n{}_{}", layer_idx, neuron_idx, id_salt))
        .height(plot_height);
    if show_legend {
        p = p.legend(Legend::default());
    }
    p.show(ui, |plot_ui| {
        for wi in 0..n_weights {
            let pts: PlotPoints = history
                .snapshots
                .iter()
                .map(|s| [s.epoch as f64, s.layers[layer_idx].neurons[neuron_idx].weights[wi]])
                .collect();
            let line = Line::new(pts);
            let line = if show_legend { line.name(format!("w{}", wi)) } else { line };
            plot_ui.line(line);
        }
        let bias_pts: PlotPoints = history
            .snapshots
            .iter()
            .map(|s| [s.epoch as f64, s.layers[layer_idx].neurons[neuron_idx].bias])
            .collect();
        let bias_line = Line::new(bias_pts).color(egui::Color32::from_rgb(180, 80, 200));
        let bias_line = if show_legend { bias_line.name("bias") } else { bias_line };
        plot_ui.line(bias_line);
    });
}

pub fn draw_weight_sparklines_single_layer(
    ui: &mut Ui,
    history: &SingleLayerHistory,
    neuron_idx: usize,
    id_salt: &str,
    scale: f32,
) {
    let nh = match history.neuron_histories.get(neuron_idx) {
        Some(h) => h,
        None => {
            ui.label("(no data for this neuron)");
            return;
        }
    };
    let n_weights = nh.snapshots.first().map(|s| s.neuron.weights.len()).unwrap_or(0);
    let plot_height = (160.0 + (n_weights.saturating_sub(4) as f32) * 4.0).min(320.0) * scale;
    let show_legend = n_weights <= LEGEND_LINE_LIMIT;

    if !show_legend {
        ui.label(format!(
            "({} weights — legend hidden; colors cycle w0..w{})",
            n_weights,
            n_weights - 1
        ));
    }

    let mut p = Plot::new(format!("sparkline_sl_n{}_{}", neuron_idx, id_salt))
        .height(plot_height);
    if show_legend {
        p = p.legend(Legend::default());
    }
    p.show(ui, |plot_ui| {
        for wi in 0..n_weights {
            let pts: PlotPoints = nh
                .snapshots
                .iter()
                .map(|s| [s.epoch as f64, s.neuron.weights[wi]])
                .collect();
            let line = Line::new(pts);
            let line = if show_legend { line.name(format!("w{}", wi)) } else { line };
            plot_ui.line(line);
        }
        let bias_pts: PlotPoints = nh
            .snapshots
            .iter()
            .map(|s| [s.epoch as f64, s.neuron.bias])
            .collect();
        let bias_line = Line::new(bias_pts).color(egui::Color32::from_rgb(180, 80, 200));
        let bias_line = if show_legend { bias_line.name("bias") } else { bias_line };
        plot_ui.line(bias_line);
    });
}
