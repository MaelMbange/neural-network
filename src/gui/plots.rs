use egui::Ui;
use egui_plot::{Legend, Line, MarkerShape, Plot, PlotPoints, Points};

use crate::history::types::{EpochSnapshot, History, MlpHistory, SingleLayerHistory};

// ── Loss curve ────────────────────────────────────────────────────────────────

pub fn draw_loss_curve(ui: &mut Ui, history: &History, id_salt: &str) {
    let points: PlotPoints = history
        .snapshots
        .iter()
        .map(|s| [s.epoch as f64, s.loss])
        .collect();

    Plot::new(format!("loss_curve_{}", id_salt))
        .legend(Legend::default())
        .height(200.0)
        .show(ui, |plot_ui| {
            plot_ui.line(Line::new(points).name("loss"));
        });
}

pub fn draw_loss_curve_mlp(ui: &mut Ui, history: &MlpHistory, id_salt: &str) {
    let points: PlotPoints = history
        .snapshots
        .iter()
        .map(|s| [s.epoch as f64, s.loss])
        .collect();

    Plot::new(format!("loss_curve_mlp_{}", id_salt))
        .legend(Legend::default())
        .height(200.0)
        .show(ui, |plot_ui| {
            plot_ui.line(Line::new(points).name("MSE"));
        });
}

pub fn draw_loss_curve_single_layer(ui: &mut Ui, history: &SingleLayerHistory, id_salt: &str) {
    Plot::new(format!("loss_sl_{}", id_salt))
        .legend(Legend::default())
        .height(200.0)
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

// ── 2D scatter + decision boundary ───────────────────────────────────────────

/// Computes a fixed [x_min, x_max, y_min, y_max] from the dataset with padding.
/// Used to clip the boundary line so auto-bounds never follows extreme slope values.
fn data_bounds_2d(dataset: &[(Vec<f64>, f64)], pad: f64) -> [f64; 4] {
    let xs: Vec<f64> = dataset.iter().map(|(inp, _)| inp[0]).collect();
    let ys: Vec<f64> = dataset.iter().map(|(inp, _)| inp[1]).collect();
    let x_min = xs.iter().cloned().fold(f64::INFINITY, f64::min) - pad;
    let x_max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + pad;
    let y_min = ys.iter().cloned().fold(f64::INFINITY, f64::min) - pad;
    let y_max = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + pad;
    [x_min, x_max, y_min, y_max]
}

/// Draw a 2D scatter of the dataset (colour by label) plus the decision boundary
/// at the state given by `snapshot`.
///
/// Decision boundary: w0*x + w1*y + b = 0  →  y = -(w0*x + b) / w1
///
/// The boundary line is clipped to the data's y range so the plot never
/// auto-zooms to follow extreme slope values when scrubbing epochs.
pub fn draw_scatter_2d_boundary(
    ui: &mut Ui,
    dataset: &[(Vec<f64>, f64)],
    snapshot: &EpochSnapshot,
    id_salt: &str,
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
                // Clamp to data y range so auto-bounds is never pulled by extreme values.
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
        .height(280.0)
        // Anchor the view to the data range regardless of what the boundary does.
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

// ── 1-D regression line ───────────────────────────────────────────────────────

pub fn draw_regression_1d(
    ui: &mut Ui,
    dataset: &[(Vec<f64>, f64)],
    snapshot: &EpochSnapshot,
    id_salt: &str,
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
            // Clamp to data y range to prevent auto-bounds from following extreme y.
            let y = (w0 * x + b).clamp(y_min, y_max);
            [x, y]
        })
        .collect();

    Plot::new(format!("regression_1d_{}", id_salt))
        .legend(Legend::default())
        .height(280.0)
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

// ── Weight sparklines for MLP ─────────────────────────────────────────────────

pub fn draw_weight_sparklines_mlp(
    ui: &mut Ui,
    history: &MlpHistory,
    layer_idx: usize,
    neuron_idx: usize,
    id_salt: &str,
) {
    let layer_exists = history
        .snapshots
        .first()
        .map(|s| s.layers.len() > layer_idx)
        .unwrap_or(false);
    let neuron_exists = history
        .snapshots
        .first()
        .and_then(|s| s.layers.get(layer_idx))
        .map(|l| l.neurons.len() > neuron_idx)
        .unwrap_or(false);

    if !layer_exists || !neuron_exists {
        ui.label("(no data for this layer/neuron)");
        return;
    }

    let n_weights = history.snapshots[0].layers[layer_idx].neurons[neuron_idx].weights.len();

    Plot::new(format!("sparkline_mlp_l{}_n{}_{}", layer_idx, neuron_idx, id_salt))
        .legend(Legend::default())
        .height(160.0)
        .show(ui, |plot_ui| {
            for wi in 0..n_weights {
                let pts: PlotPoints = history
                    .snapshots
                    .iter()
                    .map(|s| {
                        let w = s.layers[layer_idx].neurons[neuron_idx].weights[wi];
                        [s.epoch as f64, w]
                    })
                    .collect();
                plot_ui.line(Line::new(pts).name(format!("w{}", wi)));
            }
            let bias_pts: PlotPoints = history
                .snapshots
                .iter()
                .map(|s| {
                    let b = s.layers[layer_idx].neurons[neuron_idx].bias;
                    [s.epoch as f64, b]
                })
                .collect();
            plot_ui.line(
                Line::new(bias_pts)
                    .name("bias")
                    .color(egui::Color32::from_rgb(180, 80, 200)),
            );
        });
}

pub fn draw_weight_sparklines_single_layer(
    ui: &mut Ui,
    history: &SingleLayerHistory,
    neuron_idx: usize,
    id_salt: &str,
) {
    let nh = match history.neuron_histories.get(neuron_idx) {
        Some(h) => h,
        None => {
            ui.label("(no data for this neuron)");
            return;
        }
    };

    let n_weights = nh.snapshots.first().map(|s| s.neuron.weights.len()).unwrap_or(0);

    Plot::new(format!("sparkline_sl_n{}_{}", neuron_idx, id_salt))
        .legend(Legend::default())
        .height(160.0)
        .show(ui, |plot_ui| {
            for wi in 0..n_weights {
                let pts: PlotPoints = nh
                    .snapshots
                    .iter()
                    .map(|s| [s.epoch as f64, s.neuron.weights[wi]])
                    .collect();
                plot_ui.line(Line::new(pts).name(format!("w{}", wi)));
            }
            let bias_pts: PlotPoints = nh
                .snapshots
                .iter()
                .map(|s| [s.epoch as f64, s.neuron.bias])
                .collect();
            plot_ui.line(
                Line::new(bias_pts)
                    .name("bias")
                    .color(egui::Color32::from_rgb(180, 80, 200)),
            );
        });
}
