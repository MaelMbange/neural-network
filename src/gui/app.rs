use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{Duration, Instant};

use egui::{Context, ScrollArea, Slider, Ui};

use crate::gui_module::{
    experiments::{ExperimentKind, TrainingResult, all_experiments, run_experiment},
    plots::{
        draw_loss_curve, draw_loss_curve_mlp, draw_loss_curve_single_layer,
        draw_regression_1d, draw_scatter_2d_boundary, draw_weight_sparklines_mlp,
        draw_weight_sparklines_single_layer,
    },
};
use crate::history::{MlpHistory, SingleLayerHistory};
use crate::history::types::{EpochSnapshot, History};

// ── Background training channel ───────────────────────────────────────────────

type TrainMsg = Result<TrainingResult, String>;

enum TrainingState {
    Idle,
    Running,
    Done(TrainingResult),
    Error(String),
}

// ── Playback speed options ────────────────────────────────────────────────────

const SPEED_OPTIONS: &[(&str, f32)] = &[
    ("1 snap/s",   1.0),
    ("5 snap/s",   5.0),
    ("10 snap/s", 10.0),
    ("20 snap/s", 20.0),
    ("50 snap/s", 50.0),
];

// ── Per-experiment UI state ───────────────────────────────────────────────────

struct ExperimentUiState {
    epoch_slider: usize,
    /// For MLP: selected layer and neuron for sparkline.
    mlp_layer: usize,
    mlp_neuron: usize,
    /// For SingleLayer: selected neuron for sparkline.
    sl_neuron: usize,
    /// Playback
    playing: bool,
    speed_idx: usize,
    last_advance: Option<Instant>,
}

impl Default for ExperimentUiState {
    fn default() -> Self {
        Self {
            epoch_slider: 0,
            mlp_layer: 0,
            mlp_neuron: 0,
            sl_neuron: 0,
            playing: false,
            speed_idx: 1, // default 5 snap/s
            last_advance: None,
        }
    }
}

// ── Main app ──────────────────────────────────────────────────────────────────

pub struct GuiApp {
    experiment_names: Vec<&'static str>,
    selected: usize,
    verbose: bool,

    training_state: TrainingState,
    tx: Sender<TrainMsg>,
    rx: Receiver<TrainMsg>,

    ui_state: ExperimentUiState,
}

impl GuiApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let names: Vec<&'static str> = all_experiments().iter().map(|e| e.name).collect();
        let (tx, rx) = mpsc::channel();
        Self {
            experiment_names: names,
            selected: 0,
            verbose: false,
            training_state: TrainingState::Idle,
            tx,
            rx,
            ui_state: ExperimentUiState::default(),
        }
    }

    fn selected_name(&self) -> &str {
        self.experiment_names[self.selected]
    }

    fn kind_for_selected(&self) -> ExperimentKind {
        let exps = all_experiments();
        exps.into_iter()
            .find(|e| e.name == self.selected_name())
            .map(|e| e.kind)
            .unwrap_or(ExperimentKind::Regression1D)
    }

    fn dataset_for_selected(&self) -> Vec<(Vec<f64>, f64)> {
        use crate::gui_module::experiments::DatasetSource;
        let exps = all_experiments();
        let cfg = exps.into_iter().find(|e| e.name == self.selected_name()).unwrap();
        match cfg.dataset {
            DatasetSource::Inline2D(d) => d,
            DatasetSource::File { path, n_inputs, n_outputs } => {
                rna::csv_reader::load_dataset_multi(path, n_inputs, n_outputs, false)
                    .map(|(inputs, targets)| {
                        inputs
                            .into_iter()
                            .zip(targets.into_iter().map(|t| t[0]))
                            .collect()
                    })
                    .unwrap_or_default()
            }
        }
    }

    // ── Sidebar ───────────────────────────────────────────────────────────────

    fn draw_sidebar(&mut self, ui: &mut Ui, ctx: &Context) {
        ui.heading("Experiments");
        ui.separator();

        ScrollArea::vertical().max_height(300.0).show(ui, |ui| {
            for (i, name) in self.experiment_names.iter().enumerate() {
                if ui.selectable_label(self.selected == i, *name).clicked() {
                    if self.selected != i {
                        self.selected = i;
                        self.training_state = TrainingState::Idle;
                        self.ui_state = ExperimentUiState::default();
                    }
                }
            }
        });

        ui.separator();
        ui.checkbox(&mut self.verbose, "Verbose (terminal)");
        ui.separator();

        let running = matches!(self.training_state, TrainingState::Running);

        if running {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("Training…");
            });
            ctx.request_repaint();
        } else if ui.button("▶  Run").clicked() {
            let name = self.selected_name().to_string();
            let verbose = self.verbose;
            let tx = self.tx.clone();
            std::thread::spawn(move || {
                let result = run_experiment(&name, verbose);
                let _ = tx.send(result);
            });
            self.ui_state.playing = false;
            self.training_state = TrainingState::Running;
        }

        ui.separator();

        if let TrainingState::Done(ref result) = self.training_state {
            if ui.button("💾  Export JSON").clicked() {
                let json = match result {
                    TrainingResult::Single(h) => serde_json::to_string_pretty(h).ok(),
                    TrainingResult::SingleLayer(h) => serde_json::to_string_pretty(h).ok(),
                    TrainingResult::Mlp(h) => serde_json::to_string_pretty(h).ok(),
                };
                if let Some(json) = json {
                    let path = format!("{}.json", result.dataset_name());
                    if std::fs::write(&path, json).is_ok() {
                        ui.label(format!("Saved to {}", path));
                    } else {
                        ui.label("Save failed");
                    }
                }
            }
        }
    }

    // ── Playback controls bar ─────────────────────────────────────────────────

    /// Draws the Reset / Prev / Play-Pause / Next buttons + speed selector.
    /// Returns the (current epoch, max epoch) after any button interactions.
    fn draw_playback_controls(
        &mut self,
        ui: &mut Ui,
        ctx: &Context,
        max_epoch: usize,
    ) {
        ui.horizontal(|ui| {
            // Reset
            if ui.button("|◀ Reset").clicked() {
                self.ui_state.epoch_slider = 0;
                self.ui_state.playing = false;
            }
            // Prev
            if ui.button("◀ Prev").clicked() {
                self.ui_state.playing = false;
                self.ui_state.epoch_slider =
                    self.ui_state.epoch_slider.saturating_sub(1);
            }
            // Play / Pause
            let play_label = if self.ui_state.playing { "⏸ Pause" } else { "▶ Play" };
            if ui.button(play_label).clicked() {
                self.ui_state.playing = !self.ui_state.playing;
                if self.ui_state.playing {
                    self.ui_state.last_advance = Some(Instant::now());
                    // Restart from beginning if already at end.
                    if self.ui_state.epoch_slider >= max_epoch {
                        self.ui_state.epoch_slider = 0;
                    }
                }
            }
            // Next
            if ui.button("Next ▶").clicked() {
                self.ui_state.playing = false;
                self.ui_state.epoch_slider =
                    (self.ui_state.epoch_slider + 1).min(max_epoch);
            }

            ui.separator();
            ui.label("Speed:");
            egui::ComboBox::from_id_salt("speed_combo")
                .selected_text(SPEED_OPTIONS[self.ui_state.speed_idx].0)
                .show_ui(ui, |ui| {
                    for (i, (label, _)) in SPEED_OPTIONS.iter().enumerate() {
                        ui.selectable_value(&mut self.ui_state.speed_idx, i, *label);
                    }
                });
        });

        // Time-based advance while playing
        if self.ui_state.playing {
            let speed = SPEED_OPTIONS[self.ui_state.speed_idx].1;
            let interval = Duration::from_secs_f32(1.0 / speed);

            let elapsed = self
                .ui_state
                .last_advance
                .map(|t| t.elapsed())
                .unwrap_or(interval);

            if elapsed >= interval {
                let steps = (elapsed.as_secs_f32() * speed) as usize;
                let new_epoch = (self.ui_state.epoch_slider + steps).min(max_epoch);
                self.ui_state.epoch_slider = new_epoch;
                self.ui_state.last_advance = Some(Instant::now());

                if new_epoch >= max_epoch {
                    self.ui_state.playing = false;
                }
            }

            ctx.request_repaint_after(interval);
        }
    }

    // ── Central panel ─────────────────────────────────────────────────────────

    fn draw_central(&mut self, ui: &mut Ui, ctx: &Context) {
        if let Ok(msg) = self.rx.try_recv() {
            self.training_state = match msg {
                Ok(result) => TrainingState::Done(result),
                Err(e) => TrainingState::Error(e),
            };
            self.ui_state = ExperimentUiState::default();
        }

        match &self.training_state {
            TrainingState::Idle => {
                ui.label("Select an experiment and press ▶ Run.");
            }
            TrainingState::Running => {
                ui.label("Training in progress…");
            }
            TrainingState::Error(e) => {
                ui.colored_label(egui::Color32::RED, format!("Error: {e}"));
            }
            TrainingState::Done(_) => {
                self.draw_results(ui, ctx);
            }
        }
    }

    fn draw_results(&mut self, ui: &mut Ui, ctx: &Context) {
        let result = match &self.training_state {
            TrainingState::Done(r) => r.clone(),
            _ => return,
        };

        let kind = self.kind_for_selected();
        let dataset = self.dataset_for_selected();
        let name = self.selected_name().to_string();

        match result {
            TrainingResult::Single(ref h) => {
                self.draw_single_result(ui, ctx, h, &kind, &dataset, &name);
            }
            TrainingResult::SingleLayer(ref h) => {
                self.draw_single_layer_result(ui, ctx, h, &name);
            }
            TrainingResult::Mlp(ref h) => {
                self.draw_mlp_result(ui, ctx, h, &name);
            }
        }
    }

    fn draw_single_result(
        &mut self,
        ui: &mut Ui,
        ctx: &Context,
        h: &History,
        kind: &ExperimentKind,
        dataset: &[(Vec<f64>, f64)],
        name: &str,
    ) {
        let max_epoch = h.snapshots.len().saturating_sub(1);
        self.ui_state.epoch_slider = self.ui_state.epoch_slider.min(max_epoch);

        ui.heading(format!("Loss — {} ({} epochs)", name, h.total_epochs));
        draw_loss_curve(ui, h, name);

        ui.separator();
        self.draw_playback_controls(ui, ctx, max_epoch);
        ui.add(Slider::new(&mut self.ui_state.epoch_slider, 0..=max_epoch).text("Snapshot"));

        let snap = &h.snapshots[self.ui_state.epoch_slider];

        match kind {
            ExperimentKind::Perceptron2D { .. } => {
                ui.heading("Decision boundary");
                draw_scatter_2d_boundary(ui, dataset, snap, name);
            }
            ExperimentKind::Regression1D => {
                ui.heading("Regression line");
                draw_regression_1d(ui, dataset, snap, name);
            }
            _ => {}
        }

        ui.separator();
        self.draw_snapshot_panel(ui, snap);
    }

    fn draw_single_layer_result(
        &mut self,
        ui: &mut Ui,
        ctx: &Context,
        h: &SingleLayerHistory,
        name: &str,
    ) {
        ui.heading(format!(
            "Single-layer — {} ({} neurons)",
            name,
            h.neuron_histories.len()
        ));

        draw_loss_curve_single_layer(ui, h, name);

        ui.separator();

        let n_neurons = h.neuron_histories.len();
        if n_neurons > 0 {
            ui.horizontal(|ui| {
                ui.label("Neuron:");
                for i in 0..n_neurons {
                    if ui
                        .selectable_label(self.ui_state.sl_neuron == i, format!("{}", i))
                        .clicked()
                    {
                        self.ui_state.sl_neuron = i;
                        self.ui_state.epoch_slider = 0;
                        self.ui_state.playing = false;
                    }
                }
            });

            let ni = self.ui_state.sl_neuron.min(n_neurons - 1);
            ui.heading(format!("Weights — neuron {}", ni));
            draw_weight_sparklines_single_layer(ui, h, ni, name);

            let max_epoch = h.neuron_histories[ni].snapshots.len().saturating_sub(1);
            self.ui_state.epoch_slider = self.ui_state.epoch_slider.min(max_epoch);

            ui.separator();
            self.draw_playback_controls(ui, ctx, max_epoch);
            ui.add(
                Slider::new(&mut self.ui_state.epoch_slider, 0..=max_epoch).text("Snapshot"),
            );

            let snap = &h.neuron_histories[ni].snapshots[self.ui_state.epoch_slider];
            self.draw_snapshot_panel(ui, snap);
        }
    }

    fn draw_mlp_result(&mut self, ui: &mut Ui, ctx: &Context, h: &MlpHistory, name: &str) {
        let n_layers = h.snapshots.first().map(|s| s.layers.len()).unwrap_or(0);
        ui.heading(format!(
            "MLP — {} ({} epochs, {} layers)",
            name, h.total_epochs, n_layers
        ));

        draw_loss_curve_mlp(ui, h, name);

        ui.separator();

        if n_layers > 0 {
            ui.horizontal(|ui| {
                ui.label("Layer:");
                for li in 0..n_layers {
                    if ui
                        .selectable_label(self.ui_state.mlp_layer == li, format!("{}", li))
                        .clicked()
                    {
                        self.ui_state.mlp_layer = li;
                        self.ui_state.mlp_neuron = 0;
                    }
                }
            });

            let li = self.ui_state.mlp_layer.min(n_layers - 1);
            let n_neurons = h
                .snapshots
                .first()
                .and_then(|s| s.layers.get(li))
                .map(|l| l.neurons.len())
                .unwrap_or(0);

            if n_neurons > 0 {
                ui.horizontal(|ui| {
                    ui.label("Neuron:");
                    for ni in 0..n_neurons {
                        if ui
                            .selectable_label(self.ui_state.mlp_neuron == ni, format!("{}", ni))
                            .clicked()
                        {
                            self.ui_state.mlp_neuron = ni;
                        }
                    }
                });
            }

            let ni = self.ui_state.mlp_neuron;
            ui.heading(format!("Weights — layer {}, neuron {}", li, ni));
            draw_weight_sparklines_mlp(ui, h, li, ni, name);
        }

        ui.separator();

        let max_epoch = h.snapshots.len().saturating_sub(1);
        self.ui_state.epoch_slider = self.ui_state.epoch_slider.min(max_epoch);

        self.draw_playback_controls(ui, ctx, max_epoch);
        ui.add(Slider::new(&mut self.ui_state.epoch_slider, 0..=max_epoch).text("Snapshot"));

        if let Some(snap) = h.snapshots.get(self.ui_state.epoch_slider) {
            ui.separator();
            ui.heading(format!("Epoch {} — MSE = {:.6}", snap.epoch, snap.loss));

            let li = self.ui_state.mlp_layer.min(snap.layers.len().saturating_sub(1));
            let ni = self.ui_state.mlp_neuron;

            if let Some(layer) = snap.layers.get(li) {
                if let Some(neuron) = layer.neurons.get(ni) {
                    egui::Grid::new("mlp_snap_grid")
                        .num_columns(2)
                        .striped(true)
                        .show(ui, |ui| {
                            ui.label("bias");
                            ui.label(format!("{:.6}", neuron.bias));
                            ui.end_row();
                            for (i, w) in neuron.weights.iter().enumerate() {
                                ui.label(format!("w{}", i));
                                ui.label(format!("{:.6}", w));
                                ui.end_row();
                            }
                            ui.label("delta");
                            if let Some(d) = layer.deltas.get(ni) {
                                ui.label(format!("{:.6}", d));
                            }
                            ui.end_row();
                        });
                }
            }
        }
    }

    // ── Snapshot detail panel ─────────────────────────────────────────────────

    fn draw_snapshot_panel(&self, ui: &mut Ui, snap: &EpochSnapshot) {
        ui.separator();
        ui.heading(format!("Epoch {} — loss = {:.6}", snap.epoch, snap.loss));

        egui::Grid::new(format!("snap_grid_{}", snap.epoch))
            .num_columns(2)
            .striped(true)
            .show(ui, |ui| {
                ui.label("bias");
                ui.label(format!("{:.6}", snap.neuron.bias));
                ui.end_row();

                for (i, w) in snap.neuron.weights.iter().enumerate() {
                    ui.label(format!("w{}", i));
                    ui.label(format!("{:.6}", w));
                    ui.end_row();
                }

                if let Some(mc) = snap.misclassified {
                    ui.label("misclassified");
                    ui.label(format!("{}", mc));
                    ui.end_row();
                }
            });
    }
}

impl eframe::App for GuiApp {
    fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        egui::SidePanel::left("sidebar")
            .resizable(true)
            .min_width(180.0)
            .show(ctx, |ui| {
                self.draw_sidebar(ui, ctx);
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ScrollArea::vertical().show(ui, |ui| {
                self.draw_central(ui, ctx);
            });
        });
    }
}
