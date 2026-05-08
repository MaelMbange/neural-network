use serde::{Deserialize, Serialize};

// ── Snapshot kind ─────────────────────────────────────────────────────────────

/// Distinguishes the four roles a [`TrainingSnapshot`] can play.
///
/// Defaults to [`SnapshotKind::Update`] when deserialising history files that
/// predate this field, so old `history.json` files produced by the pre-refactor
/// perceptron binary remain readable.
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
#[serde(rename_all = "snake_case")]
pub enum SnapshotKind {
    /// Pre-training sentinel — the very first entry; no weight update has occurred.
    PreTraining,
    /// A per-sample weight update (Perceptron, ADALINE).
    Update,
    /// End-of-epoch snapshot after batch weights are applied (Gradient descent).
    EpochEnd,
    /// Post-convergence sentinel — the training loop terminated with zero error.
    Converged,
}

impl Default for SnapshotKind {
    fn default() -> Self {
        SnapshotKind::Update
    }
}

// ── TrainingSnapshot ──────────────────────────────────────────────────────────

/// Generic training snapshot, parameterised over the neuron's output type `O`.
///
/// | Neuron       | `O`   | primary hook          |
/// |--------------|-------|-----------------------|
/// | Perceptron   | `i32` | `on_sample_update`    |
/// | Gradient     | `f64` | `on_epoch_end`        |
/// | ADALINE      | `f64` | `on_sample_update`    |
///
/// Sentinel entries (`PreTraining`, `EpochEnd`, `Converged`) leave `input`,
/// `expected`, `predicted`, and `error` as `None`.
#[derive(Serialize, Deserialize, Clone)]
pub struct TrainingSnapshot<O> {
    /// Role of this snapshot in the training timeline.
    #[serde(default)]
    pub kind: SnapshotKind,
    pub epoch: usize,
    pub sample_index: usize,
    pub weights: Vec<f64>,
    pub bias: f64,
    /// `None` on sentinel entries.
    pub input: Option<Vec<f64>>,
    /// `None` on sentinel entries.
    pub expected: Option<O>,
    /// `None` on sentinel entries.
    pub predicted: Option<O>,
    /// `None` on sentinel entries and epoch-end snapshots.
    pub error: Option<O>,
    /// For Perceptron / ADALINE: cumulative misclassification count (cast to `f64`).
    /// For Gradient: mean squared error for the current epoch.
    pub total_error_so_far: f64,
}

// ── HistoryFile ───────────────────────────────────────────────────────────────

/// Top-level structure written to `history.json`.
///
/// Bundles the dataset alongside the snapshot list so the visualiser can plot
/// all data points, not only those that triggered weight updates.
///
/// `class_threshold` tells the visualiser where to split the dataset into two
/// display classes:
/// - `0.5` — 0 / 1-labelled data (Perceptron AND/OR gate).
/// - `0.0` — ±1-labelled data (Gradient / ADALINE).
///
/// `boundary_threshold` is the net-input level at which the neuron fires —
/// used to draw the decision boundary line and half-plane shading:
/// - `0.0` (default) — Step activation fires at net ≥ 0; also correct for
///   bipolar {−1, +1} Identity outputs.
/// - `0.5` — Identity activation trained on unipolar {0, 1} labels.
///
/// Old files that lack either field get the documented defaults.
#[derive(Serialize, Deserialize)]
pub struct HistoryFile<O> {
    pub dataset: Vec<(Vec<f64>, O)>,
    pub snapshots: Vec<TrainingSnapshot<O>>,
    #[serde(default = "default_class_threshold")]
    pub class_threshold: f64,
    /// Net-input level that defines the decision boundary (`w·x + bias = boundary_threshold`).
    /// Defaults to `0.0`, which is correct for Step activation and bipolar {−1,+1} labels.
    #[serde(default)]
    pub boundary_threshold: f64,
}

fn default_class_threshold() -> f64 {
    0.5
}

// ── TrainObserver trait ───────────────────────────────────────────────────────

/// Observer notified at key points during a neuron's training loop.
///
/// All methods have empty default implementations — implement only the hooks
/// you need.  [`NoopObserver`] is the zero-cost implementation used internally
/// by [`Trainable::train`], ensuring each neuron's loop is written only once.
pub trait TrainObserver<O> {
    /// Called once before the first epoch, with the initial weights and bias.
    fn on_train_start(&mut self, _weights: &[f64], _bias: f64) {}

    /// Called after each **per-sample** weight update (Perceptron, ADALINE).
    ///
    /// `total_error_so_far` is the cumulative unsigned error for the current
    /// epoch, cast to `f64`.
    #[allow(clippy::too_many_arguments)]
    fn on_sample_update(
        &mut self,
        _epoch: usize,
        _sample_idx: usize,
        _weights: &[f64],
        _bias: f64,
        _input: &[f64],
        _expected: &O,
        _predicted: &O,
        _error: &O,
        _total_error_so_far: f64,
    ) {
    }

    /// Called once **per epoch** after batch weights are applied (Gradient descent).
    ///
    /// `epoch_error` is the mean squared error for that epoch.
    fn on_epoch_end(
        &mut self,
        _epoch: usize,
        _n_samples: usize,
        _weights: &[f64],
        _bias: f64,
        _epoch_error: f64,
    ) {
    }

    /// Called when training converges (or the epoch limit is reached with zero
    /// error).  Not called on a timeout without convergence.
    fn on_converged(
        &mut self,
        _epoch: usize,
        _n_samples: usize,
        _weights: &[f64],
        _bias: f64,
    ) {
    }
}

// ── NoopObserver ──────────────────────────────────────────────────────────────

/// Zero-cost observer that discards all events.
///
/// Used by `Trainable::train` so each neuron's training loop is written once
/// inside `train_with_observer`, with `NoopObserver` passed when no history is
/// needed.
pub struct NoopObserver;

impl<O> TrainObserver<O> for NoopObserver {}

// ── HistoryRecorder ───────────────────────────────────────────────────────────

/// Accumulates [`TrainingSnapshot<O>`] entries as training progresses.
///
/// Pass a `&mut HistoryRecorder` to a neuron's `train_with_observer`, then
/// move `recorder.snapshots` into a [`HistoryFile`] for serialisation:
///
/// ```ignore
/// let mut recorder = HistoryRecorder::new();
/// neuron.train_with_observer(&dataset, Some(100), &mut recorder);
/// let file = HistoryFile { dataset: ..., snapshots: recorder.snapshots, class_threshold: 0.5 };
/// ```
///
/// # ADALINE
///
/// ADALINE (stochastic gradient / Widrow-Hoff delta rule) updates weights
/// per-sample like the Perceptron, so it uses the same [`on_sample_update`]
/// hook.  To add history support when implementing `src/neuron/adaline.rs`:
///
/// 1. Call `observer.on_train_start(...)` once before the loop.
/// 2. Call `observer.on_sample_update(...)` inside the loop, after each weight
///    update, passing `&error` of type `&f64`.
/// 3. Call `observer.on_converged(...)` when the stopping criterion is met.
/// 4. Add a `src/bin/adaline.rs` that passes a `HistoryRecorder<f64>` and
///    writes `HistoryFile { ..., class_threshold: 0.0 }`.
///
/// [`on_sample_update`]: TrainObserver::on_sample_update
pub struct HistoryRecorder<O> {
    pub snapshots: Vec<TrainingSnapshot<O>>,
}

impl<O> HistoryRecorder<O> {
    pub fn new() -> Self {
        Self {
            snapshots: Vec::new(),
        }
    }
}

impl<O> Default for HistoryRecorder<O> {
    fn default() -> Self {
        Self::new()
    }
}

impl<O: Clone> TrainObserver<O> for HistoryRecorder<O> {
    fn on_train_start(&mut self, weights: &[f64], bias: f64) {
        self.snapshots.push(TrainingSnapshot {
            kind: SnapshotKind::PreTraining,
            epoch: 0,
            sample_index: 0,
            weights: weights.to_vec(),
            bias,
            input: None,
            expected: None,
            predicted: None,
            error: None,
            total_error_so_far: 0.0,
        });
    }

    fn on_sample_update(
        &mut self,
        epoch: usize,
        sample_idx: usize,
        weights: &[f64],
        bias: f64,
        input: &[f64],
        expected: &O,
        predicted: &O,
        error: &O,
        total_error_so_far: f64,
    ) {
        self.snapshots.push(TrainingSnapshot {
            kind: SnapshotKind::Update,
            epoch,
            sample_index: sample_idx,
            weights: weights.to_vec(),
            bias,
            input: Some(input.to_vec()),
            expected: Some(expected.clone()),
            predicted: Some(predicted.clone()),
            error: Some(error.clone()),
            total_error_so_far,
        });
    }

    fn on_epoch_end(
        &mut self,
        epoch: usize,
        n_samples: usize,
        weights: &[f64],
        bias: f64,
        epoch_error: f64,
    ) {
        self.snapshots.push(TrainingSnapshot {
            kind: SnapshotKind::EpochEnd,
            epoch,
            sample_index: n_samples,
            weights: weights.to_vec(),
            bias,
            input: None,
            expected: None,
            predicted: None,
            error: None,
            total_error_so_far: epoch_error,
        });
    }

    fn on_converged(&mut self, epoch: usize, n_samples: usize, weights: &[f64], bias: f64) {
        self.snapshots.push(TrainingSnapshot {
            kind: SnapshotKind::Converged,
            epoch,
            sample_index: n_samples,
            weights: weights.to_vec(),
            bias,
            input: None,
            expected: None,
            predicted: None,
            error: None,
            total_error_so_far: 0.0,
        });
    }
}
