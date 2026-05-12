use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct NeuronSnapshot {
    pub weights: Vec<f64>,
    pub bias: f64,
}

/// Snapshot for a single-perceptron trainer (Linear / Gradient / Adeline).
#[derive(Debug, Clone, Serialize)]
pub struct EpochSnapshot {
    pub epoch: usize,
    pub neuron: NeuronSnapshot,
    /// Matches the loss formula used by the respective trainer:
    ///   - Linear   : sum of |error| (total_error)
    ///   - Gradient : squarred_error_sum / n  (MSE)
    ///   - Adeline  : squarred_error_sum / n  (MSE, evaluated after online update)
    pub loss: f64,
    /// Number of misclassified examples (only when class_stop is configured).
    pub misclassified: Option<usize>,
}

/// Top-level history object for single-perceptron experiments.
#[derive(Debug, Clone, Serialize)]
pub struct History {
    pub dataset_name: String,
    pub learning_rate: f64,
    pub tolerance: f64,
    pub total_epochs: usize,
    pub snapshots: Vec<EpochSnapshot>,
}

// ── MLP ──────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct LayerSnapshot {
    pub neurons: Vec<NeuronSnapshot>,
    /// Output activations after forward pass on every training input (all samples).
    pub outputs_per_sample: Vec<Vec<f64>>,
    /// Deltas after backward pass on the last sample of the epoch.
    pub deltas: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MlpEpochSnapshot {
    pub epoch: usize,
    pub layers: Vec<LayerSnapshot>,
    /// squarred_error_sum / n  (MSE, matching MLP::train formula)
    pub loss: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct MlpHistory {
    pub dataset_name: String,
    pub learning_rate: f64,
    pub tolerance: f64,
    pub total_epochs: usize,
    pub snapshots: Vec<MlpEpochSnapshot>,
}

// ── Single-layer (multi-output, one perceptron per output) ───────────────────

#[derive(Debug, Clone, Serialize)]
pub struct NeuronTraceHistory {
    pub neuron_index: usize,
    pub total_epochs: usize,
    pub snapshots: Vec<EpochSnapshot>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SingleLayerHistory {
    pub dataset_name: String,
    pub learning_rate: f64,
    pub tolerance: f64,
    pub neuron_histories: Vec<NeuronTraceHistory>,
}
