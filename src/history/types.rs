use serde::Serialize;

// ── Activation name ───────────────────────────────────────────────────────────

/// Which activation function was used — stored in MlpHistory so the GUI can
/// run a standalone forward pass through snapshot weights without the live model.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum ActivationName {
    Identity,
    Sigmoid,
    Tanh,
}

impl ActivationName {
    pub fn apply(&self, z: f64) -> f64 {
        match self {
            ActivationName::Identity => z,
            ActivationName::Sigmoid => 1.0 / (1.0 + (-z).exp()),
            ActivationName::Tanh => z.tanh(),
        }
    }
}

/// Run a full forward pass through snapshot layer weights without needing the
/// live MLP model. Used for the decision-boundary grid rendering.
pub fn forward_snapshot(
    layers: &[LayerSnapshot],
    input: &[f64],
    activation: &ActivationName,
) -> Vec<f64> {
    let mut current = input.to_vec();
    for layer in layers {
        let mut next = Vec::with_capacity(layer.neurons.len());
        for neuron in &layer.neurons {
            let z: f64 = current
                .iter()
                .zip(neuron.weights.iter())
                .map(|(x, w)| x * w)
                .sum::<f64>()
                + neuron.bias;
            next.push(activation.apply(z));
        }
        current = next;
    }
    current
}

// ── Single-perceptron snapshot ────────────────────────────────────────────────

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
    /// Epoch cap that was passed to training (usize::MAX if no cap).
    pub max_epochs: usize,
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
    /// Per-output-neuron MSE for this epoch (one value per output neuron).
    /// Enables a multi-series loss chart for experiments with several outputs.
    pub per_output_mse: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MlpHistory {
    pub dataset_name: String,
    pub learning_rate: f64,
    pub tolerance: f64,
    /// Epoch cap that was passed to training.
    pub max_epochs: usize,
    pub total_epochs: usize,
    /// Activation used for ALL layers — needed to replay forward passes from snapshots.
    pub activation: ActivationName,
    /// Number of inputs to the first layer — needed for boundary grid rendering.
    pub input_dim: usize,
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
    /// Epoch cap that was passed to training.
    pub max_epochs: usize,
    pub neuron_histories: Vec<NeuronTraceHistory>,
}
