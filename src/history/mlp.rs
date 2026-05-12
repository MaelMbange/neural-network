use rna::activation::{Activation, Derivative};
use rna::layer::MLP;

use crate::history::types::{
    ActivationName, LayerSnapshot, MlpEpochSnapshot, MlpHistory, NeuronSnapshot,
};

/// Mirrors `MLP::train` and captures a per-epoch snapshot of every layer's
/// weights, deltas, and outputs.
///
/// New parameters vs the original:
///   `activation_name` — stored in `MlpHistory` so the GUI can replay forward
///                       passes from snapshots without the live model.
///   `input_dim`       — also stored for boundary-grid rendering.
///
/// `verbose`: if true, prints epoch index and MSE to stdout.
pub fn train_mlp_with_history<A: Activation + Derivative>(
    mlp: &mut MLP<A>,
    inputs: &[Vec<f64>],
    expected: &[Vec<f64>],
    learning_rate: f64,
    tolerance: f64,
    epochs: Option<usize>,
    dataset_name: impl Into<String>,
    activation_name: ActivationName,
    input_dim: usize,
    verbose: bool,
) -> MlpHistory {
    let mut snapshots = Vec::new();
    mlp.epoch = 0;

    loop {
        let mut squarred_error_sum = 0.0;
        let num_layers = mlp.layers.len();
        let mut outputs_per_layer: Vec<Vec<Vec<f64>>> = vec![Vec::new(); num_layers];

        for (inp, exp) in inputs.iter().zip(expected.iter()) {
            let output = mlp.forward(inp);

            for (li, layer) in mlp.layers.iter().enumerate() {
                outputs_per_layer[li].push(layer.outputs.clone());
            }

            let last_idx = mlp.layers.len() - 1;
            {
                let last = mlp.layers.iter_mut().last().unwrap();
                for ((o, e), (delta, potential)) in output
                    .iter()
                    .zip(exp.iter())
                    .zip(last.deltas.iter_mut().zip(last.potentials.iter()))
                {
                    let error = e - o;
                    *delta = error * A::derivative(*potential);
                    squarred_error_sum += 0.5 * error * error;
                }
            }

            for i in (0..last_idx).rev() {
                let (current, next) = mlp.layers.split_at_mut(i + 1);
                let current_layer = &mut current[i];
                let next_layer = &next[0];

                for (j, (delta, potential)) in current_layer
                    .deltas
                    .iter_mut()
                    .zip(current_layer.potentials.iter())
                    .enumerate()
                {
                    let mut error_sum = 0.0;
                    for (k, next_neuron) in next_layer.neurons.iter().enumerate() {
                        error_sum += next_neuron.weights[j] * next_layer.deltas[k];
                    }
                    *delta = error_sum * A::derivative(*potential);
                }
            }

            for layer in mlp.layers.iter_mut() {
                let layer_inputs = layer.inputs.clone();
                for (neuron, delta) in layer.neurons.iter_mut().zip(layer.deltas.iter()) {
                    for (w, x) in neuron.weights.iter_mut().zip(layer_inputs.iter()) {
                        *w += learning_rate * delta * x;
                    }
                    neuron.bias += learning_rate * delta;
                }
            }
        }

        let mse = squarred_error_sum / inputs.len() as f64;

        let layer_snaps: Vec<LayerSnapshot> = mlp
            .layers
            .iter()
            .enumerate()
            .map(|(li, layer)| LayerSnapshot {
                neurons: layer
                    .neurons
                    .iter()
                    .map(|n| NeuronSnapshot {
                        weights: n.weights.clone(),
                        bias: n.bias,
                    })
                    .collect(),
                outputs_per_sample: outputs_per_layer[li].clone(),
                deltas: layer.deltas.clone(),
            })
            .collect();

        let snap = MlpEpochSnapshot { epoch: mlp.epoch, layers: layer_snaps, loss: mse };

        if verbose {
            println!("[mlp] epoch {:>5}  MSE = {:.6}", snap.epoch, snap.loss);
        }

        snapshots.push(snap);

        if squarred_error_sum / inputs.len() as f64 <= tolerance {
            break;
        }

        mlp.epoch += 1;
        if let Some(max) = epochs {
            if mlp.epoch + 1 >= max {
                break;
            }
        }
    }

    let total_epochs = mlp.epoch;
    MlpHistory {
        dataset_name: dataset_name.into(),
        learning_rate,
        tolerance,
        max_epochs: epochs.unwrap_or(usize::MAX),
        total_epochs,
        activation: activation_name,
        input_dim,
        snapshots,
    }
}
