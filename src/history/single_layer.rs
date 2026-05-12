use std::marker::PhantomData;

use rna::{activation::identity::Identity, perceptron::Perceptron, train::ClassificationStop};

use crate::history::{
    adeline::AdelineHistory,
    gradient::GradientHistory,
    types::{NeuronTraceHistory, SingleLayerHistory},
};

/// Mirrors the local `SingleLayer` struct from `src/bin/adeline_singlelayer_3_1.rs`
/// and `src/bin/gradient_singlelayer_3_1.rs`.
///
/// Each neuron in the layer is trained independently (one trainer instance reused)
/// on its own column of the output matrix.  History is captured per neuron.
pub enum SingleLayerTrainerKind {
    Gradient {
        learning_rate: f64,
        tolerance: f64,
        class_stop: Option<ClassificationStop>,
    },
    Adeline {
        learning_rate: f64,
        tolerance: f64,
        class_stop: Option<ClassificationStop>,
    },
}

pub fn train_single_layer_with_history(
    neuron_template: &Perceptron<Identity>,
    neuron_count: usize,
    trainer_kind: &SingleLayerTrainerKind,
    inputs: &[Vec<f64>],
    outputs: &[Vec<f64>],
    epochs: Option<usize>,
    dataset_name: impl Into<String>,
    verbose: bool,
) -> SingleLayerHistory {
    assert!(
        outputs.iter().all(|o| o.len() == neuron_count),
        "output column count must match neuron_count"
    );

    let name = dataset_name.into();
    let mut neuron_histories = Vec::with_capacity(neuron_count);

    let (learning_rate, tolerance) = match trainer_kind {
        SingleLayerTrainerKind::Gradient { learning_rate, tolerance, .. } => {
            (*learning_rate, *tolerance)
        }
        SingleLayerTrainerKind::Adeline { learning_rate, tolerance, .. } => {
            (*learning_rate, *tolerance)
        }
    };

    for i in 0..neuron_count {
        let mut neuron = Perceptron::<Identity> {
            weights: neuron_template.weights.clone(),
            bias: neuron_template.bias,
            activation: PhantomData,
        };

        let sub_dataset: Vec<(Vec<f64>, f64)> = inputs
            .iter()
            .zip(outputs.iter())
            .map(|(inp, out)| (inp.clone(), out[i]))
            .collect();

        if verbose {
            println!("[single_layer] training neuron {}/{}", i + 1, neuron_count);
        }

        let history = match trainer_kind {
            SingleLayerTrainerKind::Gradient { learning_rate, tolerance, class_stop } => {
                GradientHistory::new(*learning_rate, *tolerance, *class_stop, verbose)
                    .train_with_history(
                        &mut neuron,
                        &sub_dataset,
                        epochs,
                        format!("{} neuron {}", name, i),
                    )
            }
            SingleLayerTrainerKind::Adeline { learning_rate, tolerance, class_stop } => {
                AdelineHistory::new(*learning_rate, *tolerance, *class_stop, verbose)
                    .train_with_history(
                        &mut neuron,
                        &sub_dataset,
                        epochs,
                        format!("{} neuron {}", name, i),
                    )
            }
        };

        neuron_histories.push(NeuronTraceHistory {
            neuron_index: i,
            total_epochs: history.total_epochs,
            snapshots: history.snapshots,
        });
    }

    SingleLayerHistory {
        dataset_name: name,
        learning_rate,
        tolerance,
        max_epochs: epochs.unwrap_or(usize::MAX),
        neuron_histories,
    }
}
