use rna::{activation::Activation, perceptron::Perceptron};

use crate::history::types::{EpochSnapshot, History, NeuronSnapshot};

pub struct LinearHistory {
    pub learning_rate: f64,
    pub verbose: bool,
}

impl LinearHistory {
    pub fn new(learning_rate: f64, verbose: bool) -> Self {
        Self { learning_rate, verbose }
    }

    /// Mirrors `Linear::train` exactly but captures a snapshot after every epoch.
    /// Loss metric: sum of |error| over the dataset (matches `total_error` in original).
    pub fn train_with_history<A: Activation>(
        &self,
        perceptron: &mut Perceptron<A>,
        dataset: &[(Vec<f64>, f64)],
        epochs: Option<usize>,
        dataset_name: impl Into<String>,
    ) -> History {
        let mut snapshots = Vec::new();
        let mut epoch = 0usize;

        loop {
            let mut total_error = 0.0;

            for (inputs, target) in dataset {
                let output = perceptron.forward(inputs);
                let error = target - output;

                if error != 0.0 {
                    total_error += error.abs();
                    for (w, x) in perceptron.weights.iter_mut().zip(inputs.iter()) {
                        *w += self.learning_rate * error * x;
                    }
                    perceptron.bias += self.learning_rate * error;
                }
            }

            let snap = EpochSnapshot {
                epoch,
                neuron: NeuronSnapshot {
                    weights: perceptron.weights.clone(),
                    bias: perceptron.bias,
                },
                loss: total_error,
                misclassified: None,
            };

            if self.verbose {
                println!(
                    "[linear] epoch {:>5}  loss(|err| sum) = {:.6}  w = {:?}  b = {:.6}",
                    epoch, snap.loss, snap.neuron.weights, snap.neuron.bias
                );
            }

            snapshots.push(snap);

            if total_error == 0.0 {
                break;
            }

            epoch += 1;
            if let Some(max) = epochs {
                if epoch + 1 >= max {
                    break;
                }
            }
        }

        let total_epochs = epoch;
        History {
            dataset_name: dataset_name.into(),
            learning_rate: self.learning_rate,
            tolerance: 0.0,
            total_epochs,
            snapshots,
        }
    }
}
