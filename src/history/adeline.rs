use rna::{
    activation::Activation,
    perceptron::Perceptron,
    train::{ClassificationStop, Train, adeline::Adeline},
};

use crate::history::types::{EpochSnapshot, History, NeuronSnapshot};

pub struct AdelineHistory {
    pub learning_rate: f64,
    pub tolerance: f64,
    pub class_stop: Option<ClassificationStop>,
    pub verbose: bool,
}

impl AdelineHistory {
    pub fn new(
        learning_rate: f64,
        tolerance: f64,
        class_stop: Option<ClassificationStop>,
        verbose: bool,
    ) -> Self {
        Self { learning_rate, tolerance, class_stop, verbose }
    }

    /// Mirrors `Adeline::train` exactly (online update then separate loss eval)
    /// and captures a snapshot after every epoch.
    ///
    /// Loss metric: `squarred_error_sum / dataset.len()` — MSE.
    pub fn train_with_history<A: Activation>(
        &self,
        perceptron: &mut Perceptron<A>,
        dataset: &[(Vec<f64>, f64)],
        epochs: Option<usize>,
        dataset_name: impl Into<String>,
    ) -> History {
        let class_stop_owned = self.class_stop;
        let helper = Adeline::new(self.tolerance, self.learning_rate, &class_stop_owned);

        let mut snapshots = Vec::new();
        let mut epoch = 0usize;

        loop {
            // Online weight update (same as original: update per sample, then eval loss)
            for (inputs, expected) in dataset.iter() {
                let output = perceptron.forward(inputs);
                let error = expected - output;
                for (w, x) in perceptron.weights.iter_mut().zip(inputs.iter()) {
                    *w += self.learning_rate * error * x;
                }
                perceptron.bias += self.learning_rate * error;
            }

            // Evaluate loss with updated weights (separate pass, matching original)
            let mut squarred_error_sum = 0.0;
            for (inputs, expected) in dataset.iter() {
                let output = perceptron.forward(inputs);
                let error = expected - output;
                squarred_error_sum += 0.5 * error * error;
            }

            let mse = squarred_error_sum / dataset.len() as f64;

            let misclassified = self.class_stop.map(|cfg| {
                dataset
                    .iter()
                    .filter(|(inputs, expected)| {
                        helper.classify(perceptron, inputs, cfg.threshold, cfg.values) != *expected
                    })
                    .count()
            });

            let snap = EpochSnapshot {
                epoch,
                neuron: NeuronSnapshot {
                    weights: perceptron.weights.clone(),
                    bias: perceptron.bias,
                },
                loss: mse,
                misclassified,
            };

            if self.verbose {
                print!(
                    "[adeline] epoch {:>5}  MSE = {:.6}  w = {:?}  b = {:.6}",
                    epoch, snap.loss, snap.neuron.weights, snap.neuron.bias
                );
                if let Some(mc) = misclassified {
                    print!("  misclassified = {}", mc);
                }
                println!();
            }

            snapshots.push(snap);

            if mse <= self.tolerance {
                break;
            }

            if let Some(cfg) = &self.class_stop {
                let error_count = dataset
                    .iter()
                    .filter(|(inputs, expected)| {
                        helper.classify(perceptron, inputs, cfg.threshold, cfg.values) != *expected
                    })
                    .count();
                if error_count <= cfg.error_limit {
                    break;
                }
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
            tolerance: self.tolerance,
            max_epochs: epochs.unwrap_or(usize::MAX),
            total_epochs,
            snapshots,
        }
    }
}
