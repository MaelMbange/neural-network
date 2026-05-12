use rna::{
    activation::Activation,
    perceptron::Perceptron,
    train::{ClassificationStop, Train, gradient::Gradient},
};

use crate::history::types::{EpochSnapshot, History, NeuronSnapshot};

pub struct GradientHistory {
    pub learning_rate: f64,
    pub tolerance: f64,
    pub class_stop: Option<ClassificationStop>,
    pub verbose: bool,
}

impl GradientHistory {
    pub fn new(
        learning_rate: f64,
        tolerance: f64,
        class_stop: Option<ClassificationStop>,
        verbose: bool,
    ) -> Self {
        Self { learning_rate, tolerance, class_stop, verbose }
    }

    /// Mirrors `Gradient::train` exactly but captures a snapshot after every epoch.
    ///
    /// Loss metric: `squarred_error_sum / dataset.len()` — MSE matching the original
    /// stopping condition.
    pub fn train_with_history<A: Activation>(
        &self,
        perceptron: &mut Perceptron<A>,
        dataset: &[(Vec<f64>, f64)],
        epochs: Option<usize>,
        dataset_name: impl Into<String>,
    ) -> History {
        // We need a `Gradient` instance to reuse its `classify` method when
        // evaluating misclassification count.
        let class_stop_owned = self.class_stop;
        let helper = Gradient::new(self.tolerance, self.learning_rate, &class_stop_owned);

        let mut snapshots = Vec::new();
        let mut epoch = 0usize;

        loop {
            let mut delta_bias = 0.0;
            let mut delta_weights = vec![0.0; perceptron.weights.len()];
            let mut squarred_error_sum = 0.0;

            for (inputs, expected) in dataset.iter() {
                let output = perceptron.forward(inputs);
                let error = expected - output;
                squarred_error_sum += 0.5 * error * error;

                delta_bias += self.learning_rate * error;
                for (dw, x) in delta_weights.iter_mut().zip(inputs.iter()) {
                    *dw += self.learning_rate * error * x;
                }
            }

            perceptron.bias += delta_bias;
            for (w, dw) in perceptron.weights.iter_mut().zip(delta_weights.iter()) {
                *w += *dw;
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
                    "[gradient] epoch {:>5}  MSE = {:.6}  w = {:?}  b = {:.6}",
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
            total_epochs,
            snapshots,
        }
    }
}
