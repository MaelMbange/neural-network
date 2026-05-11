use std::fmt::Debug;

use rand::{distr::uniform::SampleRange, random_range};

use crate::neuron::classification_config::ClassificationConfig;
use crate::neuron::history::{NoopObserver, TrainObserver};
use crate::neuron::trainable::Trainable;
use crate::neuron::{Neuron, activation::Activation};

#[derive(Debug, Clone)]
pub struct Gradient<'a, A: Activation> {
    bias: f64,
    weights: Vec<f64>,
    learning_rate: f64,
    tolerance: f64,
    classification_configuration: &'a Option<ClassificationConfig>,
    mean_error: f64,
    epoch: usize,
    activation: A,
    debug: bool,
}

impl<'a, A: Activation> Gradient<'a, A> {
    pub fn new<R>(
        n_inputs: usize,
        bias: f64,
        learning_rate: f64,
        tolerance: f64,
        classification_configuration: &'a Option<ClassificationConfig>,
        weight_range: R,
        activation: A,
    ) -> Self
    where
        R: SampleRange<f64> + Clone,
    {
        Gradient {
            bias,
            weights: (0..n_inputs)
                .map(|_| random_range(weight_range.clone()))
                .collect(),
            learning_rate,
            tolerance,
            classification_configuration,
            epoch: 0,
            mean_error: f64::INFINITY,
            activation,
            debug: false,
        }
    }

    pub fn set_weights(&mut self, weights: Vec<f64>) {
        self.weights = weights;
    }

    pub fn zero_weights(&mut self) {
        self.weights = vec![0.0; self.weights.len()];
    }

    pub fn set_tolerance(&mut self, tolerance: f64) {
        self.tolerance = tolerance;
    }

    /// Trains the neuron using the gradient descent algorithm, notifying
    /// `observer` before the first epoch ([`TrainObserver::on_train_start`]),
    /// after each epoch's batch update ([`TrainObserver::on_epoch_end`]), and
    /// on convergence ([`TrainObserver::on_converged`]).
    ///
    /// [`Trainable::train`] calls this method with a [`NoopObserver`] —
    ///
    pub fn train_with_observer<Obs: TrainObserver<f64>>(
        &mut self,
        dataset: &[(Vec<f64>, f64)],
        epochs: Option<usize>,
        observer: &mut Obs,
    ) {
        observer.on_train_start(&self.weights, self.bias);

        let mut finished = false;
        let mut epoch = 0;

        loop {
            if let Some(max) = epochs {
                if epoch >= max {
                    epoch -= 1;
                    break;
                }
            }

            // reinitialise les deltas et l'erreur totale pour cette époque
            let mut delta_bias = 0.0;
            let mut delta_weights = vec![0.0; self.weights.len()];
            let mut squarred_error_sum = 0.0;

            if self.debug {
                println!("Epoch {}:", epoch);
            }

            //boucle des examples d'entraînement
            for (inputs, expected) in dataset.iter() {
                let prediction = self.predict(inputs);
                let error = expected - prediction;
                squarred_error_sum += 0.5 * error * error;

                if self.debug {
                    println!(
                        "\tInputs: {:.3?}, Weights: {:.3?}, Bias: {:.3}, \
                        Expected: {:.3}, Prediction: {:.3}, Error: {:.3}, Squarred Error Sum: {:.3}",
                        inputs,
                        self.weights,
                        self.bias,
                        expected,
                        prediction,
                        error,
                        squarred_error_sum
                    );
                }

                // on accumule les deltas pour cette époque, qui seront appliqués à la fin de l'époque
                // petite explication:
                //exemple i=1: delta_bias = 0.2 * -1 = -0.2;        delta_weights = [0.2 * -1 * 0, 0.2 * -1 * 0]                = [0, 0]
                //exemple i=2: delta_bias = -0.2 + 0.2 * -1 = -0.4; delta_weights = [0, 0] + [0.2 * -1 * 0, 0.2 * -1 * 1]       = [0, -0.2]
                //exemple i=3: delta_bias = -0.4 + 0.2 * -1 = -0.6; delta_weights = [0, -0.2] + [0.2 * -1 * 1, 0.2 * -1 * 0]    = [-0.2, -0.2]
                //exemple i=4: delta_bias = -0.6 + 0.2 * 1 = -0.4;  delta_weights = [-0.2, -0.2] + [0.2 * 1 * 1, 0.2 * 1 * 1]   = [0, 0]
                delta_bias += self.learning_rate * error;
                for (delta_w, input) in delta_weights.iter_mut().zip(inputs.iter()) {
                    *delta_w += self.learning_rate * error * input;
                }
            }

            //une fois tout les exemples traités, on applique les deltas accumulés aux poids et au biais du neurone
            self.bias += delta_bias;
            for (delta_w, w) in delta_weights.iter().zip(self.weights.iter_mut()) {
                *w += *delta_w;
            }

            let mean_error = squarred_error_sum / dataset.len() as f64;
            self.mean_error = mean_error;

            let classification_error = self.classification_configuration.as_ref().map(|config| {
                dataset
                    .iter()
                    .filter(|(inputs, expected)| {
                        self.classify(inputs, config.threshold, config.values) != *expected
                    })
                    .count()
            });
            let classification_cond = classification_error.is_some_and(|x| {
                x <= self
                    .classification_configuration
                    .as_ref()
                    .unwrap()
                    .error_limit
            });

            if self.debug {
                println!("\tMean Error: {:.3}", mean_error);
                println!(
                    "\tClassification Error: {}",
                    classification_error
                        .map(|e| e.to_string())
                        .unwrap_or_else(|| "N/A".to_string())
                );
            }

            observer.on_epoch_end(epoch, dataset.len(), &self.weights, self.bias, mean_error);

            if mean_error < self.tolerance || classification_cond {
                if self.debug {
                    println!("Training converged at epoch {epoch} with mean error {mean_error:.3}");
                }
                observer.on_converged(epoch, dataset.len(), &self.weights, self.bias);
                finished = true;
                break;
            }

            epoch += 1;
        }

        self.epoch = epoch;

        if !finished && self.debug {
            println!("Training ended after {epoch} without convergence");
        }
    }
}

impl<'a, A: Activation> Neuron for Gradient<'a, A> {
    fn predict(&self, inputs: &[f64]) -> f64 {
        let sum: f64 = self
            .weights
            .iter()
            .zip(inputs.iter())
            .map(|(w, x)| w * x)
            .sum::<f64>()
            + self.bias;

        self.activation.activate(sum)
    }

    fn classify(&self, inputs: &[f64], threshold: f64, values: (f64, f64)) -> f64 {
        if self.predict(inputs) >= threshold {
            values.1
        } else {
            values.0
        }
    }

    fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }
}

impl<'a, A: Activation> Trainable for Gradient<'a, A> {
    fn train(&mut self, dataset: &[(Vec<f64>, f64)], epochs: Option<usize>) {
        self.train_with_observer(dataset, epochs, &mut NoopObserver);
    }
}
