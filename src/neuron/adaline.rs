use crate::neuron::classification_config::ClassificationConfig;
use crate::neuron::history::{NoopObserver, TrainObserver};
use crate::neuron::{Neuron, activation::Activation, trainable::Trainable};
use rand::distr::uniform::SampleRange;
use rand::random_range;
use std::fmt::Debug;

#[derive(Debug)]
pub struct Adaline<'a, A: Activation> {
    bias: f64,
    learning_rate: f64,
    weights: Vec<f64>,
    activation: A,
    tolerance: f64,
    classification_configuration: &'a Option<ClassificationConfig>,
    mean_error: f64,
    epoch: usize,
    debug: bool,
}

impl<'a, A: Activation> Adaline<'a, A> {
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
        Adaline {
            bias,
            learning_rate,
            weights: (0..n_inputs)
                .map(|_| random_range(weight_range.clone()))
                .collect(),
            activation,
            tolerance,
            classification_configuration,
            epoch: 0,
            mean_error: f64::INFINITY,
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

    /// Entraîne le perceptron en notifiant `observer` à chaque mise à jour des
    /// poids (via [`TrainObserver::on_sample_update`]), avant la première époque
    /// ([`TrainObserver::on_train_start`]) et lors de la convergence
    /// ([`TrainObserver::on_converged`]).
    ///
    /// [`Trainable::train`] appelle cette méthode avec un [`NoopObserver`] —
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

            let mut squarred_error_sum = 0.0;

            if self.debug {
                println!("Epoch {epoch}");
            }

            for (inputs, expected) in dataset.iter() {
                let prediction = self.predict(inputs);
                let error = *expected - prediction;

                if self.debug {
                    println!(
                        "\tInputs: {:?}, Weights: {:.3?}, Bias: {}, \
                          Expected: {}, Prediction: {}, Error: {}",
                        inputs, self.weights, self.bias, expected, prediction, error,
                    );
                }

                for (x, w) in inputs.iter().zip(self.weights.iter_mut()) {
                    *w += self.learning_rate * error as f64 * x;
                }
                self.bias += self.learning_rate * error as f64;
            }

            for (inputs, expected) in dataset {
                let prediction = self.predict(inputs);
                let error = *expected - prediction;
                squarred_error_sum += 0.5 * error * error;
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
                    println!("Training converged at epoch {epoch}");
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

impl<'a, A: Activation> Neuron for Adaline<'a, A> {
    fn predict(&self, inputs: &[f64]) -> f64 {
        assert_eq!(
            inputs.len(),
            self.weights.len(),
            "Input number must match weights number"
        );

        let p: f64 = inputs
            .iter()
            .zip(self.weights.iter())
            .map(|(x, w)| x * w)
            .sum::<f64>()
            + self.bias;

        self.activation.activate(p)
    }

    fn classify(&self, inputs: &[f64], step: f64, values: (f64, f64)) -> f64 {
        if self.predict(inputs) >= step {
            values.1
        } else {
            values.0
        }
    }

    fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }
}

impl<'a, A: Activation> Trainable for Adaline<'a, A> {
    fn train(&mut self, dataset: &[(Vec<f64>, f64)], epochs: Option<usize>) {
        self.train_with_observer(dataset, epochs, &mut NoopObserver);
    }
}

impl<'a, A: Activation + Clone> Clone for Adaline<'_, A> {
    fn clone(&self) -> Self {
        Adaline {
            bias: self.bias,
            learning_rate: self.learning_rate,
            weights: self.weights.clone(),
            activation: self.activation.clone(),
            tolerance: self.tolerance,
            classification_configuration: self.classification_configuration,
            epoch: self.epoch,
            mean_error: self.mean_error,
            debug: self.debug,
        }
    }
}
