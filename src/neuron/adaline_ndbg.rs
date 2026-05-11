use crate::neuron::classification_config::ClassificationConfig;
use crate::neuron::{Neuron, activation::Activation, trainable::Trainable};
use rand::distr::uniform::SampleRange;
use rand::random_range;
use std::fmt::Debug;

#[derive(Debug, Clone)]
pub struct Adaline<'a, A: Activation> {
    bias: f64,
    learning_rate: f64,
    weights: Vec<f64>,
    activation: A,
    tolerance: f64,
    classification_configuration: &'a Option<ClassificationConfig>,
    mean_error: f64,
    epoch: usize,
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

    fn set_debug(&mut self, _debug: bool) {}
}

impl<'a, A: Activation> Trainable for Adaline<'a, A> {
    fn train(&mut self, dataset: &[(Vec<f64>, f64)], epochs: Option<usize>) {
        let mut epoch = 0;
        loop {
            if let Some(max) = epochs {
                if epoch >= max {
                    epoch -= 1;
                    break;
                }
            }

            //erreur quadratique (pas la moyenne)
            let mut squarred_error_sum = 0.0;

            for (inputs, expected) in dataset.iter() {
                let prediction = self.predict(inputs);
                let error = *expected - prediction;

                //mise a jour des poids et du biais a chaque exemple d'entrainement
                for (x, w) in inputs.iter().zip(self.weights.iter_mut()) {
                    *w += self.learning_rate * error as f64 * x;
                }
                self.bias += self.learning_rate * error as f64;
            }

            // une fois les poids mis a jour pour tous les exemples d'entrainement, on calcule l'erreur quadratique moyenne pour cette époque
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

            if mean_error < self.tolerance || classification_cond {
                break;
            }

            epoch += 1;
        }

        self.epoch = epoch;
    }
}
