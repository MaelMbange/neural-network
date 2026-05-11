use crate::neuron::{Neuron, activation::Activation, trainable::Trainable};
use rand::distr::uniform::SampleRange;
use rand::random_range;
use std::fmt::Debug;

#[derive(Debug)]
pub struct Perceptron<A: Activation> {
    bias: f64,
    learning_rate: f64,
    weights: Vec<f64>,
    epoch: usize,
    activation: A,
}

impl<A: Activation> Perceptron<A> {
    pub fn new<R>(
        n_inputs: usize,
        bias: f64,
        learning_rate: f64,
        weight_range: R,
        activation: A,
    ) -> Self
    where
        R: SampleRange<f64> + Clone,
    {
        Perceptron {
            bias,
            learning_rate,
            weights: (0..n_inputs)
                .map(|_| random_range(weight_range.clone()))
                .collect(),
            epoch: 0,
            activation,
        }
    }

    pub fn set_weights(&mut self, weights: Vec<f64>) {
        self.weights = weights;
    }

    pub fn zero_weights(&mut self) {
        self.weights = vec![0.0; self.weights.len()];
    }
}

impl<A: Activation> Neuron for Perceptron<A> {
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

impl<A: Activation> Trainable for Perceptron<A> {
    fn train(&mut self, dataset: &[(Vec<f64>, f64)], epochs: Option<usize>) {
        let mut _epoch = 0;
        loop {
            if let Some(max) = epochs {
                if _epoch >= max {
                    _epoch -= 1;
                    break;
                }
            }

            let mut total_error: usize = 0;

            for (_index, (inputs, expected)) in dataset.iter().enumerate() {
                let prediction = self.predict(inputs);
                let error = *expected - prediction;

                if error != 0f64 {
                    total_error += error.abs() as usize;

                    for (x, w) in inputs.iter().zip(self.weights.iter_mut()) {
                        *w += self.learning_rate * error as f64 * x;
                    }
                    self.bias += self.learning_rate * error as f64;
                }
            }

            if total_error == 0 {
                self.epoch = _epoch;

                break;
            }

            _epoch += 1;
        }
    }
}
