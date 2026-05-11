use crate::neuron::history::{NoopObserver, TrainObserver};
use crate::neuron::{Neuron, activation::Activation, trainable::Trainable};
use rand::distr::uniform::SampleRange;
use rand::random_range;
use std::fmt::Debug;

#[derive(Debug, Clone)]
pub struct Perceptron<A: Activation> {
    bias: f64,
    learning_rate: f64,
    weights: Vec<f64>,
    epoch: usize,
    activation: A,
    debug: bool,
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
            debug: false,
        }
    }

    pub fn set_weights(&mut self, weights: Vec<f64>) {
        self.weights = weights;
    }

    pub fn zero_weights(&mut self) {
        self.weights = vec![0.0; self.weights.len()];
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

            let mut total_error: usize = 0;

            if self.debug {
                println!("Epoch {epoch}");
            }

            for (index, (inputs, expected)) in dataset.iter().enumerate() {
                let prediction = self.predict(inputs);
                let error = *expected - prediction;

                if self.debug {
                    println!(
                        "\tInputs: {:?}, Weights: {:.3?}, Bias: {}, \
                        Expected: {}, Prediction: {}, Error: {}",
                        inputs, self.weights, self.bias, expected, prediction, error
                    );
                }

                if error != 0f64 {
                    total_error += error.abs() as usize;

                    for (x, w) in inputs.iter().zip(self.weights.iter_mut()) {
                        *w += self.learning_rate * error as f64 * x;
                    }
                    self.bias += self.learning_rate * error as f64;

                    observer.on_sample_update(
                        epoch,
                        index,
                        &self.weights,
                        self.bias,
                        inputs,
                        expected,
                        &prediction,
                        &error,
                        total_error as f64,
                    );
                }
            }

            if self.debug {
                println!("\tTotal error: {}", total_error);
            }

            if total_error == 0 {
                self.epoch = epoch;
                if self.debug {
                    println!("Training converged at epoch {epoch}");
                }
                observer.on_converged(self.epoch, dataset.len(), &self.weights, self.bias);
                finished = true;
                break;
            }

            epoch += 1;
        }

        if !finished && self.debug {
            println!("Training ended after {epoch} without convergence");
        }
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

    fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }
}

impl<A: Activation> Trainable for Perceptron<A> {
    fn train(&mut self, dataset: &[(Vec<f64>, f64)], epochs: Option<usize>) {
        self.train_with_observer(dataset, epochs, &mut NoopObserver);
    }
}
