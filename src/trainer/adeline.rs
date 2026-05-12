use super::ClassificationStop;
use crate::{activation::Activation, perceptron::Perceptron, trainer::Train};

#[derive(Debug, Clone, Copy)]
pub struct Adeline<'a> {
    pub epoch: usize,
    pub tolerance: f64,
    pub learning_rate: f64,
    pub class_stop: &'a Option<ClassificationStop>,
}

impl<'a> Adeline<'a> {
    pub fn new(
        tolerance: f64,
        learning_rate: f64,
        class_stop: &'a Option<ClassificationStop>,
    ) -> Self {
        Self {
            epoch: 0,
            tolerance,
            learning_rate,
            class_stop,
        }
    }
}

impl<'a> Train for Adeline<'a> {
    fn train<A: Activation>(
        &mut self,
        perceptron: &mut Perceptron<A>,
        dataset: &[(Vec<f64>, f64)],
        epochs: Option<usize>,
    ) {
        self.epoch = 0;

        loop {
            let mut squarred_error_sum = 0.0;

            for (inputs, expected) in dataset.iter() {
                let output = perceptron.forward(inputs);
                let error = expected - output;

                for (w, x) in perceptron.weights.iter_mut().zip(inputs.iter()) {
                    *w += self.learning_rate * error * x;
                }

                perceptron.bias += self.learning_rate * error;
            }

            for (inputs, expected) in dataset.iter() {
                let output = perceptron.forward(inputs);
                let error = expected - output;
                squarred_error_sum += 0.5 * error * error;
            }

            if squarred_error_sum / dataset.len() as f64 <= self.tolerance {
                break;
            }

            if let Some(cfg) = self.class_stop {
                let error_count = dataset
                    .iter()
                    .filter(|(inputs, expected)| {
                        self.classify(perceptron, inputs, cfg.threshold, cfg.values) != *expected
                    })
                    .count();

                if error_count <= cfg.error_limit {
                    break;
                }
            }

            self.epoch += 1;
            if let Some(max) = epochs {
                if self.epoch + 1 >= max {
                    break;
                }
            }
        }
    }

    fn epoch(&self) -> usize {
        self.epoch
    }
}
