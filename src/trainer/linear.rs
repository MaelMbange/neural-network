use crate::{activation::Activation, perceptron::Perceptron, trainer::Train};

#[derive(Debug, Clone, Copy)]
pub struct Linear {
    pub epoch: usize,
    pub learning_rate: f64,
}

impl Linear {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            epoch: 0,
            learning_rate,
        }
    }
}

impl Train for Linear {
    fn train<A: Activation>(
        &mut self,
        perceptron: &mut Perceptron<A>,
        dataset: &[(Vec<f64>, f64)],
        epochs: Option<usize>,
    ) {
        self.epoch = 0;

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

            if total_error == 0.0 {
                break;
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
