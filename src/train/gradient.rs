use crate::{
    activation::Activation,
    perceptron::Perceptron,
    train::{ClassificationStop, Train},
};

#[derive(Debug, Clone, Copy)]
pub struct Gradient<'a> {
    pub epoch: usize,
    pub tolerance: f64,
    pub learning_rate: f64,
    pub class_stop: &'a Option<ClassificationStop>,
}

impl<'a> Gradient<'a> {
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

impl<'a> Train for Gradient<'a> {
    fn train<A: Activation>(
        &mut self,
        perceptron: &mut Perceptron<A>,
        dataset: &[(Vec<f64>, f64)],
        epochs: Option<usize>,
    ) {
        self.epoch = 0;

        loop {
            let mut delta_bias = 0.0;
            let mut delta_weights = vec![0.0; perceptron.weights.len()];
            let mut squarred_error_sum = 0.0;

            // chaque exemple sera traite avec les poids et le biais actuels du neurone,
            // et les deltas seront accumulés pour être appliqués à la fin de l'époque
            for (inputs, expected) in dataset.iter() {
                let output = perceptron.forward(inputs);
                let error = expected - output;
                squarred_error_sum += 0.5 * error * error;

                // on accumule les deltas pour cette époque, qui seront appliqués à la fin de l'époque
                delta_bias += self.learning_rate * error;
                for (w, x) in delta_weights.iter_mut().zip(inputs.iter()) {
                    *w += self.learning_rate * error * x;
                }
            }

            //une fois tout les exemples traités, on applique les deltas accumulés aux poids et au biais du neurone
            perceptron.bias += delta_bias;
            for (w, dw) in perceptron.weights.iter_mut().zip(delta_weights.iter()) {
                *w += *dw;
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
