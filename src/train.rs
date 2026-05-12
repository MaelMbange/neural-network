use crate::{activation::Activation, perceptron::Perceptron};

pub mod adeline;
pub mod gradient;
pub mod linear;

#[derive(Debug, Clone, Copy)]
pub struct ClassificationStop {
    pub error_limit: usize,
    pub threshold: f64,
    pub values: (f64, f64),
}

pub trait Train {
    fn train<A: Activation>(
        &mut self,
        perceptron: &mut Perceptron<A>,
        dataset: &[(Vec<f64>, f64)],
        epochs: Option<usize>,
    );

    fn classify<A: Activation>(
        &self,
        perceptron: &Perceptron<A>,
        inputs: &[f64],
        step: f64,
        values: (f64, f64),
    ) -> f64 {
        if perceptron.forward(inputs) >= step {
            values.1
        } else {
            values.0
        }
    }

    fn epoch(&self) -> usize;
}
