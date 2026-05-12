use std::marker::PhantomData;

use rand::{distr::uniform::SampleRange, random_range};

use crate::activation::Activation;

#[derive(Debug, Clone)]
pub struct Perceptron<A: Activation> {
    pub bias: f64,
    pub weights: Vec<f64>,
    pub activation: PhantomData<A>,
}

impl<A: Activation> Perceptron<A> {
    pub fn new(weights: Vec<f64>, bias: f64) -> Self {
        Self {
            weights,
            bias,
            activation: PhantomData,
        }
    }

    pub fn new_with_random_range<R: SampleRange<f64> + Clone>(
        input_size: usize,
        bias: f64,
        weight_range: R,
    ) -> Self {
        let weights = (0..input_size)
            .map(|_| random_range(weight_range.clone()))
            .collect();

        Self {
            weights,
            bias,
            activation: PhantomData,
        }
    }

    pub fn potential(&self, inputs: &[f64]) -> f64 {
        inputs
            .iter()
            .zip(&self.weights)
            .map(|(x, w)| x * w)
            .sum::<f64>()
            + self.bias
    }

    pub fn forward(&self, inputs: &[f64]) -> f64 {
        let z: f64 = inputs
            .iter()
            .zip(&self.weights)
            .map(|(x, w)| x * w)
            .sum::<f64>()
            + self.bias;

        A::activate(z)
    }
}
