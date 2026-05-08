pub mod activation;
pub mod adaline;
pub mod classification_config;
pub mod gradient;
pub mod history;
pub mod perceptron;
pub mod trainable;

pub trait Neuron {
    fn predict(&self, inputs: &[f64]) -> f64;
    fn classify(&self, inputs: &[f64], threshold: f64, values: (f64, f64)) -> f64;
    fn set_debug(&mut self, debug: bool);
}
