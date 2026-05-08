use crate::neuron::activation::Activation;

#[derive(Debug, Clone)]
pub struct Tanh;

impl Activation for Tanh {
    fn activate(&self, x: f64) -> f64 {
        x.tanh()
    }

    fn derivative(&self, x: f64) -> f64 {
        let tanh_x = x.tanh();
        1.0 - tanh_x * tanh_x
    }
}
