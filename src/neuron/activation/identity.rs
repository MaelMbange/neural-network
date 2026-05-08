use crate::neuron::activation::Activation;

#[derive(Debug, Clone)]
pub struct Identity;

impl Activation for Identity {
    fn activate(&self, x: f64) -> f64 {
        x
    }

    fn derivative(&self, _x: f64) -> f64 {
        1.0
    }
}
