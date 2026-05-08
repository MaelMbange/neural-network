use crate::neuron::activation::Activation;

#[derive(Debug, Clone)]
pub struct Step;

impl Activation for Step {
    fn activate(&self, x: f64) -> f64 {
        if x >= 0.0 { 1.0 } else { 0.0 }
    }

    fn derivative(&self, _x: f64) -> f64 {
        0.0
    }
}
