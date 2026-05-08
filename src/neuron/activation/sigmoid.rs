use crate::neuron::activation::Activation;

#[derive(Debug, Clone)]
pub struct Sigmoid;

impl Activation for Sigmoid {
    fn activate(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn derivative(&self, x: f64) -> f64 {
        let sigmoid_x = self.activate(x);
        sigmoid_x * (1.0 - sigmoid_x)
    }
}
