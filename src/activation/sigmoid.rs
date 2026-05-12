use crate::activation::{Activation, Derivative};

#[derive(Debug, Clone, Copy)]
pub struct Sigmoid;

impl Activation for Sigmoid {
    fn activate(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
}

impl Derivative for Sigmoid {
    fn derivative(x: f64) -> f64 {
        let sigmoid_x = Sigmoid::activate(x);
        sigmoid_x * (1.0 - sigmoid_x)
    }
}
