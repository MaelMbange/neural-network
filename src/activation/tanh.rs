use crate::activation::{Activation, Derivative};

#[derive(Debug, Clone, Copy)]
pub struct Tanh;

impl Activation for Tanh {
    fn activate(x: f64) -> f64 {
        x.tanh()
    }
}

impl Derivative for Tanh {
    fn derivative(x: f64) -> f64 {
        1.0 - x.tanh().powi(2)
    }
}
