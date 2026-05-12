use crate::activation::{Activation, Derivative};

#[derive(Debug, Clone, Copy)]
pub struct Tanh;

impl Activation for Tanh {
    fn activate(&self, x: f64) -> f64 {
        x.tanh()
    }
}

impl Derivative for Tanh {
    fn derivative(&self, x: f64) -> f64 {
        1.0 - x.tanh().powi(2)
    }
}
