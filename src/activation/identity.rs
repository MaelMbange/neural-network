use crate::activation::{Activation, Derivative};

#[derive(Debug, Clone, Copy)]
pub struct Identity;

impl Activation for Identity {
    fn activate(&self, x: f64) -> f64 {
        x
    }
}

impl Derivative for Identity {
    fn derivative(&self, _x: f64) -> f64 {
        1.0
    }
}
