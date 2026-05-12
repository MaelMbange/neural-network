use crate::activation::Activation;

#[derive(Debug, Clone, Copy)]
pub struct Step;

impl Activation for Step {
    fn activate(&self, x: f64) -> f64 {
        if x >= 0.0 { 1.0 } else { 0.0 }
    }
}
