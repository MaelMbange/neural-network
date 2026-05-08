pub mod identity;
pub mod sigmoid;
pub mod step;
pub mod tanh;

pub trait Activation {
    fn activate(&self, x: f64) -> f64;
    fn derivative(&self, x: f64) -> f64;
}
