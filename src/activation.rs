pub mod identity;
pub mod sigmoid;
pub mod step;
pub mod tanh;

pub trait Activation {
    fn activate(x: f64) -> f64;
}

pub trait Derivative {
    fn derivative(x: f64) -> f64;
}
