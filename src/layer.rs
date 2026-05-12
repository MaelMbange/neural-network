use crate::{
    activation::{Activation, Derivative},
    perceptron::Perceptron,
};

#[derive(Debug, Clone)]
pub struct Layer<A: Activation + Derivative> {
    pub neurons: Vec<Perceptron<A>>,

    // pour gerer la propagation avant et la retropropagation des erreurs
    pub potentials: Vec<f64>,
    pub outputs: Vec<f64>,
    pub inputs: Vec<f64>,
}

pub struct MLP<A: Activation + Derivative> {
    pub layers: Vec<Layer<A>>,
}
