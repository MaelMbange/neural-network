use crate::neuron::Neuron;

pub trait Trainable: Neuron {
    fn train(&mut self, dataset: &[(Vec<f64>, f64)], epochs: Option<usize>);
}
