use crate::neuron::trainable::Trainable;

#[derive(Debug, Clone)]
pub struct Layer<N: Trainable> {
    pub neurons: Vec<N>,
}

impl<N: Trainable> Layer<N> {
    pub fn new(n_neurons: usize, neuron_factory: impl Fn() -> N) -> Self {
        let neurons = (0..n_neurons).map(|_| neuron_factory()).collect();
        Self { neurons }
    }
}
