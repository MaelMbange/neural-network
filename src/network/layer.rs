use crate::neuron::trainable::Trainable;

#[derive(Debug, Clone)]
pub struct Layer<N: Trainable> {
    pub neurons: Vec<N>,
}

impl<N: Trainable> Layer<N> {
    pub fn new(neuron_count: usize, neuron_factory: impl Fn() -> N) -> Self {
        let neurons = (0..neuron_count).map(|_| neuron_factory()).collect();
        Self { neurons }
    }

    pub fn predict(&self, inputs: &[f64]) -> Vec<f64> {
        self.neurons.iter().map(|n| n.predict(inputs)).collect()
    }

    pub fn classify(&self, inputs: &[f64], threshold: f64, values: (f64, f64)) -> Vec<f64> {
        self.neurons
            .iter()
            .map(|n| n.classify(inputs, threshold, values))
            .collect()
    }

    pub fn train(&mut self, dataset: &[(Vec<f64>, Vec<f64>)], epochs: Option<usize>) {
        // on vefifie juste que les sorties attendues ont la même longueur que le nombre de neurones dans la couche
        for (idx, (_, expected)) in dataset.iter().enumerate() {
            assert_eq!(
                expected.len(),
                self.neurons.len(),
                "sample {idx}: expected output length {} does not match number of neurons {}",
                expected.len(),
                self.neurons.len(),
            );
        }

        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            let sub_dataset: Vec<(Vec<f64>, f64)> = dataset
                .iter()
                .map(|(inputs, outputs)| (inputs.clone(), outputs[i]))
                .collect();
            neuron.train(&sub_dataset, epochs);
        }
    }
}

#[derive(Debug, Clone)]
pub struct Network<L: Trainable> {
    pub layers: Vec<L>,
}

impl<L: Trainable> Network<L> {
    pub fn predict(&self, inputs: &[f64]) -> Vec<f64> {}

    pub fn train(&mut self, dataset: &[(Vec<f64>, Vec<f64>)], epochs: Option<usize>) {
        loop {}
    }
}
