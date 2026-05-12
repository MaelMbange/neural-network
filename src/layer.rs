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
    pub deltas: Vec<f64>,
}

impl<A: Activation + Derivative> Layer<A> {
    pub fn new(neuron_count: usize, input_count: usize) -> Self {
        let neurons = (0..neuron_count)
            .map(|_| Perceptron::new_with_random_range(input_count, 0.0, -1.0..=1.0))
            .collect();

        Self {
            neurons,
            potentials: vec![0.0; neuron_count],
            outputs: vec![0.0; neuron_count],
            inputs: vec![0.0; input_count],
        }
    }

    pub fn new_with_weights(
        neuron_count: usize,
        input_count: usize,
        weights: Vec<Vec<f64>>,
    ) -> Self {
        assert!(
            weights.iter().filter(|w| w.len() == input_count).count() == neuron_count,
            "Le nombre de lignes de poids doit correspondre au nombre d'entrées"
        );

        let neurons = weights
            .into_iter()
            .map(|w| Perceptron::new(w, 0.0))
            .collect();

        Self {
            neurons,
            potentials: vec![0.0; neuron_count],
            outputs: vec![0.0; neuron_count],
            inputs: vec![0.0; input_count],
        }
    }

    pub fn forward(&mut self, inputs: &[f64]) -> Vec<f64> {
        self.inputs = inputs.to_vec();

        for (i, neuron) in self.neurons.iter().enumerate() {
            self.potentials[i] = neuron.potential(inputs);
            self.outputs[i] = neuron.forward(inputs);
        }

        self.outputs.clone()
    }
}

#[derive(Debug, Clone)]
pub struct MLP<A: Activation + Derivative> {
    pub layers: Vec<Layer<A>>,
}

impl<A: Activation + Derivative> MLP<A> {
    pub fn forward(&mut self, inputs: &[f64]) -> Vec<f64> {
        let mut current_inputs = inputs.to_vec();

        for layer in &mut self.layers {
            current_inputs = layer.forward(&current_inputs);
        }

        current_inputs
    }

    pub fn train(
        &mut self,
        _inputs: &[Vec<f64>],
        _expected: &[Vec<f64>],
        _learning_rate: f64,
        _epochs: Option<usize>,
    ) {
        let epoch = 0;

        loop {
            let mut delta_bias = 0.0;
            let mut delta_weights = vec![0.0; perceptron.weights.len()];
            let mut squarred_error_sum = 0.0;

            epoch += 1;
            if let Some(max) = _epochs {
                if epoch + 1 >= max {
                    break;
                }
            }
        }
    }
}
