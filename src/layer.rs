use rand::distr::uniform::SampleRange;

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
    pub fn new<R: SampleRange<f64> + Clone>(
        neuron_count: usize,
        num_inputs: usize,
        weight_range: R,
    ) -> Self {
        let neurons = (0..neuron_count)
            .map(|_| Perceptron::new_with_random_range(num_inputs, 0.0, weight_range.clone()))
            .collect();

        Self {
            neurons,
            potentials: vec![0.0; neuron_count],
            outputs: vec![0.0; neuron_count],
            inputs: vec![0.0; num_inputs],
            deltas: vec![0.0; neuron_count],
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
            deltas: vec![0.0; neuron_count],
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
    pub epoch: usize,
}

impl<A: Activation + Derivative> MLP<A> {
    pub fn new<R: SampleRange<f64> + Clone>(
        layers_sizes: &[usize],
        input_size: usize,
        weight_range: R,
    ) -> Self {
        let mut layers = Vec::new();
        let mut current_input_size = input_size;

        for &size in layers_sizes {
            layers.push(Layer::<A>::new(
                size,
                current_input_size,
                weight_range.clone(),
            ));
            current_input_size = size;
        }

        Self { layers, epoch: 0 }
    }

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
        _tolerance: f64,
        _epochs: Option<usize>,
    ) {
        self.epoch = 0;

        loop {
            let mut squarred_error_sum = 0.0;

            for (inputs, expected) in _inputs.iter().zip(_expected.iter()) {
                //etape 1 : propagation avant pour calculer les sorties et les potentiels de chaque neurone
                let output = self.forward(inputs);
                let last = self.layers.iter_mut().last().unwrap();

                //etape 2a: calcul du signal d'erreur le la couche de sortie
                for ((o, e), (delta, potential)) in output
                    .iter()
                    .zip(expected.iter())
                    .zip(last.deltas.iter_mut().zip(last.potentials.iter()))
                {
                    let error = e - o;
                    *delta = error * A::derivative(*potential); // signal d'erreur de la couche de sortie pour chaque neurone
                    squarred_error_sum += 0.5 * error * error; // on accumule l'erreur quadratique pour cette époque
                }

                //etape 2b: calcul du signal d'erreur pour les couches cachées, en remontant de la couche de sortie vers la couche d'entrée
                // mr, si vous lisez ceci, sachez que j'ai eu envie de me tirer une balle en écrivant cette partie.
                // (╯°□°）╯︵ ┻━┻
                for i in (0..self.layers.len() - 1).rev() {
                    let (current, next) = self.layers.split_at_mut(i + 1);
                    let current_layer = &mut current[i];
                    let next_layer = &next[0];

                    // pour chaque neurone on recupere son vecteur delta et son potentiel
                    for (j, (delta, potential)) in current_layer
                        .deltas
                        .iter_mut()
                        .zip(current_layer.potentials.iter())
                        .enumerate()
                    {
                        //formule: delta(C) = derivative(k_c) * sum(w_s * delta(S))
                        let mut error_sum = 0.0;
                        for (k, next_neuron) in next_layer.neurons.iter().enumerate() {
                            error_sum += next_neuron.weights[j] * next_layer.deltas[k];
                            // on multiplie le poids par le signal d'erreur de la couche suivante,
                            // ici: sum(w_s * delta(S))
                        }
                        *delta = error_sum * A::derivative(*potential); // signal d'erreur de la couche cachée pour chaque neurone
                    }
                }

                //etape 3: Correction des poids synaptique de chaque couches (a+b)
                // la partie simple
                for layer in self.layers.iter_mut() {
                    let layer_inputs = layer.inputs.clone();
                    for (neuron, delta) in layer.neurons.iter_mut().zip(layer.deltas.iter()) {
                        // on applique la formule: wsc(t+1) = wsc(t) + learning_rate * delta * y (y etant l'entrée du neurone)
                        for (w, x) in neuron.weights.iter_mut().zip(layer_inputs.iter()) {
                            *w += _learning_rate * delta * x; // correction du poids synaptique : w = w + learning_rate * delta * input
                        }
                        neuron.bias += _learning_rate * delta; // correction du biais : b = b + learning_rate * delta
                    }
                }
            }

            // on verifie si l'erreur quadratique moyenne est inférieure ou égale à la tolérance pour arrêter l'entraînement
            if squarred_error_sum / _inputs.len() as f64 <= _tolerance {
                break;
            }

            self.epoch += 1;
            if let Some(max) = _epochs {
                if self.epoch + 1 >= max {
                    break;
                }
            }
        }
    }

    pub fn classify_binary(&mut self, inputs: &[f64], threshold: f64, low: f64, high: f64) -> f64 {
        let y = self.forward(inputs);
        assert_eq!(y.len(), 1, "classify_binary attend une unique sortie");
        if y[0] >= threshold { high } else { low }
    }

    pub fn classify_argmax(&mut self, inputs: &[f64]) -> usize {
        let y = self.forward(inputs);
        assert!(
            y.len() >= 2,
            "classify_argmax attend au moins 2 sorties (classification multi-classes)"
        );
        y.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    }
}
