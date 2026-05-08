use std::fmt::Debug;

use crate::neuron::trainable::Trainable;

#[derive(Debug)]
pub struct SingleLayer<N: Trainable + Debug> {
    pub neurons: Vec<N>,
    pub debug: bool,
}

impl<N: Trainable + Debug> SingleLayer<N> {
    pub fn new(neurons: Vec<N>) -> Self {
        Self {
            neurons,
            debug: false,
        }
    }

    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
        for neuron in &mut self.neurons {
            neuron.set_debug(debug);
        }
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
        if self.debug {
            println!("Start training SingleLayer network.");
            println!("Neurons before training:");
            for (i, neuron) in self.neurons.iter().enumerate() {
                println!("Neuron {}: {:#?}", i, neuron);
            }
        }

        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            let sub_dataset: Vec<(Vec<f64>, f64)> = dataset
                .iter()
                .map(|(inputs, outputs)| (inputs.clone(), outputs[i]))
                .collect();
            if self.debug {
                println!("Neuron: {}", i);
            }
            neuron.train(&sub_dataset, epochs);
        }

        if self.debug {
            println!();
            println!("Finished training SingleLayer network.");
            println!("Neurons after training:");
            for (i, neuron) in self.neurons.iter().enumerate() {
                println!("Neuron {}: {:#?}", i, neuron);
            }
        }
    }
}
