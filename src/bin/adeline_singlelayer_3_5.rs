use std::marker::PhantomData;

use rna::{
    activation::identity::Identity,
    csv_reader::load_dataset_multi,
    perceptron::Perceptron,
    trainer::{ClassificationStop, Train, adeline::Adeline},
};

#[derive(Debug)]
struct SingleLayer<T: Train> {
    neurons: Vec<Perceptron<Identity>>,
    neurons_epochs: Vec<usize>,
    trainer: T,
}

impl<T: Train> SingleLayer<T> {
    pub fn new(neurons: Vec<Perceptron<Identity>>, trainer: T) -> Self {
        Self {
            neurons,
            neurons_epochs: vec![],
            trainer,
        }
    }

    pub fn _forward(&self, inputs: &[f64]) -> Vec<f64> {
        self.neurons
            .iter()
            .map(|neuron| neuron.forward(inputs))
            .collect()
    }

    pub fn classify(&self, inputs: &[f64], threshold: f64, values: (f64, f64)) -> Vec<f64> {
        self.neurons
            .iter()
            .map(|neuron| self.trainer.classify(neuron, inputs, threshold, values))
            .collect()
    }

    pub fn train(&mut self, inputs: &[Vec<f64>], outputs: &[Vec<f64>], epochs: Option<usize>) {
        // on vefifie juste que les sorties attendues ont la même longueur que le nombre de neurones dans la couche
        for (_, output) in outputs.iter().enumerate() {
            if output.len() != self.neurons.len() {
                panic!("Output length must match the number of neurons in the layer");
            }
        }

        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            let sub_inputs: Vec<Vec<f64>> = inputs.to_vec();
            let sub_outputs: Vec<f64> = outputs.iter().map(|output| output[i]).collect();

            let sub_dataset: Vec<(Vec<f64>, f64)> = sub_inputs
                .into_iter()
                .zip(sub_outputs.into_iter())
                .collect();
            self.trainer.train(neuron, &sub_dataset, epochs);
            self.neurons_epochs.push(self.trainer.epoch());
        }
    }
}

fn main() {
    let (inputs, targets) = load_dataset_multi("Datas/Datas/table_3_5.csv", 25, 4, false)
        .expect("Failed to load dataset");

    let _conf = Some(ClassificationStop {
        error_limit: 0,
        threshold: 0.0,
        values: (-1.0, 1.0),
    });

    let _conf = None;

    let p = Perceptron {
        bias: 0.0,
        weights: vec![0.0; 25],
        activation: PhantomData::<Identity>,
    };

    let trainer = Adeline::new(0.05, 0.001, &_conf);

    let mut layer = SingleLayer::new(vec![p.clone(); 4], trainer);

    layer.train(&inputs, &targets, Some(1_000));

    println!("{layer:#?}");

    println!("Classification results:");
    for (i, (input, target)) in inputs.iter().zip(targets.iter()).enumerate() {
        let output = layer.classify(input, 0.0, (-1.0, 1.0));
        println!("{i} - Target: {:?}, Output: {:?}", target, output);
    }

    println!("\nClassify a new data point:");
    let mut new_data = vec![0.0; 23];
    new_data.push(1.0);
    new_data.push(1.0);

    let output = layer.classify(&new_data, 0.0, (-1.0, 1.0));
    println!("New data output: {:?}", output);

    println!("Prediction:");
    let prediction = layer.classify(&new_data, 0.0, (-1.0, 1.0));
    println!("New data prediction: {:?}", prediction);
}
