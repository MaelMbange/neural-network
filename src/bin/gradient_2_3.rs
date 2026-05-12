use std::marker::PhantomData;

use rna::{
    activation::identity::Identity,
    perceptron::Perceptron,
    trainer::{Train, gradient::Gradient},
};

fn main() {
    let dataset = [
        (vec![0.0, 0.0], -1.0),
        (vec![0.0, 1.0], -1.0),
        (vec![1.0, 0.0], -1.0),
        (vec![1.0, 1.0], 1.0),
    ];

    let mut perceptron = Perceptron {
        weights: vec![0.0; 2],
        bias: 0.0,
        activation: PhantomData::<Identity>,
    };

    let mut trainer = Gradient {
        epoch: 0,
        tolerance: 0.125001,
        learning_rate: 0.2,
        class_stop: &None,
    };

    println!("Perceptron before: {:#?}", perceptron);
    trainer.train(&mut perceptron, &dataset, Some(10_000));
    println!("After epoch [{}]: {:#?}", trainer.epoch, perceptron);
}
