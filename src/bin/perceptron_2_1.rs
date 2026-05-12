use std::marker::PhantomData;

use rna::{
    activation::step::Step,
    perceptron::Perceptron,
    train::{Train, linear::Linear},
};

fn main() {
    let dataset = [
        (vec![0.0, 0.0], 0.0),
        (vec![0.0, 1.0], 0.0),
        (vec![1.0, 0.0], 0.0),
        (vec![1.0, 1.0], 1.0),
    ];

    let mut perceptron = Perceptron {
        weights: vec![0.0; 2],
        bias: 0.0,
        activation: PhantomData::<Step>,
    };

    let mut trainer = Linear {
        epoch: 0,
        learning_rate: 1.0,
    };

    println!("Perceptron before: {:#?}", perceptron);
    trainer.train(&mut perceptron, &dataset, Some(100));
    println!("After epoch [{}]: {:#?}", trainer.epoch, perceptron);
}
