use rna::{
    activation::identity::Identity,
    perceptron::Perceptron,
    trainer::{ClassificationStop, Train, gradient::Gradient},
};

fn main() {
    let dataset = [
        (vec![1.0, 2.0], 1.0),
        (vec![1.0, 4.0], -1.0),
        (vec![1.0, 5.0], 1.0),
        (vec![7.0, 5.0], -1.0),
        (vec![7.0, 6.0], -1.0),
        (vec![2.0, 1.0], -1.0),
        (vec![2.0, 3.0], 1.0),
        (vec![2.0, 4.0], 1.0),
        (vec![6.0, 2.0], 1.0),
        (vec![6.0, 4.0], -1.0),
        (vec![6.0, 5.0], -1.0),
        (vec![3.0, 1.0], -1.0),
        (vec![3.0, 2.0], -1.0),
        (vec![3.0, 4.0], 1.0),
        (vec![3.0, 5.0], 1.0),
        (vec![5.0, 3.0], -1.0),
        (vec![5.0, 4.0], -1.0),
        (vec![5.0, 6.0], 1.0),
        (vec![5.0, 7.0], 1.0),
        (vec![4.0, 2.0], -1.0),
        (vec![4.0, 3.0], 1.0),
        (vec![4.0, 5.0], 1.0),
        (vec![4.0, 6.0], 1.0),
    ];

    let mut perceptron = Perceptron {
        weights: vec![0.0; 2],
        bias: 0.0,
        activation: Identity,
    };

    let mut trainer = Gradient {
        epoch: 0,
        tolerance: 0.0,
        learning_rate: 0.0015,
        class_stop: &Some(ClassificationStop {
            error_limit: 3,
            threshold: 0.0,
            values: (-1.0, 1.0),
        }),
    };

    println!("Before: {:#?}", perceptron);
    trainer.train(&mut perceptron, &dataset, Some(1000));
    println!("After epoch [{}]: {:#?}", trainer.epoch, perceptron);
}
