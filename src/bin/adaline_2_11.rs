use rna::{
    activation::identity::Identity,
    perceptron::Perceptron,
    trainer::{Train, adeline::Adeline},
};

fn main() {
    let dataset = [
        (vec![10.0], 4.4),
        (vec![14.0], 5.6),
        (vec![12.0], 4.6),
        (vec![18.0], 6.1),
        (vec![16.0], 6.0),
        (vec![14.0], 7.0),
        (vec![22.0], 6.8),
        (vec![28.0], 10.6),
        (vec![26.0], 11.0),
        (vec![16.0], 7.6),
        (vec![23.0], 10.8),
        (vec![25.0], 10.0),
        (vec![20.0], 6.5),
        (vec![20.0], 8.2),
        (vec![24.0], 8.8),
        (vec![12.0], 5.5),
        (vec![15.0], 5.0),
        (vec![18.0], 8.0),
        (vec![14.0], 7.8),
        (vec![26.0], 9.0),
        (vec![25.0], 9.4),
        (vec![17.0], 8.5),
        (vec![12.0], 6.4),
        (vec![20.0], 7.5),
        (vec![23.0], 9.0),
        (vec![22.0], 8.1),
        (vec![26.0], 8.2),
        (vec![22.0], 10.0),
        (vec![18.0], 9.1),
        (vec![21.0], 9.0),
    ];

    let mut perceptron = Perceptron {
        weights: vec![0.0; 1],
        bias: 0.0,
        activation: Identity,
    };

    let mut trainer = Adeline {
        epoch: 0,
        tolerance: 0.56,
        learning_rate: 0.00014,
        class_stop: &None,
    };

    println!("Before: {:#?}", perceptron);
    trainer.train(&mut perceptron, &dataset, Some(10_000));
    println!("After epoch [{}]: {:#?}", trainer.epoch, perceptron);
}
