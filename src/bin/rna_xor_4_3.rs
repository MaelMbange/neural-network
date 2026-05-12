use rna::{
    activation::sigmoid::Sigmoid,
    layer::{Layer, MLP},
};

fn main() {
    let mut mlp = MLP::<Sigmoid> {
        layers: vec![
            Layer::<Sigmoid>::new_with_weights(
                2,
                3,
                vec![vec![0.1, 0.15, 0.05], vec![0.12, 0.18, 0.08]],
            ),
            Layer::<Sigmoid>::new_with_weights(
                3,
                2,
                vec![vec![0.1, 0.14], vec![0.125, 0.21], vec![0.13, 0.07]],
            ),
        ],
        tolerance: 0.01,
    };

    println!("Avant entraînement : {:?}", mlp.forward(&[0.9, 0.1, 0.9]));
    mlp.train(&[vec![0.9, 0.1, 0.9]], &[vec![0.1, 0.9, 0.9]], 0.5, Some(1));
    println!("Après entraînement : {:?}", mlp.forward(&[0.9, 0.1, 0.9]));
}
