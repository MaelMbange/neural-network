use rna::{
    activation::sigmoid::Sigmoid,
    layer::{Layer, MLP},
};

fn main() {
    let mut layer = Layer::<Sigmoid>::new_with_weights(
        2,
        3,
        vec![vec![0.1, 0.15, 0.05], vec![0.12, 0.18, 0.08]],
    );
    println!("Layer initialise avec succes !");
    println!("{:#?}", layer);

    println!("forward du layer avec les inputs [0.5, -0.5] :");
    let outputs = layer.forward(&[0.9, 0.1, 0.9]);
    println!("{:#?}", outputs);
    println!("{:#?}", layer);

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
    };

    println!("MLP initialise avec succes !");
    println!("{:#?}", mlp);

    println!("forward du MLP avec les inputs [0.9, 0.1, 0.9] :");
    mlp.forward(&[0.9, 0.1, 0.9]);
    println!("{:#?}", mlp);
}
