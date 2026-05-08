use rna::{
    network::layer::Layer,
    neuron::{
        activation::sigmoid::Sigmoid, adaline::Adaline, classification_config::ClassificationConfig,
    },
};

fn main() {
    let a = Layer::new(2, || {
        Adaline::new(
            2,
            0.0,
            0.001,
            0.05,
            &Some(ClassificationConfig {
                error_limit: 0,
                threshold: 0.0,
                values: (-1.0, 1.0),
            }),
            -1.0..=1.0,
            Sigmoid,
        )
    });

    println!("{a:#?}");
}
