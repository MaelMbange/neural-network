use rna::{
    csv_reader::load_dataset_multi,
    network::layer::Layer,
    neuron::{
        activation::identity::Identity, classification_config::ClassificationConfig,
        gradient::Gradient,
    },
};

fn main() {
    let datas = load_dataset_multi("Datas/Datas/table_3_1.csv", 2, 3, false)
        .expect("Failed to load dataset");

    let _conf = Some(ClassificationConfig {
        error_limit: 0,
        threshold: 0.0,
        values: (-1.0, 1.0),
    });

    let conf = None;

    let mut network = Layer::new(3, || {
        Gradient::new(2, 0.0, 0.0001, 0.01, &conf, 0.0..=0.0, Identity)
    });

    Layer::train(&mut network, &datas, Some(300));
}
