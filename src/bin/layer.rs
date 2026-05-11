use rna::{
    csv_reader::load_dataset_multi,
    network::single_layer::SingleLayer,
    neuron::{
        activation::identity::Identity, adaline::Adaline,
        classification_config::ClassificationConfig,
    },
};

fn main() {
    let datas: Vec<(Vec<f64>, Vec<f64>)> =
        load_dataset_multi("Datas/Datas/table_3_1.csv", 2, 3, false)
            .expect("Failed to load dataset");

    let _conf = Some(ClassificationConfig {
        error_limit: 0,
        threshold: 0.0,
        values: (-1.0, 1.0),
    });

    let _conf = None;

    let mut l = SingleLayer::new(3, || {
        Adaline::new(2, 0.0, 0.001, 0.01, &_conf, 0.0..=0.0, Identity)
    });

    l.train(&datas, Some(300));

    println!("{l:#?}");
}
