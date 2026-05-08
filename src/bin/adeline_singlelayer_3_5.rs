use rna::{
    csv_reader::load_dataset_multi,
    network::singlelayer::SingleLayer,
    neuron::{
        activation::identity::Identity, adaline::Adaline,
        classification_config::ClassificationConfig,
    },
};

fn main() {
    let datas = load_dataset_multi("Datas/Datas/table_3_5.csv", 25, 4, false)
        .expect("Failed to load dataset");

    let _conf = Some(ClassificationConfig {
        error_limit: 0,
        threshold: 0.0,
        values: (-1.0, 1.0),
    });

    let conf = None;

    let mut network = SingleLayer::new(vec![
        Adaline::new(
            25,
            0.0,
            0.001,
            0.05,
            &conf,
            0.0..=0.0,
            Identity
        );
        4
    ]);
    network.set_debug(true);

    SingleLayer::train(&mut network, &datas, Some(1_000));

    println!("Classification results:");
    for (i, (data, _)) in datas.iter().enumerate() {
        let output = network.classify(data, 0.0, (-1.0, 1.0));
        println!("Data {}: Output: {:?}", i + 1, output);
    }

    println!("\nClassify a new data point:");
    let mut new_data = vec![0.0; 23];
    new_data.push(1.0);
    new_data.push(1.0);

    let output = network.classify(&new_data, 0.0, (-1.0, 1.0));
    println!("New data output: {:?}", output);
}
