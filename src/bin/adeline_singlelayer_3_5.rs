use rna::{
    csv_reader::load_dataset_multi,
    network::single_layer::SingleLayer,
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

    let _conf = None;

    let mut network = SingleLayer::new(4, || {
        Adaline::new(25, 0.0, 0.001, 0.05, &_conf, 0.0..=0.0, Identity)
    });

    SingleLayer::train(&mut network, &datas, Some(1_000));
    println!("Trained network: {:#?}", network);

    println!("Classification results:");
    for (i, (data, _)) in datas.iter().enumerate() {
        let output = network.classify(data, 0.0, (-1.0, 1.0));
        let classname = match output.as_slice() {
            [1.0, -1.0, -1.0, -1.0] => "A",
            [-1.0, 1.0, -1.0, -1.0] => "B",
            [-1.0, -1.0, 1.0, -1.0] => "C",
            [-1.0, -1.0, -1.0, 1.0] => "D",
            _ => "Unknown",
        };

        println!("Data {}: Output: {:?}", i + 1, classname);
    }

    println!("\nClassify a new data point:");
    let mut new_data = vec![0.0; 23];
    new_data.push(1.0);
    new_data.push(1.0);

    let output = network.classify(&new_data, 0.0, (-1.0, 1.0));
    println!("New data output: {:?}", output);

    println!("Prediction:");
    let prediction = network.predict(&new_data);
    println!("New data prediction: {:?}", prediction);
}
