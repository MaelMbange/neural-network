use rna::neuron::{
    Neuron,
    activation::identity::Identity,
    classification_config::ClassificationConfig,
    gradient::Gradient,
    history::{HistoryFile, HistoryRecorder},
};

fn main() {
    let dataset = vec![
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

    let mut g = Gradient::new(
        2,
        0.0,
        0.0015,
        0.0,
        &Some(ClassificationConfig {
            error_limit: 3,
            threshold: 0.0,
            values: (-1.0, 1.0),
        }),
        -1.0..=1.0,
        Identity,
    );
    g.set_debug(true);
    g.zero_weights();

    println!("Gradient before: {:#?}", g);
    let mut recorder = HistoryRecorder::new();
    g.train_with_observer(&dataset, Some(1000), &mut recorder);
    println!("Gradient after: {:#?}", g);

    for (inputs, expected) in &dataset {
        let prediction = g.classify(inputs, 0.0, (-1.0, 1.0));
        println!(
            "Inputs: {:?}, Expected: {}, Prediction: {}",
            inputs, expected, prediction
        );
    }

    let history_file = HistoryFile {
        dataset: dataset.iter().map(|(v, l)| (v.clone(), *l)).collect(),
        snapshots: recorder.snapshots,
        class_threshold: 0.0,
        boundary_threshold: 0.0,
    };

    match serde_json::to_string_pretty(&history_file) {
        Ok(json) => match std::fs::write("history.json", json) {
            Ok(_) => println!(
                "history.json written ({} snapshots)",
                history_file.snapshots.len()
            ),
            Err(e) => eprintln!("failed to write history.json: {}", e),
        },
        Err(e) => eprintln!("failed to serialize history: {}", e),
    }
}
