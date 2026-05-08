use rna::neuron::{
    Neuron,
    activation::identity::Identity,
    adaline::Adaline,
    history::{HistoryFile, HistoryRecorder},
};

fn main() {
    let dataset = [
        (vec![0.0, 0.0], -1.0),
        (vec![0.0, 1.0], -1.0),
        (vec![1.0, 0.0], -1.0),
        (vec![1.0, 1.0], 1.0),
    ];

    let mut a = Adaline::new(2, 0.0, 0.03, 0.1251, &None, -1.0..=1.0, Identity);
    a.set_debug(true);
    a.zero_weights();

    println!("Adaline before: {:#?}", a);
    let mut recorder = HistoryRecorder::new();
    a.train_with_observer(&dataset, Some(10_000), &mut recorder);
    println!("Adaline after: {:#?}", a);

    println!("\nPredictions:");
    for (inputs, expected) in &dataset {
        let prediction = a.classify(inputs, 0.0, (-1.0, 1.0));
        println!(
            "Inputs: {:?}, Expected: {}, Prediction: {}",
            inputs, expected, prediction
        );
    }
    println!();

    // Write history.json so `cargo run --bin visualize` can replay the gradient run.
    // Labels are ±1 — split the display classes at 0.0.
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
