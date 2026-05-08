use rna::neuron::{
    Neuron,
    activation::identity::Identity,
    gradient::Gradient,
    history::{HistoryFile, HistoryRecorder},
};

fn main() {
    let dataset = [
        (vec![0.0, 0.0], -1.0),
        (vec![0.0, 1.0], -1.0),
        (vec![1.0, 0.0], -1.0),
        (vec![1.0, 1.0], 1.0),
    ];

    let mut g = Gradient::new(2, 0.0, 0.2, 0.125001, &None, -1.0..=1.0, Identity);
    g.set_debug(true);
    g.zero_weights();

    println!("Gradient before: {:#?}", g);
    let mut recorder = HistoryRecorder::new();
    g.train_with_observer(&dataset, Some(10_000), &mut recorder);
    println!("Gradient after: {:#?}", g);

    for (inputs, expected) in &dataset {
        let prediction = g.classify(inputs, 0.0, (-1.0, 1.0));
        println!(
            "Inputs: {:?}, Expected: {}, Prediction: {}",
            inputs, expected, prediction
        );
    }

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
