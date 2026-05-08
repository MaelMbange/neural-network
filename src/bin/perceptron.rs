use rna::neuron::{
    Neuron,
    activation::step::Step,
    history::{HistoryFile, HistoryRecorder},
    perceptron::Perceptron,
};

fn main() {
    let and = [
        (vec![0.0, 0.0], 0.0),
        (vec![0.0, 1.0], 0.0),
        (vec![1.0, 0.0], 0.0),
        (vec![1.0, 1.0], 1.0),
    ];

    let mut p = Perceptron::new(2, 0.0, 1.0, -1.0..=1.0, Step);
    p.set_debug(true);
    p.zero_weights();

    println!("perceptron before: {:#?}", p);
    let mut recorder = HistoryRecorder::new();
    p.train_with_observer(&and, Some(100), &mut recorder);
    println!("perceptron after: {:#?}", p);
    println!();

    let history_file = HistoryFile {
        dataset: and.iter().map(|(v, l)| (v.clone(), *l)).collect(),
        snapshots: recorder.snapshots,
        class_threshold: 0.5,
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
