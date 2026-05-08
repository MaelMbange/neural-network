use rna::neuron::{
    Neuron,
    activation::identity::Identity,
    adaline::Adaline,
    history::{HistoryFile, HistoryRecorder},
};

fn main() {
    let dataset: Vec<(Vec<f64>, f64)> = vec![
        (vec![10.0], 4.4),
        (vec![14.0], 5.6),
        (vec![12.0], 4.6),
        (vec![18.0], 6.1),
        (vec![16.0], 6.0),
        (vec![14.0], 7.0),
        (vec![22.0], 6.8),
        (vec![28.0], 10.6),
        (vec![26.0], 11.0),
        (vec![16.0], 7.6),
        (vec![23.0], 10.8),
        (vec![25.0], 10.0),
        (vec![20.0], 6.5),
        (vec![20.0], 8.2),
        (vec![24.0], 8.8),
        (vec![12.0], 5.5),
        (vec![15.0], 5.0),
        (vec![18.0], 8.0),
        (vec![14.0], 7.8),
        (vec![26.0], 9.0),
        (vec![25.0], 9.4),
        (vec![17.0], 8.5),
        (vec![12.0], 6.4),
        (vec![20.0], 7.5),
        (vec![23.0], 9.0),
        (vec![22.0], 8.1),
        (vec![26.0], 8.2),
        (vec![22.0], 10.0),
        (vec![18.0], 9.1),
        (vec![21.0], 9.0),
    ];

    let mut a = Adaline::new(1, 0.0, 0.00014, 0.56, &None, -1.0..=1.0, Identity);
    a.set_debug(true);
    a.zero_weights();

    println!("Adaline before: {:#?}", a);
    let mut recorder = HistoryRecorder::new();
    a.train_with_observer(&dataset, Some(10_000), &mut recorder);
    println!("Adaline after: {:#?}", a);
    println!();

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
