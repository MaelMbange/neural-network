use rna::{activation::sigmoid::Sigmoid, csv_reader::load_dataset_multi, layer::MLP};

fn main() {
    let (inputs, outputs) = load_dataset_multi("Datas/Datas/table_4_12.csv", 2, 1, false)
        .expect("Failed to load dataset");

    let mut mlp = MLP::<Sigmoid>::new(&[10, 1], 2, -1.0..=1.0);

    println!("Avant entraînement : {:#?}", mlp);
    mlp.train(&inputs, &outputs, 0.5, 0.001, Some(2000));
    println!("Après entraînement : {:#?}", mlp);
    println!("{:#?}", mlp);

    println!("Classification :");
    let mut correct = 0;
    for (i, (input, output)) in inputs.iter().zip(outputs.iter()).enumerate() {
        let result = mlp.classify_binary(input, 0.5, 0.0, 1.0);
        println!(
            "{i}: {:.2?} => Output: {:?} (Expected: {:?})",
            input, result, output
        );
        if result == output[0] {
            correct += 1;
        }
    }
    println!("Correct classifications: {}/{}", correct, inputs.len());
    println!(
        "Accuracy: {:.2}%",
        (correct as f64 / inputs.len() as f64) * 100.0
    );
}
