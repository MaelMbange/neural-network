use rna::{activation::tanh::Tanh, csv_reader::load_dataset_multi, layer::MLP};

fn main() {
    let (inputs, outputs) = load_dataset_multi(
        "Datas/Datas/LangageDesSignes/data_formatted.csv",
        42,
        5,
        false,
    )
    .expect("Failed to load dataset");

    let mut mlp = MLP::<Tanh>::new(&[10, 5], 42, -1.0..=1.0);

    println!("Avant entraînement : {:#?}", mlp);
    mlp.train(&inputs[..250], &outputs[..250], 0.25, 0.01, Some(1000));
    println!("Après entraînement : {:#?}", mlp);

    println!("Classification :");
    let mut correct = 0;
    for (i, (input, output)) in inputs[250..].iter().zip(outputs[250..].iter()).enumerate() {
        // let result = mlp.classify_binary(input, 0.5, 0.0, 1.0);
        let result = mlp.classify_argmax(input);
        let output = match output.iter().position(|&x| x == 1.0) {
            Some(pos) => pos,
            None => {
                println!(
                    "{i}: {:.2?} => Output: {:?} (Expected: {:?}) <",
                    input, result, output
                );
                continue;
            }
        };

        if result == output {
            correct += 1;
            println!("{i}: => Output: {:?} (Expected: {:?})", result, output);
        } else {
            println!("{i}: => Output: {:?} (Expected: {:?}) <", result, output);
        }
    }
    println!(
        "Correct classifications: {}/{}",
        correct,
        inputs[250..].len()
    );
    println!(
        "Accuracy: {:.2}%",
        (correct as f64 / inputs[250..].len() as f64) * 100.0
    );
    println!("epochs: {}", mlp.epoch);
    println!("mean_error: {}", mlp.mean_squared_error);
}
