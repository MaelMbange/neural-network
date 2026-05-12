use rna::{activation::sigmoid::Sigmoid, layer::MLP};

fn main() {
    let (inputs, outputs) = (
        vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ],
        vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]],
    );

    let mut mlp = MLP::<Sigmoid>::new(&[2, 1], 2, -1.0..=1.0);

    println!("Avant entraînement : {:#?}", mlp);
    mlp.train(&inputs, &outputs, 0.8, 0.001, Some(2000));
    println!("Après entraînement : {:#?}", mlp);
    println!("{:#?}", mlp);

    println!("Classification :");
    for (input, output) in inputs.iter().zip(outputs.iter()) {
        let result = mlp.classify_binary(input, 0.5, 0.0, 1.0);
        // let result = mlp.classify_argmax(input);
        println!(
            "Input: {:?} => Output: {:?} (Expected: {:?})",
            input, result, output
        );
    }
}
