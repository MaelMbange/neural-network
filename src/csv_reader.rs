use csv::ReaderBuilder;
use std::error::Error;
use std::path::Path;

pub fn load_dataset_multi<P: AsRef<Path>>(
    path: P,
    n_inputs: usize,
    n_classes: usize,
    has_headers: bool,
) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>), Box<dyn Error>> {
    let expected = n_inputs + n_classes;

    let mut reader = ReaderBuilder::new()
        .has_headers(has_headers)
        .delimiter(b',')
        .trim(csv::Trim::All)
        .from_path(path)?;

    let mut inputs = Vec::new();
    let mut targets = Vec::new();

    for (i, result) in reader.records().enumerate() {
        let record = result?;

        if record.len() != expected {
            return Err(format!(
                "Index {}: expected {} column ({} entries + {} classes), found {}",
                i + 1,
                expected,
                n_inputs,
                n_classes,
                record.len()
            )
            .into());
        }

        let values: Vec<f64> = record
            .iter()
            .map(|s| s.parse::<f64>())
            .collect::<Result<_, _>>()?;

        let (row_inputs, row_classes) = values.split_at(n_inputs);
        inputs.push(row_inputs.to_vec());
        targets.push(row_classes.to_vec());
    }

    Ok((inputs, targets))
}
