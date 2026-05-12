use std::marker::PhantomData;

use rna::{
    activation::{identity::Identity, sigmoid::Sigmoid, tanh::Tanh},
    layer::MLP,
    perceptron::Perceptron,
    train::ClassificationStop,
};

use crate::history::{
    MlpHistory, SingleLayerHistory,
    adeline::AdelineHistory,
    gradient::GradientHistory,
    linear::LinearHistory,
    mlp::train_mlp_with_history,
    single_layer::{SingleLayerTrainerKind, train_single_layer_with_history},
    types::History,
};
use super::params::{HyperParams, MlpActivation};

// ── Experiment kind ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum ExperimentKind {
    Perceptron2D { threshold: f64, values: (f64, f64) },
    Regression1D,
    SingleLayer,
    Mlp,
}

// ── Training result ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum TrainingResult {
    Single(History),
    SingleLayer(SingleLayerHistory),
    Mlp(MlpHistory),
}

impl TrainingResult {
    pub fn dataset_name(&self) -> &str {
        match self {
            TrainingResult::Single(h) => &h.dataset_name,
            TrainingResult::SingleLayer(h) => &h.dataset_name,
            TrainingResult::Mlp(h) => &h.dataset_name,
        }
    }
}

// ── Dataset source ────────────────────────────────────────────────────────────

pub struct ExperimentConfig {
    pub name: &'static str,
    pub kind: ExperimentKind,
    pub dataset: DatasetSource,
}

pub enum DatasetSource {
    Inline2D(Vec<(Vec<f64>, f64)>),
    /// Single-output CSV (first output column used as label for 2D display).
    File { path: &'static str, n_inputs: usize, n_outputs: usize },
}

// ── Dataset helper ────────────────────────────────────────────────────────────

/// Returns the full (inputs, targets) for an experiment.
/// For Inline2D, targets are wrapped in a single-element vec.
pub fn get_full_dataset(name: &str) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>), String> {
    let exps = all_experiments();
    let cfg = exps
        .iter()
        .find(|e| e.name == name)
        .ok_or_else(|| format!("unknown: {}", name))?;
    match &cfg.dataset {
        DatasetSource::Inline2D(d) => {
            let inputs = d.iter().map(|(i, _)| i.clone()).collect();
            let targets = d.iter().map(|(_, t)| vec![*t]).collect();
            Ok((inputs, targets))
        }
        DatasetSource::File { path, n_inputs, n_outputs } => {
            rna::csv_reader::load_dataset_multi(path, *n_inputs, *n_outputs, false)
                .map_err(|e| e.to_string())
        }
    }
}

// ── Registry ──────────────────────────────────────────────────────────────────

pub fn all_experiments() -> Vec<ExperimentConfig> {
    vec![
        // ── perceptron ────────────────────────────────────────────────────────
        ExperimentConfig {
            name: "perceptron_2_1",
            kind: ExperimentKind::Perceptron2D { threshold: 0.5, values: (0.0, 1.0) },
            dataset: DatasetSource::Inline2D(vec![
                (vec![0.0, 0.0], 0.0), (vec![0.0, 1.0], 0.0),
                (vec![1.0, 0.0], 0.0), (vec![1.0, 1.0], 1.0),
            ]),
        },
        // ── gradient perceptron ───────────────────────────────────────────────
        ExperimentConfig {
            name: "gradient_2_3",
            kind: ExperimentKind::Perceptron2D { threshold: 0.0, values: (-1.0, 1.0) },
            dataset: DatasetSource::Inline2D(vec![
                (vec![0.0, 0.0], -1.0), (vec![0.0, 1.0], -1.0),
                (vec![1.0, 0.0], -1.0), (vec![1.0, 1.0], 1.0),
            ]),
        },
        ExperimentConfig {
            name: "gradient_2_9_ls",
            kind: ExperimentKind::Perceptron2D { threshold: 0.0, values: (-1.0, 1.0) },
            dataset: DatasetSource::Inline2D(vec![
                (vec![1.0, 6.0], 1.0),  (vec![7.0, 9.0], -1.0), (vec![1.0, 9.0], 1.0),
                (vec![7.0, 10.0], -1.0),(vec![2.0, 5.0], -1.0), (vec![2.0, 7.0], 1.0),
                (vec![2.0, 8.0], 1.0),  (vec![6.0, 8.0], -1.0), (vec![6.0, 9.0], -1.0),
                (vec![3.0, 5.0], -1.0), (vec![3.0, 6.0], -1.0), (vec![3.0, 8.0], 1.0),
                (vec![3.0, 9.0], 1.0),  (vec![5.0, 7.0], -1.0), (vec![5.0, 8.0], -1.0),
                (vec![5.0, 10.0], 1.0), (vec![5.0, 11.0], 1.0), (vec![4.0, 6.0], -1.0),
                (vec![4.0, 7.0], -1.0), (vec![4.0, 9.0], 1.0),  (vec![4.0, 10.0], 1.0),
            ]),
        },
        ExperimentConfig {
            name: "gradient_2_9_nls",
            kind: ExperimentKind::Perceptron2D { threshold: 0.0, values: (-1.0, 1.0) },
            dataset: DatasetSource::Inline2D(vec![
                (vec![1.0, 2.0], 1.0),  (vec![1.0, 4.0], -1.0), (vec![1.0, 5.0], 1.0),
                (vec![7.0, 5.0], -1.0), (vec![7.0, 6.0], -1.0), (vec![2.0, 1.0], -1.0),
                (vec![2.0, 3.0], 1.0),  (vec![2.0, 4.0], 1.0),  (vec![6.0, 2.0], 1.0),
                (vec![6.0, 4.0], -1.0), (vec![6.0, 5.0], -1.0), (vec![3.0, 1.0], -1.0),
                (vec![3.0, 2.0], -1.0), (vec![3.0, 4.0], 1.0),  (vec![3.0, 5.0], 1.0),
                (vec![5.0, 3.0], -1.0), (vec![5.0, 4.0], -1.0), (vec![5.0, 6.0], 1.0),
                (vec![5.0, 7.0], 1.0),  (vec![4.0, 2.0], -1.0), (vec![4.0, 3.0], 1.0),
                (vec![4.0, 5.0], 1.0),  (vec![4.0, 6.0], 1.0),
            ]),
        },
        ExperimentConfig {
            name: "gradient_2_11",
            kind: ExperimentKind::Regression1D,
            dataset: DatasetSource::Inline2D(vec![
                (vec![10.0], 4.4),  (vec![14.0], 5.6),  (vec![12.0], 4.6),
                (vec![18.0], 6.1),  (vec![16.0], 6.0),  (vec![14.0], 7.0),
                (vec![22.0], 6.8),  (vec![28.0], 10.6), (vec![26.0], 11.0),
                (vec![16.0], 7.6),  (vec![23.0], 10.8), (vec![25.0], 10.0),
                (vec![20.0], 6.5),  (vec![20.0], 8.2),  (vec![24.0], 8.8),
                (vec![12.0], 5.5),  (vec![15.0], 5.0),  (vec![18.0], 8.0),
                (vec![14.0], 7.8),  (vec![26.0], 9.0),  (vec![25.0], 9.4),
                (vec![17.0], 8.5),  (vec![12.0], 6.4),  (vec![20.0], 7.5),
                (vec![23.0], 9.0),  (vec![22.0], 8.1),  (vec![26.0], 8.2),
                (vec![22.0], 10.0), (vec![18.0], 9.1),  (vec![21.0], 9.0),
            ]),
        },
        // ── adaline perceptron ────────────────────────────────────────────────
        ExperimentConfig {
            name: "adaline_2_3",
            kind: ExperimentKind::Perceptron2D { threshold: 0.0, values: (-1.0, 1.0) },
            dataset: DatasetSource::Inline2D(vec![
                (vec![0.0, 0.0], -1.0), (vec![0.0, 1.0], -1.0),
                (vec![1.0, 0.0], -1.0), (vec![1.0, 1.0], 1.0),
            ]),
        },
        ExperimentConfig {
            name: "adaline_2_9_ls",
            kind: ExperimentKind::Perceptron2D { threshold: 0.0, values: (-1.0, 1.0) },
            dataset: DatasetSource::Inline2D(vec![
                (vec![1.0, 6.0], 1.0),  (vec![7.0, 9.0], -1.0), (vec![1.0, 9.0], 1.0),
                (vec![7.0, 10.0], -1.0),(vec![2.0, 5.0], -1.0), (vec![2.0, 7.0], 1.0),
                (vec![2.0, 8.0], 1.0),  (vec![6.0, 8.0], -1.0), (vec![6.0, 9.0], -1.0),
                (vec![3.0, 5.0], -1.0), (vec![3.0, 6.0], -1.0), (vec![3.0, 8.0], 1.0),
                (vec![3.0, 9.0], 1.0),  (vec![5.0, 7.0], -1.0), (vec![5.0, 8.0], -1.0),
                (vec![5.0, 10.0], 1.0), (vec![5.0, 11.0], 1.0), (vec![4.0, 6.0], -1.0),
                (vec![4.0, 7.0], -1.0), (vec![4.0, 9.0], 1.0),  (vec![4.0, 10.0], 1.0),
            ]),
        },
        ExperimentConfig {
            name: "adaline_2_9_nls",
            kind: ExperimentKind::Perceptron2D { threshold: 0.0, values: (-1.0, 1.0) },
            dataset: DatasetSource::Inline2D(vec![
                (vec![1.0, 2.0], 1.0),  (vec![1.0, 4.0], -1.0), (vec![1.0, 5.0], 1.0),
                (vec![7.0, 5.0], -1.0), (vec![7.0, 6.0], -1.0), (vec![2.0, 1.0], -1.0),
                (vec![2.0, 3.0], 1.0),  (vec![2.0, 4.0], 1.0),  (vec![6.0, 2.0], 1.0),
                (vec![6.0, 4.0], -1.0), (vec![6.0, 5.0], -1.0), (vec![3.0, 1.0], -1.0),
                (vec![3.0, 2.0], -1.0), (vec![3.0, 4.0], 1.0),  (vec![3.0, 5.0], 1.0),
                (vec![5.0, 3.0], -1.0), (vec![5.0, 4.0], -1.0), (vec![5.0, 6.0], 1.0),
                (vec![5.0, 7.0], 1.0),  (vec![4.0, 2.0], -1.0), (vec![4.0, 3.0], 1.0),
                (vec![4.0, 5.0], 1.0),  (vec![4.0, 6.0], 1.0),
            ]),
        },
        ExperimentConfig {
            name: "adaline_2_11",
            kind: ExperimentKind::Regression1D,
            dataset: DatasetSource::Inline2D(vec![
                (vec![10.0], 4.4),  (vec![14.0], 5.6),  (vec![12.0], 4.6),
                (vec![18.0], 6.1),  (vec![16.0], 6.0),  (vec![14.0], 7.0),
                (vec![22.0], 6.8),  (vec![28.0], 10.6), (vec![26.0], 11.0),
                (vec![16.0], 7.6),  (vec![23.0], 10.8), (vec![25.0], 10.0),
                (vec![20.0], 6.5),  (vec![20.0], 8.2),  (vec![24.0], 8.8),
                (vec![12.0], 5.5),  (vec![15.0], 5.0),  (vec![18.0], 8.0),
                (vec![14.0], 7.8),  (vec![26.0], 9.0),  (vec![25.0], 9.4),
                (vec![17.0], 8.5),  (vec![12.0], 6.4),  (vec![20.0], 7.5),
                (vec![23.0], 9.0),  (vec![22.0], 8.1),  (vec![26.0], 8.2),
                (vec![22.0], 10.0), (vec![18.0], 9.1),  (vec![21.0], 9.0),
            ]),
        },
        // ── single-layer ──────────────────────────────────────────────────────
        ExperimentConfig {
            name: "adaline_singlelayer_3_1",
            kind: ExperimentKind::SingleLayer,
            dataset: DatasetSource::File {
                path: "Datas/Datas/table_3_1.csv",
                n_inputs: 2,
                n_outputs: 3,
            },
        },
        ExperimentConfig {
            name: "gradient_singlelayer_3_1",
            kind: ExperimentKind::SingleLayer,
            dataset: DatasetSource::File {
                path: "Datas/Datas/table_3_1.csv",
                n_inputs: 2,
                n_outputs: 3,
            },
        },
        ExperimentConfig {
            name: "adaline_singlelayer_3_5",
            kind: ExperimentKind::SingleLayer,
            dataset: DatasetSource::File {
                path: "Datas/Datas/table_3_5.csv",
                n_inputs: 25,
                n_outputs: 4,
            },
        },
        // ── MLP ───────────────────────────────────────────────────────────────
        ExperimentConfig {
            name: "rna_xor_4_3",
            kind: ExperimentKind::Mlp,
            dataset: DatasetSource::Inline2D(vec![
                (vec![0.0, 0.0], 0.0), (vec![0.0, 1.0], 1.0),
                (vec![1.0, 0.0], 1.0), (vec![1.0, 1.0], 0.0),
            ]),
        },
        ExperimentConfig {
            name: "rna_cnl_4_12",
            kind: ExperimentKind::Mlp,
            dataset: DatasetSource::File {
                path: "Datas/Datas/table_4_12.csv",
                n_inputs: 2,
                n_outputs: 1,
            },
        },
        ExperimentConfig {
            name: "rna_cnl_4_14",
            kind: ExperimentKind::Mlp,
            dataset: DatasetSource::File {
                path: "Datas/Datas/table_4_14.csv",
                n_inputs: 2,
                n_outputs: 3,
            },
        },
        ExperimentConfig {
            name: "rna_reg_1_dim_4_17",
            kind: ExperimentKind::Mlp,
            dataset: DatasetSource::File {
                path: "Datas/Datas/table_4_17.csv",
                n_inputs: 1,
                n_outputs: 1,
            },
        },
        ExperimentConfig {
            name: "rna_langue_signe",
            kind: ExperimentKind::Mlp,
            dataset: DatasetSource::File {
                path: "Datas/Datas/LangageDesSignes/data_formatted.csv",
                n_inputs: 42,
                n_outputs: 5,
            },
        },
    ]
}

// ── Dispatch ──────────────────────────────────────────────────────────────────

pub fn run_experiment(
    name: &str,
    verbose: bool,
    params: &HyperParams,
) -> Result<TrainingResult, String> {
    let exps = all_experiments();
    let cfg = exps
        .iter()
        .find(|e| e.name == name)
        .ok_or_else(|| format!("unknown experiment: {}", name))?;

    // Flatten to (Vec<f64>, f64) for single-output trainers.
    let dataset_2d: Vec<(Vec<f64>, f64)> = match &cfg.dataset {
        DatasetSource::Inline2D(d) => d.clone(),
        DatasetSource::File { path, n_inputs, n_outputs } => {
            let (inputs, targets) =
                rna::csv_reader::load_dataset_multi(path, *n_inputs, *n_outputs, false)
                    .map_err(|e| format!("CSV load error: {e}"))?;
            inputs
                .into_iter()
                .zip(targets.into_iter().map(|t| t[0]))
                .collect()
        }
    };

    // Build ClassificationStop ONLY when the user explicitly set an error limit.
    // Without this guard, error_limit = usize::MAX makes the condition always true
    // and stops training after a single epoch.
    let class_stop: Option<ClassificationStop> =
        match (params.class_neg, params.class_pos, params.class_threshold, params.class_error_limit) {
            (Some(neg), Some(pos), Some(thr), Some(error_limit)) => {
                Some(ClassificationStop { error_limit, threshold: thr, values: (neg, pos) })
            }
            _ => None,
        };

    let max_ep = Some(params.max_epochs);

    match name {
        // ── Linear / perceptron ───────────────────────────────────────────────
        "perceptron_2_1" => {
            let mut p = Perceptron::<rna::activation::step::Step> {
                weights: vec![0.0; 2], bias: 0.0, activation: PhantomData,
            };
            let h = LinearHistory::new(params.learning_rate, verbose)
                .train_with_history(&mut p, &dataset_2d, max_ep, "AND gate");
            Ok(TrainingResult::Single(h))
        }

        // ── Gradient ──────────────────────────────────────────────────────────
        "gradient_2_3" => {
            run_gradient_2d(&dataset_2d, params, class_stop, max_ep, "AND gate (gradient)", verbose)
        }
        "gradient_2_9_ls" => {
            run_gradient_2d(&dataset_2d, params, class_stop, max_ep, "2-9 LS", verbose)
        }
        "gradient_2_9_nls" => {
            run_gradient_2d(&dataset_2d, params, class_stop, max_ep, "2-9 NLS", verbose)
        }
        "gradient_2_11" => {
            run_gradient_1d(&dataset_2d, params, max_ep, "regression 2-11", verbose)
        }

        // ── Adaline ───────────────────────────────────────────────────────────
        "adaline_2_3" => {
            run_adeline_2d(&dataset_2d, params, class_stop, max_ep, "AND gate (adaline)", verbose)
        }
        "adaline_2_9_ls" => {
            run_adeline_2d(&dataset_2d, params, class_stop, max_ep, "2-9 LS (adaline)", verbose)
        }
        "adaline_2_9_nls" => {
            run_adeline_2d(&dataset_2d, params, class_stop, max_ep, "2-9 NLS (adaline)", verbose)
        }
        "adaline_2_11" => {
            run_adeline_1d(&dataset_2d, params, max_ep, "regression 2-11 (adaline)", verbose)
        }

        // ── Single-layer ──────────────────────────────────────────────────────
        "adaline_singlelayer_3_1" | "gradient_singlelayer_3_1"
        | "adaline_singlelayer_3_5" => {
            let (path, n_inputs, n_outputs) = match &cfg.dataset {
                DatasetSource::File { path, n_inputs, n_outputs } => (path, n_inputs, n_outputs),
                _ => unreachable!(),
            };
            let (inputs, targets) =
                rna::csv_reader::load_dataset_multi(path, *n_inputs, *n_outputs, false)
                    .map_err(|e| format!("CSV load error: {e}"))?;
            let template = Perceptron::<Identity> {
                weights: vec![0.0; *n_inputs], bias: 0.0, activation: PhantomData,
            };
            let trainer_kind = if name.starts_with("adaline") {
                SingleLayerTrainerKind::Adeline {
                    learning_rate: params.learning_rate,
                    tolerance: params.tolerance,
                    class_stop,
                }
            } else {
                SingleLayerTrainerKind::Gradient {
                    learning_rate: params.learning_rate,
                    tolerance: params.tolerance,
                    class_stop,
                }
            };
            let h = train_single_layer_with_history(
                &template, *n_outputs, &trainer_kind,
                &inputs, &targets, max_ep, name, verbose,
            );
            Ok(TrainingResult::SingleLayer(h))
        }

        // ── MLP ───────────────────────────────────────────────────────────────
        "rna_xor_4_3" => {
            let inputs = vec![
                vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0],
            ];
            let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];
            run_mlp_from_data(&inputs, &targets, 2, 1, "XOR (MLP)", params, verbose)
        }
        "rna_cnl_4_12" => {
            let (inputs, targets) = load_csv_for_mlp(cfg)?;
            run_mlp_from_data(&inputs, &targets, 2, 1, "CNL 4-12", params, verbose)
        }
        "rna_cnl_4_14" => {
            let (inputs, targets) = load_csv_for_mlp(cfg)?;
            run_mlp_from_data(&inputs, &targets, 2, 3, "CNL 4-14", params, verbose)
        }
        "rna_reg_1_dim_4_17" => {
            let (inputs, targets) = load_csv_for_mlp(cfg)?;
            run_mlp_from_data(&inputs, &targets, 1, 1, "Reg 1D 4-17", params, verbose)
        }

        // Trains on first 251 rows, matching the original binary's split.
        "rna_langue_signe" => {
            let (inputs, targets) = load_csv_for_mlp(cfg)?;
            let n_train = 251.min(inputs.len());
            run_mlp_from_data(
                &inputs[..n_train],
                &targets[..n_train],
                42, 5,
                "Langue des signes",
                params,
                verbose,
            )
        }

        other => Err(format!("dispatch not implemented for: {}", other)),
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

fn load_csv_for_mlp(cfg: &ExperimentConfig) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>), String> {
    match &cfg.dataset {
        DatasetSource::File { path, n_inputs, n_outputs } => {
            rna::csv_reader::load_dataset_multi(path, *n_inputs, *n_outputs, false)
                .map_err(|e| format!("CSV load error: {e}"))
        }
        _ => Err("expected file dataset".into()),
    }
}

fn run_gradient_2d(
    dataset: &[(Vec<f64>, f64)],
    params: &HyperParams,
    class_stop: Option<ClassificationStop>,
    max_ep: Option<usize>,
    label: &str,
    verbose: bool,
) -> Result<TrainingResult, String> {
    let mut p = Perceptron::<Identity> { weights: vec![0.0; 2], bias: 0.0, activation: PhantomData };
    let h = GradientHistory::new(params.learning_rate, params.tolerance, class_stop, verbose)
        .train_with_history(&mut p, dataset, max_ep, label);
    Ok(TrainingResult::Single(h))
}

fn run_gradient_1d(
    dataset: &[(Vec<f64>, f64)],
    params: &HyperParams,
    max_ep: Option<usize>,
    label: &str,
    verbose: bool,
) -> Result<TrainingResult, String> {
    let mut p = Perceptron::<Identity> { weights: vec![0.0; 1], bias: 0.0, activation: PhantomData };
    let h = GradientHistory::new(params.learning_rate, params.tolerance, None, verbose)
        .train_with_history(&mut p, dataset, max_ep, label);
    Ok(TrainingResult::Single(h))
}

fn run_adeline_2d(
    dataset: &[(Vec<f64>, f64)],
    params: &HyperParams,
    class_stop: Option<ClassificationStop>,
    max_ep: Option<usize>,
    label: &str,
    verbose: bool,
) -> Result<TrainingResult, String> {
    let mut p = Perceptron::<Identity> { weights: vec![0.0; 2], bias: 0.0, activation: PhantomData };
    let h = AdelineHistory::new(params.learning_rate, params.tolerance, class_stop, verbose)
        .train_with_history(&mut p, dataset, max_ep, label);
    Ok(TrainingResult::Single(h))
}

fn run_adeline_1d(
    dataset: &[(Vec<f64>, f64)],
    params: &HyperParams,
    max_ep: Option<usize>,
    label: &str,
    verbose: bool,
) -> Result<TrainingResult, String> {
    let mut p = Perceptron::<Identity> { weights: vec![0.0; 1], bias: 0.0, activation: PhantomData };
    let h = AdelineHistory::new(params.learning_rate, params.tolerance, None, verbose)
        .train_with_history(&mut p, dataset, max_ep, label);
    Ok(TrainingResult::Single(h))
}

/// Generic MLP dispatch — constructs topology from params.mlp_hidden_sizes + output_dim,
/// then dispatches over activation at compile time.
pub fn run_mlp_from_data(
    inputs: &[Vec<f64>],
    targets: &[Vec<f64>],
    input_dim: usize,
    output_dim: usize,
    dataset_name: &str,
    params: &HyperParams,
    verbose: bool,
) -> Result<TrainingResult, String> {
    let layer_sizes: Vec<usize> = params
        .mlp_hidden_sizes
        .iter()
        .cloned()
        .chain(std::iter::once(output_dim))
        .collect();
    if layer_sizes.is_empty() {
        return Err("MLP needs at least one output layer".into());
    }

    let lr  = params.learning_rate;
    let tol = params.tolerance;
    let ep  = Some(params.max_epochs);
    let act = params.mlp_activation.to_activation_name();

    let h = match &params.mlp_activation {
        MlpActivation::Sigmoid => {
            let mut mlp = MLP::<Sigmoid>::new(&layer_sizes, input_dim, -1.0..=1.0);
            train_mlp_with_history(&mut mlp, inputs, targets, lr, tol, ep, dataset_name, act, input_dim, verbose)
        }
        MlpActivation::Tanh => {
            let mut mlp = MLP::<Tanh>::new(&layer_sizes, input_dim, -1.0..=1.0);
            train_mlp_with_history(&mut mlp, inputs, targets, lr, tol, ep, dataset_name, act, input_dim, verbose)
        }
        MlpActivation::Identity => {
            let mut mlp = MLP::<Identity>::new(&layer_sizes, input_dim, -1.0..=1.0);
            train_mlp_with_history(&mut mlp, inputs, targets, lr, tol, ep, dataset_name, act, input_dim, verbose)
        }
    };
    Ok(TrainingResult::Mlp(h))
}
