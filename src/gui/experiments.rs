use std::marker::PhantomData;

use rna::{
    activation::{identity::Identity, sigmoid::Sigmoid},
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

// ── Experiment kind determines which visualisation panels are shown ───────────

#[derive(Debug, Clone, PartialEq)]
pub enum ExperimentKind {
    /// 2-input perceptron — scatter + decision boundary + loss curve
    Perceptron2D {
        threshold: f64,
        values: (f64, f64),
    },
    /// 1-input regression — scatter + regression line + loss curve
    Regression1D,
    /// Multiple perceptrons trained independently (singlelayer bins)
    SingleLayer,
    /// MLP with backprop — sparklines + loss curve
    Mlp,
}

// ── Training result (one of three payload types) ─────────────────────────────

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

// ── Per-experiment configuration ─────────────────────────────────────────────

pub struct ExperimentConfig {
    pub name: &'static str,
    pub kind: ExperimentKind,
    /// If Some, the dataset is embedded inline; if None, load from path.
    pub dataset: DatasetSource,
}

pub enum DatasetSource {
    Inline2D(Vec<(Vec<f64>, f64)>),
    File { path: &'static str, n_inputs: usize, n_outputs: usize },
}

// ── Registry ──────────────────────────────────────────────────────────────────

pub fn all_experiments() -> Vec<ExperimentConfig> {
    vec![
        // ── perceptron_2_1 ────────────────────────────────────────────────────
        ExperimentConfig {
            name: "perceptron_2_1",
            kind: ExperimentKind::Perceptron2D {
                threshold: 0.5,
                values: (0.0, 1.0),
            },
            dataset: DatasetSource::Inline2D(vec![
                (vec![0.0, 0.0], 0.0),
                (vec![0.0, 1.0], 0.0),
                (vec![1.0, 0.0], 0.0),
                (vec![1.0, 1.0], 1.0),
            ]),
        },
        // ── gradient_2_3 ─────────────────────────────────────────────────────
        ExperimentConfig {
            name: "gradient_2_3",
            kind: ExperimentKind::Perceptron2D {
                threshold: 0.0,
                values: (-1.0, 1.0),
            },
            dataset: DatasetSource::Inline2D(vec![
                (vec![0.0, 0.0], -1.0),
                (vec![0.0, 1.0], -1.0),
                (vec![1.0, 0.0], -1.0),
                (vec![1.0, 1.0], 1.0),
            ]),
        },
        // ── gradient_2_9_ls ───────────────────────────────────────────────────
        ExperimentConfig {
            name: "gradient_2_9_ls",
            kind: ExperimentKind::Perceptron2D {
                threshold: 0.0,
                values: (-1.0, 1.0),
            },
            dataset: DatasetSource::Inline2D(vec![
                (vec![1.0, 6.0], 1.0),
                (vec![7.0, 9.0], -1.0),
                (vec![1.0, 9.0], 1.0),
                (vec![7.0, 10.0], -1.0),
                (vec![2.0, 5.0], -1.0),
                (vec![2.0, 7.0], 1.0),
                (vec![2.0, 8.0], 1.0),
                (vec![6.0, 8.0], -1.0),
                (vec![6.0, 9.0], -1.0),
                (vec![3.0, 5.0], -1.0),
                (vec![3.0, 6.0], -1.0),
                (vec![3.0, 8.0], 1.0),
                (vec![3.0, 9.0], 1.0),
                (vec![5.0, 7.0], -1.0),
                (vec![5.0, 8.0], -1.0),
                (vec![5.0, 10.0], 1.0),
                (vec![5.0, 11.0], 1.0),
                (vec![4.0, 6.0], -1.0),
                (vec![4.0, 7.0], -1.0),
                (vec![4.0, 9.0], 1.0),
                (vec![4.0, 10.0], 1.0),
            ]),
        },
        // ── gradient_2_9_nls ──────────────────────────────────────────────────
        ExperimentConfig {
            name: "gradient_2_9_nls",
            kind: ExperimentKind::Perceptron2D {
                threshold: 0.0,
                values: (-1.0, 1.0),
            },
            dataset: DatasetSource::Inline2D(vec![
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
            ]),
        },
        // ── gradient_2_11 (1-D regression) ────────────────────────────────────
        ExperimentConfig {
            name: "gradient_2_11",
            kind: ExperimentKind::Regression1D,
            dataset: DatasetSource::Inline2D(vec![
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
            ]),
        },
        // ── adaline_2_3 ───────────────────────────────────────────────────────
        ExperimentConfig {
            name: "adaline_2_3",
            kind: ExperimentKind::Perceptron2D {
                threshold: 0.0,
                values: (-1.0, 1.0),
            },
            dataset: DatasetSource::Inline2D(vec![
                (vec![0.0, 0.0], -1.0),
                (vec![0.0, 1.0], -1.0),
                (vec![1.0, 0.0], -1.0),
                (vec![1.0, 1.0], 1.0),
            ]),
        },
        // ── adeline_singlelayer_3_1 ────────────────────────────────────────────
        ExperimentConfig {
            name: "adeline_singlelayer_3_1",
            kind: ExperimentKind::SingleLayer,
            dataset: DatasetSource::File {
                path: "Datas/Datas/table_3_1.csv",
                n_inputs: 2,
                n_outputs: 3,
            },
        },
        // ── gradient_singlelayer_3_1 ──────────────────────────────────────────
        ExperimentConfig {
            name: "gradient_singlelayer_3_1",
            kind: ExperimentKind::SingleLayer,
            dataset: DatasetSource::File {
                path: "Datas/Datas/table_3_1.csv",
                n_inputs: 2,
                n_outputs: 3,
            },
        },
        // ── rna_xor_4_3 ───────────────────────────────────────────────────────
        ExperimentConfig {
            name: "rna_xor_4_3",
            kind: ExperimentKind::Mlp,
            dataset: DatasetSource::Inline2D(vec![
                // Inputs and expected outputs are stored as (input, first_output).
                // For MLP the full Vec<Vec<f64>> is built inside run_experiment.
                (vec![0.0, 0.0], 0.0),
                (vec![0.0, 1.0], 1.0),
                (vec![1.0, 0.0], 1.0),
                (vec![1.0, 1.0], 0.0),
            ]),
        },
    ]
}

// ── Dispatch ──────────────────────────────────────────────────────────────────

/// Runs the experiment identified by `name`, returns the training result.
/// `verbose` controls per-epoch stdout printing.
pub fn run_experiment(name: &str, verbose: bool) -> Result<TrainingResult, String> {
    let exps = all_experiments();
    let cfg = exps
        .iter()
        .find(|e| e.name == name)
        .ok_or_else(|| format!("unknown experiment: {}", name))?;

    let dataset_2d: Vec<(Vec<f64>, f64)> = match &cfg.dataset {
        DatasetSource::Inline2D(d) => d.clone(),
        DatasetSource::File { path, n_inputs, n_outputs } => {
            let (inputs, targets) =
                rna::csv_reader::load_dataset_multi(path, *n_inputs, *n_outputs, false)
                    .map_err(|e| format!("CSV load error: {e}"))?;
            // Flatten targets to first column for use as single-output dataset.
            // (The real multi-output path is handled below.)
            inputs
                .into_iter()
                .zip(targets.into_iter().map(|t| t[0]))
                .collect()
        }
    };

    match name {
        "perceptron_2_1" => {
            let mut p = Perceptron::<rna::activation::step::Step> {
                weights: vec![0.0; 2],
                bias: 0.0,
                activation: PhantomData,
            };
            let h = LinearHistory::new(1.0, verbose).train_with_history(
                &mut p,
                &dataset_2d,
                Some(100),
                "AND gate",
            );
            Ok(TrainingResult::Single(h))
        }

        "gradient_2_3" => {
            let mut p = Perceptron::<Identity> {
                weights: vec![0.0; 2],
                bias: 0.0,
                activation: PhantomData,
            };
            let h = GradientHistory::new(0.2, 0.125001, None, verbose).train_with_history(
                &mut p,
                &dataset_2d,
                Some(10_000),
                "AND gate (gradient)",
            );
            Ok(TrainingResult::Single(h))
        }

        "gradient_2_9_ls" => {
            let mut p = Perceptron::<Identity> {
                weights: vec![0.0; 2],
                bias: 0.0,
                activation: PhantomData,
            };
            let class_stop = Some(ClassificationStop {
                error_limit: 0,
                threshold: 0.0,
                values: (-1.0, 1.0),
            });
            let h = GradientHistory::new(0.0011, 0.0, class_stop, verbose).train_with_history(
                &mut p,
                &dataset_2d,
                Some(1000),
                "2-9 LS",
            );
            Ok(TrainingResult::Single(h))
        }

        "gradient_2_9_nls" => {
            let mut p = Perceptron::<Identity> {
                weights: vec![0.0; 2],
                bias: 0.0,
                activation: PhantomData,
            };
            let class_stop = Some(ClassificationStop {
                error_limit: 3,
                threshold: 0.0,
                values: (-1.0, 1.0),
            });
            let h = GradientHistory::new(0.0015, 0.0, class_stop, verbose).train_with_history(
                &mut p,
                &dataset_2d,
                Some(1000),
                "2-9 NLS",
            );
            Ok(TrainingResult::Single(h))
        }

        "gradient_2_11" => {
            let mut p = Perceptron::<Identity> {
                weights: vec![0.0; 1],
                bias: 0.0,
                activation: PhantomData,
            };
            let h =
                GradientHistory::new(0.000167, 0.56, None, verbose).train_with_history(
                    &mut p,
                    &dataset_2d,
                    Some(10_000),
                    "regression 2-11",
                );
            Ok(TrainingResult::Single(h))
        }

        "adaline_2_3" => {
            let mut p = Perceptron::<Identity> {
                weights: vec![0.0; 2],
                bias: 0.0,
                activation: PhantomData,
            };
            let h = AdelineHistory::new(0.03, 0.1251, None, verbose).train_with_history(
                &mut p,
                &dataset_2d,
                Some(10_000),
                "AND gate (adaline)",
            );
            Ok(TrainingResult::Single(h))
        }

        "adeline_singlelayer_3_1" | "gradient_singlelayer_3_1" => {
            let (inputs, targets) = match &cfg.dataset {
                DatasetSource::File { path, n_inputs, n_outputs } => {
                    rna::csv_reader::load_dataset_multi(path, *n_inputs, *n_outputs, false)
                        .map_err(|e| format!("CSV load error: {e}"))?
                }
                _ => unreachable!(),
            };

            let template = Perceptron::<Identity> {
                weights: vec![0.0; 2],
                bias: 0.0,
                activation: PhantomData,
            };

            let kind = if name == "adeline_singlelayer_3_1" {
                SingleLayerTrainerKind::Adeline {
                    learning_rate: 0.001,
                    tolerance: 0.01,
                    class_stop: None,
                }
            } else {
                SingleLayerTrainerKind::Gradient {
                    learning_rate: 0.0001,
                    tolerance: 0.01,
                    class_stop: None,
                }
            };

            let h = train_single_layer_with_history(
                &template,
                3,
                &kind,
                &inputs,
                &targets,
                Some(300),
                name,
                verbose,
            );
            Ok(TrainingResult::SingleLayer(h))
        }

        "rna_xor_4_3" => {
            let inputs = vec![
                vec![0.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
            ];
            let outputs_multi = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

            let mut mlp = MLP::<Sigmoid>::new(&[2, 1], 2, -1.0..=1.0);
            let h = train_mlp_with_history(
                &mut mlp,
                &inputs,
                &outputs_multi,
                0.8,
                0.001,
                Some(2000),
                "XOR (MLP)",
                verbose,
            );
            Ok(TrainingResult::Mlp(h))
        }

        other => Err(format!("dispatch not implemented for: {}", other)),
    }
}
