use crate::history::types::ActivationName;

/// Activation choices available for MLP experiments.
/// Step is excluded because it does not implement `Derivative`.
#[derive(Debug, Clone, PartialEq)]
pub enum MlpActivation {
    Sigmoid,
    Tanh,
    Identity,
}

impl MlpActivation {
    pub fn label(&self) -> &'static str {
        match self {
            MlpActivation::Sigmoid => "Sigmoid",
            MlpActivation::Tanh => "Tanh",
            MlpActivation::Identity => "Identity",
        }
    }

    pub fn to_activation_name(&self) -> ActivationName {
        match self {
            MlpActivation::Sigmoid => ActivationName::Sigmoid,
            MlpActivation::Tanh => ActivationName::Tanh,
            MlpActivation::Identity => ActivationName::Identity,
        }
    }
}

/// All tunable hyperparameters for a single experiment run.
/// Defaults always reproduce the behaviour of the original `src/bin/*.rs` files.
#[derive(Debug, Clone)]
pub struct HyperParams {
    pub learning_rate: f64,
    pub tolerance: f64,
    pub max_epochs: usize,

    // ── Classification ────────────────────────────────────────────────────────
    /// None for regression / multi-output experiments.
    pub class_neg: Option<f64>,
    pub class_pos: Option<f64>,
    pub class_threshold: Option<f64>,
    /// When Some(n), stop if misclassified count ≤ n. When None, not used.
    pub class_error_limit: Option<usize>,

    // ── MLP-specific ──────────────────────────────────────────────────────────
    /// Neurons in each hidden layer. Output layer size is derived from the
    /// dataset. The input layer size is always derived from the dataset.
    pub mlp_hidden_sizes: Vec<usize>,
    pub mlp_activation: MlpActivation,
}

impl HyperParams {
    /// Return default hyperparameters matching the original `src/bin/<name>.rs`.
    pub fn defaults_for(name: &str) -> Self {
        match name {
            "perceptron_2_1" => Self {
                learning_rate: 1.0,
                tolerance: 0.0,
                max_epochs: 100,
                class_neg: Some(0.0),
                class_pos: Some(1.0),
                class_threshold: Some(0.5),
                class_error_limit: None,
                mlp_hidden_sizes: vec![],
                mlp_activation: MlpActivation::Sigmoid,
            },
            "gradient_2_3" => Self {
                learning_rate: 0.2,
                tolerance: 0.125001,
                max_epochs: 10_000,
                class_neg: Some(-1.0),
                class_pos: Some(1.0),
                class_threshold: Some(0.0),
                class_error_limit: None,
                mlp_hidden_sizes: vec![],
                mlp_activation: MlpActivation::Sigmoid,
            },
            "gradient_2_9_ls" => Self {
                learning_rate: 0.0011,
                tolerance: 0.0,
                max_epochs: 1000,
                class_neg: Some(-1.0),
                class_pos: Some(1.0),
                class_threshold: Some(0.0),
                class_error_limit: Some(0),
                mlp_hidden_sizes: vec![],
                mlp_activation: MlpActivation::Sigmoid,
            },
            "gradient_2_9_nls" => Self {
                learning_rate: 0.0015,
                tolerance: 0.0,
                max_epochs: 1000,
                class_neg: Some(-1.0),
                class_pos: Some(1.0),
                class_threshold: Some(0.0),
                class_error_limit: Some(3),
                mlp_hidden_sizes: vec![],
                mlp_activation: MlpActivation::Sigmoid,
            },
            "gradient_2_11" => Self {
                learning_rate: 0.000167,
                tolerance: 0.56,
                max_epochs: 10_000,
                class_neg: None,
                class_pos: None,
                class_threshold: None,
                class_error_limit: None,
                mlp_hidden_sizes: vec![],
                mlp_activation: MlpActivation::Sigmoid,
            },
            "adaline_2_3" => Self {
                learning_rate: 0.03,
                tolerance: 0.1251,
                max_epochs: 10_000,
                // original has class_stop: &None — no classification-based early stop
                class_neg: Some(-1.0),
                class_pos: Some(1.0),
                class_threshold: Some(0.0),
                class_error_limit: None,
                mlp_hidden_sizes: vec![],
                mlp_activation: MlpActivation::Sigmoid,
            },
            "adaline_2_9_ls" => Self {
                learning_rate: 0.012,
                tolerance: 0.0,
                max_epochs: 1000,
                class_neg: Some(-1.0),
                class_pos: Some(1.0),
                class_threshold: Some(0.0),
                class_error_limit: Some(0),
                mlp_hidden_sizes: vec![],
                mlp_activation: MlpActivation::Sigmoid,
            },
            "adaline_2_9_nls" => Self {
                learning_rate: 0.0015,
                tolerance: 0.0,
                max_epochs: 1000,
                class_neg: Some(-1.0),
                class_pos: Some(1.0),
                class_threshold: Some(0.0),
                class_error_limit: Some(3),
                mlp_hidden_sizes: vec![],
                mlp_activation: MlpActivation::Sigmoid,
            },
            "adaline_2_11" => Self {
                learning_rate: 0.00014,
                tolerance: 0.56,
                max_epochs: 10_000,
                class_neg: None,
                class_pos: None,
                class_threshold: None,
                class_error_limit: None,
                mlp_hidden_sizes: vec![],
                mlp_activation: MlpActivation::Sigmoid,
            },
            "adaline_singlelayer_3_5" => Self {
                learning_rate: 0.001,
                tolerance: 0.05,
                max_epochs: 1000,
                class_neg: None, // 25-input, multi-output — no class controls
                class_pos: None,
                class_threshold: None,
                class_error_limit: None,
                mlp_hidden_sizes: vec![],
                mlp_activation: MlpActivation::Sigmoid,
            },
            "adaline_singlelayer_3_1" => Self {
                learning_rate: 0.001,
                tolerance: 0.01,
                max_epochs: 300,
                class_neg: None, // multi-output CSV, class controls hidden
                class_pos: None,
                class_threshold: None,
                class_error_limit: None,
                mlp_hidden_sizes: vec![],
                mlp_activation: MlpActivation::Sigmoid,
            },
            "gradient_singlelayer_3_1" => Self {
                learning_rate: 0.0001,
                tolerance: 0.01,
                max_epochs: 300,
                class_neg: None,
                class_pos: None,
                class_threshold: None,
                class_error_limit: None,
                mlp_hidden_sizes: vec![],
                mlp_activation: MlpActivation::Sigmoid,
            },
            "rna_cnl_4_12" => Self {
                learning_rate: 0.5,
                tolerance: 0.001,
                max_epochs: 2000,
                class_neg: Some(0.0),
                class_pos: Some(1.0),
                class_threshold: Some(0.5),
                class_error_limit: None,
                mlp_hidden_sizes: vec![10],
                mlp_activation: MlpActivation::Sigmoid,
            },
            "rna_cnl_4_14" => Self {
                learning_rate: 1.2,
                tolerance: 0.001,
                max_epochs: 2000,
                class_neg: None, // multi-class argmax — no binary class labels
                class_pos: None,
                class_threshold: None,
                class_error_limit: None,
                mlp_hidden_sizes: vec![10],
                mlp_activation: MlpActivation::Sigmoid,
            },
            "rna_reg_1_dim_4_17" => Self {
                learning_rate: 0.001,
                tolerance: 0.04,
                max_epochs: 10_000,
                class_neg: None,
                class_pos: None,
                class_threshold: None,
                class_error_limit: None,
                mlp_hidden_sizes: vec![10],
                mlp_activation: MlpActivation::Sigmoid,
            },
            "rna_langue_signe" => Self {
                learning_rate: 0.03,
                tolerance: 0.03,
                max_epochs: 300,
                class_neg: None, // 5-class argmax — no binary class controls
                class_pos: None,
                class_threshold: None,
                class_error_limit: None,
                mlp_hidden_sizes: vec![10],
                mlp_activation: MlpActivation::Tanh,
            },
            "rna_xor_4_3" => Self {
                learning_rate: 0.8,
                tolerance: 0.001,
                max_epochs: 2000,
                class_neg: Some(0.0),
                class_pos: Some(1.0),
                class_threshold: Some(0.5),
                class_error_limit: None,
                mlp_hidden_sizes: vec![2], // one hidden layer, 2 neurons
                mlp_activation: MlpActivation::Sigmoid,
            },
            _ => Self::default(),
        }
    }
}

impl Default for HyperParams {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            tolerance: 0.001,
            max_epochs: 1000,
            class_neg: Some(-1.0),
            class_pos: Some(1.0),
            class_threshold: Some(0.0),
            class_error_limit: None,
            mlp_hidden_sizes: vec![4],
            mlp_activation: MlpActivation::Sigmoid,
        }
    }
}
