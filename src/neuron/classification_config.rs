#[derive(Debug)]
pub struct ClassificationConfig {
    pub error_limit: usize,
    pub threshold: f64,
    pub values: (f64, f64),
}
