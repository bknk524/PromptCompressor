#[derive(Debug, Default, Clone, Copy)]
pub struct SimpleTokenEstimator;

impl SimpleTokenEstimator {
    pub fn estimate(&self, input: &str, _tokenizer_profile: &str) -> usize {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return 0;
        }

        trimmed.split_whitespace().count().max(1)
    }
}
