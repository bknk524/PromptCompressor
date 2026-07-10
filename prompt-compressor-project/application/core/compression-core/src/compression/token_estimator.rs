#[derive(Debug, Default, Clone, Copy)]
pub struct SimpleTokenEstimator;

impl SimpleTokenEstimator {
    pub fn estimate(&self, input: &str, _tokenizer_profile: &str) -> usize {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return 0;
        }

        let mut estimate = 0;
        let mut japanese_characters = 0;
        let mut latin_characters = 0;
        let mut number_characters = 0;
        let mut active_run = None;

        #[derive(Clone, Copy, PartialEq, Eq)]
        enum RunKind {
            Japanese,
            Latin,
            Number,
        }

        let flush_runs = |estimate: &mut usize,
                          japanese_characters: &mut usize,
                          latin_characters: &mut usize,
                          number_characters: &mut usize| {
            if *japanese_characters > 0 {
                // Japanese text is usually split more densely than whitespace-delimited words.
                *estimate += (*japanese_characters * 2).div_ceil(3);
                *japanese_characters = 0;
            }
            if *latin_characters > 0 {
                // Codex-style BPE tokenizers average roughly four ASCII characters per token.
                *estimate += (*latin_characters).div_ceil(4);
                *latin_characters = 0;
            }
            if *number_characters > 0 {
                *estimate += (*number_characters).div_ceil(3);
                *number_characters = 0;
            }
        };

        for character in trimmed.chars() {
            let run_kind = if is_japanese_character(character) {
                Some(RunKind::Japanese)
            } else if character.is_ascii_alphabetic() || character == '_' {
                Some(RunKind::Latin)
            } else if character.is_ascii_digit() {
                Some(RunKind::Number)
            } else {
                None
            };

            let Some(run_kind) = run_kind else {
                flush_runs(
                    &mut estimate,
                    &mut japanese_characters,
                    &mut latin_characters,
                    &mut number_characters,
                );
                active_run = None;
                if !character.is_whitespace() {
                    estimate += 1;
                }
                continue;
            };

            if active_run != Some(run_kind) {
                flush_runs(
                    &mut estimate,
                    &mut japanese_characters,
                    &mut latin_characters,
                    &mut number_characters,
                );
            }

            match run_kind {
                RunKind::Japanese => japanese_characters += 1,
                RunKind::Latin => latin_characters += 1,
                RunKind::Number => number_characters += 1,
            }
            active_run = Some(run_kind);
        }

        flush_runs(
            &mut estimate,
            &mut japanese_characters,
            &mut latin_characters,
            &mut number_characters,
        );
        estimate.max(1)
    }
}

fn is_japanese_character(character: char) -> bool {
    matches!(character, '\u{3040}'..='\u{30ff}' | '\u{3400}'..='\u{9fff}')
}

#[cfg(test)]
mod tests {
    use super::SimpleTokenEstimator;

    #[test]
    fn estimates_japanese_text_as_more_than_one_token() {
        let estimate = SimpleTokenEstimator.estimate(
            "検索ボタンを押したときだけAPIを呼び出してください。",
            "codex_default",
        );

        assert!(estimate > 10);
    }

    #[test]
    fn estimates_a_shorter_japanese_prompt_as_fewer_tokens() {
        let estimator = SimpleTokenEstimator;
        let original = "検索ボタンを押したときだけAPIを呼び出し、URLクエリを保持してください。既存のuseSearchParamsを使い、大きなリファクタは避けてください。";
        let compressed = "検索ボタン押下時のみAPIを呼び出し、URLクエリとuseSearchParamsを保持。大規模なリファクタは避ける。";

        assert!(
            estimator.estimate(original, "codex_default")
                > estimator.estimate(compressed, "codex_default")
        );
    }

    #[test]
    fn estimates_english_words_and_identifiers() {
        let estimate = SimpleTokenEstimator.estimate(
            "Keep useSearchParams and URL query parameters.",
            "codex_default",
        );

        assert!(estimate >= 8);
    }
}
