use crate::runtime::backend::{normalized_verification_input, preserves_requested_negations};
use crate::types::{CompressionRequest, PreservedRequirement, RiskFlag, RiskSeverity};

#[derive(Debug, Clone)]
pub struct VerificationSummary {
    pub preserved_requirements: Vec<PreservedRequirement>,
    pub risk_flags: Vec<RiskFlag>,
    pub should_send_original: bool,
    pub fallback_reason: Option<String>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct SimpleVerifier;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CheckStatus {
    NotApplicable,
    Preserved,
    Missing,
}

impl SimpleVerifier {
    pub fn verify(
        &self,
        request: &CompressionRequest,
        distilled_prompt: &str,
    ) -> VerificationSummary {
        if distilled_prompt.trim().is_empty() {
            return VerificationSummary {
                preserved_requirements: Vec::new(),
                risk_flags: vec![RiskFlag {
                    code: "EMPTY_OUTPUT".to_string(),
                    severity: RiskSeverity::High,
                    message: "Compression produced an empty output; sending the original is safer."
                        .to_string(),
                }],
                should_send_original: true,
                fallback_reason: Some("empty_distilled_prompt".to_string()),
            };
        }

        let verification_input = normalized_verification_input(&request.input_text);
        let checks = [
            (
                "preserve_code_blocks",
                check_exact_values(
                    request.constraints.preserve_code_blocks,
                    extract_fenced_code_blocks(&verification_input),
                    extract_fenced_code_blocks(distilled_prompt),
                ),
            ),
            (
                "preserve_file_names",
                check_case_insensitive_values(
                    request.constraints.preserve_file_names,
                    extract_file_names(&verification_input),
                    extract_file_names(distilled_prompt),
                ),
            ),
            (
                "preserve_error_messages",
                check_values_in_text(
                    request.constraints.preserve_error_messages,
                    extract_error_literals(&verification_input),
                    distilled_prompt,
                ),
            ),
            (
                "preserve_numbers",
                check_exact_values(
                    request.constraints.preserve_numbers,
                    extract_numbers(&verification_input),
                    extract_all_numbers(distilled_prompt),
                ),
            ),
            (
                "preserve_negations",
                check_negations(request, distilled_prompt),
            ),
        ];

        let preserved_requirements = checks
            .iter()
            .filter(|(_, status)| *status == CheckStatus::Preserved)
            .map(|(name, _)| PreservedRequirement {
                kind: "constraint".to_string(),
                text: (*name).to_string(),
            })
            .collect();
        let missing: Vec<_> = checks
            .iter()
            .filter(|(_, status)| *status == CheckStatus::Missing)
            .map(|(name, _)| *name)
            .collect();
        let should_send_original = !missing.is_empty();
        let fallback_reason =
            should_send_original.then(|| format!("verification_failed: {}", missing.join(", ")));
        let risk_flags = should_send_original
            .then(|| RiskFlag {
                code: "VERIFICATION_FAILED".to_string(),
                severity: RiskSeverity::High,
                message: format!(
                    "Compression omitted protected content for: {}. Sending the original is safer.",
                    missing.join(", ")
                ),
            })
            .into_iter()
            .collect();

        VerificationSummary {
            preserved_requirements,
            risk_flags,
            should_send_original,
            fallback_reason,
        }
    }
}

fn check_exact_values(enabled: bool, input: Vec<String>, output: Vec<String>) -> CheckStatus {
    if !enabled || input.is_empty() {
        return CheckStatus::NotApplicable;
    }
    if input.iter().all(|value| output.contains(value)) {
        CheckStatus::Preserved
    } else {
        CheckStatus::Missing
    }
}

fn check_case_insensitive_values(
    enabled: bool,
    input: Vec<String>,
    output: Vec<String>,
) -> CheckStatus {
    if !enabled || input.is_empty() {
        return CheckStatus::NotApplicable;
    }
    if input
        .iter()
        .all(|value| output.iter().any(|item| item.eq_ignore_ascii_case(value)))
    {
        CheckStatus::Preserved
    } else {
        CheckStatus::Missing
    }
}

fn check_values_in_text(enabled: bool, input: Vec<String>, output: &str) -> CheckStatus {
    if !enabled || input.is_empty() {
        return CheckStatus::NotApplicable;
    }
    if input.iter().all(|value| output.contains(value)) {
        CheckStatus::Preserved
    } else {
        CheckStatus::Missing
    }
}

fn check_negations(request: &CompressionRequest, output: &str) -> CheckStatus {
    if !request.constraints.preserve_negations || !contains_constraint_marker(&request.input_text) {
        return CheckStatus::NotApplicable;
    }
    if preserves_requested_negations(request, output) {
        CheckStatus::Preserved
    } else {
        CheckStatus::Missing
    }
}

fn contains_constraint_marker(input: &str) -> bool {
    let lower = input.to_ascii_lowercase();
    [
        "しない",
        "せず",
        "禁止",
        "避け",
        "変えない",
        "変更しない",
        "変更せず",
        "維持",
        "保持",
        "だけ",
        "のみ",
        "must not",
        "do not",
        "without",
        "preserve",
        "keep",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

fn extract_fenced_code_blocks(input: &str) -> Vec<String> {
    input
        .replace("\r\n", "\n")
        .split("```")
        .enumerate()
        .filter(|(index, _)| index % 2 == 1)
        .map(|(_, block)| block.trim().to_string())
        .filter(|block| !block.is_empty())
        .collect()
}

fn extract_file_names(input: &str) -> Vec<String> {
    const EXTENSIONS: &[&str] = &[
        "bat", "cmd", "css", "csv", "dll", "env", "exe", "go", "graphql", "html", "java", "js",
        "json", "jsx", "kt", "lock", "log", "md", "pdf", "ps1", "py", "rs", "sh", "sql", "swift",
        "toml", "ts", "tsx", "txt", "yaml", "yml",
    ];
    const BASENAMES: &[&str] = &["Dockerfile", "Makefile", "README"];

    let mut values = Vec::new();
    for token in ascii_path_tokens(input) {
        let token = token.trim_end_matches([':', '*']);
        let components: Vec<_> = token
            .split(['/', '\\'])
            .filter(|component| !component.is_empty())
            .collect();
        let file_component_count = components
            .iter()
            .filter(|component| file_name_like(component, EXTENSIONS, BASENAMES))
            .count();
        if file_component_count >= 2 {
            for component in components {
                collect_file_names(component, EXTENSIONS, BASENAMES, &mut values);
            }
        } else {
            collect_file_names(token, EXTENSIONS, BASENAMES, &mut values);
        }
    }
    values
}

fn file_name_like(token: &str, extensions: &[&str], basenames: &[&str]) -> bool {
    let lower = token.to_ascii_lowercase();
    basenames
        .iter()
        .any(|name| token.eq_ignore_ascii_case(name))
        || extensions
            .iter()
            .any(|extension| lower.contains(&format!(".{extension}")))
}

fn collect_file_names(
    token: &str,
    extensions: &[&str],
    basenames: &[&str],
    values: &mut Vec<String>,
) {
    let lower = token.to_ascii_lowercase();
    if lower == "node.js" {
        return;
    }
    if basenames
        .iter()
        .any(|name| token.eq_ignore_ascii_case(name))
    {
        push_unique(values, token.to_string());
        return;
    }
    for extension in extensions {
        let suffix = format!(".{extension}");
        let Some(index) = lower.find(&suffix) else {
            continue;
        };
        let end = index + suffix.len();
        push_unique(values, token[..end].to_string());
    }
}

fn ascii_path_tokens(input: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for character in input.chars() {
        if character.is_ascii_alphanumeric()
            || matches!(
                character,
                '_' | '-' | '.' | '/' | '\\' | ':' | '*' | '{' | '}'
            )
        {
            current.push(character);
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

fn extract_error_literals(input: &str) -> Vec<String> {
    let mut values = Vec::new();
    for clause in input.split(['。', '！', '？', '\n', ';']) {
        let lower = clause.to_ascii_lowercase();
        if !["エラー", "error", "失敗", "例外", "返", "throw"]
            .iter()
            .any(|marker| lower.contains(marker))
        {
            continue;
        }
        for token in ascii_identifier_tokens(clause) {
            let upper = token.to_ascii_uppercase();
            let excluded = ["API", "GET", "HTTP", "JSON", "POST", "PUT", "DELETE"];
            let is_error_code = token.len() >= 4
                && token
                    .chars()
                    .any(|character| character.is_ascii_alphabetic())
                && token
                    .chars()
                    .all(|character| !character.is_ascii_lowercase())
                && !excluded.contains(&upper.as_str());
            if is_error_code {
                push_unique(&mut values, token);
            }
        }
        for (open, close) in [('"', '"'), ('「', '」'), ('『', '』')] {
            for value in quoted_values(clause, open, close) {
                push_unique(&mut values, value);
            }
        }
    }
    values
}

fn ascii_identifier_tokens(input: &str) -> Vec<String> {
    input
        .split(|character: char| {
            !(character.is_ascii_alphanumeric() || matches!(character, '_' | '-'))
        })
        .filter(|token| !token.is_empty())
        .map(str::to_string)
        .collect()
}

fn quoted_values(input: &str, open: char, close: char) -> Vec<String> {
    let mut values = Vec::new();
    let mut remainder = input;
    while let Some(start) = remainder.find(open) {
        let after_open = &remainder[start + open.len_utf8()..];
        let Some(end) = after_open.find(close) else {
            break;
        };
        let value = after_open[..end].trim();
        if !value.is_empty() {
            values.push(value.to_string());
        }
        remainder = &after_open[end + close.len_utf8()..];
    }
    values
}

fn extract_numbers(input: &str) -> Vec<String> {
    let mut values = Vec::new();
    for clause in input.split(['。', '！', '？', '\n', ';']) {
        if [
            "関係ありません",
            "関係ない",
            "意味ない",
            "無視してください",
            "タッチミス",
        ]
        .iter()
        .any(|marker| clause.contains(marker))
        {
            continue;
        }
        let effective = corrected_clause_tail(clause);
        let mut current = String::new();
        for character in effective.chars() {
            if character.is_ascii_digit() {
                current.push(character);
            } else if !current.is_empty() {
                push_unique(&mut values, std::mem::take(&mut current));
            }
        }
        if !current.is_empty() {
            push_unique(&mut values, current);
        }
    }
    values
}

pub(crate) fn preserves_requested_numbers(input: &str, output: &str) -> bool {
    let required = extract_numbers(input);
    let actual = extract_all_numbers(output);
    required.iter().all(|value| actual.contains(value))
}

fn extract_all_numbers(input: &str) -> Vec<String> {
    let mut values = Vec::new();
    let mut current = String::new();
    for character in input.chars() {
        if character.is_ascii_digit() {
            current.push(character);
        } else if !current.is_empty() {
            push_unique(&mut values, std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        push_unique(&mut values, current);
    }
    values
}

fn corrected_clause_tail(clause: &str) -> &str {
    ["最終的には", "最終的に", "正しくは", "ここは", "ではなく"]
        .iter()
        .filter_map(|marker| clause.rfind(marker).map(|index| (index, *marker)))
        .max_by_key(|(index, _)| *index)
        .map_or(clause, |(index, marker)| &clause[index + marker.len()..])
}

fn push_unique(values: &mut Vec<String>, value: String) {
    if !values.iter().any(|existing| existing == &value) {
        values.push(value);
    }
}

#[cfg(test)]
mod tests {
    use super::SimpleVerifier;
    use crate::types::{
        CompressionConstraints, CompressionLevel, CompressionRequest, RequestSource, RequestTarget,
    };

    #[test]
    fn rejects_missing_protected_values_without_reporting_them_as_preserved() {
        let request = test_request(
            "config.yamlでHTTP 400のINVALID_CUSTOMERを返してください。既存画面の表示文言は変えないでください。",
        );

        let summary = SimpleVerifier.verify(&request, "入力エラーを返すよう修正する。");

        assert!(summary.should_send_original);
        assert_eq!(
            summary.fallback_reason.as_deref(),
            Some("verification_failed: preserve_file_names, preserve_error_messages, preserve_numbers, preserve_negations")
        );
        for missing in [
            "preserve_file_names",
            "preserve_error_messages",
            "preserve_numbers",
            "preserve_negations",
        ] {
            assert!(!summary
                .preserved_requirements
                .iter()
                .any(|requirement| requirement.text == missing));
        }
        assert!(summary
            .risk_flags
            .iter()
            .any(|risk| risk.code == "VERIFICATION_FAILED"));
    }

    #[test]
    fn reports_only_constraints_verified_against_the_output() {
        let request = test_request(
            "config.yamlでHTTP 400のINVALID_CUSTOMERを返してください。既存画面の表示文言は変えないでください。",
        );
        let output =
            "config.yaml: HTTP 400でINVALID_CUSTOMERを返す。既存画面の表示文言は変更しない。";

        let summary = SimpleVerifier.verify(&request, output);

        assert!(!summary.should_send_original);
        assert!(summary.risk_flags.is_empty());
        for preserved in [
            "preserve_file_names",
            "preserve_error_messages",
            "preserve_numbers",
            "preserve_negations",
        ] {
            assert!(summary
                .preserved_requirements
                .iter()
                .any(|requirement| requirement.text == preserved));
        }
    }

    #[test]
    fn omits_preservation_claims_when_the_constraint_is_not_applicable() {
        let request = test_request("依頼を短くまとめてください。");

        let summary = SimpleVerifier.verify(&request, "依頼を短縮する。");

        assert!(!summary.should_send_original);
        assert!(summary.preserved_requirements.is_empty());
    }

    #[test]
    fn ignores_noise_and_superseded_values_before_verifying_numbers() {
        let input = "指数バックオフは1秒、2秒、4秒、最大30秒、jitterは±20%。zzzz 987654はタッチミスなので関係ありません。60秒後に0へ戻し、4001/4003では再接続しない。10回で停止。最大20秒とも考えましたが、最終的には30秒です。";
        let output = "指数バックオフ1/2/4秒、最大30秒、±20% jitter。60秒後0へ戻す。4001/4003では再接続しない。10回で停止。";
        let request = test_request(input);

        let summary = SimpleVerifier.verify(&request, output);

        assert!(!summary.should_send_original, "{summary:?}");
        assert!(summary
            .preserved_requirements
            .iter()
            .any(|requirement| requirement.text == "preserve_numbers"));
    }

    #[test]
    fn verifies_corrected_error_code_and_final_status_only() {
        let input = "INVALID_CUSTMERではなくINVALID_CUSTOMERを返してください。たぶんHTTP 404とも考えましたが、ここはHTTP 400が正しいです。";
        let output = "HTTP 400でINVALID_CUSTOMERを返す。";
        let mut request = test_request(input);
        request.constraints.preserve_negations = false;

        let summary = SimpleVerifier.verify(&request, output);

        assert!(!summary.should_send_original, "{summary:?}");
        for preserved in ["preserve_error_messages", "preserve_numbers"] {
            assert!(summary
                .preserved_requirements
                .iter()
                .any(|requirement| requirement.text == preserved));
        }
    }

    #[test]
    fn accepts_compact_search_state_and_rebuild_avoidance() {
        let request = test_request(
            "React の検索画面について相談です。今は検索欄に文字を入力している途中でも API が呼ばれてしまい、通信回数が多くなって画面の反応も重く感じます。検索ボタンを押した時だけ API を呼び出すように変更してください。既存の useSearchParams による URL クエリ管理は維持し、ページ番号を変更しても検索条件と検索状態が消えないようにしてください。TypeScript の既存構造はなるべく活かし、大規模なリファクタリングや画面全体の作り直しは避けてください。",
        );
        let output = "React の検索画面について相談。検索ボタンを押した時だけ API を呼び出すように変更して。既存の useSearchParams による URL クエリ管理は維持し。ページ変更時も検索条件/状態維持。TypeScript の既存構造はなるべく活かし、大規模リファクタリング/画面作り直し回避";

        let summary = SimpleVerifier.verify(&request, output);

        assert!(
            !summary.should_send_original,
            "{:?}",
            summary.fallback_reason
        );
    }

    #[test]
    fn accepts_compact_model_readme_and_ci_constraints() {
        let readme_request = test_request(
            "アプリ内 Model フォルダの役割を README に追記してください。採用中の Sarashina 2.2 3B GGUF の配置先、モデル本体を Git 管理しない理由、LM Studio 接続はユーザーが任意のローカルモデルを検証するために残すことを明記してください。exe 化した後でも、アプリ内モデルと LM Studio 接続の役割が分かるように説明してください。",
        );
        let readme_output = "アプリ内 Model フォルダの役割を README に追記して。Sarashina 2.2 3B GGUF の配置先、モデル本体を Git 管理しない理由、LM Studio 接続は任意ローカルモデル検証用に残すを明記して。exe 化した後でも、アプリ内モデルと LM Studio 接続の役割が分かるように説明して";
        let readme_summary = SimpleVerifier.verify(&readme_request, readme_output);
        assert!(
            !readme_summary.should_send_original,
            "{:?}",
            readme_summary.fallback_reason
        );

        let ci_request = test_request(
            "GitHub Actions の Node.js CI が毎回依存関係を再インストールしていて遅いです。actions/cache を使って npm のキャッシュを有効化してください。ただし package-lock.json をキーに含め、テストコマンド npm test と lint は変更しないでください。キャッシュが効かなかった場合でも CI が失敗しないようにし、ログでキャッシュヒットの有無を確認できるようにしてください。",
        );
        let ci_output = "GitHub Actions の Node.js CI の再インストール遅延。actions/cache でnpmキャッシュ有効化。ただし package-lock.json をキーに含め、npm test/lint変更しない。キャッシュ無効でもCI失敗しない。ログでキャッシュヒット確認";
        let ci_summary = SimpleVerifier.verify(&ci_request, ci_output);
        assert!(
            !ci_summary.should_send_original,
            "{:?}",
            ci_summary.fallback_reason
        );
    }

    #[test]
    fn verifies_fenced_code_blocks_only_when_the_block_survives() {
        let input = "次のコードを保ったまま短くしてください。\n```rust\nfn main() {}\n```";
        let request = test_request(input);

        let preserved =
            SimpleVerifier.verify(&request, "実行条件を整理。\n```rust\nfn main() {}\n```");
        let missing = SimpleVerifier.verify(&request, "実行条件を整理。");

        assert!(preserved
            .preserved_requirements
            .iter()
            .any(|requirement| requirement.text == "preserve_code_blocks"));
        assert!(missing.should_send_original);
    }

    #[test]
    fn verifies_standalone_file_name_before_a_colon() {
        let request = test_request("Modelフォルダの役割をREADMEに追記してください。");

        let summary = SimpleVerifier.verify(&request, "README: Modelフォルダの役割を追記。");

        assert!(!summary.should_send_original, "{summary:?}");
        assert!(summary
            .preserved_requirements
            .iter()
            .any(|requirement| requirement.text == "preserve_file_names"));
    }

    #[test]
    fn keeps_output_numbers_that_appear_before_a_contrast_marker() {
        let request = test_request(
            "Windows WebView2アプリです。通知はPowerShellではなくアプリ本体から出してください。",
        );

        let summary = SimpleVerifier.verify(
            &request,
            "Windows WebView2通知: PowerShellではなくアプリ本体から出す。",
        );

        assert!(!summary.should_send_original, "{summary:?}");
        assert!(summary
            .preserved_requirements
            .iter()
            .any(|requirement| requirement.text == "preserve_numbers"));
    }

    #[test]
    fn verifies_multiple_file_names_joined_by_a_slash() {
        let request = test_request("ja.jsonとen.jsonの不足キーを検出してください。");

        let summary = SimpleVerifier.verify(&request, "ja.json/en.json不足キーを検出。");

        assert!(!summary.should_send_original, "{summary:?}");
        assert!(summary
            .preserved_requirements
            .iter()
            .any(|requirement| requirement.text == "preserve_file_names"));
    }

    fn test_request(input_text: &str) -> CompressionRequest {
        CompressionRequest {
            input_text: input_text.to_string(),
            compression_level: CompressionLevel::from_u8(2).expect("valid level"),
            profile: "internal_llm".to_string(),
            constraints: CompressionConstraints::default(),
            target: RequestTarget::codex_default(),
            source: RequestSource::Cli,
        }
    }
}
