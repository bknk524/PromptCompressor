#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PromptRole {
    CurrentState,
    Request,
    Constraint,
    Verification,
    Target,
    Context,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PromptFact {
    role: PromptRole,
    label: String,
    text: String,
    required_terms: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PromptStructure {
    facts: Vec<PromptFact>,
}

impl PromptStructure {
    pub(super) fn analyze(input: &str, required_terms: &[String]) -> Self {
        let clauses = input_clauses(input);
        let has_actionable_clause = clauses.iter().any(|clause| {
            is_constraint_clause(clause)
                || (!is_current_state_clause(clause) && is_request_clause(clause))
                || (is_current_state_clause(clause)
                    && is_request_clause(clause)
                    && has_current_to_request_transition(clause))
        });
        let non_current_text = clauses
            .iter()
            .filter(|clause| {
                is_constraint_clause(clause)
                    || !is_current_state_clause(clause)
                    || (is_request_clause(clause) && has_current_to_request_transition(clause))
            })
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join("\n");
        let mut facts = Vec::new();

        for clause in clauses {
            if is_redundant_context(&clause)
                && !required_terms
                    .iter()
                    .any(|term| contains_ascii_case_insensitive(&clause, term))
            {
                continue;
            }
            if is_constraint_clause(&clause) {
                for item in atomic_constraint_items(&clause) {
                    let role = if is_verification_prompt_clause(item) {
                        PromptRole::Verification
                    } else {
                        PromptRole::Constraint
                    };
                    let label = if role == PromptRole::Verification {
                        "検証".to_string()
                    } else {
                        constraint_label(item)
                    };
                    push_fact(&mut facts, role, label, item, required_terms);
                }
                continue;
            }

            let is_current = is_current_state_clause(&clause);
            let is_request = is_request_clause(&clause);
            let is_transition =
                is_current && is_request && has_current_to_request_transition(&clause);
            if is_current && !is_transition && has_actionable_clause {
                let has_unique_required_term = required_terms.iter().any(|term| {
                    contains_ascii_case_insensitive(&clause, term)
                        && !contains_ascii_case_insensitive(&non_current_text, term)
                });
                if !has_unique_required_term {
                    continue;
                }
            }

            let (role, label) = if is_transition {
                (PromptRole::Request, "現状→要求")
            } else if is_current {
                (PromptRole::CurrentState, "現状")
            } else if is_verification_prompt_clause(&clause) {
                (PromptRole::Verification, "検証")
            } else if is_request {
                (PromptRole::Request, "要求")
            } else if required_terms
                .iter()
                .any(|term| contains_ascii_case_insensitive(&clause, term))
            {
                (PromptRole::Target, "対象")
            } else {
                (PromptRole::Context, "文脈")
            };
            push_fact(&mut facts, role, label.to_string(), &clause, required_terms);
        }

        Self { facts }
    }

    pub(super) fn render_for_model(&self) -> String {
        self.facts
            .iter()
            .map(|fact| {
                if fact.required_terms.is_empty() {
                    format!("[{}] {}", fact.label, fact.text)
                } else {
                    format!(
                        "[{}|必須語:{}] {}",
                        fact.label,
                        fact.required_terms.join(","),
                        fact.text
                    )
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub(super) fn protected_expressions(&self) -> Vec<String> {
        let mut expressions = Vec::new();
        for fact in &self.facts {
            if fact.role != PromptRole::Constraint {
                continue;
            }
            collect_protected_expressions(&fact.text, &fact.required_terms, &mut expressions);
        }
        expressions
    }

    pub(super) fn compact_candidate(&self) -> Option<String> {
        if !self.facts.iter().any(|fact| {
            matches!(
                fact.role,
                PromptRole::Request | PromptRole::Constraint | PromptRole::Verification
            )
        }) {
            return None;
        }

        let mut compacted = Vec::new();
        for fact in &self.facts {
            if fact.role == PromptRole::Context
                && fact.required_terms.is_empty()
                && is_redundant_context(&fact.text)
            {
                continue;
            }

            let text = compact_surface_text(&fact.text);
            if text.is_empty()
                || compacted
                    .iter()
                    .any(|existing: &String| existing.eq_ignore_ascii_case(&text))
            {
                continue;
            }
            compacted.push(text);
        }

        let candidate = compacted.join("。");
        (!candidate.is_empty()).then_some(candidate)
    }
}

fn collect_protected_expressions(
    text: &str,
    required_terms: &[String],
    expressions: &mut Vec<String>,
) {
    for marker in ["だけ", "のみ", "以外"] {
        for (marker_start, _) in text.match_indices(marker) {
            let marker_end = marker_start + marker.len();
            let nearest_term = required_terms
                .iter()
                .filter_map(|term| {
                    text[..marker_start]
                        .rfind(term)
                        .map(|start| (start, start + term.len(), term))
                })
                .filter(|(_, end, _)| marker_start.saturating_sub(*end) <= 3)
                .max_by_key(|(_, end, _)| *end);
            let expression = nearest_term
                .map(|(start, _, _)| text[start..marker_end].to_string())
                .unwrap_or_else(|| suffix_through_marker(text, marker_start, marker_end, 14));
            push_unique(expressions, expression);
        }
    }

    for marker in [
        "しない",
        "せず",
        "ないで",
        "禁止",
        "不可",
        "除外",
        "変更せず",
        "増やさない",
    ] {
        for (marker_start, _) in text.match_indices(marker) {
            let marker_end = marker_start + marker.len();
            push_unique(
                expressions,
                suffix_through_marker(text, marker_start, marker_end, 20),
            );
        }
    }

    for marker in ["なら", "場合", "とき", "時は", "unless", "only if"] {
        if contains_ascii_case_insensitive(text, marker)
            && !expressions
                .iter()
                .any(|expression| contains_ascii_case_insensitive(expression, marker))
        {
            push_unique(expressions, marker.to_string());
        }
    }
}

fn suffix_through_marker(
    text: &str,
    marker_start: usize,
    marker_end: usize,
    max_chars: usize,
) -> String {
    let prefix = &text[..marker_start];
    let start = prefix
        .char_indices()
        .rev()
        .take(max_chars)
        .last()
        .map(|(index, _)| index)
        .unwrap_or(0);
    text[start..marker_end]
        .trim_start_matches(['。', '、', ',', '，', ';', '；', ' ', '\t', '\n'])
        .to_string()
}

fn push_unique(values: &mut Vec<String>, value: String) {
    let value = value.trim().to_string();
    if !value.is_empty()
        && !values
            .iter()
            .any(|existing| existing.eq_ignore_ascii_case(&value))
    {
        values.push(value);
    }
}

fn push_fact(
    facts: &mut Vec<PromptFact>,
    role: PromptRole,
    label: String,
    text: &str,
    required_terms: &[String],
) {
    let required_terms = required_terms
        .iter()
        .filter(|term| contains_ascii_case_insensitive(text, term))
        .cloned()
        .collect();
    facts.push(PromptFact {
        role,
        label,
        text: text.trim().to_string(),
        required_terms,
    });
}

fn input_clauses(input: &str) -> Vec<String> {
    input
        .split_inclusive(['。', '！', '？', '\n', ';', '；'])
        .map(|segment| {
            segment
                .trim()
                .trim_end_matches(['。', '！', '？', ';', '；'])
                .trim()
                .to_string()
        })
        .filter(|clause| !clause.is_empty())
        .collect()
}

fn atomic_constraint_items(clause: &str) -> Vec<&str> {
    let items = clause
        .split(['、', ','])
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .collect::<Vec<_>>();
    if items.len() > 1 && items.iter().all(|item| is_constraint_clause(item)) {
        items
    } else {
        vec![clause]
    }
}

fn constraint_label(clause: &str) -> String {
    let mut kinds = Vec::new();
    if contains_any_marker(clause, &["だけ", "のみ", "only"]) {
        kinds.push("限定");
    }
    if contains_any_marker(
        clause,
        &[
            "場合",
            "なら",
            "ときは",
            "時は",
            "際は",
            "超過時",
            "失敗時",
            "成功時",
        ],
    ) {
        kinds.push("条件対応");
    }
    if contains_any_marker(
        clause,
        &[
            "しない",
            "せず",
            "禁止",
            "避け",
            "変えず",
            "変えない",
            "変更しない",
            "消さない",
            "行わない",
            "不要",
            "must not",
            "do not",
            "without",
        ],
    ) {
        kinds.push("禁止");
    }
    if contains_any_marker(
        clause,
        &["維持", "保持", "残す", "復元", "keep", "preserve"],
    ) {
        kinds.push("維持");
    }
    if kinds.is_empty() {
        "制約".to_string()
    } else {
        format!("制約:{}", kinds.join("+"))
    }
}

fn is_constraint_clause(clause: &str) -> bool {
    contains_any_marker(
        clause,
        &[
            "避け",
            "禁止",
            "しない",
            "できない",
            "不要",
            "ではなく",
            "でなく",
            "行わない",
            "読み込まず",
            "読まない",
            "下げない",
            "廃止",
            "変えず",
            "入れない",
            "影響させない",
            "削除しない",
            "消えない",
            "消さない",
            "変更不可",
            "実通信しない",
            "送信しない",
            "二重作成しない",
            "再接続しない",
            "再試行しない",
            "超えたら",
            "再推論せず",
            "戻さない",
            "参照しない",
            "混ぜない",
            "のみ",
            "だけ",
            "必ず",
            "維持",
            "残す",
            "テスト",
            "確認できる",
            "確認したい",
            "場合",
            "なら",
            "ときは",
            "時は",
            "際は",
            "失敗時",
            "成功時",
            "変えない",
            "変更せず",
            "変更しない",
            "改変せず",
            "増やさない",
            "せず",
            "must",
            "must not",
            "do not",
            "don't",
            "avoid",
            "only",
            "without",
            "preserve",
            "keep",
        ],
    )
}

fn is_current_state_clause(clause: &str) -> bool {
    contains_any_marker(
        clause,
        &[
            "今の実装",
            "現在",
            "現状",
            "今は",
            "いまは",
            "発生",
            "起きて",
            "なって",
            "してしま",
            "できない",
            "届かない",
            "重い",
            "遅い",
            "不便",
            "問題",
            "困って",
            "言われて",
        ],
    )
}

fn is_request_clause(clause: &str) -> bool {
    contains_any_marker(
        clause,
        &[
            "修正",
            "追加",
            "を実装",
            "実装して",
            "実装する",
            "実装を",
            "作成",
            "更新",
            "調査",
            "整理",
            "提案",
            "検証",
            "確認",
            "返却",
            "コピー",
            "保持",
            "維持",
            "直して",
            "直したい",
            "してほしい",
            "したい",
            "対応して",
            "使って",
            "使用して",
            "用いて",
        ],
    )
}

fn is_verification_prompt_clause(clause: &str) -> bool {
    let implements_validation = contains_any_marker(
        clause,
        &[
            "検証を追加",
            "検証を実装",
            "検証処理",
            "検証機能",
            "検証ロジック",
        ],
    );
    contains_any_marker(clause, &["テスト", "確認", "test", "spec", "assert"])
        || (clause.contains("検証") && !implements_validation)
}

fn has_current_to_request_transition(clause: &str) -> bool {
    contains_any_marker(
        clause,
        &["ので", "ため", "から", "ですが", "けれど", "一方で"],
    )
}

fn is_redundant_context(text: &str) -> bool {
    contains_any_marker(
        text,
        &[
            "前にもお願い",
            "以前にもお願い",
            "何度かお願い",
            "念のため背景",
            "参考までに",
            "余談ですが",
            "背景説明は長く",
            "背景説明が長く",
            "必要なのはこの変更だけ",
            "必要なのは上記だけ",
        ],
    )
}

fn compact_surface_text(text: &str) -> String {
    let mut compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
    for (from, to) in [
        ("なんですけど ", "で"),
        ("なんですけど、", "で"),
        ("を直してほしいです", "を修正"),
        ("を呼び出してください", "呼出"),
        ("を作成してください", "作成"),
        ("を追加してください", "追加"),
        ("を実装してください", "実装"),
        ("を修正してください", "修正"),
        ("を確認してください", "確認"),
        ("を調べてください", "調査"),
        ("を整理してください", "整理"),
        ("をまとめてください", "整理"),
        ("を提案してください", "提案"),
        ("を検証してください", "検証"),
        ("を更新してください", "更新"),
        ("を設定してください", "設定"),
        ("を適用してください", "適用"),
        ("を保持してください", "保持"),
        ("を維持してください", "維持"),
        ("を返してください", "返却"),
        ("をコピーしてください", "コピー"),
        ("を出してください", "出力"),
        ("をお願いしたいです", ""),
        ("お願いいたします", ""),
        ("お願い致します", ""),
    ] {
        compact = compact.replace(from, to);
    }
    compact = compact.trim_start_matches("えっと").trim().to_string();
    if let Some((prefix, request)) = compact.split_once("ですが、") {
        if contains_any_marker(
            prefix,
            &["相談", "お願い", "依頼", "共有", "背景", "前にも", "以前"],
        ) && !request.trim().is_empty()
        {
            compact = request.trim().to_string();
        }
    }
    for prefix in [
        "できれば、",
        "可能であれば、",
        "念のため、",
        "要するに、",
        "今回については、",
    ] {
        if let Some(rest) = compact.strip_prefix(prefix) {
            compact = rest.trim().to_string();
        }
    }
    for suffix in [
        "していただけますでしょうか",
        "していただきたいです",
        "してもらいたいです",
        "してほしいです",
        "するようにしてください",
        "をお願いします",
        "お願いします",
        "ください",
        "です",
    ] {
        if let Some(stem) = compact.strip_suffix(suffix) {
            compact = stem.trim().to_string();
            break;
        }
    }
    compact
}

fn contains_any_marker(value: &str, markers: &[&str]) -> bool {
    let normalized = value.to_ascii_lowercase();
    markers
        .iter()
        .any(|marker| normalized.contains(&marker.to_ascii_lowercase()))
}

fn contains_ascii_case_insensitive(haystack: &str, needle: &str) -> bool {
    haystack
        .to_ascii_lowercase()
        .contains(&needle.to_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::PromptStructure;

    #[test]
    fn renders_conditional_constraints_and_verification_as_distinct_facts() {
        let input = "customerIdが空ならHTTP 400を返してください。既存のJSONキーは変えないでください。Vitestで境界値を確認してください。";
        let terms = vec![
            "customerId".to_string(),
            "HTTP".to_string(),
            "Vitest".to_string(),
        ];

        let rendered = PromptStructure::analyze(input, &terms).render_for_model();

        assert!(rendered.contains("[制約:条件対応"), "{rendered}");
        assert!(rendered.contains("[制約:禁止"), "{rendered}");
        assert!(rendered.contains("[検証"), "{rendered}");
    }

    #[test]
    fn builds_candidate_for_unseen_domain_without_fixture_markers() {
        let input = "前にもお願いした内容ですが、参考までに共有します。Zigのbuild.zigを更新してください。release-fastだけを対象にし、既存のinstall手順は変更しないでください。最後にzig build testで確認してください。";
        let terms = vec![
            "Zig".to_string(),
            "build.zig".to_string(),
            "release-fast".to_string(),
            "install".to_string(),
        ];

        let candidate = PromptStructure::analyze(input, &terms)
            .compact_candidate()
            .expect("generic candidate");

        assert!(!candidate.contains("前にもお願い"), "{candidate}");
        assert!(candidate.contains("build.zig"), "{candidate}");
        assert!(candidate.contains("release-fastだけ"), "{candidate}");
        assert!(candidate.contains("install手順は変更しない"), "{candidate}");
        assert!(candidate.contains("zig build test"), "{candidate}");
        assert!(candidate.chars().count() < input.chars().count());
    }

    #[test]
    fn removes_conversation_history_without_dropping_exclusive_constraints() {
        let input = "前にも少し相談した件ですが、Zigのbuild.zigを整理してください。release-fastだけはLTOを有効にし、release-safeは安全性チェックを無効にしないでください。背景説明は長くなりましたが、必要なのはこの変更だけです。";
        let terms = vec![
            "Zig".to_string(),
            "build.zig".to_string(),
            "release-fast".to_string(),
            "release-safe".to_string(),
            "LTO".to_string(),
        ];

        let candidate = PromptStructure::analyze(input, &terms)
            .compact_candidate()
            .expect("generic candidate");

        assert!(!candidate.contains("前にも"), "{candidate}");
        assert!(!candidate.contains("背景説明"), "{candidate}");
        assert!(candidate.contains("release-fastだけ"), "{candidate}");
        assert!(candidate.contains("無効にしない"), "{candidate}");
    }

    #[test]
    fn extracts_literal_constraint_expressions_for_model_contract() {
        let input = "release-fastだけはLTOを有効にし、release-safeでは安全性チェックを無効にしないでください。Linuxの場合だけ実行してください。";
        let terms = vec![
            "release-fast".to_string(),
            "release-safe".to_string(),
            "Linux".to_string(),
        ];

        let expressions = PromptStructure::analyze(input, &terms).protected_expressions();

        assert!(expressions.iter().any(|value| value == "release-fastだけ"));
        assert!(expressions
            .iter()
            .any(|value| value.contains("無効にしない")));
        assert!(expressions.iter().any(|value| value.contains("場合")));
        assert!(!expressions.iter().any(|value| value == "場合"));
    }
}
