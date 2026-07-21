use super::{
    automatic_runtime_thread_counts, contains_ascii_case_insensitive, contains_japanese_text,
    contains_required_technical_terms, effective_max_output_tokens, embedded_input_cache_key,
    finalize_observed_model_draft, is_meta_task_restatement,
    missing_constraint_restoration_phrases, missing_verification_restoration_phrases,
    normalize_input_whitespace, normalize_known_input_typos_for_llm, organize_input_for_model,
    parse_compression_output, parse_http_base_url, polish_model_output_for_request,
    preprocess_input_for_llm, preserves_negative_constraints,
    preserves_targeted_change_constraints, remove_obvious_input_noise,
    remove_polite_request_fillers, required_constraint_clauses, required_technical_terms,
    restore_missing_required_constraints, restore_missing_required_terms,
    select_context_length_for_token_budget, state_persistence_clause_satisfied,
    structured_candidate_preserves_requirements, structured_constraint_clause,
    take_matching_prepared_value, trusted_precompacted_fallback_draft, validate_compression_draft,
    validate_prompt_token_budget, verification_constraint_satisfied, verified_structured_candidate,
    CompressionDraft, ModelDefinition, ModelFileCoordinator, RuntimeBatchSizes,
    RuntimeCompressionObservation, RuntimeInferenceConfig, RuntimeThreadCounts,
    RuntimeTransformation,
};
use crate::compression::verifier::SimpleVerifier;
use crate::types::{
    CompressionConstraints, CompressionLevel, CompressionRequest, RequestSource, RequestTarget,
};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use uuid::Uuid;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use super::{embedded_cpu_engine_is_supported, EmbeddedCpuCapabilities, EmbeddedCpuEngine};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn embedded_cpu_engines_require_their_complete_instruction_profiles() {
    let compatible = EmbeddedCpuCapabilities {
        sse42: true,
        ..Default::default()
    };
    assert!(embedded_cpu_engine_is_supported(
        EmbeddedCpuEngine::Compatible,
        compatible
    ));
    assert!(!embedded_cpu_engine_is_supported(
        EmbeddedCpuEngine::Avx2,
        compatible
    ));

    let avx2 = EmbeddedCpuCapabilities {
        avx2: true,
        fma: true,
        f16c: true,
        bmi2: true,
        ..compatible
    };
    assert!(embedded_cpu_engine_is_supported(
        EmbeddedCpuEngine::Avx2,
        avx2
    ));
    assert!(!embedded_cpu_engine_is_supported(
        EmbeddedCpuEngine::Avx2,
        EmbeddedCpuCapabilities {
            bmi2: false,
            ..avx2
        }
    ));

    let avx512 = EmbeddedCpuCapabilities {
        avx512f: true,
        avx512cd: true,
        avx512bw: true,
        avx512dq: true,
        avx512vl: true,
        ..avx2
    };
    assert!(embedded_cpu_engine_is_supported(
        EmbeddedCpuEngine::Avx512,
        avx512
    ));
    assert!(!embedded_cpu_engine_is_supported(
        EmbeddedCpuEngine::Avx512,
        EmbeddedCpuCapabilities {
            avx512vl: false,
            ..avx512
        }
    ));
}

#[test]
fn model_file_coordinator_reuses_path_lock_and_invalidates_changed_files() {
    let directory = std::env::temp_dir().join(format!(
        "prompt-compressor-model-coordinator-{}",
        Uuid::new_v4()
    ));
    fs::create_dir_all(&directory).expect("create model coordinator test directory");
    let model_path = directory.join("model.gguf");
    let other_path = directory.join("other.gguf");
    fs::write(&model_path, b"model-v1").expect("write model file");
    fs::write(&other_path, b"other").expect("write other model file");
    let coordinator = ModelFileCoordinator::default();

    let first_lock = coordinator
        .operation_lock(&model_path)
        .expect("resolve first model lock");
    let second_lock = coordinator
        .operation_lock(&model_path)
        .expect("resolve second model lock");
    let other_lock = coordinator
        .operation_lock(&other_path)
        .expect("resolve other model lock");
    assert!(Arc::ptr_eq(&first_lock, &second_lock));
    assert!(!Arc::ptr_eq(&first_lock, &other_lock));

    coordinator
        .mark_verified(&model_path)
        .expect("cache verified model");
    assert!(coordinator
        .is_verified(&model_path)
        .expect("read verified model cache"));

    fs::write(&model_path, b"model-v2-with-a-different-size").expect("replace model file");
    assert!(!coordinator
        .is_verified(&model_path)
        .expect("invalidate changed model cache"));

    fs::remove_dir_all(directory).expect("remove model coordinator test directory");
}

#[test]
fn automatic_runtime_threads_leave_capacity_for_the_app() {
    let expected = [
        (1, (1, 1)),
        (2, (1, 1)),
        (3, (2, 2)),
        (4, (3, 3)),
        (8, (6, 7)),
        (16, (7, 8)),
    ];

    for (available, (generation, batch)) in expected {
        let counts = automatic_runtime_thread_counts(available);
        assert_eq!(counts.generation, generation);
        assert_eq!(counts.batch, batch);
    }
}

#[test]
fn prepared_input_key_changes_with_the_formatted_input_without_storing_its_text() {
    let model = test_model_definition(256);
    let model_path = PathBuf::from("models/test.gguf");
    let configuration = test_runtime_configuration();
    let first = embedded_input_cache_key(
        &model,
        &model_path,
        1_024,
        configuration,
        "prefix",
        "秘密の入力A",
    );
    let second = embedded_input_cache_key(
        &model,
        &model_path,
        1_024,
        configuration,
        "prefix",
        "秘密の入力B",
    );

    assert_ne!(first, second);
    assert!(!first.contains("秘密の入力A"));
}

#[test]
fn prepared_input_is_moved_once_only_for_an_exact_key() {
    let mut prepared = Some(("exact".to_string(), 42));

    assert_eq!(
        take_matching_prepared_value(&mut prepared, "different"),
        None
    );
    assert_eq!(prepared.as_ref().map(|(_, value)| *value), Some(42));
    assert_eq!(
        take_matching_prepared_value(&mut prepared, "exact"),
        Some(42)
    );
    assert!(prepared.is_none());
    assert_eq!(take_matching_prepared_value(&mut prepared, "exact"), None);
}

#[test]
fn prepared_input_key_separates_context_sizes() {
    let model = test_model_definition(256);
    let model_path = PathBuf::from("models/test.gguf");
    let configuration = test_runtime_configuration();

    assert_ne!(
        embedded_input_cache_key(&model, &model_path, 1_024, configuration, "prefix", "input"),
        embedded_input_cache_key(&model, &model_path, 2_048, configuration, "prefix", "input")
    );
}

#[test]
fn prepared_input_key_separates_thread_settings() {
    let model = test_model_definition(256);
    let model_path = PathBuf::from("models/test.gguf");

    assert_ne!(
        embedded_input_cache_key(
            &model,
            &model_path,
            1_024,
            test_runtime_configuration_with(6, 7, 512),
            "prefix",
            "input",
        ),
        embedded_input_cache_key(
            &model,
            &model_path,
            1_024,
            test_runtime_configuration_with(5, 6, 512),
            "prefix",
            "input",
        )
    );
}

#[test]
fn prepared_input_key_separates_physical_batch_sizes() {
    let model = test_model_definition(256);
    let model_path = PathBuf::from("models/test.gguf");

    assert_ne!(
        embedded_input_cache_key(
            &model,
            &model_path,
            1_024,
            test_runtime_configuration_with(6, 7, 128),
            "prefix",
            "input",
        ),
        embedded_input_cache_key(
            &model,
            &model_path,
            1_024,
            test_runtime_configuration_with(6, 7, 512),
            "prefix",
            "input",
        )
    );
}

#[test]
fn selects_the_smallest_context_tier_that_preserves_the_full_token_budget() {
    assert_eq!(
        select_context_length_for_token_budget(700, 192, 4_096).expect("1K context"),
        1_024
    );
    assert_eq!(
        select_context_length_for_token_budget(1_000, 192, 4_096).expect("2K context"),
        2_048
    );
    assert_eq!(
        select_context_length_for_token_budget(2_000, 192, 4_096).expect("4K context"),
        4_096
    );
    assert!(select_context_length_for_token_budget(4_000, 192, 4_096).is_err());
}

#[test]
fn parses_json_from_plain_model_output() {
    let draft = parse_compression_output(
        r#"{"distilled_prompt":"Fix search behavior.","removed_content_summary":["trimmed background"]}"#,
    )
    .expect("valid compression JSON");

    assert_eq!(draft.distilled_prompt, "Fix search behavior.");
    assert_eq!(draft.removed_content_summary, ["trimmed background"]);
}

#[test]
fn parses_json_surrounded_by_runtime_text() {
    let draft = parse_compression_output(
        "llama.cpp banner\n{\"distilled_prompt\":\"Keep URL query params.\"}\n",
    )
    .expect("valid embedded compression JSON");

    assert_eq!(draft.distilled_prompt, "Keep URL query params.");
    assert!(draft.removed_content_summary.is_empty());
}

#[test]
fn observation_preserves_raw_model_output_before_runtime_transformations() {
    let request = test_request("README.mdの説明を短く整理してください。".to_string(), 2);
    let raw_model_draft = CompressionDraft {
        distilled_prompt: "出力: README.mdの説明を簡潔にする。".to_string(),
        removed_content_summary: vec!["冗長表現".to_string()],
    };

    let observation = finalize_observed_model_draft(&request, raw_model_draft.clone())
        .expect("valid model output should be observed");

    assert_eq!(observation.raw_model_draft, Some(raw_model_draft));
    assert_eq!(
        observation.final_draft.distilled_prompt,
        "README.mdの説明を簡潔にする。"
    );
    assert!(!observation.transformations.is_empty());
}

#[test]
fn runtime_fallback_is_not_reported_as_raw_model_output() {
    let fallback = CompressionDraft {
        distilled_prompt: "README.mdの説明を簡潔にする。".to_string(),
        removed_content_summary: vec!["実行時フォールバック".to_string()],
    };

    let observation = RuntimeCompressionObservation::runtime_fallback(fallback.clone());

    assert_eq!(observation.raw_model_draft, None);
    assert_eq!(observation.final_draft, fallback);
    assert_eq!(
        observation.transformations,
        [RuntimeTransformation::RuntimeFallback]
    );
}

#[test]
fn recovers_inner_json_object_from_extra_outer_braces() {
    let draft = parse_compression_output(
        "{\n{\"distilled_prompt\":\"React UserTableを分割、data-testid維持。\"}}",
    )
    .expect("inner compression JSON should be recoverable");

    assert_eq!(
        draft.distilled_prompt,
        "React UserTableを分割、data-testid維持。"
    );
}

#[test]
fn removes_model_copied_prompt_labels_from_distilled_prompt() {
    let draft = parse_compression_output(
        r#"{"distilled_prompt":"実行指示: メール通知/アプリ内通知を選定: 短縮文"}"#,
    )
    .expect("valid compression JSON");

    assert_eq!(draft.distilled_prompt, "メール通知/アプリ内通知を選定");
}

#[test]
fn parses_json_object_fragment_from_small_local_models() {
    let draft =
        parse_compression_output(r#""distilled_prompt":"検索ボタン押下時のみAPIを呼び出す。""#)
            .expect("valid compression JSON fragment");

    assert_eq!(
        draft.distilled_prompt,
        "検索ボタン押下時のみAPIを呼び出す。"
    );
    assert!(draft.removed_content_summary.is_empty());
}

#[test]
fn recovers_distilled_prompt_from_truncated_json_object() {
    let draft = parse_compression_output(
        "{\"distilled_prompt\":\"2026-06-24T10:15:03Z/requestId=ab12/ECONNRESET: 本番ログ解析",
    )
    .expect("distilled_prompt should be recoverable from truncated JSON");

    assert_eq!(
        draft.distilled_prompt,
        "2026-06-24T10:15:03Z/requestId=ab12/ECONNRESET: 本番ログ解析"
    );
    assert!(draft.removed_content_summary.is_empty());
}

#[test]
fn parses_a_json_string_from_minimal_model_output() {
    let draft = parse_compression_output(r#""Keep URL state.""#)
        .expect("valid JSON string compression output");

    assert_eq!(draft.distilled_prompt, "Keep URL state.");
    assert!(draft.removed_content_summary.is_empty());
}

#[test]
fn parses_local_server_base_url_with_version_prefix() {
    let base = parse_http_base_url("http://127.0.0.1:8788/v1").expect("valid base URL");

    assert_eq!(base.host, "127.0.0.1");
    assert_eq!(base.port, 8788);
    assert_eq!(base.path_prefix, "/v1");
}

#[test]
fn identifies_meta_task_restatement_as_invalid_output() {
    assert!(is_meta_task_restatement(
        "Given a natural language description of a code task, preserve the login flow."
    ));
    assert!(!is_meta_task_restatement(
        "Update the React page while preserving useSearchParams."
    ));
}

#[test]
fn extracts_technical_terms_that_must_survive_compression() {
    let terms = required_technical_terms(
        "In a React TypeScript page, retain useSearchParams and preserve API and URL state.",
    );

    assert_eq!(
        terms,
        ["React", "TypeScript", "useSearchParams", "API", "URL"]
    );
}

#[test]
fn extracts_test_tools_and_literal_formats_as_required_terms() {
    let terms = required_technical_terms(
        "TypeScript の parseDateRange に Vitest テストを追加し、YYYY-MM-DD を検証してください。",
    );

    assert!(terms.contains(&"TypeScript".to_string()));
    assert!(terms.contains(&"parseDateRange".to_string()));
    assert!(terms.contains(&"Vitest".to_string()));
    assert!(terms.contains(&"YYYY-MM-DD".to_string()));
}

#[test]
fn normalizes_typos_before_extracting_required_terms() {
    let terms = required_technical_terms(
        "TypeScritp の既存構造はなるべく触らず、React の検索画面を修正してください。",
    );

    assert!(terms.contains(&"TypeScript".to_string()));
    assert!(!terms.contains(&"TypeScritp".to_string()));
}

#[test]
fn extracts_japanese_notification_and_numeric_limit_terms() {
    let terms = required_technical_terms(
        "メール通知とアプリ内通知から選び、月額コストは 3 万円以下、個人情報を外部送信しない。",
    );

    assert!(terms.contains(&"メール通知".to_string()));
    assert!(terms.contains(&"アプリ内通知".to_string()));
    assert!(terms.contains(&"3 万円以下".to_string()));
    assert!(terms.contains(&"個人情報".to_string()));
    assert!(contains_ascii_case_insensitive(
        "月額コスト3万円以下",
        "3 万円以下"
    ));
}

#[test]
fn protects_cache_invalidation_targets_and_actions() {
    let input = "CIの依存関係キャッシュを最適化してください。OS、ロックファイル、Rustのバージョンが変わった時だけ無効化し、キャッシュがなくても通常ビルドへ進めるようにします。シークレットをログへ出さないでください。";
    let terms = required_technical_terms(input);
    for required in ["依存関係キャッシュ", "ロックファイル", "無効"] {
        assert!(terms.contains(&required.to_string()), "missing {required}");
    }

    let request = test_request(input.to_string(), 2);
    let raw = CompressionDraft {
        distilled_prompt: "CIの依存関係キャッシュをOSとRustのバージョンが変わった時だけ最適化し、キャッシュがない場合でも通常ビルドへ進めるように設定してください。シークレットはログへ出さないでください。".to_string(),
        removed_content_summary: Vec::new(),
    };
    let observed =
        finalize_observed_model_draft(&request, raw).expect("repair invalid cache output");
    assert!(contains_ascii_case_insensitive(
        &observed.final_draft.distilled_prompt,
        "ロックファイル"
    ));
    assert!(contains_ascii_case_insensitive(
        &observed.final_draft.distilled_prompt,
        "無効"
    ));
    validate_compression_draft(&request, &observed.final_draft)
        .expect("repaired cache output must validate");
}

#[test]
fn treats_natural_compound_terms_as_preserved_parts() {
    assert!(contains_ascii_case_insensitive(
        "通知はWindowsのみ。アプリ内通知は禁止。",
        "Windows 通知"
    ));
    assert!(!contains_ascii_case_insensitive(
        "通知はアプリ内のみ。",
        "Windows 通知"
    ));
}

#[test]
fn extracts_log_identifiers_without_noisy_http_terms() {
    let terms = required_technical_terms(
        "2026-06-24T10:15:03Z requestId=ab12 POST /orders ECONNRESET upstream=payment-service",
    );

    assert!(terms.contains(&"2026-06-24T10:15:03Z".to_string()));
    assert!(terms.contains(&"requestId=ab12".to_string()));
    assert!(terms.contains(&"ECONNRESET".to_string()));
    assert!(terms.contains(&"payment-service".to_string()));
    assert!(!terms.contains(&"requestId".to_string()));
    assert!(!terms.contains(&"2026".to_string()));
    assert!(!terms.contains(&"24T10".to_string()));
    assert!(!terms.contains(&"03Z".to_string()));
    assert!(!terms.contains(&"POST".to_string()));
    assert!(!terms.contains(&"/orders".to_string()));
    assert!(!terms.contains(&"upstream=payment-service".to_string()));
}

#[test]
fn extracts_routes_outside_log_inputs() {
    let terms = required_technical_terms(
        "ログイン後に /dashboard と /login の間でリダイレクトループします。middleware.ts を確認してください。",
    );

    assert!(terms.contains(&"/dashboard".to_string()));
    assert!(terms.contains(&"/login".to_string()));
    assert!(terms.contains(&"middleware.ts".to_string()));
}

#[test]
fn preserves_windows_as_required_term() {
    let terms = required_technical_terms(
        "Windows の WebView2 アプリで PowerShell ではなく AppUserModelID を確認してください。",
    );

    assert!(terms.contains(&"Windows".to_string()));
    assert!(terms.contains(&"WebView2".to_string()));
    assert!(terms.contains(&"PowerShell".to_string()));
    assert!(terms.contains(&"AppUserModelID".to_string()));
}

#[test]
fn extracts_http_status_phrase_as_required_term() {
    let terms =
        required_technical_terms("HTTP 400 と INVALID_CUSTOMER エラーコードを返してください。");

    assert!(terms.contains(&"HTTP 400".to_string()));
    assert!(terms.contains(&"INVALID_CUSTOMER".to_string()));
}

#[test]
fn detects_japanese_text_in_default_output() {
    assert!(contains_japanese_text(
        "検索ボタンを押したときだけ API を呼び出す。"
    ));
    assert!(!contains_japanese_text(
        "Only call the API after clicking search."
    ));
}

#[test]
fn extracts_constraint_clauses_for_the_model_prompt() {
    let clauses = required_constraint_clauses(
        "API は検索ボタンを押したときだけ呼び出してください。大規模なリファクタリングは避けてください。",
    );

    assert_eq!(
        clauses,
        [
            "API は検索ボタンを押したときだけ呼び出してください",
            "大規模なリファクタリングは避けてください",
        ]
    );

    let csv_clauses = required_constraint_clauses(
        "管理画面の CSV インポートで Shift_JIS と UTF-8 BOM を判定してください。既存の columns マッピング、dryRun オプション、エラー行番号の表示は維持してください。10MB を超えるファイルは読み込まず、INVALID_FILE_SIZE を返してください。",
    );
    assert!(csv_clauses
        .contains(&"10MB を超えるファイルは読み込まず、INVALID_FILE_SIZE を返してください"));

    let lazy_state_clauses = required_constraint_clauses(
        "useSearchParams と URL クエリ管理は消さないでください。ページ番号を変更しても検索条件と検索状態が消えないようにしてください。",
    );
    assert!(lazy_state_clauses
        .iter()
        .any(|clause| clause.contains("検索条件と検索状態が消えない")));
}

#[test]
fn restores_search_state_persistence_with_its_target() {
    let input = "React の検索画面で、検索ボタンを押したときだけ API を呼び出してください。既存の useSearchParams による URL クエリ管理は維持し、ページ番号を変更しても検索条件と検索状態が消えないようにしてください。TypeScript の既存構造を活かし、大規模なリファクタリングは避けてください。";
    let mut draft = CompressionDraft {
        distilled_prompt: "React検索画面、検索ボタン押下時のみAPI呼び出し、useSearchParamsでURLクエリ管理維持、TypeScript既存構造活用、大規模リファクタリング回避。"
            .to_string(),
        removed_content_summary: Vec::new(),
    };
    let request = test_request(input.to_string(), 2);

    assert!(!preserves_negative_constraints(
        input,
        &draft.distilled_prompt
    ));
    let phrases = missing_constraint_restoration_phrases(input, &draft.distilled_prompt);
    assert!(phrases
        .iter()
        .any(|phrase| phrase.contains("ページ変更時も検索条件/状態維持")));
    assert!(!phrases
        .iter()
        .any(|phrase| phrase.contains("useSearchParams で URL 管理")));

    restore_missing_required_constraints(&request, &mut draft);

    assert!(draft
        .distilled_prompt
        .contains("ページ変更時も検索条件/状態維持"));
    assert!(!draft.distilled_prompt.contains("; useSearchParams"));
    assert!(
        contains_required_technical_terms(input, &draft.distilled_prompt),
        "missing required terms: {:?}; output={}",
        required_technical_terms(input)
            .into_iter()
            .filter(|term| !contains_ascii_case_insensitive(&draft.distilled_prompt, term))
            .collect::<Vec<_>>(),
        draft.distilled_prompt
    );
    assert!(
        preserves_negative_constraints(input, &draft.distilled_prompt),
        "{}",
        draft.distilled_prompt
    );
    assert!(draft.distilled_prompt.chars().count() < input.chars().count());
}

#[test]
fn normalizes_required_term_typos_before_prefix_restoration() {
    let input = "React の検索画面で TypeScript の既存構造を活かしてください。";
    let mut draft = CompressionDraft {
        distilled_prompt: "React検索画面でTypeScrip既存構造活用。".to_string(),
        removed_content_summary: Vec::new(),
    };
    let request = test_request(input.to_string(), 2);

    restore_missing_required_terms(&request, &mut draft);

    assert_eq!(
        draft.distilled_prompt,
        "React検索画面でTypeScript既存構造活用。"
    );
    assert!(!draft.distilled_prompt.starts_with("TypeScript:"));
}

#[test]
fn extracts_lowercase_identifiers_connected_to_japanese_particles() {
    let terms = required_technical_terms(
        "新規検索時はpageを1に戻し、ページ移動時はkeywordとstatusを保持してください。",
    );

    assert!(terms.contains(&"page".to_string()));
    assert!(terms.contains(&"keyword".to_string()));
    assert!(terms.contains(&"status".to_string()));
}

#[test]
fn removes_lowercase_components_already_covered_by_longer_literals() {
    let terms = required_technical_terms(
        "POST /api/orders の JSON error.code を変更してください。orders と code は識別子の一部です。",
    );

    assert!(terms.contains(&"/api/orders".to_string()));
    assert!(terms.contains(&"error.code".to_string()));
    assert!(!terms.contains(&"orders".to_string()));
    assert!(!terms.contains(&"code".to_string()));
}

#[test]
fn classifies_conditional_outcomes_as_constraints() {
    let input = "customerIdが空ならHTTP 400とINVALID_CUSTOMERを返してください。requestIdがない場合はINVALID_REQUEST_IDを返してください。";
    let terms = required_technical_terms(input);

    let organized = organize_input_for_model(input, &terms);

    assert!(organized.contains("[制約:条件対応"), "{organized}");
    assert!(organized.contains("INVALID_CUSTOMER"));
    assert!(organized.contains("INVALID_REQUEST_ID"));
    assert_eq!(organized.matches("[制約:条件対応").count(), 2);
}

#[test]
fn does_not_corrupt_correct_required_term_during_typo_normalization() {
    let input = "React の TypeScript 構造は変更しないでください。";
    let mut draft = CompressionDraft {
        distilled_prompt: "ReactのTypeScript構造を維持。".to_string(),
        removed_content_summary: Vec::new(),
    };
    let request = test_request(input.to_string(), 2);

    restore_missing_required_terms(&request, &mut draft);

    assert_eq!(draft.distilled_prompt, "ReactのTypeScript構造を維持。");
}

#[test]
fn corrects_single_edit_typo_in_required_ascii_identifier() {
    let input = "AbortControllerを使って古い通信を中断してください。";
    let mut draft = CompressionDraft {
        distilled_prompt: "AbortConrollerで古い通信を中断。".to_string(),
        removed_content_summary: Vec::new(),
    };
    let request = test_request(input.to_string(), 2);

    restore_missing_required_terms(&request, &mut draft);

    assert_eq!(draft.distilled_prompt, "AbortControllerで古い通信を中断。");
}

#[test]
fn restores_missing_mechanism_term_next_to_its_related_target() {
    let input = "検索条件はuseSearchParamsでURLクエリへ保存してください。";
    let mut draft = CompressionDraft {
        distilled_prompt: "検索条件のURLクエリ保存。".to_string(),
        removed_content_summary: Vec::new(),
    };
    let request = test_request(input.to_string(), 2);

    restore_missing_required_terms(&request, &mut draft);

    assert_eq!(
        draft.distilled_prompt,
        "検索条件をuseSearchParamsでURLクエリ保存。"
    );
}

#[test]
fn restores_missing_literal_from_explicit_target_as_natural_context() {
    let input = "対象は POST /api/orders です。customerIdの入力検証を追加してください。";
    let mut draft = CompressionDraft {
        distilled_prompt: "customerIdの入力検証を追加。".to_string(),
        removed_content_summary: Vec::new(),
    };
    let request = test_request(input.to_string(), 2);

    restore_missing_required_terms(&request, &mut draft);

    assert!(
        draft
            .distilled_prompt
            .starts_with("POST /api/ordersを対象に、"),
        "{}",
        draft.distilled_prompt
    );
    assert!(!draft.distilled_prompt.starts_with("/api/orders:"));
}

#[test]
fn normalizes_singular_column_mapping_when_columns_is_required() {
    let input = "管理画面の CSV インポートで、既存の columns マッピング、dryRun オプション、エラー行番号の表示は維持してください。";
    let mut draft = CompressionDraft {
        distilled_prompt: "CSVインポートでcolumn mappings、dryRun、エラー行番号表示を維持。"
            .to_string(),
        removed_content_summary: Vec::new(),
    };
    let request = test_request(input.to_string(), 2);

    restore_missing_required_terms(&request, &mut draft);

    assert!(draft.distilled_prompt.contains("columns mapping"));
    assert!(!draft.distilled_prompt.starts_with("columns:"));
    assert!(
        contains_required_technical_terms(input, &draft.distilled_prompt),
        "missing required terms: {:?}; output={}",
        required_technical_terms(input)
            .into_iter()
            .filter(|term| !contains_ascii_case_insensitive(&draft.distilled_prompt, term))
            .collect::<Vec<_>>(),
        draft.distilled_prompt
    );
}

#[test]
fn removes_typo_prefix_when_required_term_exists_in_body() {
    let input = "TypeScritp の既存構造はなるべく触らず、React の検索画面を修正してください。";
    let mut draft = CompressionDraft {
        distilled_prompt: "TypeScritp: React検索画面を修正し、TypeScript既存構造は触らない。"
            .to_string(),
        removed_content_summary: Vec::new(),
    };
    let request = test_request(input.to_string(), 2);

    restore_missing_required_terms(&request, &mut draft);

    assert_eq!(
        draft.distilled_prompt,
        "React検索画面を修正し、TypeScript既存構造は触らない。"
    );
}

#[test]
fn restores_explicit_test_verification_requirements() {
    let input = "Next.js の POST /api/orders で、customerId が空のまま送られた時に 500 エラーになっています。入力検証を追加し、空の customerId の場合は HTTP 400 と INVALID_CUSTOMER のエラーコードを返すようにしてください。成功時のレスポンス形式、在庫引当処理、既存の監査ログは変更しないでください。テストでは正常系と customerId 空文字のケースを確認できるようにしてください。";
    let mut draft = CompressionDraft {
        distilled_prompt: "Next.js POST /api/orders: 空customerId時HTTP 400+INVALID_CUSTOMER返却。成功レスポンス/在庫引当/監査ログ変更しない。"
            .to_string(),
        removed_content_summary: Vec::new(),
    };
    let request = test_request(input.to_string(), 2);

    assert!(!preserves_negative_constraints(
        input,
        &draft.distilled_prompt
    ));

    restore_missing_required_constraints(&request, &mut draft);

    assert!(draft.distilled_prompt.contains("正常系"));
    assert!(draft.distilled_prompt.contains("customerId"));
    assert!(draft.distilled_prompt.contains("空文字"));
    assert!(["テスト", "確認", "検証"]
        .iter()
        .any(|marker| draft.distilled_prompt.contains(marker)));
    assert!(
        preserves_negative_constraints(input, &draft.distilled_prompt),
        "{}",
        draft.distilled_prompt
    );
    assert!(draft.distilled_prompt.chars().count() < input.chars().count());
}

#[test]
fn accepts_compact_search_state_persistence_phrase() {
    let input = "ページ番号を変更しても検索条件と検索状態が消えないようにしてください。";
    let output = "ページ変更時も検索条件/状態維持。";

    assert!(preserves_negative_constraints(input, output));
}

#[test]
fn accepts_named_state_persistence_without_repeating_generic_state_word() {
    let clause = "検索条件はuseSearchParamsでURLクエリへ保存し、戻る・進むでも復元できる状態を維持してください";
    let output = "useSearchParamsでURLクエリの検索条件を保存し、戻る・進むでも維持。";

    assert!(state_persistence_clause_satisfied(clause, output));
}

#[test]
fn rejects_generic_verification_when_enumerated_targets_are_missing() {
    let clause =
        "Vitestではボタン、Enter、入力中に呼ばれないこと、古いリクエストの中断を確認してください";

    assert!(!verification_constraint_satisfied(
        clause,
        "Vitestによる検証を実施"
    ));
    assert!(verification_constraint_satisfied(
        clause,
        "Vitestでボタン、Enter、入力中の非呼び出し、古いリクエストの中断を確認"
    ));
}

#[test]
fn restores_enumerated_verification_after_other_missing_constraints() {
    let input = "React と TypeScript で作っている管理画面の検索一覧を直してください。今回は検索ボタンを押した時、または検索欄で Enter を押した時だけ GET /api/customers を呼んでください。入力中は通信しないでください。検索条件は useSearchParams で URL クエリへ保存してください。既存コンポーネントの分割方法や CSS は変えず、画面全体の作り直しは避けてください。Vitest ではボタン、Enter、入力中に呼ばれないこと、古いリクエストの中断を確認してください。";
    let mut draft = CompressionDraft {
        distilled_prompt: "ReactとTypeScriptの検索一覧を修正し、検索ボタンまたはEnter時だけGET /api/customersを呼ぶ。入力中は通信禁止。URLクエリへ保存。CSS変更と作り直し禁止。Vitestで検証。".to_string(),
        removed_content_summary: Vec::new(),
    };
    let request = test_request(input.to_string(), 2);

    let phrases = missing_constraint_restoration_phrases(input, &draft.distilled_prompt);
    let verification_phrases =
        missing_verification_restoration_phrases(input, &draft.distilled_prompt);
    assert!(
        phrases
            .iter()
            .any(|phrase| phrase.contains("古いリクエスト")),
        "all={phrases:?}; verification={verification_phrases:?}"
    );
    restore_missing_required_constraints(&request, &mut draft);

    assert!(
        draft.distilled_prompt.contains("ボタン"),
        "{}",
        draft.distilled_prompt
    );
    assert!(
        draft.distilled_prompt.contains("Enter"),
        "{}",
        draft.distilled_prompt
    );
    assert!(
        draft.distilled_prompt.contains("入力中"),
        "{}",
        draft.distilled_prompt
    );
    assert!(
        draft.distilled_prompt.contains("古いリクエスト"),
        "{}",
        draft.distilled_prompt
    );
}

#[test]
fn generic_fallback_handles_unregistered_domain_and_paraphrased_constraints() {
    let input = "前にもお願いした内容ですが、参考までに共有します。Zig の build.zig を更新してください。release-fast だけを対象にし、既存の install 手順は変更しないでください。失敗した場合は HTTP 503 を返し、BUILD_FAILED をログへ残してください。最後に zig build test で成功系と失敗系を確認してください。";
    let request = test_request(input.to_string(), 2);

    let candidate = verified_structured_candidate(input).expect("generic candidate");
    let draft = trusted_precompacted_fallback_draft(&request).expect("generic fallback");

    assert_eq!(draft.distilled_prompt, candidate);
    assert!(!candidate.contains("前にもお願い"), "{candidate}");
    for expected in [
        "build.zig",
        "release-fast だけ",
        "install 手順は変更しない",
        "HTTP 503",
        "BUILD_FAILED",
        "zig build test",
    ] {
        assert!(
            candidate.contains(expected),
            "missing {expected}: {candidate}"
        );
    }
    assert!(structured_candidate_preserves_requirements(
        input, &candidate
    ));
}

#[test]
fn generic_candidate_repairs_unregistered_model_output_without_fixture_code() {
    let input = "前にもお願いした内容ですが、参考までに共有します。Zig の build.zig を更新してください。release-fast だけを対象にし、既存の install 手順は変更しないでください。失敗した場合は HTTP 503 を返し、BUILD_FAILED をログへ残してください。最後に zig build test で成功系と失敗系を確認してください。";
    let request = test_request(input.to_string(), 2);
    let mut draft = CompressionDraft {
        distilled_prompt: "build.zigを更新。".to_string(),
        removed_content_summary: Vec::new(),
    };

    polish_model_output_for_request(&request, &mut draft);

    assert!(structured_candidate_preserves_requirements(
        input,
        &draft.distilled_prompt
    ));
    assert!(draft.distilled_prompt.contains("BUILD_FAILED"));
    assert!(draft.distilled_prompt.chars().count() < input.chars().count());
}

#[test]
fn accepts_saved_and_restored_state_without_repeating_state_label() {
    let clause = "検索条件はuseSearchParamsでURLクエリへ保存し、戻る・進むでも復元できる状態を維持してください";
    let output = "useSearchParamsでURLクエリに保存し、戻る・進むで復元。";

    assert!(state_persistence_clause_satisfied(clause, output));
}

#[test]
fn restores_long_log_constraint_by_trimming_model_output() {
    let input = "本番ログを解析し、注文送信が失敗する原因候補を優先度順に整理してください。2026-06-24T10:15:03Z requestId=ab12 POST /orders ECONNRESET upstream=payment-service。時刻、requestId、エラー文字列は改変せず、追加で確認すべきログと暫定対応を示してください。";
    let mut draft = CompressionDraft {
        distilled_prompt: "/orders/upstream=payment-service: 2026/06/24T10/15/03Z/POST/ECONNRESET/2026-06-24T10:15:03Z/requestId=ab12/追加確認:payment-serviceログとネットワーク障害ログ。暫定対応:再試行設定と接続タイムアウト調整".to_string(),
        removed_content_summary: Vec::new(),
    };
    let request = test_request(input.to_string(), 1);

    restore_missing_required_constraints(&request, &mut draft);

    assert!(draft.distilled_prompt.contains("改変せず"));
    assert!(contains_required_technical_terms(
        input,
        &draft.distilled_prompt
    ));
    assert!(draft.distilled_prompt.chars().count() < input.chars().count());
}

#[test]
fn preprocesses_input_whitespace_before_llm_prompt() {
    let input = "React　の検索画面で、  検索ボタンを押したときだけ  API を呼び出してください。\r\n\r\n\r\nTypeScript の既存構造を活かしてください。";

    let preprocessed = preprocess_input_for_llm(input);

    assert!(!preprocessed.contains('\r'));
    assert!(!preprocessed.contains("  "));
    assert!(!preprocessed.contains("\n\n\n"));
    assert!(preprocessed.contains("検索ボタンを押したときだけ"));
    assert!(preprocessed.contains("API 呼出"));
    assert!(preprocessed.contains("TypeScript"));
}

#[test]
fn preprocess_keeps_negative_polite_constraints() {
    let input = "既存の監査ログは変更しないでください。UI 内の完了トーストは追加しないでください。結果をコピーしてください。";

    let preprocessed = preprocess_input_for_llm(input);

    assert!(preprocessed.contains("変更しないでください"));
    assert!(preprocessed.contains("追加しないでください"));
    assert!(preprocessed.contains("結果コピー"));
    assert!(!preprocessed.contains("コピーしてください"));
}

#[test]
fn preprocess_removes_obvious_noise_but_keeps_lazy_request_details() {
    let input = "こんにちｈ。今日はｄあさ、これは関係ないです。検索のところが重いので直してください。React の検索画面で入力中に API が何度も呼ばれているので、検索ボタンを押した時だけ API を呼ぶようにしたいです。useSearchParams と URL クエリ管理は消さないでください。TypeScritp の既存構造は大きく変えないでください。";

    let preprocessed = preprocess_input_for_llm(input);

    assert!(!preprocessed.contains("こんにちｈ"));
    assert!(!preprocessed.contains("今日はｄあさ"));
    assert!(preprocessed.contains("React"));
    assert!(preprocessed.contains("API"));
    assert!(preprocessed.contains("useSearchParams"));
    assert!(preprocessed.contains("URL"));
    assert!(preprocessed.contains("TypeScript"));
    assert!(preprocessed.contains("消さない"));
    assert!(preprocessed.contains("変えない"));
}

#[test]
fn preprocess_normalizes_only_high_confidence_typos_for_llm() {
    let input = "TypeScritp の処理で PawerShell 通知ではなく AppUserModelID を使い、DataLoder と UTF8 BOM と HTTP400 の表記も確認してください。--dryrun は --dry-run として扱ってください。";

    let preprocessed = preprocess_input_for_llm(input);

    assert!(preprocessed.contains("TypeScript"));
    assert!(preprocessed.contains("PowerShell"));
    assert!(preprocessed.contains("DataLoader"));
    assert!(preprocessed.contains("UTF-8 BOM"));
    assert!(preprocessed.contains("HTTP 400"));
    assert!(preprocessed.contains("--dry-run"));
    assert!(!preprocessed.contains("TypeScritp"));
    assert!(!preprocessed.contains("PawerShell"));
    assert!(!preprocessed.contains("DataLoder"));
}

#[test]
fn typo_normalization_preserves_literals_and_identifier_substrings() {
    let input = "ログには `Recat` と「今日はｄあさ」をそのまま残し、説明文の Recat は直してください。soketManager は既存識別子で、soket は誤字です。";

    let normalized = normalize_known_input_typos_for_llm(input);

    assert!(normalized.contains("`Recat`"), "{normalized}");
    assert!(normalized.contains("「今日はｄあさ」"), "{normalized}");
    assert!(normalized.contains("説明文の React"), "{normalized}");
    assert!(normalized.contains("soketManager"), "{normalized}");
    assert!(normalized.contains("socket は誤字"), "{normalized}");
}

#[test]
fn preprocessing_preserves_literal_spacing_and_polite_text() {
    let input = "表示例は `a  b` と「を修正してください」。実装  を確認してください。";

    let normalized = normalize_input_whitespace(input);
    let cleaned = remove_polite_request_fillers(&normalized);

    assert!(normalized.contains("`a  b`"), "{normalized}");
    assert!(cleaned.contains("「を修正してください」"), "{cleaned}");
    assert!(cleaned.contains("実装 確認"), "{cleaned}");
}

#[test]
fn preprocesses_self_corrected_order_request_without_unrelated_noise() {
    let input = "Nex.js、Nextjs かな、正式には Next.js の POST /api/odrers、すみません POST /api/orders を直してください。custmerID、項目名は customerId です。HTTP400、表記は HTTP 400。reqestId、正しくは requestId。[]{}っっっっ これは貼り付けの残りで意味はありません。";

    let preprocessed = preprocess_input_for_llm(input);

    for expected in [
        "Next.js",
        "POST /api/orders",
        "customerId",
        "HTTP 400",
        "requestId",
    ] {
        assert!(
            preprocessed.contains(expected),
            "missing {expected}: {preprocessed}"
        );
    }
    for removed in [
        "Nex.js",
        "Nextjs",
        "/api/odrers",
        "custmerID",
        "reqestId",
        "貼り付けの残り",
    ] {
        assert!(
            !preprocessed.contains(removed),
            "kept {removed}: {preprocessed}"
        );
    }
}

#[test]
fn preprocesses_self_corrected_settings_request_without_conversion_noise() {
    let input = "デスクトップアプリのせってい保存。圧縮れべるとウインドウサイズを保持。user-setings.json と書きかけましたが user-settings.json が正しい名前です。aplication/local/stete じゃなくて application/local/state です。123あいう ここは意味ないので消してよいです。";

    let preprocessed = preprocess_input_for_llm(input);

    for expected in [
        "設定",
        "圧縮レベル",
        "ウィンドウサイズ",
        "user-settings.json",
        "application/local/state",
    ] {
        assert!(
            preprocessed.contains(expected),
            "missing {expected}: {preprocessed}"
        );
    }
    for removed in [
        "せってい",
        "圧縮れべる",
        "ウインドウ",
        "user-setings.json",
        "aplication/local/stete",
        "123あいう",
    ] {
        assert!(
            !preprocessed.contains(removed),
            "kept {removed}: {preprocessed}"
        );
    }
}

#[test]
fn preprocesses_self_corrected_websocket_request_without_unrelated_story() {
    let input = "WebSoket、正しくは WebSocket。指数ばっくおふで1秒、2病、いや2秒、4秒、最大30秒。プラマイ20%、±20% jitter。soket、socketを2本作らない。message hander は message handler です。10会、10回失敗。fake timre、fake timerで確認。zzzz 今日はカレーでした 987654 これはタッチミスと関係ない話なので依頼には含めないでください。";

    let preprocessed = preprocess_input_for_llm(input);

    for expected in [
        "WebSocket",
        "指数バックオフ",
        "2秒",
        "±20%",
        "socket",
        "message handler",
        "10回",
        "fake timer",
    ] {
        assert!(
            preprocessed.contains(expected),
            "missing {expected}: {preprocessed}"
        );
    }
    for removed in [
        "WebSoket",
        "指数ばっくおふ",
        "2病",
        "プラマイ20%",
        "soket",
        "message hander",
        "10会",
        "fake timre",
        "今日はカレー",
    ] {
        assert!(
            !preprocessed.contains(removed),
            "kept {removed}: {preprocessed}"
        );
    }
}

#[test]
fn preprocesses_explicit_corrections_and_declared_noise_generically() {
    let input = "入力は最大25MB、いや最終的に20MBです。変な入力xyzxyzは無視で。timeoutは最初5000msと書きましたが正しくは3000msです。意味なし文字%%abc123。外部formatterを呼ぶ部分を直してほしです。";

    let preprocessed = preprocess_input_for_llm(input);

    assert!(preprocessed.contains("20MB"));
    assert!(preprocessed.contains("3000ms"));
    assert!(preprocessed.contains("formatterを呼ぶ部分を修正"));
    assert!(!preprocessed.contains("最初3000ms"), "{preprocessed}");
    for removed in ["25MB", "5000ms", "xyzxyz", "abc123", "意味なし文字"] {
        assert!(
            !preprocessed.contains(removed),
            "kept {removed}: {preprocessed}"
        );
    }
}

#[test]
fn preprocesses_discourse_corrections_confirmations_and_declared_gibberish() {
    let nix_input = "えっとNixのflake.nixなんですけど devShellをmacOSとLinuxでそろえたいです。nodejs_22とpnpmを両方入れて、たぶんPythonも、いやPythonは今回は不要です。asdfgh 998877は関係ない文字です。既存のnix flake checkは変更しないで、nix develop後にpnpm testが通ることを確認してください。";
    let protobuf_input = "gRPCのuser.proto更新をお願いしたいですです。display_nameをfield 7で追加、あっ7で合ってます。以前消したfield 4はreservedのまま再利用しないでください。文字列 qqq!! は意味ありません。既存クライアントが未知fieldを受けても壊れないことと、buf breakingで後方互換性を確認してください。";

    let nix = preprocess_input_for_llm(nix_input);
    for removed in ["えっと", "たぶんPythonも", "asdfgh", "998877"] {
        assert!(!nix.contains(removed), "kept {removed}: {nix}");
    }
    assert!(nix.contains("Pythonは今回は不要"), "{nix}");

    let protobuf = preprocess_input_for_llm(protobuf_input);
    for removed in ["ですです", "あっ7で合ってます", "qqq", "意味ありません"] {
        assert!(!protobuf.contains(removed), "kept {removed}: {protobuf}");
    }
    for required in [
        "display_name",
        "field 7",
        "field 4",
        "reserved",
        "buf breaking",
    ] {
        assert!(
            protobuf.contains(required),
            "missing {required}: {protobuf}"
        );
    }
}

#[test]
fn compacts_noisy_source_derived_candidates_below_level_two_budget() {
    for input in [
        "えっとNixのflake.nixなんですけど devShellをmacOSとLinuxでそろえたいです。nodejs_22とpnpmを両方入れて、たぶんPythonも、いやPythonは今回は不要です。asdfgh 998877は関係ない文字です。既存のnix flake checkは変更しないで、nix develop後にpnpm testが通ることを確認してください。",
        "gRPCのuser.proto更新をお願いしたいですです。display_nameをfield 7で追加、あっ7で合ってます。以前消したfield 4はreservedのまま再利用しないでください。文字列 qqq!! は意味ありません。既存クライアントが未知fieldを受けても壊れないことと、buf breakingで後方互換性を確認してください。",
    ] {
        let normalized = super::normalized_verification_input(input);
        let candidate = verified_structured_candidate(&normalized).expect("verified candidate");

        assert!(
            candidate.chars().count() * 100 <= input.chars().count() * 85,
            "candidate too long: {candidate}"
        );
    }
}

#[test]
fn extracts_compound_names_routes_commands_and_generic_types() {
    let input = "ASP.NET CoreとFeature Managementを使い、POST /ordersは維持する。Web Componentsからshadow DOMを操作し、cmake --build --preset arm64-linuxを実行する。戻り値Promise<Uint8Array>は変更しない。";

    let terms = required_technical_terms(input);

    for expected in [
        "ASP.NET Core",
        "Feature Management",
        "POST /orders",
        "Web Components",
        "shadow DOM",
        "cmake --build --preset arm64-linux",
        "Promise<Uint8Array>",
    ] {
        assert!(
            terms.iter().any(|term| term == expected),
            "missing {expected}: {terms:?}"
        );
    }
}

#[test]
fn required_ascii_words_do_not_match_longer_identifiers() {
    assert!(!contains_ascii_case_insensitive(
        "Escapingで閉じる",
        "Escape"
    ));
    assert!(contains_ascii_case_insensitive("Escapeで閉じる", "Escape"));
}

#[test]
fn rejects_required_items_moved_into_an_exclusion_group() {
    let input = "tool_aとtool_bを両方入れて、tool_cは不要です。";
    let invalid = "tool_aを追加し、tool_bとtool_cは不要。";
    let valid = "tool_aとtool_bを追加し、tool_cは不要。";

    assert!(!super::preserves_constraint_clause_roles(input, invalid));
    assert!(super::preserves_constraint_clause_roles(input, valid));
}

#[test]
fn rejects_distinct_verification_cases_merged_into_one_item() {
    let input = "timeoutは3000msです。既存の:Formatコマンド名は維持し、plenary.nvimで成功、timeout、非0 exit codeをテストしてください。";
    let invalid = "timeoutは3000ms。既存の:Formatコマンド名を維持し、plenary.nvimで成功、timeout時の非0 exit codeをテスト。";
    let valid = "timeoutは3000ms。既存の:Formatコマンド名を維持し、plenary.nvimで成功、timeout、非0 exit codeをテスト。";

    assert!(!super::preserves_constraint_clause_roles(input, invalid));
    assert!(super::preserves_constraint_clause_roles(input, valid));
}

#[test]
fn builds_verified_candidate_for_compound_level_two_constraints() {
    let input = "前にも少し相談した件ですが、Zigのbuild.zigでrelease-fastとrelease-safeの2構成を扱えるように整理してください。release-fastだけはLTOを有効にし、release-safeでは安全性チェックを無効にしないでください。既存のzig build testとinstall手順は変更せず、LinuxとWindowsの両方で生成物名が一致することを確認してください。背景説明は長くなりましたが、必要なのはこの変更だけです。";
    let normalized = super::normalized_verification_input(input);
    let terms = super::required_technical_terms(&normalized);
    let candidate = super::PromptStructure::analyze(&normalized, &terms)
        .compact_candidate()
        .expect("structured candidate");

    assert!(super::preserves_constraint_clause_roles(
        &normalized,
        &candidate
    ));
    assert!(
        super::trusted_precompacted_fallback_draft(&test_request(input.to_string(), 2)).is_some()
    );
}

#[test]
fn preprocess_does_not_drop_noise_sentence_with_protected_content() {
    let input = "これは関係ないかもしれませんが、API は検索ボタンを押した時だけ呼んでください。今日はｄあさ。URL と useSearchParams は維持してください。";

    let preprocessed = preprocess_input_for_llm(input);

    assert!(preprocessed.contains("API"));
    assert!(preprocessed.contains("検索ボタンを押した時だけ"));
    assert!(preprocessed.contains("URL"));
    assert!(preprocessed.contains("useSearchParams"));
    assert!(preprocessed.contains("維持"));
    assert!(!preprocessed.contains("今日はｄあさ"));
}

#[test]
fn noise_filter_preserves_quoted_markers_and_actionable_sentences() {
    let input = "API で「これは意味ないです」を表示し、この文言は変更しないでください。これは意味ないと言われていますが、timeout を修正してください。";

    let cleaned = remove_obvious_input_noise(input);

    assert!(cleaned.contains("「これは意味ないです」"), "{cleaned}");
    assert!(cleaned.contains("文言は変更しない"), "{cleaned}");
    assert!(cleaned.contains("timeout を修正"), "{cleaned}");
}

#[test]
fn noise_filter_still_removes_explicitly_discarded_segments() {
    let input = "API を更新してください。zzzz 123 これは意味ないので依頼には含めないでください。";

    let cleaned = remove_obvious_input_noise(input);

    assert!(cleaned.contains("API を更新"), "{cleaned}");
    assert!(!cleaned.contains("zzzz"), "{cleaned}");
    assert!(!cleaned.contains("123"), "{cleaned}");
}

#[test]
fn organizes_each_prompt_clause_once_by_semantic_role() {
    let input = "今の実装では入力中にもAPIが呼ばれて重いです。検索画面の通信を直したいです。検索ボタンを押した時だけAPIを呼んでください。URL状態は維持してください。既存CSSの作り直しは避けてください。テストで入力中に呼ばれないことを確認してください。";

    let organized = organize_input_for_model(input, &[]);

    assert!(!organized.contains("今の実装では"), "{organized}");
    assert!(organized.contains("[要求] 検索画面の通信を直したいです"));
    assert!(organized.contains("[制約:限定] 検索ボタンを押した時だけAPIを呼んでください"));
    assert!(organized.contains("[制約:維持] URL状態は維持してください"));
    assert!(organized.contains("[制約:禁止] 既存CSSの作り直しは避けてください"));
    assert!(organized.contains("[検証] テストで入力中に呼ばれないことを確認してください"));
    assert_eq!(organized.matches("今の実装では").count(), 0);
    assert_eq!(organized.matches("検索ボタンを押した時だけ").count(), 1);
}

#[test]
fn structures_shared_predicate_and_conditional_value_lists() {
    let conditional = structured_constraint_clause(
        "customerIdが未指定、null、空文字、空白だけのいずれかならHTTP 400を返してください",
    )
    .expect("conditional list");
    let shared_predicate = structured_constraint_clause(
        "成功時のレスポンス形式、orderIdの採番、在庫引当、監査ログ形式は変更しないでください",
    )
    .expect("shared predicate list");

    assert!(
        conditional.contains("customerId=未指定/null/空文字/空白のいずれかなら"),
        "{conditional}"
    );
    assert!(
        shared_predicate.contains("成功レスポンス形式/orderId採番/在庫引当/監査ログ形式変更しない"),
        "{shared_predicate}"
    );
}

#[test]
fn restores_missing_items_from_generalized_constraint_lists() {
    let input = "最近入力漏れによる失敗が増えており、利用者への案内も分かりづらいため、既存実装を確認しながら入力検証を整理したいです。以前にも同じ相談があり、今回は必要な条件と変更範囲を明確にした上で安全に対応する予定です。関係者との確認内容やこれまでの経緯も多いですが、実装者が判断できる情報を中心にまとめます。customerIdが未指定、null、空文字、空白だけのいずれかならHTTP 400を返してください。成功時のレスポンス形式、orderIdの採番、在庫引当、決済予約、監査ログ形式は変更しないでください。";
    let mut draft = CompressionDraft {
        distilled_prompt: "customerId未指定/nullならHTTP 400。orderId/監査ログ変更しない。"
            .to_string(),
        removed_content_summary: Vec::new(),
    };
    let request = test_request(input.to_string(), 2);

    let phrases = missing_constraint_restoration_phrases(input, &draft.distilled_prompt);
    assert!(
        phrases.iter().any(|phrase| phrase.contains("空文字")),
        "{phrases:?}"
    );
    assert!(
        phrases.iter().any(|phrase| phrase.contains("決済予約")),
        "{phrases:?}"
    );
    restore_missing_required_constraints(&request, &mut draft);

    for expected in ["空文字", "空白", "成功レスポンス", "在庫引当", "決済予約"]
    {
        assert!(
            draft.distilled_prompt.contains(expected),
            "missing {expected}: {}",
            draft.distilled_prompt
        );
    }
}

#[test]
fn restores_only_missing_target_from_two_item_prohibition() {
    let input = "今回の目的は入力検証なので、注文処理全体のリファクタリングやDBスキーマ変更までは行わないでください。";
    let output = "DBスキーマ変更は行わない。";

    let phrases = missing_constraint_restoration_phrases(input, output);

    assert!(
        phrases
            .iter()
            .any(|phrase| phrase.contains("リファクタリング行わない")),
        "{phrases:?}"
    );
    assert!(
        phrases
            .iter()
            .all(|phrase| !phrase.contains("DBスキーマ変更/")),
        "{phrases:?}"
    );
}

#[test]
fn retains_current_state_when_it_has_unique_required_evidence() {
    let input = "現在POST /api/ordersがHTTP 500になります。入力検証を追加してください。";
    let terms = required_technical_terms(input);

    let organized = organize_input_for_model(input, &terms);

    assert!(organized.contains("[現状"), "{organized}");
    assert!(organized.contains("HTTP 500"), "{organized}");
    assert!(organized.contains("[要求] 入力検証を追加してください"));
}

#[test]
fn does_not_treat_current_implementation_noun_as_requested_action() {
    let input = "今の実装では入力中にもAPIが呼ばれるので画面が重いです。検索ボタン押下時だけAPIを呼んでください。";
    let terms = required_technical_terms(input);

    let organized = organize_input_for_model(input, &terms);

    assert!(!organized.contains("[現状→要求]"), "{organized}");
    assert!(!organized.contains("今の実装では"), "{organized}");
    assert!(organized.contains("[制約:限定"), "{organized}");
}

#[test]
fn classifies_literal_only_context_as_target() {
    let input = "対象はPOST /api/ordersです。入力検証を追加してください。";
    let terms = required_technical_terms(input);

    let organized = organize_input_for_model(input, &terms);

    assert!(organized.contains("[対象|必須語:"), "{organized}");
    assert!(organized.contains("POST /api/orders"));
}

#[test]
fn classifies_framework_verification_without_test_word_as_verification() {
    let input = "Vitestでは入力中にAPIが呼ばれないことと中断を確認してください。";
    let terms = required_technical_terms(input);

    let organized = organize_input_for_model(input, &terms);

    assert!(
        organized.contains("[検証|必須語:Vitest,API]"),
        "{organized}"
    );
}

#[test]
fn classifies_tool_usage_as_request_action() {
    let input = "AbortControllerを使って古い通信を中断してください。";
    let terms = required_technical_terms(input);

    let organized = organize_input_for_model(input, &terms);

    assert!(
        organized.contains("[要求|必須語:AbortController]"),
        "{organized}"
    );
}

#[test]
fn anchors_required_terms_to_their_source_clause() {
    let input = "React検索画面を修正。URL状態はuseSearchParamsで維持してください。";
    let terms = required_technical_terms(input);

    let organized = organize_input_for_model(input, &terms);

    assert!(organized.contains("[要求|必須語:React]"), "{organized}");
    assert!(
        organized.contains("必須語:URL,useSearchParams"),
        "{organized}"
    );
}

#[test]
fn separates_only_independent_constraint_fragments() {
    let input = "既存CSSは変えず、画面全体の作り直しは避けてください。";

    let organized = organize_input_for_model(input, &[]);

    assert!(organized.contains("[制約:禁止] 既存CSSは変えず"));
    assert!(organized.contains("[制約:禁止] 画面全体の作り直しは避けてください"));
    assert_eq!(organized.matches("[制約:禁止]").count(), 2);
}

#[test]
fn normalizes_level_two_csv_read_skip_constraint() {
    let input = "管理画面の CSV インポートで Shift_JIS と UTF-8 BOM を判定し、文字化けを防いでください。既存の columns マッピング、dryRun オプション、エラー行番号の表示は維持してください。10MB を超えるファイルは読み込まず、INVALID_FILE_SIZE を返してください。";
    let mut draft = CompressionDraft {
        distilled_prompt: "CSVインポート: Shift_JIS/UTF-8 BOM判定、columns mapping/dryRun/エラー行番号表示維持。10MB を超ファイル読み込み拒否、INVALID_FILE_SIZE返却。".to_string(),
        removed_content_summary: Vec::new(),
    };
    let request = test_request(input.to_string(), 2);

    restore_missing_required_constraints(&request, &mut draft);

    assert!(draft.distilled_prompt.contains("読み込まず"));
    assert!(!draft.distilled_prompt.contains("読み込み拒否"));
    assert!(contains_required_technical_terms(
        input,
        &draft.distilled_prompt
    ));
    assert!(
        preserves_negative_constraints(input, &draft.distilled_prompt),
        "{}",
        draft.distilled_prompt
    );
    assert!(draft.distilled_prompt.chars().count() < input.chars().count());
}

#[test]
fn removes_level_two_polite_request_fillers() {
    let input = "Prisma の User テーブルに lastLoginAt を追加する migration を作成してください。既存データは NULL のまま許容し、email の unique 制約や createdAt の default は変更しないでください。ロールバック手順も短く添えてください。";
    let mut draft = CompressionDraft {
        distilled_prompt: "PrismaのUserテーブルにlastLoginAtを追加し、既存データをNULL許容で維持、emailのunique制約とcreatedAtのdefaultを変更しない、ロールバック手順も添えたmigrationを作成してください。".to_string(),
        removed_content_summary: Vec::new(),
    };
    let request = test_request(input.to_string(), 2);

    polish_model_output_for_request(&request, &mut draft);

    assert!(!draft.distilled_prompt.contains("してください"));
    assert!(
        contains_required_technical_terms(input, &draft.distilled_prompt),
        "missing required terms: {:?}; output={}",
        required_technical_terms(input)
            .into_iter()
            .filter(|term| !contains_ascii_case_insensitive(&draft.distilled_prompt, term))
            .collect::<Vec<_>>(),
        draft.distilled_prompt
    );
    assert!(preserves_negative_constraints(
        input,
        &draft.distilled_prompt
    ));
    assert!(
        draft.distilled_prompt.chars().count() * 100 <= input.chars().count() * 78,
        "{}",
        draft.distilled_prompt
    );
}

#[test]
fn restores_missing_http_status_phrase() {
    let input = "Next.js の POST /api/orders が空の customerId で 500 を返します。入力検証を追加し、HTTP 400 と既存の INVALID_CUSTOMER エラーコードを返してください。成功時のレスポンス形式、在庫引当処理、既存の監査ログは変更しないでください。";
    let mut draft = CompressionDraft {
        distilled_prompt: "Next.js POST /api/orders に customerId 空時、HTTP 500 回避。400 と INVALID_CUSTOMER エラー追加。成功/在庫引当/監査ログ変更なし。".to_string(),
        removed_content_summary: Vec::new(),
    };
    let request = test_request(input.to_string(), 2);

    restore_missing_required_terms(&request, &mut draft);

    assert!(draft.distilled_prompt.contains("HTTP 400"));
    assert!(contains_required_technical_terms(
        input,
        &draft.distilled_prompt
    ));
}

#[test]
fn removes_duplicate_assignment_value_after_key_value_term() {
    let input =
        "2026-06-24T10:15:03Z requestId=ab12 POST /orders ECONNRESET upstream=payment-service。";
    let mut draft = CompressionDraft {
        distilled_prompt:
            "2026-06-24T10:15:03Z/requestId=ab12/payment-service: ab12, ECONNRESET原因整理。"
                .to_string(),
        removed_content_summary: Vec::new(),
    };
    let request = test_request(input.to_string(), 3);

    polish_model_output_for_request(&request, &mut draft);

    assert!(draft.distilled_prompt.contains("requestId=ab12"));
    assert!(!draft.distilled_prompt.contains(": ab12,"));
    assert!(contains_required_technical_terms(
        input,
        &draft.distilled_prompt
    ));
}

#[test]
fn normalizes_privacy_exclusion_to_negative_constraint_marker() {
    let input = "エラー本文に受け取った customerId や個人情報を丸ごと入れないでください。";
    let mut draft = CompressionDraft {
        distilled_prompt: "customerId検証、個人情報エラー本文除外。".to_string(),
        removed_content_summary: Vec::new(),
    };
    let request = test_request(input.to_string(), 2);

    restore_missing_required_constraints(&request, &mut draft);

    assert!(
        draft.distilled_prompt.contains("個人情報を含めない"),
        "{}",
        draft.distilled_prompt
    );
    assert!(preserves_negative_constraints(
        input,
        &draft.distilled_prompt
    ));
}

#[test]
fn fallback_uses_verified_long_graphql_dataloader_candidate() {
    let input = "GraphQL の users クエリで各ユーザーの posts と comments を表示すると N+1 が発生しており、一覧100件で数百回 SQL が実行されています。DataLoader を導入して userId ごとの posts と postId ごとの comments をそれぞれバッチ取得してください。ただしキャッシュは HTTP リクエスト単位だけに作り、別リクエスト、別テナント、別ユーザーの結果を共有しないでください。権限チェックより前にキャッシュしたデータを返すことも避け、resolver ごとの認可処理は維持してください。既存の schema.graphql、users の引数、posts と comments のフィールド名、ページネーション形式は変更しないでください。削除済み post は今までどおり除外し、取得順も入力されたキーの順番と対応させてください。部分的に DB エラーが起きた場合は全ユーザーを同じエラーにせず、該当キーだけへエラーを関連付けてください。テストでは SQL 発行回数、テナント分離、権限不足、キー順、空配列を確認してください。";
    let request = test_request(input.to_string(), 2);

    let draft =
        trusted_precompacted_fallback_draft(&request).expect("verified graphql fallback candidate");

    assert!(contains_japanese_text(&draft.distilled_prompt));
    assert!(draft.distilled_prompt.chars().count() < input.chars().count());
    assert!(contains_required_technical_terms(
        input,
        &draft.distilled_prompt
    ));
    assert!(preserves_negative_constraints(
        input,
        &draft.distilled_prompt
    ));
    assert!(
        !SimpleVerifier
            .verify(&request, &draft.distilled_prompt)
            .should_send_original
    );
    assert!(draft.distilled_prompt.contains("100件"));
    assert!(draft.distilled_prompt.contains("該当キーだけ"));
    assert!(draft.distilled_prompt.contains("空配列"));
}

#[test]
fn timeout_fallback_uses_verified_inventory_sync_lock_candidate() {
    let input = "毎日午前2時に動く在庫同期ジョブが、前日の処理が長引いた日に二重起動して在庫数を上書きすることがあります。scheduler の設定だけで避けるのではなく、ジョブ開始時に PostgreSQL advisory lock を取得し、同じ warehouseId の処理がすでに動いていれば新しい実行はスキップしてください。別 warehouseId は並行実行して構いません。ロックを取得できなかった場合はエラー終了ではなく skipped として記録し、監視アラートを発報しないでください。外部在庫 API は1ページ100件で、429 の場合だけ Retry-After に従い最大3回再試行し、400 や 401 は再試行しないでください。途中で失敗した場合は最後に完了した cursor を保存し、次回はそこから再開してください。ただし同期対象日の異なる cursor を誤って再利用しないでください。ジョブ終了時は成功・失敗にかかわらずロックを解放し、warehouseId、対象日、処理件数、再試行回数、最終 cursor を構造化ログへ残してください。商品名や仕入価格はログへ出さないでください。二重起動、別倉庫、429、401、中断再開を統合テストで確認してください。";
    let request = test_request(input.to_string(), 2);

    let draft = trusted_precompacted_fallback_draft(&request).expect("verified timeout fallback");

    assert!(draft.distilled_prompt.chars().count() < input.chars().count());
    assert!(contains_required_technical_terms(
        input,
        &draft.distilled_prompt
    ));
    assert!(preserves_negative_constraints(
        input,
        &draft.distilled_prompt
    ));
    assert!(draft
        .distilled_prompt
        .contains("成功・失敗にかかわらずロックを解放"));
    assert!(draft
        .distilled_prompt
        .contains("商品名や仕入価格はログへ出さない"));
}

#[test]
fn rejects_websocket_candidate_that_drops_an_unrelated_added_constraint() {
    let input = "WebSocket 切断時に指数バックオフで再接続するようにしてください。初回は 1 秒、最大 30 秒まで伸ばし、手動ログアウト後は再接続しないでください。既存の message handler と認証トークン更新処理は変更しないでください。ネットワーク復帰後に重複接続が増えないことも確認してください。既存画面の表示文言は変えないでください。";
    let request = test_request(input.to_string(), 2);
    let draft = CompressionDraft {
        distilled_prompt: "WebSocket再接続: 1秒/最大30秒指数バックオフ。手動ログアウト後再接続しない。message handler/認証トークン更新処理は変更しない。ネットワーク復帰後も重複接続しないことを確認。".to_string(),
        removed_content_summary: Vec::new(),
    };

    assert!(
        validate_compression_draft(&request, &draft).is_err(),
        "an unrelated added constraint must not be covered by the WebSocket-specific validator"
    );
}

#[test]
fn matches_single_target_change_constraints_to_their_output_clause() {
    let input = "ログ出力形式は変えないでください。APIレスポンス形式も変更しないでください。";

    assert!(preserves_targeted_change_constraints(
        input,
        "ログ出力形式変更なし。APIレスポンス形式維持。"
    ));
    assert!(!preserves_targeted_change_constraints(
        input,
        "APIレスポンス形式変更なし。画面レイアウト維持。"
    ));
}

#[test]
fn compacts_i18n_eval_case_below_case_budget() {
    let input = "i18n の ja.json と en.json で不足しているキーを検出するスクリプトを追加してください。キーの並び順は既存ファイルに合わせ、翻訳文の自動生成はしないでください。不足キー一覧を CI のログに出し、差分がある場合は exit code 1 で失敗するようにしてください。既存の翻訳文は変更しないでください。";
    let mut draft = CompressionDraft {
        distilled_prompt: "i18n不足キーを検出する。".to_string(),
        removed_content_summary: Vec::new(),
    };

    polish_model_output_for_request(&test_request(input.to_string(), 2), &mut draft);

    for marker in [
        "ja.json",
        "en.json",
        "CI",
        "exit code 1",
        "自動生成",
        "変更しない",
    ] {
        assert!(
            draft.distilled_prompt.contains(marker),
            "missing {marker}: {}",
            draft.distilled_prompt
        );
    }
    assert!(draft.distilled_prompt.chars().count() * 100 <= input.chars().count() * 92);
}

#[test]
fn keeps_level_two_order_api_restoration_below_case_budget() {
    let input = "Next.js の POST /api/orders で、customerId が空のまま送られた時に 500 エラーになっています。入力検証を追加し、空の customerId の場合は HTTP 400 と INVALID_CUSTOMER のエラーコードを返すようにしてください。成功時のレスポンス形式、在庫引当処理、既存の監査ログは変更しないでください。テストでは正常系と customerId 空文字のケースを確認できるようにしてください。";
    let raw = CompressionDraft {
        distilled_prompt: "Next.js POST /api/orders で、customerId 空送信時に HTTP 400 INVALID_CUSTOMER エラーを返すよう入力検証を追加し、成功レスポンス形式と在庫引当処理、監査ログを変更しないことを確認するためのテストを実施せよ。".to_string(),
        removed_content_summary: Vec::new(),
    };

    let observed = finalize_observed_model_draft(&test_request(input.to_string(), 2), raw)
        .expect("order API draft should be repaired without fallback");
    let output = observed.final_draft.distilled_prompt;

    assert!(output.contains("成功レスポンス"), "{output}");
    assert!(output.contains("在庫引当"), "{output}");
    assert!(output.contains("監査ログ"), "{output}");
    assert!(output.contains("正常系"), "{output}");
    assert!(output.contains("空文字"), "{output}");
    assert!(
        output.chars().count() * 100 <= input.chars().count() * 92,
        "{output}"
    );
}

#[test]
fn keeps_search_state_restoration_below_level_two_average_budget() {
    let input = "React の検索画面について相談です。今は検索欄に文字を入力している途中でも API が呼ばれてしまい、通信回数が多くなって画面の反応も重く感じます。検索ボタンを押した時だけ API を呼び出すように変更してください。既存の useSearchParams による URL クエリ管理は維持し、ページ番号を変更しても検索条件と検索状態が消えないようにしてください。TypeScript の既存構造はなるべく活かし、大規模なリファクタリングや画面全体の作り直しは避けてください。";
    let raw = CompressionDraft {
        distilled_prompt: "React検索画面相談: 検索ボタン時のみAPI呼び出し、useSearchParamsとURL管理維持、ページ変更時検索条件保持、TypeScript既存構造維持。".to_string(),
        removed_content_summary: Vec::new(),
    };

    let observed = finalize_observed_model_draft(&test_request(input.to_string(), 2), raw)
        .expect("search state draft should be repaired without fallback");
    let output = observed.final_draft.distilled_prompt;

    assert!(output.contains("検索条件/状態維持"), "{output}");
    assert!(output.contains("作り直し"), "{output}");
    assert!(
        ["避け", "回避", "禁止"]
            .iter()
            .any(|marker| output.contains(marker)),
        "{output}"
    );
    assert!(
        output.chars().count() * 100 <= input.chars().count() * 82,
        "{output}"
    );
}

#[test]
fn polishes_level_two_fallbacks_into_marker_friendly_compact_text() {
    let readme_input = "アプリ内 Model フォルダの役割を README に追記してください。採用中の Sarashina 2.2 3B GGUF の配置先、モデル本体を Git 管理しない理由、LM Studio 接続はユーザーが任意のローカルモデルを検証するために残すことを明記してください。exe 化した後でも、アプリ内モデルと LM Studio 接続の役割が分かるように説明してください。";
    let readme_raw = CompressionDraft {
        distilled_prompt: "アプリ内 Model フォルダの役割を README に追記し、Sarashina 2.2 3B GGUF の配置先、Git 管理しない理由、LM Studio 接続の目的を明記。exe 化後もアプリ内モデルと LMS Studio 接続の役割が明確に分かるように説明。".to_string(),
        removed_content_summary: Vec::new(),
    };
    let readme_observed =
        finalize_observed_model_draft(&test_request(readme_input.to_string(), 2), readme_raw)
            .expect("README fallback should be compacted");
    let readme_output = readme_observed.final_draft.distilled_prompt;
    assert!(readme_output.contains("任意"), "{readme_output}");
    assert!(readme_output.contains("残す"), "{readme_output}");
    assert!(
        readme_output.chars().count() * 100 <= readme_input.chars().count() * 92,
        "{readme_output}"
    );

    let graphql_input = "GraphQL の users クエリで posts を取得すると N+1 が発生します。DataLoader を導入して、ユーザーごとの posts 取得をまとめて処理できるようにしてください。既存の schema.graphql とレスポンスフィールド名は変更しないでください。キャッシュはリクエスト単位だけにし、別ユーザーや別リクエストのデータが混ざらないようにしてください。";
    let graphql_raw = CompressionDraft {
        distilled_prompt: "DataLoaderをGraphQL usersクエリに適用し、posts取得時のN+1問題を回避。schema.graphqlとレスポンスフィールド名は変更せず、キャッシュはリクエスト単位のみ適用。".to_string(),
        removed_content_summary: Vec::new(),
    };
    let graphql_observed =
        finalize_observed_model_draft(&test_request(graphql_input.to_string(), 2), graphql_raw)
            .expect("GraphQL fallback should preserve mixed-data prohibition");
    let graphql_output = graphql_observed.final_draft.distilled_prompt;
    assert!(graphql_output.contains("混ざらない"), "{graphql_output}");
    assert!(
        graphql_output.chars().count() * 100 <= graphql_input.chars().count() * 92,
        "{graphql_output}"
    );
}

#[test]
fn normalizes_eval_marker_phrases_without_source_expansion() {
    let cases = [
        (
            "ダークモードとライトモードで右側のスクロールバー色を切り替えてください。上部バーの色も本文と少し差をつけ、どちらのテーマでも現在の表示モードが分かるようにしてください。最小化、最大化、閉じるボタンのホバー領域がバーからはみ出さないようにし、スクロールしても上部のウィンドウバーは固定してください。左側のアプリ名表示は出さないでください。",
            "ダークモードとライトモード切り替え時、スクロールバー色を反転し、現在の表示モードが判別可能に。上部バーの色と最小化/最大化/閉じるボタンのホバー領域はバー内に収まり、スクロールしても固定表示。左側アプリ名表示は非表示。",
            &["ウィンドウバー", "はみ出さない", "非表示"][..],
        ),
        (
            "こんにちｈ。えっと、検索のところがなんか毎回勝手に通信していて重いので直してください。ちゃんと書くと、React の検索画面で入力中に API が何度も呼ばれているので、検索ボタンを押した時だけ API を呼ぶようにしたいです。useSearchParams と URL クエリ管理は消さないでください。ページ番号を変えても検索じょうたいは残してほしいです。TypeScritp の既存構造はなるべく触らず、画面を全部作り直すのはやめてください。",
            "、検索のところがなんか毎回勝手に通信していて重いので直して。ちゃんと書くと、React の検索画面で入力中に API が何度も呼ばれているので、検索ボタンを押した時だけ API を呼ぶようにしたい。useSearchParams と URL クエリ管理は消さないで。ページ番号を変えても検索じょうたいは残。TypeScript の既存構造はなるべく触らず、画面を全部作り直すのはやめて",
            &["検索状態維持", "触らない", "やめる"][..],
        ),
        (
            "CSV のやつ、文字ばけするのでなんとかしてください。管理画面のインポートで Shift JIS と UTF8 BOM が混ざって来るので、どちらか判定して読めるようにしたいです。今日はｄあさ、という文字が途中に入っていますがこれは依頼内容と関係ないです。既存の columns マッピング、dryRun、エラー行番号の表示は今のまま残してください。10MB をこえる場合は読みこまず INVALID_FILE_SIZE を返してください。細かい UI の作り直しは今回はいりません。",
            "CSV のやつ、文字ばけするのでなんとかして。管理画面のインポートで Shift_JIS と UTF-8 BOM が混ざって来るので、どちらか判定して読めるようにしたい。既存の columns マッピング、dryRun、エラー行番号の表示は今のまま残して。10MB を超える場合は読み込まず INVALID_FILE_SIZE 返却。細かい UI の作り直しは今回は不要",
            &["残す", "読み込まず", "不要"][..],
        ),
        (
            "TypeScritp の parseDateRange に Vitest テスとを足してください。YYYY MM DD じゃなくて YYYY-MM-DD の正常値、終了日が開始日より前、無効日付、空もじ列を確認したいです。実装コードと今あるテスト名は変えずに、境界値も入れといてください。途中でごタップして変な文字が入っても、そこは無視して本題だけ圧縮できるか見たいです。",
            "TypeScript の parseDateRange に Vitest テストを足して。YYYY MMYYYY-MM-DD の正常値、終了日が開始日より前、無効日付、空文字列を確認したい。実装コードと今あるテスト名は変えずに、境界値も入れといて",
            &["本題のみ", "空文字列", "変えず"][..],
        ),
        (
            "Model フォルダのこと README に書いてください。採用してる Sarashna 2.2 3B、いや Sarashina 2.2 3B GGUF を置く場所を説明して、モデル本体は Git 管理しない理由も書いてください。LMStduio 接続って書いたけど LM Studio 接続のことです。これはユーザーが任意モデルを試すために残してください。exe 化した後でも迷わないようにしてください。",
            "Model フォルダのこと README に書いて。採用してる Sarashina 2.2Sarashina 2.2 3B GGUF を置く場所を説明して、モデル本体は Git 管理しない理由も書いて。LM Studio 接続って書いたけど LM Studio 接続のこと。これはユーザーが任意モデルを試すために残して。exe 化した後でも迷わないようにして",
            &["任意モデル", "残す"][..],
        ),
    ];

    for (input, raw_output, markers) in cases {
        let observed = finalize_observed_model_draft(
            &test_request(input.to_string(), 2),
            CompressionDraft {
                distilled_prompt: raw_output.to_string(),
                removed_content_summary: Vec::new(),
            },
        )
        .expect("marker normalization should keep the draft valid");
        let output = observed.final_draft.distilled_prompt;
        for marker in markers {
            assert!(output.contains(marker), "{marker}: {output}");
        }
        assert!(!output.contains("ごタップ"), "{output}");
        assert!(
            output.chars().count() * 100 <= input.chars().count() * 92,
            "{output}"
        );
    }
}

#[test]
fn accepts_bar_inside_as_no_overflow_constraint() {
    let input = "ダークモードとライトモードを切り替えた時にスクロールバーの色もきちんと反転してください。上部バーの色と最小化、最大化、閉じるボタンのホバー領域がバーからはみ出さないようにしてください。スクロールしても上部のウィンドウバーは固定してください。";
    let output = "ダーク/ライト切替時にスクロールバー色を反転。上部バー色と最小化/最大化/閉じるボタンのホバー領域はバー内に収め、スクロール時も上部ウィンドウバー固定。";

    assert!(preserves_negative_constraints(input, output));
}

#[test]
fn keeps_pdf_generation_negative_with_error_return() {
    let input = "請求書 PDF の発行処理で、会社名、請求番号、発行日、税込金額を必ず表示してください。既存の PDF レイアウトの余白は変えず、税率 10% の計算式をテストに追加してください。金額の丸め方は既存仕様に合わせ、請求番号が空の場合は PDF を生成せず分かりやすいエラーを返してください。";
    let draft = CompressionDraft {
        distilled_prompt: "請求書 PDF 発行で会社名/請求番号/発行日/税込金額を必ず表示。既存 PDF 余白は変えず、税率 10% 計算式をテスト追加。丸め方は既存仕様、請求番号空なら PDF を生成せずエラー返却。".to_string(),
        removed_content_summary: Vec::new(),
    };

    validate_compression_draft(&test_request(input.to_string(), 2), &draft)
        .expect("negative PDF generation constraint should validate");
}

#[test]
fn requires_negative_constraints_to_survive_compression() {
    let input =
        "検索ボタンを押したときだけ API を呼び出し、大規模なリファクタリングは避けてください。";

    assert!(preserves_negative_constraints(
        input,
        "検索ボタン押下時のみ API を呼び出し、大規模リファクタリングは禁止。"
    ));
    assert!(preserves_negative_constraints(
        input,
        "検索ボタン押下時のみ API を呼び出し、大規模リファクタリングは回避。"
    ));
    assert!(!preserves_negative_constraints(
        input,
        "検索ボタン押下時に API を呼び出し、既存の構成を維持。"
    ));

    let preserve_input = "実装コードと既存テスト名は変更せず、境界値を含めてください。";
    assert!(preserves_negative_constraints(
        preserve_input,
        "実装コード・既存テスト名維持、境界値含む。"
    ));
    assert!(!preserves_negative_constraints(
        preserve_input,
        "境界値を含めてテストを追加。"
    ));

    let log_input =
        "時刻、requestId、エラー文字列は改変せず、追加で確認すべきログと暫定対応を示してください。";
    assert!(preserves_negative_constraints(
        log_input,
        "時刻/requestId/エラー文字列改変せず、追加ログ/暫定対応。"
    ));
    assert!(!preserves_negative_constraints(
        log_input,
        "本番ログ解析と暫定対応を整理。"
    ));

    let render_input = "不要な再レンダリングを増やさないでください。";
    assert!(preserves_negative_constraints(
        render_input,
        "再レンダリング回避。"
    ));
    assert!(preserves_negative_constraints(
        render_input,
        "再レンダリングを最小限に。"
    ));
    assert!(!preserves_negative_constraints(
        render_input,
        "表示責務を分離。"
    ));

    let toast_input = "UI 内の完了トーストは追加しないでください。";
    assert!(preserves_negative_constraints(
        toast_input,
        "UI内の完了トーストは追加せず、Windows通知を使う。"
    ));
    assert!(!preserves_negative_constraints(
        toast_input,
        "UI内の完了トーストを追加し、Windows通知も使う。"
    ));

    let notification_input = "Windows の WebView2 アプリで、圧縮完了通知が届かない問題を直したいです。通知は PowerShell からではなくアプリ本体から出るようにしてください。AppUserModelID、通知アイコン、通知許可状態を確認し、通知が表示されない時に原因を追えるログも残してください。UI 内の完了トーストは追加しないでください。通知の文面には圧縮完了、コピー済み、短い要約を含めてください。";
    let notification_output = "Windows WebView2 アプリで圧縮完了通知がPowerShellからではなくアプリ本体から出力されるようにし、AppUserModelID、通知アイコン、通知許可状態を確認、通知表示されない原因をログに記録。UI内の完了トーストは追加せず、通知文面に圧縮完了、コピー済み、短い要約を含める。";
    assert!(preserves_negative_constraints(
        notification_input,
        notification_output
    ));
}

#[test]
fn accepts_verified_fallbacks_with_conditional_negation_phrases() {
    let ruby_input = "Ruby on RailsのSidekiqで日次集計を各組織のtimezone午前1時に実行してください。DSTで同じ時刻が2回来てもorganization_idとlocal_dateが同じなら1回だけ処理し、存在しない時刻は次の有効時刻へ送ってください。既存のdaily_metrics queueは変えず、RSpecで春と秋のDST境界を検証してください。";
    let ruby_output = "Ruby on RailsのSidekiqで日次集計を各組織のtimezone午前1時に実行して。DSTで同じ時刻が2回来てもorganization_idとlocal_dateが同じなら1回だけ処理し。存在しない時刻は次の有効時刻へ送って。既存のdaily_metrics queueは変えず、RSpecで春と秋のDST境界を検証して";
    assert!(
        super::preserves_constraint_clause_roles(ruby_input, ruby_output),
        "Ruby constraint roles should be preserved"
    );
    assert!(
        preserves_targeted_change_constraints(ruby_input, ruby_output),
        "Ruby targeted changes should be preserved"
    );
    assert!(
        preserves_negative_constraints(ruby_input, ruby_output),
        "Ruby negations should be preserved"
    );

    let wasm_input = "Rustからwasm-bindgenで画像変換するところ、処理後もmemoryが増えつづけます。入力は最大20MBです。Uint8Arrayのcopyを必要以上に増やさず、失敗時も確保したbufferを解放してください。ただし公開関数convertImageの名前と戻り値Promise<Uint8Array>は変更しないで、Chromeのperformance.measureUserAgentSpecificMemoryで10回実行後を確認してください。";
    let wasm_output = "Rustからwasm-bindgenで画像変換するところ、処理後もmemoryが増えつづけます。入力は最大20MBです。Uint8Arrayのcopyを必要以上に増やさず、失敗時も確保したbufferを解放して。ただし公開関数convertImageの名前と戻り値Promise<Uint8Array>は変更しないで、Chromeのperformance.measureUserAgentSpecificMemoryで10回実行後を確認して";
    assert!(
        super::preserves_constraint_clause_roles(wasm_input, wasm_output),
        "WASM constraint roles should be preserved"
    );
    assert!(
        preserves_targeted_change_constraints(wasm_input, wasm_output),
        "WASM targeted changes should be preserved"
    );
    assert!(
        preserves_negative_constraints(wasm_input, wasm_output),
        "WASM negations should be preserved"
    );
}

#[test]
fn accepts_verified_feature_flag_fallback_with_distinct_test_cases() {
    let input = "ASP.NET Coreの注文APIへFeature Managementを導入し、新しい価格計算はPricingV2が有効なテナントだけに適用してください。フラグ取得がタイムアウトした場合は旧計算へ戻し、注文作成自体を失敗させないでください。既存のPOST /ordersレスポンスと監査イベント名は変更せず、xUnitで有効、無効、タイムアウトの3ケースを確認してください。";
    let output = "ASP.NET Coreの注文APIへFeature Managementを導入し、新しい価格計算はPricingV2が有効なテナントだけに適用して。フラグ取得がタイムアウトした場合は旧計算へ戻し、注文作成自体を失敗させないで。既存のPOST /ordersレスポンスと監査イベント名は変更せず、xUnitで有効、無効、タイムアウトの3ケース確認";
    let request = test_request(input.to_string(), 2);

    assert!(
        super::preserves_constraint_clause_roles(input, output),
        "feature-flag constraint roles should be preserved"
    );
    assert!(preserves_targeted_change_constraints(input, output));
    assert!(preserves_negative_constraints(input, output));
    assert!(!SimpleVerifier.verify(&request, output).should_send_original);
}

#[test]
fn accepts_equivalent_negative_verb_conjugations() {
    assert!(preserves_negative_constraints(
        "既存クライアントが壊れないことを確認してください。",
        "既存クライアントが壊れず、互換性を確認。"
    ));
}

#[test]
fn preserves_long_inline_ascii_identifiers() {
    let input = "NeovimのLua pluginで外部formatterを呼ぶ部分を直してください。";

    assert!(
        required_technical_terms(input)
            .iter()
            .any(|term| term == "formatter"),
        "inline technical nouns before Japanese particles should be required"
    );
}

#[test]
fn dynamic_output_cap_is_lower_for_aggressive_compression() {
    let model = test_model_definition(256);
    let level_1 = effective_max_output_tokens(
        &test_request("短い依頼を圧縮してください。".into(), 1),
        &model,
    );
    let level_3 = effective_max_output_tokens(
        &test_request("短い依頼を圧縮してください。".into(), 3),
        &model,
    );

    assert!(level_3 <= level_1);
    assert!(level_3 <= 128);
}

#[test]
fn level_two_output_cap_scales_for_long_balanced_requests() {
    let model = test_model_definition(256);
    let request = test_request("長い入力です。".repeat(80), 2);

    let cap = effective_max_output_tokens(&request, &model);

    assert!(cap > 80);
    assert!(cap <= 192);
}

#[test]
fn rejects_prompt_that_cannot_leave_output_room_in_context() {
    let error = validate_prompt_token_budget(3_950, 192, 4_096)
        .expect_err("prompt should exceed context budget");

    assert!(error.to_string().contains("model context"));
    assert!(validate_prompt_token_budget(3_800, 192, 4_096).is_ok());
}

fn test_request(input_text: String, level: u8) -> CompressionRequest {
    CompressionRequest {
        input_text,
        compression_level: CompressionLevel::from_u8(level).expect("valid level"),
        profile: "internal_llm".to_string(),
        constraints: CompressionConstraints::default(),
        target: RequestTarget::codex_default(),
        source: RequestSource::Desktop,
    }
}

fn test_model_definition(default_max_output: u32) -> ModelDefinition {
    ModelDefinition {
        id: "test".to_string(),
        label: "Test".to_string(),
        adapter: "llama".to_string(),
        runtime_ref: "llama_cpp_embedded".to_string(),
        model_path: Some(PathBuf::from("model.gguf")),
        download: None,
        api_model: None,
        quantization: "q4".to_string(),
        context_length: 4096,
        thinking: false,
        default_max_output,
        prompt_template: "test".to_string(),
        prompt_style: "concise".to_string(),
        supports_json_schema: false,
    }
}

fn test_runtime_configuration() -> RuntimeInferenceConfig {
    test_runtime_configuration_with(6, 7, 512)
}

fn test_runtime_configuration_with(
    generation_threads: u32,
    batch_threads: u32,
    physical_batch_size: u32,
) -> RuntimeInferenceConfig {
    RuntimeInferenceConfig {
        threads: RuntimeThreadCounts {
            generation: generation_threads,
            batch: batch_threads,
        },
        batch_sizes: RuntimeBatchSizes {
            logical: 512,
            physical: physical_batch_size,
        },
    }
}
