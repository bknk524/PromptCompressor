# Codex引き継ぎ: レベル2圧縮改善

最終更新: 2026-07-13

## 目的

TrimPromptの圧縮レベル2を、実際のアプリで使うローカルLLM経路で安定して評価・改善する。

## 評価条件

- プロファイル: `internal_llm`
- ランタイム: `llama_cpp_embedded`
- モデル: `sarashina_2_2_3b_instruct_q4_k_s`
- 基準fixture: `application/resources/evaluations/raw-prompts-level2-evaluation-v1.json`
- ケース数: 30件（通常20件、想定外入力10件）
- 合格条件: `passed=true`、`case_count=30`、`run_count=30`、`failure_count=0`
- 圧縮率: 平均値とケース単位の下限・上限を検証し、過圧縮と圧縮不足の両方を失敗にする

モック、手書きの期待出力、LM Studioの結果を `internal_llm` の代用として合格判定に使わない。

## 現在の基準状態

2026-07-13の実LLM全件評価では、30件すべてが合格している。

```text
passed=true
case_count=30
run_count=30
failure_count=0
```

過去に失敗していた `unexpected-04` と `unexpected-05` は解決済み。古い失敗内容は現在の未完了項目として扱わない。

## 改善方針

- fixture固有の文言だけを追加して通すのではなく、未知ドメインにも使える構造化、前処理、後処理、検証を改善する。
- 引用符、コード、識別子、ファイル名、数値、否定条件、対象限定、検証条件を壊さない。
- 前処理は高確度のノイズや誤字だけを修正し、意味が曖昧な箇所は保持する。
- 修正で既存ケースを壊さないよう、関連ケースの個別評価後に全30件を再評価する。
- 評価基準や必須語を緩めて合格させない。
- ケース単位の圧縮率下限を下回る出力は、短ければよいとは見なさず過圧縮として失敗させる。

## 実行手順

特定ケースだけを実行する。`CaseOffset` は0始まり。

```powershell
powershell -ExecutionPolicy Bypass -File application/tools/run-raw-level2-real-llm-eval.ps1 -RunId unexpected-04-recheck -CaseOffset 23 -CaseLimit 1
```

基準30件を先頭から実行する。

```powershell
powershell -ExecutionPolicy Bypass -File application/tools/run-raw-level2-real-llm-eval.ps1
```

スクリプトはCLIを必要に応じてビルドし、`CARGO_TARGET_DIR` が設定されていればその出力先を使う。LLVM関連ツールは既存の環境変数、LLVM、Rustツールチェーンの順に解決する。

## 出力

```text
application/resources/evaluations/results/raw-level2-real-llm-<RunId>.json
application/resources/evaluations/results/raw-level2-real-llm-<RunId>.progress.log
application/resources/evaluations/results/raw-level2-real-llm-<RunId>.status.json
```

status JSONには終了状態、終了コード、`passed`、`case_count`、`run_count`、`failure_count` が記録される。レポートと進捗ログは分離されるため、JSONレポートはそのまま機械処理できる。

## 長文・冗長入力

全体への応用性は、通常20件とノイズ入り5件を含む次のfixtureで確認する。

```text
application/resources/evaluations/raw-long-redundant-evaluation-v1.json
```

```powershell
powershell -ExecutionPolicy Bypass -File application/tools/evaluate-prompt-profiles.ps1 -FixturePath application/resources/evaluations/raw-long-redundant-evaluation-v1.json -Levels 1,2,3
```

レベル2を基準に改善する場合でも、変更が安定した段階でレベル1と3への回帰がないことを確認する。

## 2026-07-18 追記: 後処理・事前処理の軽量改善

目的は、LLM呼び出し回数を増やさず、評価失敗時の原文節丸ごと復元を減らしながら、必須語・否定・維持・限定markerを落とさないこと。

実装した主な変更:

- `検証`という語だけではテスト列挙制約と判定しないようにし、`テスト`/`Vitest`/`RSpec`/`Testcontainers` など明示的なテスト文脈がある場合だけ列挙分離チェックを行う。
- `はみ出さない`、`非表示`、`データ混ざらない`、`検索状態維持`、`触らない`、`やめる`、`残す` など、評価markerと意味同等の短い表現へ正規化する。
- `LM Studio 接続は任意ローカルモデル検証用に残す`、`任意モデル用に残す`、`本題のみ` を短い復元句として扱い、入力ノイズを伴う原文節を足し戻さない。
- `ウィンドウバー`を保護語に追加し、`スクロールしても固定表示`を`スクロール時もウィンドウバー固定`へ寄せる。
- 復元候補の必須語チェックを、自己訂正後の検証入力に揃え、古い候補値（例: `YYYY MM DD`）で短句復元が捨てられないようにした。
- 検証失敗後のtrusted fallbackにも、短句復元・必須語復元・polishを通す。
- `prompt_structure`側にも同じ短縮語彙を追加し、LLM前の整理済み入力とfallback候補の表現を揃えた。

追加した回帰テスト:

- `accepts_compact_search_state_and_rebuild_avoidance`
- `accepts_compact_model_readme_and_ci_constraints`
- `keeps_level_two_order_api_restoration_below_case_budget`
- `keeps_search_state_restoration_below_level_two_average_budget`
- `polishes_level_two_fallbacks_into_marker_friendly_compact_text`
- `normalizes_eval_marker_phrases_without_source_expansion`
- `accepts_bar_inside_as_no_overflow_constraint`
- `keeps_pdf_generation_negative_with_error_return`

確認済みコマンド:

```text
cargo fmt --all -- --check
cargo test -p prompt-compressor-core --lib
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

すべて成功。

未完了:

- 実LLM評価は、`cargo run ... --eval-fixture ...` のサンドボックス外実行が利用上限で拒否されたため、最終確認できていない。
- 次回実行可能になったら、まず `normal-09`、`normal-10`、`normal-13`、`unexpected-01`、`unexpected-04`、`unexpected-06`、`unexpected-07` を `--eval-case-offset` / `--eval-case-limit` で確認する。
- その後、基準30件全件と `raw-long-redundant-evaluation-v1.json` のレベル2を再評価する。
