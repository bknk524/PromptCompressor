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
