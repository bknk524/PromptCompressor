# 評価メモ

このフォルダには、プロンプト圧縮の評価方針、反復結果、引き継ぎ情報を保存します。実行対象のfixtureは `application/resources/evaluations/`、実行スクリプトは `application/tools/` にあります。

## 現行資料

- `Codex引き継ぎ_レベル2圧縮改善.md`: 現在のレベル2基準、合格状態、再現手順
- `長文冗長入力_レベル2反復評価.md`: 長文・冗長入力を改善した際の反復履歴
- `llama.cpp更新候補_b9982.md`: llama.cpp 更新候補、事前ベンチマーク、採用条件

## 現行fixture

- `application/resources/evaluations/raw-prompts-level2-evaluation-v1.json`
- `application/resources/evaluations/raw-long-redundant-evaluation-v1.json`
- `application/resources/evaluations/level-profile-evaluation-v1.json`

## 運用ルール

- 合格判定には `internal_llm` の実LLM結果を使う。
- モック、手書き出力、LM Studioの結果を `internal_llm` の代用にしない。
- 局所修正後は関連ケースを確認し、最後に対象fixtureを先頭から全件実行する。
- 過去の失敗記録は履歴として扱い、現在状態は最新のstatus JSONと引き継ぎ資料で確認する。
- 生成されたレポート、進捗ログ、status JSONは `application/resources/evaluations/results/` に保存し、Gitには追加しない。
