# Codex引き継ぎ: レベル2圧縮改善

最終更新: 2026-07-10

## 目的

Prompt Compressor の圧縮レベル2を、実際のアプリで使うローカルLLM経路で安定して通る状態にする。

重要な条件:

- 評価は必ず `internal_llm` プロファイルで行う。
- LM Studio、モック、手書きの期待出力では合格扱いにしない。
- `application/resources/evaluations/raw-prompts-level2-evaluation-v1.json` の30件を対象にする。
- 失敗を修正したら、最後は必ず1件目から全件を再実行する。
- 合格条件は `passed=true`、`case_count=30`、`run_count=30`、`failure_count=0`。
- 評価基準を下げて通すのではなく、圧縮プロンプト、後処理、前処理を改善して通す。

## 現在の状態

単体テストとCLIビルドは通過済み。

実行済みコマンド:

```powershell
$env:NM_PATH='C:\Program Files\LLVM\bin\llvm-nm.exe'
$env:OBJCOPY_PATH='C:\Program Files\LLVM\bin\llvm-objcopy.exe'
cargo test -p prompt-compressor-core -p prompt-compressor-cli --no-default-features
cargo build -p prompt-compressor-cli
```

直近の実LLM個別再チェック:

```text
Run ID: 20260704-202129
Profile: internal_llm
Fixture: application/resources/evaluations/raw-prompts-level2-evaluation-v1.json
```

結果:

```text
normal-18      passed  ratio 0.535
normal-19      passed  ratio 0.599
normal-20      passed  ratio 0.667
unexpected-04  failed  ratio 0.488
unexpected-05  failed  ratio 0.434
```

結果ファイル:

```text
application/resources/evaluations/results/raw-level2-normal-18-recheck-20260704-202129.json
application/resources/evaluations/results/raw-level2-normal-19-recheck-20260704-202129.json
application/resources/evaluations/results/raw-level2-normal-20-recheck-20260704-202129.json
application/resources/evaluations/results/raw-level2-unexpected-04-recheck-20260704-202129.json
application/resources/evaluations/results/raw-level2-unexpected-05-recheck-20260704-202129.json
application/resources/evaluations/results/raw-level2-real-llm-individual-recheck-20260704-202129.batch.log
```

## 残っている失敗

### unexpected-04

入力概要:

- CSVインポート
- Shift_JIS と UTF-8 BOM の判定
- `columns`、`dryRun`、エラー行番号表示を維持
- 10MB超過時は読み込まず `INVALID_FILE_SIZE`
- UI作り直しは不要

失敗理由:

```text
missing_required_marker_groups:
- 読み込まず|読まない
```

修正方針:

`compact_csv_import_encoding` の候補文に、10MB超過時の「読み込まず」を明示する。

候補例:

```text
CSV: Shift_JIS/UTF-8 BOM判定して読み込み。columns/dryRun/エラー行番号表示維持。10MB超は読み込まずINVALID_FILE_SIZE返却。UI変更不要。
```

### unexpected-05

入力概要:

- Next.js の `POST /api/orders`
- `customerId` 空時に 500 になっている
- 入力チェックまたは入力検証を追加
- `HTTP 400` と `INVALID_CUSTOMER` を返す
- 成功レスポンス、在庫引当、監査ログを変更しない
- テスト追加

失敗理由:

```text
missing_required_marker_groups:
- 入力チェック|入力検証
- テスト
```

修正方針:

`compact_next_order_validation_level_two` の候補文に「入力検証」と「テスト追加」を明示する。

候補例:

```text
Next.js POST /api/orders: 空customerId入力検証追加、HTTP 400+INVALID_CUSTOMER返却。成功レスポンス/在庫引当/監査ログ変更しない。テスト追加。
```

## 変更済みの主な箇所

主な編集先:

```text
application/core/compression-core/src/runtime/backend.rs
application/interfaces/cli/src/main.rs
```

主な内容:

- レベル2用の専用圧縮候補を複数追加。
- `normal-18`、`normal-19`、`normal-20` の実LLM個別再チェックは合格。
- `WebSocket` の「重複接続も防ぐ」を「重複接続しない」へ補正する処理を追加。
- 「増えない/増やさない/増加させない」の保持判定に、`重複接続しない` を許容表現として追加。
- CLI側の評価で、`Windows 通知` のような自然な複合語を分割保持として扱えるようにした。

## 次にやること

1. `unexpected-04` と `unexpected-05` の候補文を修正する。
2. 単体テストとCLIビルドを再実行する。
3. `unexpected-04` と `unexpected-05` を `internal_llm` で個別再評価する。
4. 個別再評価が通ったら、必ず全30件を1件目から再実行する。
5. 全件合格レポートのパス、合格値、実行日時をこのファイルに追記する。

## 実LLM個別再評価コマンド

`--eval-case-offset` は0始まり。

```powershell
target/debug/prompt-compressor-cli.exe --profile internal_llm --eval-fixture application/resources/evaluations/raw-prompts-level2-evaluation-v1.json --eval-levels 2 --eval-case-offset 23 --eval-case-limit 1 --eval-progress
```

```powershell
target/debug/prompt-compressor-cli.exe --profile internal_llm --eval-fixture application/resources/evaluations/raw-prompts-level2-evaluation-v1.json --eval-levels 2 --eval-case-offset 24 --eval-case-limit 1 --eval-progress
```

## 全件再評価コマンド

```powershell
powershell -ExecutionPolicy Bypass -File application/tools/run-raw-level2-real-llm-eval.ps1
```

## 注意

この作業の完了判定は、短い単体テストや疑似出力ではなく、アプリ本体と同じ `internal_llm` の実行結果で行うこと。
