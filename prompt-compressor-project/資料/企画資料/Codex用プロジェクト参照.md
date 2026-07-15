# Codex Project Reference: TrimPrompt

このファイルは、Codex の新しいプロジェクトとして TrimPrompt を始めるための参照元です。

新しい Codex プロジェクトでは、まずこのファイルと `AGENTS.md` を読んでください。

## 1. プロジェクト概要

TrimPrompt は、Codex に送る前の長い依頼文を、ローカル LLM で短く安全な実行プロンプトへ整えるアプリです。

初期フォーカス:

- Codex 向け
- コーディング依頼、バグ修正、リファクタ、ログ解析、設計相談
- ローカル処理
- API キー不要
- 自動送信なし
- review-before-send
- Windows / macOS 対応

MVP では、クラウド LLM の API キー管理、API プロキシ、ブラウザ拡張、チーム管理、コスト可視化は扱いません。

## 2. ワークスペース

ワークスペース root:

```text
C:\Users\sl2ca\OneDrive\ドキュメント\New project\prompt-compressor-project
```

## 3. 最重要参照ファイル

まず読む順番:

1. `AGENTS.md`
2. `企画資料\Codex用プロジェクト参照.md`
3. `企画資料\進捗状況.md`
4. `企画資料\作業リスト.md`
5. `企画資料\システム設計_ローカルプロンプト圧縮.md`
6. `企画資料\要件定義_ローカルプロンプト圧縮.md`

絶対パス:

```text
C:\Users\sl2ca\OneDrive\ドキュメント\New project\prompt-compressor-project\AGENTS.md
C:\Users\sl2ca\OneDrive\ドキュメント\New project\prompt-compressor-project\企画資料\Codex用プロジェクト参照.md
C:\Users\sl2ca\OneDrive\ドキュメント\New project\prompt-compressor-project\企画資料\進捗状況.md
C:\Users\sl2ca\OneDrive\ドキュメント\New project\prompt-compressor-project\企画資料\作業リスト.md
C:\Users\sl2ca\OneDrive\ドキュメント\New project\prompt-compressor-project\企画資料\システム設計_ローカルプロンプト圧縮.md
C:\Users\sl2ca\OneDrive\ドキュメント\New project\prompt-compressor-project\企画資料\要件定義_ローカルプロンプト圧縮.md
```

## 4. 文書の役割

- `local-llm-prompt-compressor-product-brief.md`
  - 製品概要
  - 何をするアプリか、何ができるか、売り込み用の説明

- `企画資料\要件定義_ローカルプロンプト圧縮.md`
  - 要件定義
  - 機能要件、非機能要件、UX、MVP、モデル方針、MCP 方針

- `企画資料\システム設計_ローカルプロンプト圧縮.md`
  - 設計
  - Core / CLI / MCP / Local App の構成、データ構造、実装順

- `local-llm-prompt-compressor-market-research.md`
  - 市場調査
  - 需要、競合、売上見込み、AI PC メーカー向け別案

- `企画資料\進捗状況.md`
  - 現在地
  - 何ができていて、何がまだか

- `企画資料\作業リスト.md`
  - 次にやること
  - Codex プロジェクトでの実装タスク管理

- `README.md`
  - リポジトリ入口
  - Rust 導入後に叩くコマンド

## 5. 現在の実装状態

作成済み:

- Rust workspace scaffold
- Rust / Cargo によるビルド確認
- shared core crate
- CLI scaffold
- development local Web UI
- MCP server scaffold
- llama.cpp process backend
- runtime fail-open handling
- desktop placeholder
- config files
- prompt templates
- JSON schemas
- evaluation directory

未実装:

- `application/local/runtimes/llama-cli` executable and GGUF model files
- MCP stdio transport
- desktop app
- real token estimation
- real verifier
- model download / setup flow

現在の主な blocker:

```text
llama.cpp process backend は接続済みだが、`application/local/runtimes/llama-cli` と GGUF モデルが未配置
```

## 6. コード上の入口

Workspace:

```text
C:\Users\sl2ca\OneDrive\ドキュメント\New project\prompt-compressor-project\Cargo.toml
```

Core:

```text
C:\Users\sl2ca\OneDrive\ドキュメント\New project\prompt-compressor-project\application\core\compression-core\src\lib.rs
C:\Users\sl2ca\OneDrive\ドキュメント\New project\prompt-compressor-project\application\core\compression-core\src\compression\service.rs
C:\Users\sl2ca\OneDrive\ドキュメント\New project\prompt-compressor-project\application\core\compression-core\src\runtime\backend.rs
C:\Users\sl2ca\OneDrive\ドキュメント\New project\prompt-compressor-project\application\core\compression-core\src\types.rs
C:\Users\sl2ca\OneDrive\ドキュメント\New project\prompt-compressor-project\application\core\compression-core\src\config\profile.rs
```

CLI:

```text
C:\Users\sl2ca\OneDrive\ドキュメント\New project\prompt-compressor-project\application\interfaces\cli\src\main.rs
```

MCP:

```text
C:\Users\sl2ca\OneDrive\ドキュメント\New project\prompt-compressor-project\application\interfaces\mcp\src\main.rs
```

Desktop placeholder:

```text
C:\Users\sl2ca\OneDrive\ドキュメント\New project\prompt-compressor-project\資料\将来構想\デスクトップアプリ構想.md
```

## 7. 設定ファイル

Profiles:

```text
C:\Users\sl2ca\OneDrive\ドキュメント\New project\prompt-compressor-project\application\config\compression-profiles.yaml
```

Models:

```text
C:\Users\sl2ca\OneDrive\ドキュメント\New project\prompt-compressor-project\application\config\model-catalog.yaml
```

Runtimes:

```text
C:\Users\sl2ca\OneDrive\ドキュメント\New project\prompt-compressor-project\application\config\runtime-backends.yaml
```

Policies:

```text
C:\Users\sl2ca\OneDrive\ドキュメント\New project\prompt-compressor-project\application\config\compression-policies\balanced-codex-compression-policy-v1.yaml
C:\Users\sl2ca\OneDrive\ドキュメント\New project\prompt-compressor-project\application\config\compression-policies\code-focused-codex-compression-policy-v1.yaml
C:\Users\sl2ca\OneDrive\ドキュメント\New project\prompt-compressor-project\application\config\compression-policies\safe-codex-compression-policy-v1.yaml
```

## 8. 初期モデル方針

Profiles:

- `standard`
  - model: `qwen3_1_7b`
  - role: 標準圧縮

- `code_focused`
  - model: `qwen2_5_coder_1_5b`
  - role: コード依頼向け

- `lightweight_safe`
  - model: `gemma3_1b_it`
  - role: 軽量 fallback

Runtime:

- MVP では `llama.cpp` process backend
- 将来的に `LlamaCppEmbeddedBackend` へ移行可能な境界を残す

## 9. 最初にやること

最初の開発タスク:

1. `runtime-backends.yaml` の `local/runtimes/llama-cli` を実行できるようにする
2. GGUF モデルを `application/local/models/` に配置する
3. 実 runtime で CLI 圧縮を確認する
4. MCP stdio transport を実装する

## 10. 開発ルール

- 要件定義は、実装仕様変更が必要なときだけ編集する
- 製品説明や市場性の修正だけなら、要件定義は触らない
- コアロジックは `application/core/compression-core` に寄せる
- CLI / MCP / Desktop は Core の薄い入口にする
- 失敗時は原文返却を優先する
- ログに入力全文を残さない
- モデル名や runtime をコードに直書きしない
- config schema は `schema_version` を保つ

## 11. Codex プロジェクトでの開始指示

新しい Codex プロジェクトでは、最初に次の指示を使う。

```text
このプロジェクトは TrimPrompt です。

まず AGENTS.md と 資料\企画資料\Codex用プロジェクト参照.md を読み、資料\企画資料\進捗状況.md と 資料\企画資料\作業リスト.md で現在地を確認してください。

目的は、Codex 向けの長い依頼文をローカル LLM で短く安全に整えるアプリを作ることです。
MVPでは API キー管理、API プロキシ、ブラウザ拡張、チーム管理、コスト可視化は扱いません。

Rust workspace のビルド確認、CLI scaffold の起動、llama.cpp process backend の接続は完了しています。
次に、application/local/runtimes/llama-cli と GGUF モデルを配置し、実 runtime で CLI 圧縮を確認してください。
```
