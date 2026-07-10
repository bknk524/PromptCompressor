# Prompt Compressor

Prompt Compressor is a local-first compression layer for Codex-oriented prompts.
It turns long draft requests into shorter execution prompts while keeping a
review-before-use workflow.

Folder purposes are summarized in `資料/フォルダ構成.md`.

## Current Status

This project is a working Rust workspace with:

- a shared compression core library
- a command-line compressor
- a local Web UI for browser-based testing
- a native Windows desktop shell using WebView2
- an MCP server scaffold
- settings for the bundled local model, embedded runtime, and compression policies
- planning and operational materials

The default runtime is self-contained at application runtime. It uses the
embedded `llama.cpp` backend compiled into the Rust application and the local
Sarashina GGUF model under `application/local/models/`. It does not require
LM Studio, `llama.exe`, `llama-server.exe`, a local LLM HTTP port, or a runtime
download while the app is running.

The optional `lmstudio_local` profile remains available for testing user-owned
local models through LM Studio. That profile connects to
`http://127.0.0.1:1234/v1` and automatically uses the model currently loaded
in LM Studio. It is not required for the bundled-model workflow.

If the configured GGUF model file is unavailable or the local runtime fails,
compression returns the original prompt and marks `should_send_original=true`
in JSON output.

## Repository Layout

```text
application/
  core/                           圧縮ロジックとランタイム実装
  ui/                             ローカル Web UI と Windows デスクトップシェル
  interfaces/                     CLI と MCP サーバー
  config/                         Git 管理するモデル・プロファイル・ポリシー設定
  resources/                      LLM テンプレートと JSON スキーマ
  local/                          PC ごとのモデル・ログ・状態
資料/
  企画資料/                       企画、設計、要件、進捗、作業リスト
  評価メモ/                       評価設計と結果メモ
  将来構想/                       未実装機能の構想資料
  フォルダ構成.md                 各フォルダの詳細説明
```

## First Commands

```powershell
cargo run -p prompt-compressor-cli -- --list-profiles
```

```powershell
cargo run -p prompt-compressor-cli -- "Fix the React search behavior while preserving URL query parameters."
```

```powershell
cargo run -p prompt-compressor-local-ui -- --host 127.0.0.1 --port 8787
```

Then open:

```text
http://127.0.0.1:8787
```

For the Windows native desktop shell:

```powershell
cargo run -p prompt-compressor-desktop
```

The desktop shell does not start an HTTP server and does not bind a localhost
port. It serves the same UI through an internal WebView2 custom protocol
(`prompt-compressor://`) and routes app API calls in-process, so the packaged
exe behaves like a normal Windows application instead of a browser-hosted web
app.

UI settings are persisted under `application/local/state/ui-settings.json`.
The saved values include the selected profile, mode, task type, compression
level, and light/dark theme. Browser `localStorage` is still used as a fallback,
but the application-side settings file keeps the same choices after the app is
stopped and started again.

To create a separated runnable desktop package:

```powershell
powershell -ExecutionPolicy Bypass -File application/tools/package-desktop-release.ps1 -Clean
```

This writes the runnable app to the sibling folder
`../prompt-compressor-project-exe/` and leaves the source project folder
unchanged. The package contains `PromptCompressor.exe`, `application/config/`,
`application/resources/`, and the configured local model. The runtime is
embedded in the executable, so no `application/local/runtimes/` binaries are
copied into the release package.

To use the release, open `PromptCompressor.exe` inside the package folder. You
do not need to start the development Web UI, LM Studio, or any separate local
LLM server. Keep the `application/` folder next to the exe; the GGUF model is
too large to embed directly into the exe, so copying only `PromptCompressor.exe`
to another folder will not be enough to compress text. If required files are
missing, the desktop app writes `PromptCompressor_STARTUP_ERROR.txt` and opens
it in Notepad.

To choose a different output folder:

```powershell
powershell -ExecutionPolicy Bypass -File application/tools/package-desktop-release.ps1 -Clean -OutputPath "C:\path\to\PromptCompressor"
```

## Build Tools

The release package is self-contained at runtime, but building the embedded
`llama.cpp` backend requires LLVM build tools on the development machine.
The packaging script looks for `libclang.dll` and `llvm-nm.exe` under
`C:\Program Files\LLVM\bin` or the folder pointed to by `LIBCLANG_PATH`.

## トークン削減の計算ロジック

圧縮結果には、トークン（推定）と文字数を `入力 → 出力` の形式で表示する。表示する
`比率` は、圧縮後が圧縮前の何パーセントかを表す。

```text
トークン比 = 出力トークン推定 / 入力トークン推定 × 100
文字数比   = 出力文字数 / 入力文字数 × 100
削減率     = 100 - 比率
```

たとえばトークン比が `78%` の場合、推定トークン数は約 `22%` 削減されている。ランタイムの
失敗などで原文を返す場合は、入力と出力が同じになるため比率は `100%` になる。

### トークン推定

現在はすべてのプロファイルで、モデル固有の厳密な tokenizer ではなく、ローカルで高速に動く
共通のヒューリスティックな概算を使う。日本語を空白で区切らないために `1` トークンと誤表示
しないよう、連続する文字種ごとに以下を加算する。

- 日本語文字列: `ceil(文字数 × 2 / 3)`
- 英字と `_` の連続文字列: `ceil(文字数 / 4)`
- 数字の連続文字列: `ceil(文字数 / 3)`
- 空白以外の記号: 1 文字ごとに 1

実際の消費トークン数は、選択したモデルと tokenizer により異なる。将来はプロファイルごとに
実 tokenizer を接続できるよう、推定器には `target_tokenizer_profile` を渡している。

### 文字数

文字数は Rust の Unicode 文字単位で数える。日本語、英字、数字、空白、改行、記号を含むため、
トークン推定よりも入力と出力の長さを直感的に比較しやすい。

## Embedded Local LLM Runtime

The `internal_llm` profile runs the bundled local model directly through the
embedded `llama.cpp` backend.

1. Put the configured GGUF model under `application/local/models/`.
2. Select `internal_llm` in the Web UI or pass `--profile internal_llm` to the CLI.

When the Web UI starts, the local development server becomes available first
and then preloads the default embedded GGUF model in the background. The
desktop shell uses the same route handlers in-process through WebView2 instead
of opening a port. In both modes, `/api/runtime-status` reports whether the
model is loading, ready, skipped, or failed.

Loaded embedded models are held in the shared runtime cache for the lifetime of
the application process. The preload path and the compression path use the same
cache, so a warmed model is reused instead of being loaded again.

Runtime details, including thread count and model path, stay in the YAML
settings rather than the UI.

```powershell
cargo run -p prompt-compressor-cli -- --profile internal_llm "Shorten this Codex request while preserving constraints."
```

## Optional LM Studio Connection

The `lmstudio_local` profile is kept as a manual test path for local models
that the user loads in LM Studio.

1. Start LM Studio and enable its local server on `http://127.0.0.1:1234/v1`.
2. Load the model you want to compare.
3. Select `LM Studio（ローカルモデル自由選択）` in the Web UI, or pass
   `--profile lmstudio_local` to the CLI.

This path is intentionally separate from the default bundled-model runtime.
If LM Studio is not running, the app automatically falls back to `internal_llm`
so the desktop package can still compress text without external services.

See `資料/企画資料/内部LLM実行方針.md` for the lifecycle and deployment notes.

## Next Steps

1. Add a clearer user-facing startup error when WebView2 or the GGUF model is missing.
2. Implement MCP stdio transport.
