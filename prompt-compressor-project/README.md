# Prompt Compressor

Prompt Compressor is a local-first compression layer for Codex-oriented prompts.
It turns long draft requests into shorter execution prompts while keeping a
review-before-use workflow.

Folder purposes are summarized in `フォルダ構成説明.md`.

## Current Status

This project is a working Rust workspace with:

- a shared compression core library
- a command-line compressor
- a local Web UI for browser-based testing
- an MCP server scaffold
- settings for models, profiles, runtimes, LM Studio, and policies
- presentation and planning materials

The runtime backend is fail-open. If `local-runtime-binaries/llama-cli.exe`, the
configured GGUF model files, or a configured LM Studio server are unavailable,
compression returns the original prompt and marks `should_send_original=true` in
JSON output.

## Repository Layout

```text
applications/
  command-line-compressor/        CLI entry point
  local-web-compressor-ui/        Local browser UI on localhost
  codex-mcp-compressor-server/    MCP server scaffold
  desktop-app-notes/              Desktop app planning notes
libraries/
  compression-core/               Shared compression service and runtime code
settings/
  compression-profiles.yaml       Profile selection rules
  model-catalog.yaml              Local model definitions
  runtime-backends.yaml           llama.cpp and LM Studio runtime settings
  compression-policies/           Compression policy files
prompt-templates/                 Runtime prompt templates
json-schemas/                     Request and result JSON schemas
評価メモ/                 Evaluation planning notes
local-runtime-binaries/           Place llama.cpp executables here
local-model-files/                Place GGUF model files here
成果物/
  発表資料/                       Presentation narrative
企画資料/                         Product, design, status, and task documents
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

## LM Studio Runtime

LM Studio can be used through its OpenAI-compatible local API server.

1. Start LM Studio's local server from the Developer tab or with `lms server start`.
2. Keep the default base URL as `http://localhost:1234/v1`, or edit `settings/runtime-backends.yaml`.
3. Select the `lmstudio` profile in the UI or CLI.

```powershell
cargo run -p prompt-compressor-cli -- --profile lmstudio "Shorten this Codex request while preserving constraints."
```

`settings/model-catalog.yaml` uses `api_model: auto` for LM Studio, so the
runtime asks `/v1/models` and uses the first visible model. Set `api_model` to a
specific LM Studio model identifier when you want a fixed model.

## Next Steps

1. Add `local-runtime-binaries/llama-cli.exe` or start LM Studio's local server.
2. Add the configured GGUF model files under `local-model-files/` when using llama.cpp.
3. Run CLI and local Web UI checks against a real local model.
4. Implement MCP stdio transport.
5. Add the desktop shell.
