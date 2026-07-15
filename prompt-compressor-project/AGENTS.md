# TrimPrompt Project Instructions

## Project summary

TrimPrompt is a local-first prompt compression layer for Codex-oriented workflows.

The near-term goal is to compress long natural-language coding requests into shorter,
safer, structured prompts before they are sent to Codex.

## Product constraints

- Local-first by default
- No API key storage or handling in the MVP
- No automatic cloud forwarding
- Review-before-send UX
- CPU-friendly local model execution
- Windows and macOS first
- Codex-first before broader LLM support

## Architecture direction

- Shared Rust core for compression logic
- CLI, MCP server, and future desktop shell all call the same core
- Profiles, runtimes, and policies must stay configurable
- Runtime backend should remain replaceable
- Keep fail-open behavior: when unsafe, prefer returning the original input

## Current project state

- Workspace scaffold exists
- Core types and service scaffold exist
- CLI scaffold exists
- MCP server scaffold exists
- Config files for models, profiles, runtimes, and policies exist
- Rust toolchain is installed locally
- Real llama.cpp process backend is implemented
- CLI scaffold builds and runs with fail-open runtime behavior

## Working rules

- Keep requirements, product brief, market research, and design aligned
- Do not edit the requirements document when only product brief or market positioning changes
- Prefer small, reviewable changes
- Keep new runtime/model features behind clear boundaries
- Avoid adding background services or complex state unless clearly necessary

## Immediate implementation priorities

1. Add or configure a `llama-cli` binary and GGUF model files
2. Run the CLI against a real llama.cpp runtime
3. Implement MCP stdio transport
4. Add an evaluation set
5. Add the future Tauri desktop shell
