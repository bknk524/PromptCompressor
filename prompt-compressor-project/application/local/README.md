# Local Application Data

This directory contains machine-specific data and is not the source of truth for model selection.

- `models/`: GGUF model files used by the embedded local runtime.
- `runtimes/`: Reserved for future runtime assets. The current default runtime is embedded in the Rust executable.
- `cache/`: Rebuildable runtime cache.
- `logs/`: Local diagnostic logs.
- `state/`: Runtime-generated inventory, UI settings, and WebView2 state.

Model and runtime definitions remain in `../config/`. The application can safely recreate `cache/`, `logs/`, and `state/`.

## Saved UI Settings

The Web UI writes user-facing settings to:

```text
state/ui-settings.json
```

The file stores the selected profile, mode, task type, compression level, and
light/dark theme. It is excluded from Git and copied behavior is local to each
application folder.

## Current Local Model

The adopted local model is stored at:

```text
models/sarashina2.2-3b-instruct-v0.1/sarashina2.2-3b-instruct-v0.1-Q4_K_S.gguf
```

Source: `mmnga/sarashina2.2-3b-instruct-v0.1-gguf` on Hugging Face.
Base model: `sbintuitions/sarashina2.2-3b-instruct-v0.1`.
The `models/` directory is excluded from Git because model files are large and machine-specific.

The default app workflow does not require `llama.exe`, `llama-server.exe`, or
LM Studio. LM Studio remains available only through the optional
`lmstudio_local` profile for testing a user-loaded local model.

The default embedded model is preloaded after UI startup and remains cached in
the application process until the app exits.
