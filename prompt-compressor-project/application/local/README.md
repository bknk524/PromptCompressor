# Local Application Data

This directory contains machine-specific data and is not the source of truth for model selection.

- `models/`: GGUF model files used by the embedded local runtime.
- `runtimes/`: Reserved for optional user-managed runtime assets. The default CPU engines are stored under `../runtime/cpu/`.
- `cache/`: Rebuildable runtime cache.
- `logs/`: Local diagnostic logs.
- `state/`: Runtime-generated inventory, UI settings, and WebView2 state.

Model and runtime definitions remain in `../config/`. The application can safely recreate `cache/`, `logs/`, and `state/`.

## Saved UI Settings

The Web UI writes user-facing settings to:

```text
state/ui-settings.json
```

The file stores the selected profile, mode, task type, compression level,
light/dark theme, CPU-engine preference, and automatic/manual generation and
batch thread preferences. It is excluded from Git and its behavior is local to
each application folder. The launcher validates manual values against detected
CPU capabilities and logical processor count before passing them to the runtime.

## Current Local Model

The adopted local model is downloaded on first use and stored at:

```text
models/sarashina2.2-3b-instruct-v0.1/sarashina2.2-3b-instruct-v0.1-Q4_K_S.gguf
```

Source: `mmnga/sarashina2.2-3b-instruct-v0.1-gguf` on Hugging Face.
Base model: `sbintuitions/sarashina2.2-3b-instruct-v0.1`.
The source revision and SHA-256 are pinned in `../config/model-catalog.yaml`.
The `models/` directory is excluded from Git because model files are large and machine-specific.

The default app workflow does not require `llama.exe`, `llama-server.exe`, or
LM Studio. LM Studio remains available only through the optional
`lmstudio_local` profile for testing a user-loaded local model.

The default embedded model is preloaded after UI startup and remains cached in
the application process until the app exits.

Compression outputs are not persisted or reused. The running app may retain
model and prompt-evaluation state in memory, but every compression request runs
LLM generation.

CPU thread tuning records are stored under `state/inference-tuning-v1/`. They
contain only the detected CPU instruction profile, selected CPU engine,
hardware, model, and runtime identities plus selected thread counts. No prompt
text or compression output is stored there. Records are automatically separated
when the CPU, CPU engine, model file, or tuning contract changes.

After the model is available, missing or stale records are created from the
visible initial-setup screen, which explains that the diagnostics may take a
little time. Tuning measures a representative 128-token input batch and 12
single-token decode steps across lower-thread and high-thread candidates for
each CPU engine. It then measures 128, 256, and 512-token physical microbatches
with a 512-token input and changes the default 512 only for a median gain of at
least three percent. A faster non-default candidate must also produce exactly
the same greedy output as 512 across three representative compression prompts,
up to 192 generated tokens each; otherwise tuning keeps 512. The alternate
engine probe saves its complete inference record during the same setup run. The
desktop app restarts once before it uses newly measured values. Later launches
skip diagnostics while the records remain valid.

The AVX2/AVX-512 comparison is stored in
`state/cpu-engine-selection-v1.json`. The record contains its schema, package
build ID, inference compatibility ID, CPU identity, and selected engine. Five
fixed built-in prompts are used for the comparison. Their generated text does
not need to match across engines; each result must preserve the case's required
targets, conditions, numbers, and prohibitions. User input and generated output
are not written to disk. Temporary `state/cpu-engine-probe-*.json` files contain
only elapsed time and the aggregate quality result and are removed after a
successful comparison. A CPU change, an inference-affecting source/configuration
change, or the UI's CPU optimization reset invalidates the selection. Rebuilding
the same inference implementation or changing only static UI assets keeps the
compatible selection.

Manual CPU-engine and thread preferences in `state/ui-settings.json` override
these automatic records without deleting them. Returning both controls to
automatic mode re-enables compatible saved tuning data.
