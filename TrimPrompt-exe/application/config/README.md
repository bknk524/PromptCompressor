# Shared Configuration

This directory contains version-controlled application defaults.

- `model-catalog.yaml`: Model capabilities, local paths, and verified Hugging Face download sources.
- `compression-profiles.yaml`: User-selectable profiles that map model, policy, runtime, and fallback behavior. The UI currently exposes only LM Studio free selection and Sarashina 2.2 3B Instruct.
- `runtime-backends.yaml`: Embedded llama.cpp settings plus the optional LM Studio connection.
- `compression-policies/`: Compression behavior and preservation rules. `level-prompt-profiles-v1.yaml` defines the shared baseline and the one-request prompt profile used for the three compression levels.
- `../resources/evaluations/level-profile-evaluation-v1.json`: Acceptance thresholds and preservation checks for the level prompt profiles.
- `user/`: Reserved for machine-specific overrides and excluded from Git.

The UI and CLI choose profiles. The default profile is the bundled local model;
the LM Studio profile is only an optional external connection for trying a
user-loaded local model.

The Web UI starts serving first, then warms the default embedded model. When
saved CPU tuning is missing or stale, the normal workspace remains gated while
the explicit initial-setup screen runs diagnostics. Runtime warmup state is
exposed through `/api/runtime-status`; tuning readiness is exposed through
`/api/runtime-setup-status`.
