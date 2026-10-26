---
layout: default
title: Home
---

# Local LLM workloads, one CLI

Manage a host-based [LiteLLM](https://github.com/BerriAI/litellm) proxy and
[vLLM](https://github.com/vllm-project/vllm) backends running in NVIDIA
GPU-enabled Docker containers. Clients connect through an OpenAI-compatible API.

The launcher handles model selection, named model groups, container reuse,
startup monitoring, and logs. Deployment settings live in configuration files,
not per-start CLI overrides.

## Get started

1. Read the [requirements and installation guide](https://github.com/Harinezumigel/harinezumigel-llm-stack#install-from-any-checkout-location).
2. Download complete model files; the launcher does not download them.
3. Configure models, aliases, groups, and runtime settings using the
   [configuration guide](https://github.com/Harinezumigel/harinezumigel-llm-stack#configure-the-deployment).
4. Preview your selection with `--dry-run`, then start it.

For a group named `rag` configured in your deployment:

```bash
harinezumigel-llm-stack --group rag --start --dry-run
harinezumigel-llm-stack --group rag --start
harinezumigel-llm-stack --group rag --ps
harinezumigel-llm-stack --group rag --stop
```

Starting a group does not stop other groups or unselected models. Stop the old
workload explicitly before switching if both cannot fit in memory.

## Preflight checks

- Starts and dry-runs check the selected models' memory budgets. NVIDIA GB10
  systems use shared system memory when dedicated VRAM metrics are unavailable.
- Starts and dry-runs also probe NVIDIA image/driver compatibility.
- Dry-run is not offline: the compatibility probe can pull a missing container
  image, but does not download model files.
- Group recreation stops the selected backends before checking available memory
  and launching replacements. A dry-run does not stop them and cannot verify
  post-stop availability; it previews total capacity instead when needed.

See the [full operating guide](https://github.com/Harinezumigel/harinezumigel-llm-stack#readme)
for configuration, recreation, networking, logs, and troubleshooting.

## Project resources

- [Source code and README](https://github.com/Harinezumigel/harinezumigel-llm-stack)
- [Environment example](https://github.com/Harinezumigel/harinezumigel-llm-stack/blob/main/config/env.example)
- [Model configuration example](https://github.com/Harinezumigel/harinezumigel-llm-stack/blob/main/config/config.yaml.example)
- [Changelog](https://github.com/Harinezumigel/harinezumigel-llm-stack/blob/main/CHANGELOG.md)
- [Contributing](https://github.com/Harinezumigel/harinezumigel-llm-stack/blob/main/CONTRIBUTING.md)
- [Report an issue](https://github.com/Harinezumigel/harinezumigel-llm-stack/issues)

## Important

Provided as is, without warranty. Read the
[disclaimer](https://github.com/Harinezumigel/harinezumigel-llm-stack#disclaimer)
and validate before production use. This project is not affiliated with
LLMStack (llmstack.ai / Promptly).

LiteLLM exposes an admin UI at `/ui`; do not assume API authentication hides it.
Restrict network access or block that route at a reverse proxy.
