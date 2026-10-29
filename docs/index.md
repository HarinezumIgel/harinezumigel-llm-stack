---
layout: default
title: Home
---

# Local LLM workloads, one CLI

![Hedgehog mascot](https://raw.githubusercontent.com/HarinezumIgel/RAG-LCC/main/docs/assets/AI_Igel.png)

Manage a host-based [LiteLLM](https://github.com/BerriAI/litellm) proxy and
[vLLM](https://github.com/vllm-project/vllm) backends running in NVIDIA
GPU-enabled Docker containers. Clients connect through an OpenAI-compatible API.

The launcher handles model selection, named model groups, container reuse,
startup monitoring, and logs. Deployment settings live in configuration files,
not per-start CLI overrides.

## Use case: experiment with RAG encoders and rerankers

Imagine you have a RAG system and want to compare embedding encoders and
rerankers. You will define a group for each combination in `config.yaml`.

Both groups also include the LLMs: Mistral (inference), Guard, Judge, and LiteLLM entries.
They are configured in `model_groups`:

```yaml
model_groups:
  rag: [mistral, guard, judge, bge-m3, bge-reranker-v2-m3, litellm]
  rag_old: [mistral, guard, judge, mmarco-reranker, snowflake-embedder, litellm]
```

Then run simply with --start which will create not existing containers.
Start a group to create and start any backend containers that do not exist yet.
The launcher reuses those containers on later starts; stopping them does not
delete them.

**`--start` creates a model's container if it does not exist, then starts it.**
If the container already exists, it is reused: stopped containers are started,
and running containers are left running.

```bash
# Create and start the BGE-based RAG group the first time.
harinezumigel-llm-stack --group rag --start --dry-run
harinezumigel-llm-stack --group rag --start

# After stopping client traffic, switch to the alternative encoder/reranker.
harinezumigel-llm-stack --group rag --stop &&
  harinezumigel-llm-stack --group rag_old --start

# Stop the group, check recent logs, or follow one backend's logs.
harinezumigel-llm-stack --group rag_old --stop
harinezumigel-llm-stack --group rag_old --show-log --tail 50
harinezumigel-llm-stack snowflake-embedder --stream-log
```

## How the components fit together

```text
Client (Copilot, RAG app, ...)
               |
               | OpenAI-compatible HTTP API
               v
+---------------------------------------------------------+
| Target host: 192.168.100.50                             |
|                                                         |
| +-----------------------------------------------------+ |
| | LiteLLM proxy (host process, LAN port 4000)         | |
| +------------+----------------------------------------+ |
|              |                                          |
|              | Local backend API requests               |
|              v                                          |
| Docker host ports bound to 127.0.0.1                    |
|   8003 -> [vLLM: Guard]                                 |
|   8001 -> [vLLM: Mistral]                               |
|   8013 -> [vLLM: Judge]                                 |
|   8014 -> [vLLM: BGE M3 embeddings]                     |
|   8015 -> [vLLM: BGE reranker]                          |
|   8005 -> [vLLM: Coder]  (alternative workload)         |
|              |                                          |
|       NVIDIA GPU / shared memory                        |
|       Model files mounted from /opt/models              |
|                                                         |
| harinezumigel-llm-stack CLI                             |
|   reads config/env + config/config.yaml                 |
|   manages Docker backends and the host LiteLLM process  |
+---------------------------------------------------------+
```

The CLI controls services; inference requests go through LiteLLM directly to a
backend. A RAG app coordinates guard, inference, and judge calls itself.
These backends are workload choices, not a requirement to run all models at
once. Groups make it easier to manage the services belonging to each workload.
The host IP and ports shown here match the configuration example; adapt them
to your deployment.

## LiteLLM October 2026 Incident

For detailed discussion of the LiteLLM October 2026 compromise,
forensic indicators, audit records, and project-specific mitigations,
see [project-specific mitigations](https://github.com/HarinezumIgel/harinezumigel-llm-stack/blob/main/security_incidents/litellm-oct-2026.md).

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

## Day-to-day commands

CLI model names and CLI aliases are interchangeable. Use a single model,
several model names, or `--group rag` to select a workload.

For example, start an embedding encoder or cross-encoder using the supplied
CLI aliases:

```bash
# Embedding encoder (BGE M3).
harinezumigel-llm-stack bge-m3-embedder --start

# Cross-encoder (BGE reranker).
harinezumigel-llm-stack bge-reranker --start

# Embedding encoder (Snowflake).
harinezumigel-llm-stack snowflake-embedder --start

# Cross-encoder (mMARCO).
harinezumigel-llm-stack mmarco-reranker --start
```

These commands start individual backends; they do not require a model group
or start LiteLLM. Use `snowflake-embedder` for Snowflake embeddings and
`mmarco-reranker` for mMARCO cross-encoding.

### List models and check status

```bash
# List configured aliases and groups.
harinezumigel-llm-stack --list

# Check one model, a group, or all configured backends and LiteLLM.
harinezumigel-llm-stack guard --ps
harinezumigel-llm-stack --group rag --ps
harinezumigel-llm-stack all --ps
```

### View and follow logs

```bash
# View recent logs for a model or every member of a group.
harinezumigel-llm-stack guard --show-log --tail 100
harinezumigel-llm-stack --group rag --show-log --tail 50

# Continuously follow one backend or the LiteLLM proxy.
harinezumigel-llm-stack guard --stream-log
harinezumigel-llm-stack litellm --stream-log

# Start one backend and follow its loading and request logs.
harinezumigel-llm-stack guard --start --stream-log

# Find Docker log paths and the proxy log directory.
harinezumigel-llm-stack all --show-log-path
```

`--stream-log` follows one target indefinitely; group commands do not support
indefinite log following. Group starts monitor backend readiness and then return.
After group/batch startup, separate log-following commands are printed for each
selected target, including LiteLLM when listed, whether started or reused.
Dry-runs do not print these reminders.
Startup summaries and log-following hints group targets by `model_info.mode`:
Inference (`inference`, `chat`, or omitted), Embeddings (`embedding`), Reranking
(`rerank`), and Proxy (LiteLLM). Other modes retain their own category. Categories
appear in first-selection order; execution order and other output are unchanged.
Backend streaming shows only the latest container run, not earlier shutdowns.
`--show-log` without following shows all saved history unless limited with
`--tail N`; neither operation deletes logs.
For a single backend, `--start --startup-logs` monitors startup until readiness
without remaining attached indefinitely.

### Stop workloads

```bash
harinezumigel-llm-stack guard --stop
harinezumigel-llm-stack --group rag --stop
harinezumigel-llm-stack all --stop
```

Stopping leaves containers in place for reuse. A group stops only its selected
members, including LiteLLM if listed. `all --stop` stops configured backends and
the managed LiteLLM proxy, not unrelated Docker containers. Stop commands verify
shutdown and report an error if the selected services remain running.

### Clean logs

```bash
# Preview before deleting diagnostic history.
harinezumigel-llm-stack --group rag --clean-log --dry-run

# Clean a model, the proxy, a group, or all configured services.
harinezumigel-llm-stack guard --clean-log
harinezumigel-llm-stack litellm --clean-log
harinezumigel-llm-stack --group rag --clean-log
harinezumigel-llm-stack all --clean-log
```

Backend cleanup truncates managed Docker log files. Proxy cleanup removes
managed `.log` files under `/opt/litellm/logs`. Group cleanup includes proxy logs
only if LiteLLM belongs to the group. Cleanup does not stop or recreate services,
delete model weights, or change configuration. Save needed diagnostics first;
running services may immediately write more logs.

### Recreate after backend startup changes

```bash
harinezumigel-llm-stack --group rag --start --recreate --dry-run
harinezumigel-llm-stack --group rag --start --recreate
```

Normal starts reuse containers and their original startup parameters.
`--start --recreate` replaces selected backend containers using current settings
and requires `rebuild: true` on every selected backend. Container-local data is
lost; mounted model files stay on the host. LiteLLM is restarted if selected.
There is no standalone container-removal option.
Proxy-only routing and API-alias changes require only a LiteLLM restart, as
shown above, not backend recreation.

Use `harinezumigel-llm-stack --help` for the full command reference.

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

## Credits

This tool would not be possible without the following open-source projects,
technologies, and the people behind them:

- [vLLM](https://github.com/vllm-project/vllm) for model serving and inference.
- [LiteLLM](https://github.com/BerriAI/litellm) for the OpenAI-compatible proxy
  and model routing.
- [NVIDIA](https://www.nvidia.com/) for GPU computing, the
  [NGC vLLM images](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm),
  and [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)
  for GPU access from containers.
- [Docker](https://www.docker.com/) for the backend container runtime.
- [PyYAML](https://github.com/yaml/pyyaml) for YAML configuration parsing.
- [Hugging Face](https://huggingface.co/) and the upstream model authors for
  model distribution, weights, tokenizers, and configurations, subject to each
  model's license and access conditions.

Thank you to their maintainers and communities. Naming these projects does not
imply affiliation, sponsorship, or endorsement.

## Important

Provided as is, without warranty. Read the
[disclaimer](https://github.com/Harinezumigel/harinezumigel-llm-stack#disclaimer)
and validate before production use. This project is not affiliated with
LLMStack (llmstack.ai / Promptly).

LiteLLM exposes an admin UI at `/ui`; do not assume API authentication hides it.
Restrict network access or block that route at a reverse proxy.
