# harinezumigel-llm-stack - Unified LiteLLM + vLLM Launcher

A Python CLI tool for managing [LiteLLM](https://github.com/BerriAI/litellm) proxy servers and [vLLM](https://github.com/vllm-project/vllm) model backends in Docker containers.

⚠️ **Disclaimer**: This project is provided "as is" without warranty of any kind and is not intended for production use without proper validation and testing. This project is not affiliated with LLMStack (llmstack.ai / Promptly).

> **⚠️ Security notice**: LiteLLM always serves the admin web UI at `/ui` — it cannot be disabled via configuration. Restrict access to the LiteLLM port at the network or reverse proxy level if you do not want the UI reachable.

## Overview

`harinezumigel-llm-stack` provides a unified interface to orchestrate multiple language model deployments using:

- **LiteLLM**: Universal API proxy that provides an OpenAI-compatible interface to multiple LLM providers
- **vLLM**: High-throughput, memory-efficient inference engine for LLMs

The script manages the entire lifecycle of your LLM infrastructure:
- Start/stop LiteLLM proxy server
- Launch vLLM model backends in Docker containers
- Automatic port management and container reuse
- Runtime parameter overrides
- Log viewing and monitoring

## Features

- **Configuration-driven**: All settings from `.env` and `config.yaml` (no hardcoded values)
- **Model aliases**: Define short aliases for models (e.g., `coder` for `Qwen3-Coder-Next-AWQ`)
- **Dataclass-based configuration**: `LLMStack` class manages all state and lifecycle operations
- **Dry-run mode**: Explicit flags for destructive operations, container reuse by default
- **Port management**: Manual or automatic port allocation, runtime parameter overrides
- **Log viewing**: JSON stacktrace formatting, container status inspection
- **Log formatting**: Expands JSON-escaped stack traces to readable multi-line output
- **Docker-based**: Uses Docker for isolation, GPU passthrough, and resource management
- **NGC vLLM image**: Uses NVIDIA NGC container image

## Prerequisites

- Python 3.10+ with `pyyaml` package
- Docker with NVIDIA GPU support
- `nvidia-smi` available (for GPU detection)
- LiteLLM installed in a virtual environment (for proxy server)
- Pre-downloaded model files in a designated directory

### Quick Install

Run the provided installation script:

```bash
cd scripts/github_deploy
./install.sh
```

The installer will:
- Check Python, Docker, and GPU availability
- Install PyYAML (system package preferred)
- Create `/opt/litellm` directory structure
- Copy example configuration files
- Create a symlink to `~/.local/bin/harinezumigel-llm-stack`
- Guide you through setup steps

### Manual Installation

**PyYAML:**
```bash
# Ubuntu/Debian (recommended)
sudo apt install python3-yaml

# Fedora/RHEL
sudo dnf install python3-pyyaml

# Or via pip if needed
pip3 install --user pyyaml
```

**LiteLLM:**
```bash
python3 -m venv /opt/litellm/venv
source /opt/litellm/venv/bin/activate
pip install 'litellm[proxy]'
```

## Installation

### Automated Installation (Recommended)

```bash
cd scripts/github_deploy
./install.sh
```

### Manual Installation

1. **Clone this repository**:
   ```bash
   git clone <your-repo-url>
   cd <repo-directory>
   ```

2. **Create directory structure**:
   ```bash
   sudo mkdir -p /opt/litellm
   sudo chown $USER:$USER /opt/litellm
   ```

3. **Copy example configs**:
   ```bash
   cp .env.example /opt/litellm/.env
   cp config.yaml.example /opt/litellm/config.yaml
   ```

4. **Create symlink**:
   ```bash
   mkdir -p ~/.local/bin
   ln -s $(pwd)/harinezumigel-llm-stack.py ~/.local/bin/harinezumigel-llm-stack
   chmod +x harinezumigel-llm-stack.py
   ```

5. **Add to PATH** (if needed):
   ```bash
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
   source ~/.bashrc
   ```

## Configuration

### 1. Environment File (`/opt/litellm/.env`)

Create a `.env` file with deployment settings:

```bash
# Core paths
LITELLM_CONFIG=/opt/litellm/config.yaml
MODEL_ROOT=/opt/models

LITELLM_VENV_ACTIVATE=/opt/litellm/venv/bin/activate
LITELLM_BIN=/opt/litellm/venv/bin/litellm

# LiteLLM proxy bind
LITELLM_BIND_HOST=0.0.0.0
LITELLM_PORT=4000

# Shared vLLM host / bind
VLLM_HOST=localhost
VLLM_BIND_HOST=0.0.0.0
VLLM_CONTAINER_PORT=8000

# Docker runtime image and mounts
# Use NVIDIA NGC vLLM image for best GPU support
VLLM_DOCKER_IMAGE=nvcr.io/nvidia/vllm:26.05.post1-py3
VLLM_MODEL_VOLUME=/opt/models:/models
VLLM_CACHE_VOLUME=/opt/vllm-cache:/root/.cache/huggingface

# Optional auto-port range
VLLM_AUTO_PORT_START=8001
VLLM_AUTO_PORT_END=8010

# API keys
LITELLM_MASTER_KEY=sk-change-me
LOCAL_VLLM_API_KEY=sk-change-me

# Backend ports (one per model)
LLAMA_GUARD3_8B_PORT=8001
QWEN3_CODER_NEXT_PORT=8002

# Full API bases (use variable expansion)
LLAMA_GUARD3_8B_API_BASE=http://${VLLM_HOST}:${LLAMA_GUARD3_8B_PORT}/v1
QWEN3_CODER_3_NEXT_API_BASE=http://${VLLM_HOST}:${QWEN3_CODER_NEXT_PORT}/v1
```

### 2. LiteLLM Config (`/opt/litellm/config.yaml`)

Define your model deployments:

```yaml
model_list:
  # Safety / moderation model
  - model_name: llama_guard3_8b
    litellm_params:
      litellm_provider: openai
      model: openai/llama_guard3_8b
      api_base: os.environ/LLAMA_GUARD3_8B_API_BASE
      api_key: os.environ/LOCAL_VLLM_API_KEY
      request_timeout: 60
      temperature: 0.0
      top_p: 1.0
    model_info:
      model_dir: llama-guard-3-8b
      context_length: 2048
      max_input_tokens: 1536
      max_output_tokens: 64
      max_tokens: 64
      gpu_memory_utilization: 0.20
      max_num_seqs: 8
      max_num_batched_tokens: 2048
      dtype: auto
      description: "Llama Guard 3 8B safety/moderation model"

  # Primary coding / agentic model
  - model_name: Qwen3-Coder-Next-AWQ
    litellm_params:
      litellm_provider: openai
      model: openai/Qwen3-Coder-Next-AWQ
      api_base: os.environ/QWEN3_CODER_3_NEXT_API_BASE
      api_key: os.environ/LOCAL_VLLM_API_KEY
      request_timeout: 240
      temperature: 0.2
      top_p: 0.9
      top_k: 20
    model_info:
      alias: coder  # Optional: short alias for command-line use
      model_dir: Qwen3-Coder-Next-AWQ
      context_length: 262144
      max_input_tokens: 250000
      max_output_tokens: 4096
      max_tokens: 4096
      gpu_memory_utilization: 0.90
      max_num_seqs: 1
      max_num_batched_tokens: 8192
      quantization: compressed-tensors
      kv_cache_dtype: fp8
      generation_config: vllm
      enable_auto_tool_choice: true
      tool_call_parser: qwen3_coder
      enable_prefix_caching: true
      attention_backend: flashinfer
      enforce_eager: false
      dtype: auto
      description: "Qwen3-Coder-Next AWQ big-context profile"

router_settings:
  routing_strategy: simple-shuffle

litellm_settings:
  drop_params: true
  set_verbose: false
  request_timeout: 240
  json_logs: true
  turn_off_message_logging: true
  redact_user_api_key_info: true

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
```

## Usage

### List Configured Models

```bash
harinezumigel-llm-stack --list
```

This displays all configured models with their settings, including any defined aliases.

### Start a Model Backend

```bash
# Start with default settings (reuses existing container if available)
harinezumigel-llm-stack llama_guard3_8b --start

# Start using an alias (if configured in model_info.alias)
harinezumigel-llm-stack coder --start

# Start with explicit port (works with both name and alias)
harinezumigel-llm-stack Qwen3-Coder-Next-AWQ --start --port 8003
harinezumigel-llm-stack coder --start --port 8003  # same, using alias

# Start with automatic port allocation
harinezumigel-llm-stack llama_guard3_8b --start --auto-port

# Recreate container (removes and rebuilds)
harinezumigel-llm-stack Qwen3-Coder-Next-AWQ --start --recreate
harinezumigel-llm-stack coder --start --recreate  # same, using alias
```

### Override Runtime Parameters

```bash
# Override context length and GPU memory (works with alias)
harinezumigel-llm-stack Qwen3-Coder-Next-AWQ --start \
  --context-length 131072 \
  --max-input-tokens 125000 \
  --max-output-tokens 4096 \
  --gpu-memory-utilization 0.85

# Same using alias
harinezumigel-llm-stack coder --start \
  --context-length 131072 \
  --gpu-memory-utilization 0.85

# Limit concurrent requests
harinezumigel-llm-stack llama_guard3_8b --start --max-num-seqs 4
```

### Dry Run Mode

Preview changes without executing:

```bash
harinezumigel-llm-stack llama_guard3_8b --start --recreate --dry-run
```

### Start LiteLLM Proxy

```bash
# Start in background
harinezumigel-llm-stack litellm --start

# Start and follow logs (foreground, with formatted JSON stacktraces)
harinezumigel-llm-stack litellm --start --show-log

# Stop the proxy
harinezumigel-llm-stack litellm --stop
```

The proxy will be available at `http://localhost:4000` (or your configured port).

**Admin UI**: The `/ui` route is always served by LiteLLM — it cannot be disabled via configuration. All API calls require `LITELLM_MASTER_KEY`. To block the UI, use a reverse proxy (e.g. nginx `location /ui { return 403; }`) or a firewall rule.

**Note**: When using `--show-log`, JSON log lines with `stacktrace` fields are automatically formatted to display stack traces with real newlines for readability.

### View Logs

```bash
# View recent logs (last 200 lines, works with alias)
harinezumigel-llm-stack Qwen3-Coder-Next-AWQ --show-log
harinezumigel-llm-stack coder --show-log  # same, using alias

# Follow logs in real-time
harinezumigel-llm-stack llama_guard3_8b --show-log --follow
harinezumigel-llm-stack coder --show-log --follow  # using alias

# Show more log lines
harinezumigel-llm-stack Qwen3-Coder-Next-AWQ --show-log --tail 500

# Show log file path
harinezumigel-llm-stack llama_guard3_8b --show-log-path

# Show all log paths
harinezumigel-llm-stack all --show-log-path

# Clean/truncate log files
harinezumigel-llm-stack all --clean-log
```

**Log Formatting**: The tool automatically formats JSON log output from LiteLLM and vLLM. Stack traces embedded in JSON (with escaped `\n`) are expanded to real newlines for readability.

### Stop Containers

```bash
# Stop a specific model (works with alias)
harinezumigel-llm-stack Qwen3-Coder-Next-AWQ --stop
harinezumigel-llm-stack coder --stop  # same, using alias

# Stop all configured models
harinezumigel-llm-stack all --stop

# Stop LiteLLM proxy
harinezumigel-llm-stack litellm --stop
```

## Model Directory

Each model must have `model_info.model_dir` set explicitly in `config.yaml`.
The value must exactly match a directory name inside `MODEL_ROOT` (set in `.env`).

```yaml
model_info:
  model_dir: Qwen3-Coder-Next-AWQ   # must exist as $MODEL_ROOT/Qwen3-Coder-Next-AWQ
```

The path resolved at runtime is:

```
$MODEL_ROOT / model_dir
```

If `model_dir` is missing from the config or the directory does not exist, the
start command will fail with an error before any Docker operation is attempted.

**Example layout:**
```
/opt/models/
  llama-guard3-8b/
  llama3.1-8b/
  mistral-7b-instruct-v0.3/
  qwen2.5-32b-awq/
  Qwen3-Coder-Next-AWQ/
  Qwen3.6-27B/
  deepseek-coder-6.7b-instruct/
  deepseek-v4-pro-70b/
```

Corresponding `config.yaml` entries:
```yaml
model_info:
  model_dir: llama-guard3-8b          # llama_guard3_8b
  model_dir: llama3.1-8b              # llama_3_1_8b
  model_dir: mistral-7b-instruct-v0.3 # mistral_7b
  model_dir: qwen2.5-32b-awq          # qwen2_5_32b_awq
  model_dir: Qwen3-Coder-Next-AWQ     # Qwen3-Coder-Next-AWQ
  model_dir: Qwen3.6-27B              # qwen3.6-27b
  model_dir: deepseek-coder-6.7b-instruct  # deepseek-coder-6.7b-instruct
  model_dir: deepseek-v4-pro-70b      # deepseek-v4-pro-70b
```

## Model Aliases

You can define short, memorable aliases for models in `config.yaml` to simplify command-line usage. Aliases are optional and defined in the `model_info` section:

```yaml
model_list:
  - model_name: Qwen3-Coder-Next-AWQ
    litellm_params:
      # ... your litellm params ...
    model_info:
      alias: coder  # Optional short alias
      model_dir: Qwen3-Coder-Next-AWQ
      # ... rest of model_info ...

  - model_name: llama_guard3_8b
    litellm_params:
      # ... your litellm params ...
    model_info:
      alias: guard  # Optional short alias
      model_dir: llama-guard3-8b
      # ... rest of model_info ...
```

**Using Aliases:**

All commands that accept a model name also accept aliases:

```bash
# These are equivalent:
harinezumigel-llm-stack Qwen3-Coder-Next-AWQ --start
harinezumigel-llm-stack coder --start

# Works with all operations:
harinezumigel-llm-stack coder --show-log --follow
harinezumigel-llm-stack guard --ps
harinezumigel-llm-stack coder --stop
```

**Alias Rules:**

- Aliases must be unique across all models
- Aliases are case-sensitive
- The `--list` command shows both model names and their aliases
- When an alias is used, the tool displays which model it resolved to

**API Behavior with Aliases:**

If you start a backend using an alias, LiteLLM exposes that alias as the model identifier for callers. In practice, clients must send the alias in API requests.

```bash
# If alias is "coder", callers should use:
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-..." \
  -H "Content-Type: application/json" \
  -d '{"model":"coder","messages":[{"role":"user","content":"Hello"}]}'
```

**Example Output:**

```bash
$ harinezumigel-llm-stack coder --start
Resolved alias 'coder' to model 'Qwen3-Coder-Next-AWQ'
Starting vLLM container for model: Qwen3-Coder-Next-AWQ
...
```

## Safety Features

- **No model modification**: Does not alter model files on disk
- **Explicit operations**: Container removal only occurs with `--recreate`
- **Dry-run mode**: Preview all changes with `--dry-run`
- **Container reuse**: Default behavior reuses existing containers
- **Scoped operations**: Only manages containers with `vllm-` prefix
- **Process safety**: Kills only LiteLLM processes matching specific patterns
- **Port validation**: Checks port availability before binding
- **Self-protection**: Won't kill its own process when stopping LiteLLM
- **API authentication**: All LiteLLM API calls require `LITELLM_MASTER_KEY`; the `/ui` static files load but are non-functional without auth

See [SECURITY.md](SECURITY.md) for vulnerability reporting and security scope.

⚠️ **Important**:
Do NOT store models or any non-reproducible data inside containers.
Using `--recreate` removes containers, and any internal container data will be permanently lost.
Always use host-mounted directories or Docker volumes for persistence.

## Container Lifecycle

### Default Behavior (Safe)

```bash
harinezumigel-llm-stack mistral-7b
```

1. Checks if container `vllm-mistral-7b-8001` exists
2. If exists and running → reuses it (no change)
3. If exists but stopped → starts it
4. If doesn't exist → creates new container

### Recreate Behavior

```bash
harinezumigel-llm-stack mistral-7b --recreate
```

1. Stops existing container (if running)
2. Removes container
3. Creates fresh container with current settings

Use `--recreate` when:
- Changing Docker run parameters
- Updating vLLM image version
- Changing model directory
- Troubleshooting container issues

## Advanced Examples

### Deployment Script

```bash
#!/bin/bash
# Start all models using aliases for cleaner commands

harinezumigel-llm-stack guard --start --recreate
harinezumigel-llm-stack coder --start --port 8002 --recreate

# Start LiteLLM proxy
harinezumigel-llm-stack litellm --start

echo "All models started. LiteLLM available at http://localhost:4000"
```

### Health Check Script

```bash
#!/bin/bash
# Check model container status using aliases

for model in guard coder; do
    echo "=== $model ==="
    harinezumigel-llm-stack $model --show-log --tail 10
    echo
done

# Or check all at once
harinezumigel-llm-stack all --ps
```

### Quick Start/Stop with Aliases

```bash
#!/bin/bash
# Quick commands using short aliases

# Start models
harinezumigel-llm-stack guard --start
harinezumigel-llm-stack coder --start --gpu-memory-utilization 0.85

# Check status
harinezumigel-llm-stack guard --ps
harinezumigel-llm-stack coder --ps

# View logs
harinezumigel-llm-stack coder --show-log --follow &

# Stop when done
harinezumigel-llm-stack all --stop
```

## Troubleshooting

### Container won't start

```bash
# Check if port is in use
netstat -tuln | grep 8001

# View container logs (can use alias)
harinezumigel-llm-stack guard --show-log --tail 100
harinezumigel-llm-stack coder --show-log --tail 100

# Recreate with dry-run to see command
harinezumigel-llm-stack guard --start --recreate --dry-run

# Force recreate (can use alias)
harinezumigel-llm-stack coder --start --recreate
harinezumigel-llm-stack llama_guard3_8b --start --recreate
```

### API key issues

Ensure `LOCAL_VLLM_API_KEY` is set in `.env` and referenced in `config.yaml`:

```yaml
litellm_params:
  api_key: os.environ/LOCAL_VLLM_API_KEY
```

### Model directory not found

Set explicit path in `config.yaml`:

```yaml
model_info:
  model_dir: exact-directory-name
```

### Out of memory

Reduce GPU memory utilization or concurrent requests:

```bash
harinezumigel-llm-stack Qwen3-Coder-Next-AWQ --start \
  --gpu-memory-utilization 0.70 \
  --max-num-seqs 1 \
  --recreate
```

### Unreadable log output / stack traces

The tool automatically formats JSON logs with embedded stack traces. If you see escaped `\n` sequences:

1. Use the tool's `--show-log` command (formatting is automatic)
2. For direct Docker logs: `docker logs vllm-<container>` won't have formatting
3. LiteLLM logs with `--show-log` flag expand stacktraces for readability

## Environment Variable Override

Override the default `.env` location:

```bash
export HLS_ENV_FILE=/custom/path/.env
harinezumigel-llm-stack --list
```

## Architecture

```
┌─────────────────────────────────────┐
│  harinezumigel-llm-stack CLI        │
│  (LLMStack class manages all state) │
└───────────────┬─────────────────────┘
                │
    ┌───────────┴───────────┐
    │                       │
    ▼                       ▼
┌──────────────────┐    ┌────────────────────────────┐
│  LiteLLM Proxy   │    │   vLLM Containers          │
│  (Host Process)  │    │   (Docker + NVIDIA GPU)    │
│                  │    │                            │
│  Port: 4000      │◄───┤  llama_guard3_8b: 8001     │
│                  │    │  Qwen3-Coder-Next: 8002    │
│  Unified API     │    │  (NGC vLLM image)          │
└─────────┬────────┘    └──────────┬─────────────────┘
          │                        │
          ▼                        ▼
    OpenAI API              Model Files
    Compatible              (/models volume)
```

### Code Structure

The tool is built around a single `LLMStack` class that manages:

- **Configuration**: Loads `.env` and `config.yaml`
- **Model management**: Resolves model names, finds directories, builds Docker commands
- **Container lifecycle**: Start, stop, recreate, reuse existing containers
- **Logs**: View, follow, and clean container logs (with JSON stacktrace formatting)
- **LiteLLM proxy**: Start/stop the unified API gateway

Pure utility functions remain at module level for string manipulation, type coercion, and subprocess execution.

## Performance Tuning

### Context Length vs. Memory

Larger context windows require more GPU memory:

- 262K context (Qwen3-Coder-Next): ~0.90 GPU memory, fp8 kv_cache, FlashInfer backend
- 32K context (typical): ~0.85-0.90 GPU memory utilization
- 16K context: ~0.75-0.85 GPU memory utilization
- 2K context (Llama Guard): ~0.20 GPU memory (small safety model)

### Concurrent Requests

`max_num_seqs` controls maximum concurrent requests:

- Higher values → better throughput, more memory
- Lower values → lower memory, potential queuing
- Recommended: 1 for very large context models, 2-8 for smaller models

### Advanced vLLM Features

For large context models (>100K tokens):

```yaml
model_info:
  attention_backend: flashinfer  # Required for very large context
  kv_cache_dtype: fp8            # Compress KV cache
  quantization: compressed-tensors  # AWQ/GPTQ models
  enable_prefix_caching: true    # Cache system prompts
```

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests if applicable
4. Update documentation
5. Submit a pull request

## Dependencies

This project requires the following Python package:

- **[PyYAML](https://pyyaml.org/)** - YAML 1.1 parser and emitter for Python (MIT License)

Install via:
```bash
# Ubuntu/Debian
sudo apt install python3-yaml

# Fedora/RHEL
sudo dnf install python3-pyyaml

# Or via pip
pip install pyyaml
```

## License

[MIT License](LICENSE) - feel free to use in your projects.

## Acknowledgments

Built on these open-source projects:

- **[PyYAML](https://github.com/yaml/pyyaml)** - YAML parser for Python (MIT License)
- **[LiteLLM](https://github.com/BerriAI/litellm)** - LLM proxy with OpenAI-compatible API
- **[vLLM](https://github.com/vllm-project/vllm)** - High-throughput inference engine
- **[Docker](https://www.docker.com/)** - Containerization platform

## Support

For issues and questions:

- 🐛 [Open an issue](https://github.com/Harinezumigel/harinezumigel-llm-stack/issues)
- 📖 [Read the docs](https://github.com/Harinezumigel/harinezumigel-llm-stack/wiki)
- 💬 [Discussions](https://github.com/Harinezumigel/harinezumigel-llm-stack/discussions)

---

**Note**: This tool is designed for managing local LLM deployments. For production use at scale, consider additional monitoring, health checks, and orchestration tools (Kubernetes, Docker Compose, etc.).

## ⚠️ Disclaimer

**USE AT YOUR OWN RISK**

**Always validate in your own environment before relying on this in any critical system.**


This software is provided "as is" without warranty of any kind, express or implied. The authors and contributors are not responsible for:

- **Resource Usage**: GPU memory allocation, disk space consumption, network bandwidth, or compute costs
- **System Stability**: Docker container failures, process crashes, or system resource exhaustion
- **Data Loss**: Configuration corruption, container data loss, or log file issues
- **Security**: Exposed APIs, authentication vulnerabilities, or unauthorized access to deployed models
- **Model Behavior**: LLM outputs, model hallucinations, biases, or harmful content generation
- **Production Deployments**: Service downtime, performance degradation, or breaking changes

**Important Notes:**
- This tool manages Docker containers with GPU access and can consume significant system resources
- LLM deployments may expose API endpoints on your network - ensure proper firewall and authentication
- Always test in a non-production environment first
- Review all configurations before deploying models
- Monitor resource usage and costs when using cloud infrastructure
- **Comply with model licenses and usage restrictions**

## Model Licenses

This configuration references third-party models. Model licenses, usage rights, and restrictions are determined by the upstream model publishers. Consult the `upstream` field in `config.yaml` (or the output of `harinezumigel-llm-stack --list`) for the current license and terms applicable to each model.

By using this software, you accept full responsibility for all consequences of its deployment and operation.
