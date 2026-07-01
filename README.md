# harinezumigel-llm-stack - Unified LiteLLM + vLLM Launcher

A Python CLI tool for managing [LiteLLM](https://github.com/BerriAI/litellm) proxy servers and [vLLM](https://github.com/vllm-project/vllm) model backends in Docker containers.

⚠️ **Disclaimer**: This project is provided "as is" without warranty of any kind and is not intended for production use without proper validation and testing. This project is not affiliated with LLMStack (llmstack.ai / Promptly).

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

✨ **Key Features**:

- **Configuration-driven**: All settings from `.env` and `config.yaml` (no hardcoded values)
- **Defensive**: Dry-run mode, explicit flags for destructive operations, container reuse
- **Flexible deployment**: Manual or automatic port allocation, runtime parameter overrides
- **Easy monitoring**: Built-in log viewing, container status inspection
- **Docker-native**: Leverages Docker for isolation, GPU passthrough, and resource management

## Prerequisites

- Python 3.10+ with `pyyaml` package
- Docker with NVIDIA Container Toolkit (for GPU support)
- LiteLLM installation (for proxy server)
- Pre-downloaded model files in a designated directory

### Installing PyYAML

**Ubuntu/Debian:**
```bash
sudo apt install python3-yaml
```

**Fedora/RHEL:**
```bash
sudo dnf install python3-pyyaml
```

**Using pip (if system packages unavailable):**
```bash
# Create a virtual environment (recommended)
python3 -m venv ~/setupllm-venv
source ~/setupllm-venv/bin/activate
pip install pyyaml

# Or use pipx for isolated installation
pipx install pyyaml
```

## Installation

1. **Clone this repository**:
   ```bash
   git clone <your-repo-url>
   cd <repo-directory>
   ```

2. **Install Python dependencies**:
   ```bash
   pip install pyyaml
   ```

3. **Ensure Docker is running**:
   ```bash
   docker --version
   nvidia-docker --version  # or docker with nvidia runtime
   ```

4. **Set up directory structure**:
   ```bash
   sudo mkdir -p /opt/litellm
   sudo chown $USER:$USER /opt/litellm
   ```

## Configuration

### 1. Environment File (`/opt/litellm/.env`)

Create a `.env` file with deployment settings:

```bash
# LiteLLM Configuration
LITELLM_CONFIG=/opt/litellm/config.yaml
LITELLM_VENV_ACTIVATE=/path/to/venv/bin/activate
LITELLM_BIN=/path/to/venv/bin/litellm
LITELLM_BIND_HOST=0.0.0.0
LITELLM_PORT=4000

# Model Storage
MODEL_ROOT=/path/to/models

# vLLM Configuration
VLLM_HOST=localhost
VLLM_BIND_HOST=0.0.0.0
VLLM_CONTAINER_PORT=8000
VLLM_DOCKER_IMAGE=vllm/vllm-openai:latest

# Docker Volumes
VLLM_MODEL_VOLUME=/path/to/models:/models:ro
VLLM_CACHE_VOLUME=/path/to/cache:/root/.cache/huggingface:rw

# Port Range for Auto-allocation
VLLM_AUTO_PORT_START=8001
VLLM_AUTO_PORT_END=8010

# Optional: API Key for vLLM endpoints
LOCAL_VLLM_API_KEY=your-secret-key-here
```

### 2. LiteLLM Config (`/opt/litellm/config.yaml`)

Define your model deployments:

```yaml
model_list:
  - model_name: mistral-7b
    litellm_params:
      model: openai/mistral-7b
      api_base: http://localhost:8001/v1
      api_key: os.environ/LOCAL_VLLM_API_KEY
    model_info:
      context_length: 32768
      max_input_tokens: 28672
      max_output_tokens: 4096
      gpu_memory_utilization: 0.90
      max_num_seqs: 4
      enable_prefix_caching: true

  - model_name: qwen-coder
    litellm_params:
      model: openai/qwen-coder
      api_base: http://localhost:8002/v1
      api_key: os.environ/LOCAL_VLLM_API_KEY
    model_info:
      context_length: 32768
      max_input_tokens: 28672
      max_output_tokens: 4096
      gpu_memory_utilization: 0.85
      max_num_seqs: 2
      model_dir: Qwen2.5-Coder-32B-Instruct-AWQ
      enable_auto_tool_choice: true
      tool_call_parser: llama3_json
```

## Usage

### List Configured Models

```bash
harinezumigel-llm-stack --list
```

### Start a Model Backend

```bash
# Start with default settings (reuses existing container if available)
harinezumigel-llm-stack mistral-7b

# Start with explicit port
harinezumigel-llm-stack qwen-coder --port 8003

# Start with automatic port allocation
harinezumigel-llm-stack qwen2_5_32b_awq --auto-port

# Recreate container (removes and rebuilds)
harinezumigel-llm-stack mistral-7b --recreate
```

### Override Runtime Parameters

```bash
# Override context length and GPU memory
harinezumigel-llm-stack qwen-coder \
  --context-length 16384 \
  --max-input-tokens 14336 \
  --max-output-tokens 2048 \
  --gpu-memory-utilization 0.80

# Limit concurrent requests
harinezumigel-llm-stack mistral-7b --max-num-seqs 1
```

### Dry Run Mode

Preview changes without executing:

```bash
harinezumigel-llm-stack mistral-7b --recreate --dry-run
```

### Start LiteLLM Proxy

```bash
harinezumigel-llm-stack --litellm
```

The proxy will be available at `http://localhost:4000` (or your configured port).

### View Logs

```bash
# View recent logs
harinezumigel-llm-stack qwen-coder --logs

# Follow logs in real-time
harinezumigel-llm-stack mistral-7b --logs --follow

# Show more log lines
harinezumigel-llm-stack qwen-coder --logs --tail 500

# Show log file path
harinezumigel-llm-stack mistral-7b --log-path
```

### Stop Containers

```bash
# Stop a specific model
harinezumigel-llm-stack --stop qwen-coder

# Stop all configured models
harinezumigel-llm-stack --stop all

# Stop LiteLLM proxy
harinezumigel-llm-stack --stop-litellm
```

## Model Directory Discovery

The script automatically finds model directories using fuzzy matching:

1. **Explicit path**: Set `model_info.model_dir` in config.yaml
2. **Fuzzy matching**: Matches directory names against model name
3. **Token matching**: Finds directories containing model name tokens

Example:
- Model name: `qwen-2.5-32b-awq`
- Matches directory: `Qwen2.5-32B-Instruct-AWQ` ✓

## Safety Features


🛡️ **Built-in Safety**:

- **No model modification**: Does not alter model files on disk
- **Explicit operations**: Container removal only occurs with `--recreate`
- **Dry-run mode**: Preview all changes with `--dry-run`
- **Container reuse**: Default behavior reuses existing containers
- **Scoped operations**: Only manages containers with `vllm-` prefix
- **Process safety**: Kills only LiteLLM processes matching specific patterns
- **Port validation**: Checks port availability before binding
- **Self-protection**: Won't kill its own process when stopping LiteLLM

See [SAFETY_ANALYSIS.md](SAFETY_ANALYSIS.md) for detailed security analysis.

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
# Start all production models

harinezumigel-llm-stack mistral-7b --recreate
harinezumigel-llm-stack qwen-coder --port 8002 --recreate
harinezumigel-llm-stack llama3-70b --port 8003 --max-num-seqs 2

# Start LiteLLM proxy
harinezumigel-llm-stack --litellm

echo "All models started. LiteLLM available at http://localhost:4000"
```

### Health Check Script

```bash
#!/bin/bash
# Check model container status

for model in mistral-7b qwen-coder llama3-70b; do
    echo "=== $model ==="
    harinezumigel-llm-stack $model --logs --tail 10
    echo
done
```

## Troubleshooting

### Container won't start

```bash
# Check if port is in use
netstat -tuln | grep 8001

# View container logs
harinezumigel-llm-stack MODEL_NAME --logs --tail 100

# Recreate with dry-run to see command
harinezumigel-llm-stack MODEL_NAME --recreate --dry-run

# Force recreate
harinezumigel-llm-stack MODEL_NAME --recreate
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
harinezumigel-llm-stack MODEL_NAME \
  --gpu-memory-utilization 0.70 \
  --max-num-seqs 1 \
  --recreate
```

## Environment Variable Override

Override the default `.env` location:

```bash
export HLS_ENV_FILE=/custom/path/.env
harinezumigel-llm-stack --list
```

## Architecture

```
┌───────────────────────────────┐
│  harinezumigel-llm-stack CLI  │
└───────────────┬───────────────┘
                │
    ┌───────────┴───────────┐
    │                       │
    ▼                       ▼
┌──────────────────┐    ┌──────────────────────┐
│  LiteLLM Proxy   │    │   vLLM Containers    │
│  (Host Process)  │    │   (Docker + NVIDIA)  │
│                  │    │                      │
│  Port: 4000      │◄───┤  mistral-7b: 8001    │
│                  │    │  qwen-coder: 8002    │
│  Unified API     │    │  llama3-70b: 8003    │
└─────────┬────────┘    └──────────┬───────────┘
          │                        │
          ▼                        ▼
    OpenAI API              Model Files
    Compatible              (/models volume)
```

## Performance Tuning

### Context Length vs. Memory

Larger context windows require more GPU memory:

- 32K context: ~0.85-0.90 GPU memory utilization
- 16K context: ~0.75-0.85 GPU memory utilization
- 8K context: ~0.65-0.75 GPU memory utilization

### Concurrent Requests

`max_num_seqs` controls maximum concurrent requests:

- Higher values → better throughput, more memory
- Lower values → lower memory, potential queuing
- Recommended: 2-8 for smaller models, 1 for large models

### Prefix Caching

Enable for models with repetitive prompts (system messages):

```yaml
model_info:
  enable_prefix_caching: true
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

Built with these excellent open-source projects:

- **[PyYAML](https://github.com/yaml/pyyaml)** - YAML parser for Python (MIT License)
- **[LiteLLM](https://github.com/BerriAI/litellm)** - Universal LLM proxy with OpenAI-compatible API
- **[vLLM](https://github.com/vllm-project/vllm)** - High-throughput, memory-efficient inference engine
- **[Docker](https://www.docker.com/)** - Containerization platform for isolated deployments

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
- Comply with model licenses and usage restrictions

By using this software, you accept full responsibility for all consequences of its deployment and operation.
