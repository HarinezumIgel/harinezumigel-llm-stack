# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned Features
- Subcommand-based CLI (start/stop/logs as subcommands)
- JSON output mode for scripting
- Container health check command
- Shell completion scripts
- Docker Compose support

## [1.1.0] - 2026-07-07

### Added
- `model_info.license` and `model_info.upstream` fields in `config.yaml` for all models
- `ModelDeployment.license` and `ModelDeployment.upstream` dataclass fields
- `--list` output now shows `license` and `upstream` for each model
- `config_metadata` block in `config.yaml` with repository license and third-party model notice
- `config.yaml.example` expanded from 2 models to 8 configured models:
  - `llama_guard3_8b`, `llama_3_1_8b`, `mistral_7b`
  - `qwen2_5_32b_awq`, `Qwen3-Coder-Next-AWQ`, `qwen3.6-27b`
  - `deepseek-coder-6.7b-instruct`, `deepseek-v4-pro-70b`
- Full vLLM runtime parameters per model: `quantization`, `kv_cache_dtype`,
  `generation_config`, `attention_backend`, `enable_prefix_caching`,
  `enforce_eager`, `enable_auto_tool_choice`, `tool_call_parser`,
  `max_num_seqs`, `max_num_batched_tokens`, `gpu_memory_utilization`
- `top_k` added to `litellm_params` for applicable models
- `install.sh` now auto-creates the LiteLLM virtual environment (`/opt/litellm/venv`) with user consent
- `install.sh` detects an existing venv and offers upgrade instead of blind recreation
- `install.sh` offers to open both `.env` and `config.yaml` for editing at the end
- `install.sh` shows GPU name, driver version, and memory for each detected GPU
- `install.sh` verifies Docker daemon is running (not just installed)
- `install.sh` steps are numbered with clear section headers
- `README.md` dedicated **Model Licenses** section
- `SECURITY.md` with responsible-disclosure policy (replaces `SAFETY_ANALYSIS.md`)

### Changed
- `install.sh` is significantly more verbose: every check and action is narrated
- `install.sh` summary shows venv step only if it was skipped
- `install.sh` next-steps section uses bullet points and clearer wording

### Fixed
- `install.sh`: `rm -f` on symlink path now distinguishes between symlinks and regular files;
  removing a regular file requires explicit confirmation to avoid accidental data loss
- `install.sh`: `sed -i` on `.env` now uses a `.bak` backup, removed only on success,
  so an interrupted write cannot corrupt the file
- `harinezumigel-llm-stack.py`: removed dead `force=True` parameter from
  `_docker_remove_container` — passing it would have bypassed the stop-before-remove
  safety sequence
- `harinezumigel-llm-stack.py`: `clean_vllm_logs` now validates that the Docker-reported
  log path starts with `/var/lib/docker/` and ends with `.log` before running
  `sudo truncate`, preventing privilege escalation via unexpected log paths
- `config.yaml` `qwen3.6-27b` upstream URL corrected from the org page
  (`https://huggingface.co/Qwen`) to the specific model page
  (`https://huggingface.co/Qwen/Qwen3.6-27B`)

### Removed
- `SAFETY_ANALYSIS.md` replaced by `SECURITY.md`

## [1.0.0] - 2026-06-30

### Added
- Initial public release as **harinezumigel-llm-stack**
- Core functionality for managing LiteLLM and vLLM deployments
- Docker container lifecycle management (start/stop/recreate)
- Automatic and manual port allocation
- Configuration-driven deployments (.env and config.yaml)
- Dry-run mode for safe testing
- Log viewing and inspection (`--logs`, `--follow`, `--log-path`)
- Runtime parameter overrides (context length, GPU memory, max seqs)
- Container reuse to avoid unnecessary rebuilds
- Fuzzy model directory matching
- Self-PID protection in LiteLLM stop operation
- Explicit SIGTERM signal for graceful process shutdown
- Comprehensive documentation (README, CONTRIBUTING, SECURITY)
- Example configuration files (.env.example, config.yaml.example)

### Security
- API key redaction in command output
- Read-only access to model files (Docker volumes)
- Scoped container operations (only manages vllm-* containers)
- No file modification operations (read-only script)
- Explicit flags required for destructive operations

### Safety Features
- Dry-run mode for all destructive operations
- Container reuse as default behavior (no accidental recreation)
- Port conflict detection before starting containers
- Detailed information display before destructive operations
- Self-process protection when stopping LiteLLM
- Graceful shutdown signals (SIGTERM) instead of force-kill

### Documentation
- Comprehensive README.md with usage examples
- SECURITY.md with vulnerability reporting policy
- ARGS_SUGGESTIONS.md with CLI improvement proposals
- CONTRIBUTING.md for contributor guidelines
- Example configuration files with detailed comments
- Inline code documentation and docstrings

### Technical Details
- Python 3.10+ with type hints
- Dataclass-based configuration management
- Environment variable expansion (shell-style and LiteLLM-style)
- Docker container state inspection and management
- Process management with pattern-based PID detection
- Port availability checking (system and Docker)

---

## Version History

### [1.0.0] - Initial Release
First stable release with core functionality and comprehensive documentation.

---

## Upgrade Notes

### From Pre-1.0 to 1.0.0

If you were using a pre-release version:

1. **Environment file location**: Ensure `.env` is at `/opt/litellm/.env` or set `HLS_ENV_FILE`
2. **Config.yaml structure**: Verify `model_info` section includes required fields:
   - `context_length`
   - `max_input_tokens`
   - `max_output_tokens`
   - `gpu_memory_utilization`
3. **Container naming**: Old containers may need manual cleanup if naming changed
4. **API keys**: Now properly redacted in output - verify `LOCAL_VLLM_API_KEY` is set

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

---

## Links

- [Repository](https://github.com/your-username/setupllm)
- [Issues](https://github.com/your-username/setupllm/issues)
- [Discussions](https://github.com/your-username/setupllm/discussions)
