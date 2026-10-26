# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [10/26/2026 Revision 81]

### Changed
- Grouped recreation stops all running containers for selected backends before
  checking the combined new memory budget. Dry-runs leave them running and
  explicitly distinguish total-capacity previews from post-stop availability.
- GB10 memory preflight uses shared system memory (`MemAvailable` / `MemTotal`)
  when NVIDIA GPU memory counters are unavailable, for both starts and dry runs.
- The installer now backs up legacy config files instead of copying them into
  active paths, initializes missing active configs from current examples, and
  reports exact backup paths. Fresh installs clearly report when no old config
  needed backing up.
- Omitted `model_info.generation_config` now explicitly selects `vllm`, rather
  than inheriting model generation defaults through vLLM's `auto` default.
  Set `auto` explicitly to retain the previous behavior; recreate backend
  containers to apply changed startup defaults.
- Sampling defaults in the example configuration and README Mistral sample now
  live in `model_info.override_generation_config`, not `litellm_params`.
  Qwen3.5 uses the selected repetition penalty of `1.0` and an integer `top_k: -1`.
- Duplicate parameters in `litellm_params` and
  `model_info.override_generation_config` are rejected during configuration
  loading. Summaries identify backend and proxy defaults explicitly.
- `--list` now prints compact alias and model-group tables only; invocation
  without arguments retains detailed model settings.
- `--help` no longer includes the model runtime-values dump; command examples
  and usage guidance remain.

### Added
- Shared terminal styling across lifecycle commands and batch startup, including
  readiness/failure summaries; raw backend log output is left unchanged.
- Theme-aware ANSI styling for help and alias/group listings, automatically
  disabled for redirected output, unset/dumb terminals, and `NO_COLOR`.
- Multiple positional model names/aliases and ordered `model_groups` selections
  through `--group`, including status, stop, and finite log operations.
- Sequential batch startup with full selection preflight, HTTP readiness checks,
  bounded live startup logs, automatic failure diagnostics, and final summaries.
- `--startup-timeout`, `--startup-logs` for single-model opt-in, and `--quiet`;
  batch streaming conflicts fail before launch.
- Regression coverage for readiness, failure, interruption, and follower cleanup.

### Changed
- Removed CLI overrides for backend ports and vLLM runtime settings. Configure
  ports in `litellm_params.api_base` and runtime settings in `model_info` in
  `config.yaml`; request-level sampling overrides remain supported.
- Removed the CLI auto-port range settings from the example environment.
- Group resolution and bounded startup monitoring live in focused sibling modules.
  Installation checks their presence; renamed and symlinked commands resolve them
  relative to the source script.

## [10/21/2026 Revision 66]

### Changed
- `README.md`: clarified that the launcher is not limited to NVIDIA DGX Spark and documented Unix host, Docker/NVIDIA GPU passthrough, architecture, GPU, and driver compatibility requirements for the selected NVIDIA vLLM image
- `harinezumigel-llm-stack.py`: help examples, startup banner, parser usage errors, and command hints now derive the executable name from the invoked script or symlink

### Added
- Regression tests for renamed script and symlink names across help output, parser errors, startup banner, and command hints

## [10/21/2026 Revision 67]

### Added
- `install.sh`: final red reminder asking whether `<INSTALL_DIR>/.env` has been updated to match the user's LLM setup
- `harinezumigel-llm-stack.py`: support for `model_info.compilation_config` and `model_info.structured_outputs_config`, forwarded as JSON through `--compilation-config` and `--structured-outputs-config`; both settings appear in model listings, dynamic help, and startup summaries
- `config.yaml.example` and `README.md`: nested examples for compilation mode and the `xgrammar` structured-output backend, with vLLM version-compatibility guidance in the README
- Regression coverage for JSON config forwarding, installation-directory startup validation, and LiteLLM log-directory creation and cleanup
- `harinezumigel-llm-stack.py`: startup NVIDIA compatibility probe before vLLM model launch (`--start` paths), using a lightweight `docker run --gpus all` check
- `harinezumigel-llm-stack.py`: mismatch parser for NVIDIA runtime error output to report required and detected driver versions when available

### Changed
- `harinezumigel-llm-stack.py`: LiteLLM configuration is always read from `INSTALLED_BASE / "config.yaml"`; `LITELLM_CONFIG` is no longer required or used as an environment override
- `harinezumigel-llm-stack.py`: installation paths use `INSTALLED_BASE` (`/opt/litellm`), with deployment settings in `/opt/litellm/config/env`
- `harinezumigel-llm-stack.py`: all operations abort at script startup if the installation directory is missing, before environment loading
- LiteLLM logs now use only `/opt/litellm/logs`; the directory is created when needed, except during dry runs and cleanup. Cleanup checks for the directory before removing `.log` files and leaves outside files untouched; symlinks are not followed
- `install.sh`: LiteLLM setup no longer force-pins `fastapi<0.115.0` and `sse-starlette<2.0.0`; it now relies on `litellm[proxy]` dependency resolution and validates with `pip check`
- `harinezumigel-llm-stack.py`: deprecated `model_info.enable_reasoning` / `model_info.enable_thinking` startup mapping was replaced with `model_info.reasoning_config` (`--reasoning-config`), and `model_info.attention_config` (`--attention-config`) was added
- `harinezumigel-llm-stack.py`: startup NVIDIA compatibility probe is warning-only (startup continues) instead of hard-fail
- `harinezumigel-llm-stack.py`: `--recreate` authorization now reads only top-level `rebuild` per `model_list` entry (no `model_info.rebuild` fallback)
- `harinezumigel-llm-stack.py`: when top-level `rebuild` is missing, recreate is blocked by default and prints an explicit message requiring `rebuild: true`
- `README.md` and `config.yaml.example`: documented strict top-level `rebuild: true` requirement for `--recreate`
- `harinezumigel-llm-stack.py`: `stop_litellm` process matching is now scoped to commands that include `--config <LITELLM_CONFIG>` (removed broad binary-only match)
- `harinezumigel-llm-stack.py`: `start_litellm` now enforces a single-instance policy per `LITELLM_CONFIG` and refuses startup when another process uses the same config path
- `README.md`: API authentication safety wording now clarifies that auth enforcement is controlled by LiteLLM `master_key` configuration

### Fixed
- `harinezumigel-llm-stack.py`: strip whitespace-separated inline comments from `.env` values before variable expansion, preserving quoted and embedded `#` characters; this prevents false host mismatch warnings and API URLs losing their configured backend ports and falling back to port 80
- `harinezumigel-llm-stack.py`: report invalid bind hosts, permission failures, and other bind errors explicitly instead of incorrectly reporting that the port is already in use
- `install.sh`: avoided circular MCP conflict guidance where recreating the venv could reproduce the same incompatibility introduced by forced dependency pinning
- `harinezumigel-llm-stack.py`: reduced startup warning noise in recent vLLM images by removing `VLLM_LOG_LEVEL` injection and unsetting image-level `VLLM_VERSION`, `VLLM_FLASH_ATTN_SRC_DIR`, and `VLLM_LOG_LEVEL` before launching the API server

### Removed
- `BOOTSTRAP_ENV_FILE` and the unused `litellm_log_file` configuration field
- Legacy `LITELLM_LOG_FILE` configuration and fallback log-file handling; existing log files outside `/opt/litellm/logs` are no longer read or cleaned

## [1.2.0] - 2026-08-05

### Added
- `install.sh`: ask user before creating `/opt/litellm` (previously only asked when directory already existed)
- `install.sh`: ask user before copying each example config file individually
- `install.sh`: back up existing `.env` and `config.yaml` to date-and-time-stamped `.bak` files before overwriting (e.g. `.env_20260805_143022.bak`)
- `install.sh`: after installing `litellm[proxy]`, pin `fastapi<0.115.0` and `sse-starlette<2.0.0` to fix two incompatibilities with newer package versions
- `harinezumigel-llm-stack.py`: added `--stream-log` as a shortcut for `--show-log --follow`
- `harinezumigel-llm-stack.py`: `model_info.ipc` now accepts `host`, `ipc=<mode>`, or `--ipc=<mode>` and passes the normalized Docker IPC flag to `docker run`

### Changed
- `install.sh`: `.bak` filenames now include the time (`_HHMMSS`) in addition to the date, preventing collisions when backing up multiple times on the same day
- `install.sh`: directory-exists prompt now states that config files will be backed up and overwritten (not preserved)

### Fixed
- `install.sh`: when pinning LiteLLM proxy dependencies, the script now detects `mcp` in the LiteLLM venv, explains the `sse-starlette` incompatibility (`mcp>=3` vs LiteLLM `<2`), and instructs manual venv removal plus rerunning `install.sh` (no automatic deletion)
- `harinezumigel-llm-stack.py`: `--stream-log` is now mutually exclusive with `--show-log` and `--follow` to prevent contradictory log flag combinations
- `harinezumigel-llm-stack.py`: `--recreate` is now rejected when `--stop` is given
- `harinezumigel-llm-stack.py`: `--recreate` is now rejected unless both `--start` and a model/alias are given (previously only checked for `--start`)
- `harinezumigel-llm-stack.py`: `--recreate` is now rejected for the `litellm` target regardless of other flags
- `harinezumigel-llm-stack.py`: `start_litellm` command now sets `PYTHONPATH` to litellm's proxy directory before invoking the binary, fixing `ModuleNotFoundError: No module named 'proxy_server'`
- `install.sh`: litellm proxy startup failure due to `ImportError: cannot import name 'get_flat_dependant'` — caused by `fastapi>=0.115` removing the symbol; fixed by pinning `fastapi<0.115.0`
- `install.sh`: `sse-starlette` version conflict with pinned starlette resolved by adding `sse-starlette<2.0.0` to the pin step
- `harinezumigel-llm-stack.py`: `--start --show-log` was dumping hundreds of lines of logs from previous container runs; fixed by using `docker logs --since <timestamp>` (captured just before start) instead of `--tail 200`
- `harinezumigel-llm-stack.py`: `--show-log` was printing "Resolved alias" twice — fixed by passing the already-resolved model name to `show_vllm_logs` instead of the raw user input
- `harinezumigel-llm-stack.py`: `--clean-log` now prints a red warning when the container is running, explaining that `docker logs` caches the old EOF offset and a `docker restart` is needed to see new entries

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

1. **Environment file location**: Ensure deployment settings are at `/opt/litellm/config/env`
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
