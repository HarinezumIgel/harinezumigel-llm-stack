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
- Comprehensive documentation (README, CONTRIBUTING, SAFETY_ANALYSIS)
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
- SAFETY_ANALYSIS.md detailing security review
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
