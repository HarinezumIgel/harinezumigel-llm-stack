#!/usr/bin/env python3
# pylint: disable=too-many-lines,too-few-public-methods
"""harinezumigel-llm-stack - Unified LiteLLM + vLLM launcher.

This script provides a unified interface for managing LiteLLM proxy servers and
vLLM model backends running in Docker containers.

DISCLAIMER: This project is not affiliated with LLMStack (llmstack.ai / Promptly)

## Overview

The script reads configuration from two sources:
- Deployment settings: /opt/litellm/.env (or HLS_ENV_FILE)
- Model runtime settings: /opt/litellm/config.yaml (LITELLM_CONFIG)

It manages:
1. LiteLLM proxy server (native Python process)
2. vLLM model backends (Docker containers)

## Features

- Start/stop LiteLLM proxy and vLLM model containers
- Automatic port allocation or manual port specification
- Container reuse to avoid unnecessary rebuilds
- Dry-run mode for safe testing
- Log viewing and inspection
- Runtime parameter overrides (context length, GPU memory, etc.)

## Safety

- Only manages containers with 'vllm-' prefix
- Respects --dry-run flag for all destructive operations
- Requires explicit --recreate flag to remove containers
  ! Do NOT store models or any non-reproducible data inside containers.
  ! Using `--recreate` removes containers, and any internal container data will be permanently lost.
  ! Always use host-mounted directories or Docker volumes for persistence.
- Shows detailed information before making changes
- Does not modify model files

## Usage Examples

    # List configured models (shows model names and aliases)
    harinezumigel-llm-stack --list

    # Start a model (reuses existing container if available)
    harinezumigel-llm-stack mistral_7b --start

    # Start using an alias (if configured in model_info.alias)
    harinezumigel-llm-stack coder --start

    # Start a model AND follow logs (stays attached)
    harinezumigel-llm-stack mistral_7b --start --log --follow

    # Start with explicit port (works with both name and alias)
    harinezumigel-llm-stack qwen-coder --start --port 8003

    # Recreate container with new settings
    harinezumigel-llm-stack qwen2_5_32b_awq --start --recreate --max-num-seqs 1

    # Preview changes without executing
    harinezumigel-llm-stack mistral_7b --start --dry-run

    # Start LiteLLM proxy
    harinezumigel-llm-stack litellm --start

    # Stop LiteLLM proxy
    harinezumigel-llm-stack litellm --stop

    # View logs (container must be running, works with alias)
    harinezumigel-llm-stack qwen-coder --show-log
    harinezumigel-llm-stack coder --show-log --follow  # using alias

    # Show log file paths
    harinezumigel-llm-stack qwen-coder --show-log-path
    harinezumigel-llm-stack all --show-log-path
    harinezumigel-llm-stack all --show-log  # shows paths for 'all'

    # Clean log files (truncate to free space)
    harinezumigel-llm-stack qwen-coder --clean-log
    harinezumigel-llm-stack all --clean-log

    # Check container status (works with alias)
    harinezumigel-llm-stack qwen-coder --ps
    harinezumigel-llm-stack coder --ps  # using alias
    harinezumigel-llm-stack all --ps

    # Stop containers (works with alias)
    harinezumigel-llm-stack qwen-coder --stop
    harinezumigel-llm-stack coder --stop  # using alias
    harinezumigel-llm-stack all --stop
The script does not hardcode model ports, bind hosts, Docker image names, or
model paths. All values come from .env and config.yaml files.

See README.md for configuration details.

Author: harinezumigel-llm-stack contributors
License: MIT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import ParseResult, urlparse

import yaml


# Default environment file path (can be overridden via HLS_ENV_FILE)
BOOTSTRAP_ENV_FILE = os.environ.get("HLS_ENV_FILE", "/opt/litellm/.env")


# ---------------------------------------------------------------------------
# Environment loading
# ---------------------------------------------------------------------------
# These functions safely load and parse environment variables from .env files
# and provide type-safe access with validation.


def load_env_file(path: str) -> None:
    """Load a shell-style KEY=value env file into os.environ.

    Supported forms:
      KEY=value
      export KEY=value
      KEY=${OTHER_KEY}
      KEY=$OTHER_KEY

    Values from the env file override already-set process variables. This keeps
    /opt/litellm/.env authoritative for this launcher.

    Safety: This function only reads files and sets environment variables.
    It does not execute shell commands or modify files.
    """
    env_path = Path(path)

    if not env_path.exists():
        return

    loaded: dict[str, str] = {}

    with env_path.open("r", encoding="utf-8") as file_handle:
        for raw_line in file_handle:
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            if line.startswith("export "):
                line = line[len("export ") :].strip()

            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            # Build environment for variable expansion (current env + newly loaded)
            expansion_env = dict(os.environ)
            expansion_env.update(loaded)

            def make_replacer(current_env: dict[str, str]):
                """Create a replacer bound to the current environment snapshot."""

                def replace_var(match: re.Match[str]) -> str:
                    """Expand one shell-style environment variable reference."""
                    var_name = match.group(1) or match.group(2)
                    return current_env.get(var_name, "")

                return replace_var

            # Expand shell-style variable references: ${VAR} or $VAR
            value = re.sub(
                r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)",
                make_replacer(expansion_env),
                value,
            )

            loaded[key] = value

    # Apply all loaded variables to the process environment
    for key, value in loaded.items():
        os.environ[key] = value


def env_string(name: str, default: str | None = None, *, required: bool = False) -> str:
    """Read a string environment variable."""
    value = os.environ.get(name, default)

    if required and not value:
        print(f"ERROR: Required environment variable is not set: {name}")
        sys.exit(1)

    return value or ""


def env_int(name: str, default: str | None = None, *, required: bool = False) -> int:
    """Read an integer environment variable."""
    value = env_string(name, default, required=required)

    try:
        return int(value)
    except (TypeError, ValueError):
        print(f"ERROR: Environment variable {name} must be an integer.")
        print(f"Current value: {value}")
        sys.exit(1)


# Load environment variables early (before AppConfig initialization)
load_env_file(BOOTSTRAP_ENV_FILE)


@dataclass(frozen=True)
class AppConfig:  # pylint: disable=too-many-instance-attributes
    """Application-wide settings read from .env.

    This dataclass holds all deployment configuration such as paths,
    ports, and Docker settings. Values are loaded from environment
    variables set by load_env_file().
    """

    env_file: str
    litellm_config: str
    model_root: str
    litellm_venv_activate: str
    litellm_bin: str
    litellm_bind_host: str
    litellm_port: int
    vllm_host: str
    vllm_bind_host: str
    vllm_container_port: int
    vllm_docker_image: str
    vllm_model_volume: str
    vllm_cache_volume: str
    vllm_auto_port_start: int
    vllm_auto_port_end: int

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Build AppConfig from environment variables."""
        return cls(
            env_file=BOOTSTRAP_ENV_FILE,
            litellm_config=env_string("LITELLM_CONFIG", required=True),
            model_root=env_string("MODEL_ROOT", required=True),
            litellm_venv_activate=env_string("LITELLM_VENV_ACTIVATE", required=True),
            litellm_bin=env_string("LITELLM_BIN", required=True),
            litellm_bind_host=env_string("LITELLM_BIND_HOST", required=True),
            litellm_port=env_int("LITELLM_PORT", required=True),
            vllm_host=env_string("VLLM_HOST", required=True),
            vllm_bind_host=env_string("VLLM_BIND_HOST", required=True),
            vllm_container_port=env_int("VLLM_CONTAINER_PORT", required=True),
            vllm_docker_image=env_string("VLLM_DOCKER_IMAGE", required=True),
            vllm_model_volume=env_string("VLLM_MODEL_VOLUME", required=True),
            vllm_cache_volume=env_string("VLLM_CACHE_VOLUME", required=True),
            vllm_auto_port_start=env_int("VLLM_AUTO_PORT_START", default="8001"),
            vllm_auto_port_end=env_int("VLLM_AUTO_PORT_END", default="8010"),
        )


# Global application configuration (initialized from environment)
APP = AppConfig.from_env()


# ---------------------------------------------------------------------------
# Pure utility functions
# ---------------------------------------------------------------------------
# These functions carry no instance state and have no dependency on AppConfig
# or model data. They remain at module level.


def resolve_env_value(value: Any) -> Any:
    """Resolve LiteLLM-style os.environ/VAR and shell-style env refs.

    Handles:
    - os.environ/VAR_NAME (LiteLLM convention)
    - $VAR or ${VAR} (shell-style)
    """
    if value is None:
        return ""

    if not isinstance(value, str):
        return value

    value = os.path.expandvars(value)

    if value.startswith("os.environ/"):
        env_name = value.split("/", 1)[1]
        return os.environ.get(env_name, "")

    return value


def docker_safe_name(name: str) -> str:
    """Make a Docker-safe container-name fragment."""
    name = name.lower()
    name = re.sub(r"[^a-z0-9_.-]+", "-", name)
    name = re.sub(r"-+", "-", name)
    return name.strip("-")


def backend_model_without_provider(model: str) -> str:
    """Strip a LiteLLM provider prefix such as openai/foo."""
    if not model:
        return ""

    if "/" in model:
        return model.split("/", 1)[1]

    return model


def as_int(value: Any, default: int | None = None) -> int | None:
    """Safely cast a value to int."""
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def as_float(value: Any, default: float | None = None) -> float | None:
    """Safely cast a value to float."""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_bool(value: Any, default: bool = False) -> bool:
    """Safely cast common config values to bool."""
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")

    return bool(value)


def port_in_use(port: int, host: str) -> bool:
    """Return True if host:port cannot be bound.

    Attempts to bind to the port to check availability.
    This is a local system check (not Docker-specific).
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        sock.bind((host, port))
        return False
    except OSError:
        return True
    finally:
        sock.close()


def run_command(
    command: list[str],
    *,
    capture: bool = True,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess command with consistent defaults."""
    return subprocess.run(
        command,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=True,
        check=check,
    )


def start_background_process(command: list[str]) -> None:
    """Start a long-running background process.

    A context manager is intentionally not used here. The child process should
    continue running after this CLI script exits.

    Safety: This creates detached processes (servers, containers) that persist
    after the script completes. This is the intended behavior.
    """
    # pylint: disable=consider-using-with
    subprocess.Popen(command)


def _format_log_line(line: str) -> str:
    """Expand escaped newlines in a JSON log line's stacktrace field.

    LiteLLM json_logs mode serialises multi-line stack traces as a single JSON
    string with \\n escape sequences.  This function detects that pattern and
    emits the stacktrace separately so it is readable in the terminal.
    """
    try:
        obj = json.loads(line)
        stacktrace = obj.get("stacktrace")
        if stacktrace:
            without_trace = {k: v for k, v in obj.items() if k != "stacktrace"}
            return json.dumps(without_trace) + "\n" + stacktrace
        return line
    except (json.JSONDecodeError, ValueError):
        return line


def _stream_formatted_logs(command: list[str]) -> int:
    """Run *command* and stream its merged stdout/stderr with log formatting."""
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ) as proc:
        assert proc.stdout is not None
        for raw in proc.stdout:
            print(_format_log_line(raw.rstrip("\n")), flush=True)
        proc.wait()
    return proc.returncode


def redact_command(command: list[str]) -> list[str]:
    """Redact secret values in a printed command.

    Replaces values following --api-key, --token, or --password flags
    with ***REDACTED*** for safe logging.

    Security: Prevents accidental exposure of API keys in logs.
    """
    safe_command: list[str] = []
    redact_next = False

    for part in command:
        if redact_next:
            safe_command.append("***REDACTED***")
            redact_next = False
            continue

        safe_command.append(part)

        if part in ("--api-key", "--token", "--password"):
            redact_next = True

    return safe_command


def require_model_int(model_name: str, model_info: dict[str, Any], key: str) -> int:
    """Require an integer model_info field."""
    value = as_int(model_info.get(key), None)

    if value is None:
        print(f"ERROR: model_info.{key} missing or invalid for {model_name}")
        sys.exit(1)

    return value


def require_model_float(model_name: str, model_info: dict[str, Any], key: str) -> float:
    """Require a float model_info field."""
    value = as_float(model_info.get(key), None)

    if value is None:
        print(f"ERROR: model_info.{key} missing or invalid for {model_name}")
        sys.exit(1)

    return value


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ModelDeployment:  # pylint: disable=too-many-instance-attributes
    """A model deployment loaded from LiteLLM config.yaml.

    Represents a single model's deployment configuration including
    API endpoints, resource limits, and runtime parameters.
    """

    name: str
    backend_model: str
    api_base: str
    api_key: str
    api_base_host: str
    api_base_port: int | None
    model_info: dict[str, Any]
    context_length: int
    max_input_tokens: int
    max_output_tokens: int
    gpu_memory_utilization: float
    max_num_seqs: int | None
    dtype: str
    license: str | None
    upstream: str | None
    alias: str | None


@dataclass(frozen=True)
class StartOptions:
    """Options for starting or recreating a vLLM backend."""

    port: int
    runtime: dict[str, int | float | str | None]
    dry_run: bool
    reuse_existing: bool
    recreate: bool


# ---------------------------------------------------------------------------
# LLMStack
# ---------------------------------------------------------------------------


class LLMStack:
    """Unified manager for LiteLLM proxy and vLLM Docker backends.

    Holds all deployment configuration and model data, and exposes methods for
    the full lifecycle of both LiteLLM and vLLM: starting, stopping, log
    viewing, and status inspection.

    Typical usage::

        stack = LLMStack(APP)
        stack.run(parse_args())
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.models: dict[str, ModelDeployment] = {}

    # ------------------------------------------------------------------
    # Configuration loading
    # ------------------------------------------------------------------

    def _parse_api_base(self, api_base: str) -> tuple[str, int]:
        """Extract host and port from an api_base URL."""
        resolved = resolve_env_value(api_base)
        parsed = cast(ParseResult, urlparse(resolved))

        host: str = parsed.hostname or self.config.vllm_host

        if parsed.port:
            port: int = parsed.port
        elif parsed.scheme == "https":
            port = 443
        else:
            port = 80

        return host, port

    def load_models(self) -> None:
        """Load model deployments from config.yaml into self.models.

        Parses the LiteLLM configuration file and extracts model deployment
        information including API endpoints, context lengths, and resource
        limits.

        Raises:
            SystemExit: If config file not found or required fields missing.
        """
        config_path = Path(self.config.litellm_config)

        if not config_path.exists():
            print(f"ERROR: LiteLLM config not found: {self.config.litellm_config}")
            sys.exit(1)

        with config_path.open("r", encoding="utf-8") as fh:
            cfg: dict[str, Any] = yaml.safe_load(fh) or {}

        deployments: dict[str, ModelDeployment] = {}

        for entry in cfg.get("model_list", []):
            model_name = entry.get("model_name")
            params = entry.get("litellm_params", {})
            model_info = entry.get("model_info", {})

            if not model_name:
                continue

            api_base = resolve_env_value(params.get("api_base", ""))
            api_key = resolve_env_value(params.get("api_key", ""))
            backend_model = resolve_env_value(params.get("model", ""))
            license_info = model_info.get("license")
            upstream = model_info.get("upstream")
            alias = model_info.get("alias") or None

            if api_base:
                api_base_host, api_base_port = self._parse_api_base(api_base)
            else:
                api_base_host, api_base_port = self.config.vllm_host, None

            deployments[model_name] = ModelDeployment(
                name=model_name,
                backend_model=backend_model,
                api_base=api_base,
                api_key=api_key,
                api_base_host=api_base_host,
                api_base_port=api_base_port,
                model_info=model_info,
                context_length=require_model_int(model_name, model_info, "context_length"),
                max_input_tokens=require_model_int(model_name, model_info, "max_input_tokens"),
                max_output_tokens=require_model_int(model_name, model_info, "max_output_tokens"),
                gpu_memory_utilization=require_model_float(
                    model_name, model_info, "gpu_memory_utilization"
                ),
                max_num_seqs=as_int(model_info.get("max_num_seqs"), None),
                dtype=str(model_info.get("dtype", "auto")),
                license=license_info,
                upstream=upstream,
                alias=alias,
            )

        # Validate alias uniqueness
        seen_aliases: dict[str, str] = {}
        for model_name, deployment in deployments.items():
            if deployment.alias:
                if deployment.alias in seen_aliases:
                    print(
                        f"ERROR: Duplicate alias '{deployment.alias}' on models "
                        f"'{seen_aliases[deployment.alias]}' and '{model_name}'."
                    )
                    sys.exit(1)
                seen_aliases[deployment.alias] = model_name

        self.models = deployments

    # ------------------------------------------------------------------
    # Internal model helpers
    # ------------------------------------------------------------------

    def _resolve_model_name(self, user_input: str) -> str | None:
        """Resolve user input to a configured model name (exact match only).

        Checks both model_name and model_info.alias.
        """
        for model_name, model in self.models.items():
            if model_name == user_input:
                return model_name
            if model.alias and model.alias == user_input:
                print(f"Resolved alias '{user_input}' to model '{model_name}'")
                return model_name

        return None

    def _find_model_dir(self, model: ModelDeployment, *, dry_run: bool = False) -> Path:
        """Find the local model directory for a deployment.

        Requires model_info.model_dir to be set explicitly in config.yaml.

        Returns:
            Path to the model directory

        Raises:
            RuntimeError: If model_dir is not configured or the directory does not exist

        Safety: This function only reads directories, never modifies them.
        """
        explicit_model_dir = model.model_info.get("model_dir")

        if not explicit_model_dir:
            raise RuntimeError(
                f"model_info.model_dir is not set for model '{model.name}'. "
                f"Add model_dir to the model_info section in config.yaml."
            )

        explicit_path = Path(self.config.model_root) / explicit_model_dir

        if explicit_path.is_dir():
            return explicit_path

        raise RuntimeError(
            f"Model directory does not exist for '{model.name}': {explicit_path}"
        )

    def _find_free_port(self) -> int:
        """Find a free vLLM port in the configured auto-port range.

        Returns:
            First available port number.

        Raises:
            RuntimeError: If no free ports available in range.
        """
        for p in range(self.config.vllm_auto_port_start, self.config.vllm_auto_port_end + 1):
            if not port_in_use(p, self.config.vllm_bind_host) and not self._docker_port_in_use(p):
                return p

        raise RuntimeError(
            f"No free ports available in range "
            f"{self.config.vllm_auto_port_start}-{self.config.vllm_auto_port_end}"
        )

    # ------------------------------------------------------------------
    # Docker helpers
    # ------------------------------------------------------------------

    def _docker_container_state(self, name: str) -> str | None:
        """Return Docker container state or None if not found."""
        result = run_command(
            ["docker", "inspect", "-f", "{{.State.Status}}", name],
            capture=True,
        )

        if result.returncode != 0:
            return None

        return result.stdout.strip() or None

    def _docker_port_in_use(self, port: int) -> bool:
        """Return True if a Docker container already publishes the port."""
        result = run_command(["docker", "ps", "--format", "{{.Ports}}"], capture=True)
        output = result.stdout
        host = self.config.vllm_bind_host

        return (
            f":{port}->" in output
            or f"{host}:{port}->" in output
            or f"0.0.0.0:{port}->" in output
        )

    def _docker_list_container_names(self, *, all_containers: bool) -> list[str]:
        """List Docker container names."""
        command = ["docker", "ps", "--format", "{{.Names}}"]

        if all_containers:
            command = ["docker", "ps", "-a", "--format", "{{.Names}}"]

        result = run_command(command, capture=True)

        if result.returncode != 0:
            print("ERROR: Failed to query Docker containers")
            print(result.stderr.strip())
            sys.exit(1)

        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    def _docker_start_container(self, name: str, *, dry_run: bool = False) -> None:
        """Start an existing Docker container."""
        print(f"Starting existing Docker container: {name}")

        if dry_run:
            print("Dry run only. Container was not started.")
            return

        result = run_command(["docker", "start", name], capture=True)

        if result.returncode != 0:
            print(f"ERROR: Failed to start existing container: {name}")
            print(result.stderr.strip())
            sys.exit(result.returncode)

        print(result.stdout.strip())

    def _docker_stop_container(self, name: str, *, dry_run: bool = False) -> None:
        """Stop a Docker container gracefully."""
        print(f"Stopping Docker container: {name}")

        if dry_run:
            print("Dry run only. Container was not stopped.")
            return

        result = run_command(["docker", "stop", name], capture=True)

        if result.returncode != 0:
            print(f"ERROR: Failed to stop container: {name}")
            print(result.stderr.strip())
            sys.exit(result.returncode)

        print(result.stdout.strip())

    def _docker_remove_container(
        self,
        name: str,
        *,
        dry_run: bool = False,
    ) -> None:
        """Remove a Docker container.

        Safety: containers must be stopped before calling this. Respects dry_run flag.
        """
        command = ["docker", "rm", name]

        print(f"Removing Docker container: {name}")

        if dry_run:
            print("Command:", " ".join(command))
            print("Dry run only. Container was not removed.")
            return

        result = run_command(command, capture=True)

        if result.returncode != 0:
            print(f"ERROR: Failed to remove container: {name}")
            print(result.stderr.strip())
            sys.exit(result.returncode)

        print(result.stdout.strip())

    def _docker_container_log_path(self, name: str) -> str | None:
        """Return Docker container log file path."""
        result = run_command(
            ["docker", "inspect", "-f", "{{.LogPath}}", name],
            capture=True,
        )

        if result.returncode != 0:
            return None

        return result.stdout.strip() or None

    def _find_all_vllm_containers(self, model_name: str) -> list[str]:
        """Find all vLLM container names for a model (running and stopped)."""
        prefix = f"vllm-{docker_safe_name(model_name)}-"
        return [n for n in self._docker_list_container_names(all_containers=True) if n.startswith(prefix)]

    def _find_running_vllm_containers(self, model_name: str) -> list[str]:
        """Find running vLLM container names for a model."""
        prefix = f"vllm-{docker_safe_name(model_name)}-"
        return [n for n in self._docker_list_container_names(all_containers=False) if n.startswith(prefix)]

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _print_container_reuse_notice(self, container_name: str, model_name: str, port: int) -> None:
        """Print a notice for reused containers."""
        print()
        print("=== Reusing existing vLLM container ===")
        print(f"Container: {container_name}")
        print(f"Model:     {model_name}")
        print(f"Endpoint:  http://{self.config.vllm_bind_host}:{port}/v1")
        print()
        print("Important:")
        print("  Existing containers keep original Docker run settings.")
        print("  Use --recreate to rebuild the container with current settings.")
        print()

    def show_container_status(self, model_name: str) -> None:
        """Show container status for a specific model."""
        containers = self._find_all_vllm_containers(model_name)

        if not containers:
            print(f"\nModel: {model_name}")
            print("Status: No container found")
            print(f"\nStart with: harinezumigel-llm-stack {model_name} --start")
            return

        print(f"\nModel: {model_name}")
        print("Containers:")

        for container in containers:
            state = self._docker_container_state(container)
            port_str = container.split("-")[-1] if "-" in container else "unknown"
            symbol = "✓" if state == "running" else "✗"

            print(f"  {symbol} {container}")
            print(f"    State: {state or 'unknown'}")

            if state == "running":
                print(f"    Endpoint: http://{self.config.vllm_bind_host}:{port_str}/v1")

            print()

    def show_all_container_status(self) -> None:
        """Show status for all configured models."""
        print("\n=== Container Status ===\n")

        running_count = 0
        stopped_count = 0
        total = 0

        for model_name in sorted(self.models.keys()):
            for container in self._find_all_vllm_containers(model_name):
                state = self._docker_container_state(container)
                port_str = container.split("-")[-1] if "-" in container else "unknown"
                total += 1

                if state == "running":
                    running_count += 1
                    endpoint = f"http://{self.config.vllm_bind_host}:{port_str}/v1"
                    print(f"✓ {model_name:<30} running    port {port_str}    {endpoint}")
                else:
                    stopped_count += 1
                    print(f"✗ {model_name:<30} {state or 'unknown':<10} port {port_str}")

        print()
        print(f"Total: {total} containers ({running_count} running, {stopped_count} stopped)")
        print()

    def list_models(self) -> None:
        """Print configured model deployments."""
        print("\nModels from LiteLLM config:")
        print("---------------------------")

        interesting_keys = [
            "model_dir",
            "quantization",
            "kv_cache_dtype",
            "generation_config",
            "max_num_batched_tokens",
            "max_num_seqs",
            "attention_backend",
            "enable_prefix_caching",
            "enforce_eager",
            "enable_auto_tool_choice",
            "tool_call_parser",
            "trust_remote_code",
        ]

        for model in self.models.values():
            raw_info = model.model_info
            backend_served = backend_model_without_provider(model.backend_model)

            print(f"{model.name}")
            if model.alias:
                print(f"  alias:                    {model.alias}")
            print(f"  api_base:                 {model.api_base}")
            print(f"  api_base_host:            {model.api_base_host}")
            print(f"  api_base_port:            {model.api_base_port}")
            print(f"  vllm_host:                {self.config.vllm_host}")
            print(f"  vllm_bind_host:           {self.config.vllm_bind_host}")
            print(f"  backend_model:            {model.backend_model}")
            print(f"  backend_served_model:     {backend_served}")
            print(f"  context_length:           {model.context_length}")
            print(f"  max_input_tokens:         {model.max_input_tokens}")
            print(f"  max_output_tokens:        {model.max_output_tokens}")
            print(f"  gpu_memory_utilization:   {model.gpu_memory_utilization}")
            print(f"  max_num_seqs:             {model.max_num_seqs or '(vLLM default)'}")

            for key in interesting_keys:
                if key in raw_info:
                    print(f"  {key:<25} {raw_info.get(key)}")

            if model.license:
                print(f"  license:                  {model.license}")
            if model.upstream:
                print(f"  upstream:                 {model.upstream}")

            if model.backend_model and backend_served != model.name:
                print("  WARNING: backend served model does not match model_name")
                print(f"  Recommended: model: openai/{model.name}")

            print()

        aliases = [(m.alias, m.name) for m in self.models.values() if m.alias]
        if aliases:
            col = max(len(a) for a, _ in aliases)
            print("Aliases")
            print("-------")
            for alias, model_name in aliases:
                print(f"  {alias:<{col}}  →  {model_name}")
            print()

    def generate_help(self) -> str:
        """Generate dynamic help text."""
        keys = [
            "model_dir", "quantization", "kv_cache_dtype", "generation_config",
            "max_num_batched_tokens", "max_num_seqs", "attention_backend",
            "enable_prefix_caching", "enforce_eager", "enable_auto_tool_choice",
            "tool_call_parser", "license", "upstream",
        ]

        model_lines: list[str] = []

        for model in self.models.values():
            extras = [
                f"{k}={model.model_info.get(k)}"
                for k in keys
                if k in model.model_info
            ]
            if model.license:
                extras.append(f"license={model.license}")
            if model.upstream:
                extras.append(f"upstream={model.upstream}")

            # Show alias if present
            name_display = model.name
            if model.alias:
                name_display = f"{model.name} (alias: {model.alias})"

            model_lines.append(
                "    "
                f"{name_display:<30} "
                f"port={str(model.api_base_port):<5} "
                f"context={str(model.context_length):<6} "
                f"input={str(model.max_input_tokens):<6} "
                f"output={str(model.max_output_tokens):<6} "
                f"gpu_mem={model.gpu_memory_utilization} "
                f"max_num_seqs={model.max_num_seqs or '(default)'} "
                f"api_base={model.api_base} "
                f"{' '.join(extras)}"
            )

        model_list = "\n".join(model_lines)

        return f"""
harinezumigel-llm-stack — Unified LiteLLM + vLLM Launcher
-------------------------------------------

Config:
  {self.config.litellm_config}

Env:
  {self.config.env_file}

Important:
  Runtime values come from .env and config.yaml.
  Model aliases can be defined in config.yaml (model_info.alias) for shorter commands.

Examples:
  Show available (configured) models:
  -----------------------------------

  harinezumigel-llm-stack --list

  Start models (model name or alias):
  -----------------------------------

  harinezumigel-llm-stack mistral_7b --start                # start using model name
    harinezumigel-llm-stack coder --start                     # start using alias (if configured)
    harinezumigel-llm-stack mistral_7b --start --show-log    # start and follow logs (stays attached)
    harinezumigel-llm-stack mistral_7b --start --dry-run      # preview what would be done
    harinezumigel-llm-stack mistral_7b --start --recreate     # force recreate container

  View logs:
  ----------

  harinezumigel-llm-stack mistral_7b --show-log             # show last 200 lines
  harinezumigel-llm-stack mistral_7b --show-log --follow    # follow logs (container must be running)

  Show log paths:
  ---------------

  harinezumigel-llm-stack mistral_7b --show-log-path        # show Docker log file path
  harinezumigel-llm-stack all --show-log-path               # show log paths for all models
  harinezumigel-llm-stack all --show-log                    # shows paths for all models

  Clean logs:
  -----------

  harinezumigel-llm-stack mistral_7b --clean-log            # truncate log file for one model
  harinezumigel-llm-stack all --clean-log                   # truncate log files for all models
  Stop models:
  ------------

  harinezumigel-llm-stack mistral_7b --stop
  harinezumigel-llm-stack all --stop

  Override runtime settings:
  --------------------------

  harinezumigel-llm-stack mistral_7b --start --max-num-seqs 1 --dry-run
  harinezumigel-llm-stack mistral_7b --start --context-length 32768 --max-input-tokens 28672 --max-output-tokens 4096 --max-num-seqs 1 --dry-run

  LiteLLM proxy:
  --------------

  harinezumigel-llm-stack litellm --start
  harinezumigel-llm-stack litellm --stop

Current model values from LiteLLM config:
{model_list}
"""

    # ------------------------------------------------------------------
    # Log operations
    # ------------------------------------------------------------------

    def show_container_logs(
        self,
        container_name: str,
        *,
        tail: str = "200",
        follow: bool = False,
        show_log_path: bool = False,
    ) -> None:
        """Show Docker logs for one container."""
        state = self._docker_container_state(container_name)

        print()
        print("=== Docker logs ===")
        print(f"Container: {container_name}")
        print(f"State:     {state or 'unknown'}")

        if show_log_path:
            log_path = self._docker_container_log_path(container_name)
            print(f"Log path:  {log_path or '(not available)'}")

        print()

        command = ["docker", "logs", "--tail", str(tail)]

        if follow:
            command.append("-f")

        command.append(container_name)

        returncode = _stream_formatted_logs(command)

        if returncode != 0:
            print(f"ERROR: Failed to read logs for container: {container_name}")
            sys.exit(returncode)

    def show_vllm_logs(
        self,
        target: str,
        *,
        tail: str,
        follow: bool,
        show_log_path: bool,
    ) -> None:
        """Show logs for one model or all models."""
        if target == "all":
            containers: list[str] = []

            for model_name in self.models:
                containers.extend(self._find_all_vllm_containers(model_name))

            containers = sorted(set(containers))
        else:
            resolved = self._resolve_model_name(target)

            if resolved is None:
                print(f"ERROR: Unknown model '{target}'")
                print(self.generate_help())
                sys.exit(1)

            containers = self._find_all_vllm_containers(resolved)

        if not containers:
            print("No vLLM containers found.")
            return

        if follow and len(containers) > 1:
            print("ERROR: --follow can only be used when exactly one container matches.")

            for container in containers:
                print(f"  {container}")

            sys.exit(1)

        for container in containers:
            self.show_container_logs(
                container,
                tail=tail,
                follow=follow,
                show_log_path=show_log_path,
            )

    def show_vllm_log_paths(self, target: str) -> None:
        """Show Docker log file paths."""
        if target == "all":
            containers: list[str] = []

            for model_name in self.models:
                containers.extend(self._find_all_vllm_containers(model_name))

            containers = sorted(set(containers))
        else:
            resolved = self._resolve_model_name(target)

            if resolved is None:
                print(f"ERROR: Unknown model '{target}'")
                print(self.generate_help())
                sys.exit(1)

            containers = self._find_all_vllm_containers(resolved)

        if not containers:
            print("No vLLM containers found.")
            return

        print()
        print("=== Docker log paths ===")

        for container in containers:
            state = self._docker_container_state(container)
            path = self._docker_container_log_path(container)

            print(f"{container}")
            print(f"  state:    {state or 'unknown'}")
            print(f"  log_path: {path or '(not available)'}")
            print()

    def clean_vllm_logs(self, target: str, *, dry_run: bool = False) -> None:
        """Clean/truncate Docker log files for vLLM containers.

        Safety:
            - Requires sudo permissions to truncate Docker log files
            - Only affects containers for models in configuration
            - Respects dry_run flag
        """
        if target == "all":
            containers: list[str] = []

            for model_name in self.models:
                containers.extend(self._find_all_vllm_containers(model_name))

            containers = sorted(set(containers))
        else:
            resolved = self._resolve_model_name(target)

            if resolved is None:
                print(f"ERROR: Unknown model '{target}'")
                print(self.generate_help())
                sys.exit(1)

            containers = self._find_all_vllm_containers(resolved)

        if not containers:
            print("No vLLM containers found.")
            return

        print()
        print("=== Cleaning Docker logs ===")

        for container in containers:
            log_path = self._docker_container_log_path(container)

            if not log_path:
                print(f"{container}: No log path found")
                continue

            print(f"{container}")
            print(f"  Log path: {log_path}")

            if dry_run:
                print("  Action:   Would truncate (dry run)")
            else:
                # Validate path is a Docker-managed log file before running sudo truncate
                if not (log_path.startswith("/var/lib/docker/") and log_path.endswith(".log")):
                    print(f"  Status:   ✗ Skipped — unexpected log path: {log_path}")
                    continue

                result = run_command(["sudo", "truncate", "-s", "0", log_path], capture=True)

                if result.returncode == 0:
                    print("  Status:   ✓ Cleaned")
                else:
                    print(f"  Status:   ✗ Failed - {result.stderr.strip()}")

            print()

    # ------------------------------------------------------------------
    # vLLM lifecycle
    # ------------------------------------------------------------------

    def _build_vllm_command(
        self,
        model: ModelDeployment,
        model_dir: Path,
        port: int,
        runtime: dict[str, int | float | str | None],
    ) -> list[str]:
        """Build the Docker command for a vLLM backend."""
        raw_info = model.model_info

        command = [
            "docker", "run", "-d",
            "--name", f"vllm-{docker_safe_name(model.name)}-{port}",
            "--runtime=nvidia",
            "--gpus=all",
            "--ipc=host",
            "-p", f"{self.config.vllm_bind_host}:{port}:{self.config.vllm_container_port}",
            "-v", self.config.vllm_model_volume,
            "-v", self.config.vllm_cache_volume,
            "-v", "/etc/timezone:/etc/timezone:ro",
            "-v", "/etc/localtime:/etc/localtime:ro",
            "-e", "VLLM_CONFIGURE_LOGGING=1",
            "-e", "VLLM_LOG_LEVEL=INFO",
            self.config.vllm_docker_image,
            "python3", "-u", "-m", "vllm.entrypoints.openai.api_server",
            "--model", f"/models/{model_dir.name}",
            "--served-model-name", model.name,
            "--dtype", str(runtime.get("dtype", "auto")),
            "--max-model-len", str(runtime["context_length"]),
            "--gpu-memory-utilization", str(runtime["gpu_memory_utilization"]),
            "--host", "0.0.0.0",
        ]

        max_num_batched_tokens = raw_info.get("max_num_batched_tokens")

        if max_num_batched_tokens is not None:
            command.extend(["--max-num-batched-tokens", str(max_num_batched_tokens)])

        max_num_seqs = runtime.get("max_num_seqs")

        if max_num_seqs is not None:
            command.extend(["--max-num-seqs", str(max_num_seqs)])

        if as_bool(raw_info.get("enforce_eager", False)):
            command.append("--enforce-eager")

        if as_bool(raw_info.get("trust_remote_code", False)):
            command.append("--trust-remote-code")

        if as_bool(raw_info.get("enable_prefix_caching", False)):
            command.append("--enable-prefix-caching")

        if raw_info.get("attention_backend"):
            command.extend(["--attention-backend", str(raw_info["attention_backend"])])

        if raw_info.get("generation_config"):
            command.extend(["--generation-config", str(raw_info["generation_config"])])

        if raw_info.get("quantization"):
            command.extend(["--quantization", str(raw_info["quantization"])])

        if raw_info.get("kv_cache_dtype"):
            command.extend(["--kv-cache-dtype", str(raw_info["kv_cache_dtype"])])

        if as_bool(raw_info.get("enable_auto_tool_choice", False)):
            command.append("--enable-auto-tool-choice")

        if raw_info.get("tool_call_parser"):
            command.extend(["--tool-call-parser", str(raw_info["tool_call_parser"])])

        api_key = model.api_key or os.environ.get("LOCAL_VLLM_API_KEY", "")

        if api_key:
            command.extend(["--api-key", api_key])

        return command

    def _print_vllm_start_summary(
        self,
        model: ModelDeployment,
        model_dir: Path,
        port: int,
        runtime: dict[str, int | float | str | None],
        command: list[str],
    ) -> None:
        """Print a readable vLLM startup summary."""
        raw_info = model.model_info

        print(f"=== Starting vLLM model: {model.name} ===")
        print(f"Container:                vllm-{docker_safe_name(model.name)}-{port}")
        print(f"Model directory:           {model_dir}")
        print(f"Directory basename:        {model_dir.name}")
        print(f"Served model name:         {model.name}")
        print(f"License:                   {model.license}")
        print(f"Upstream:                  {model.upstream}")
        print(
            f"Bind:                      "
            f"{self.config.vllm_bind_host}:{port} -> "
            f"container:{self.config.vllm_container_port}"
        )
        print(f"Context length:            {runtime['context_length']}")
        print(f"Max input tokens:          {runtime['max_input_tokens']}")
        print(f"Max output tokens:         {runtime['max_output_tokens']}")
        print(f"GPU memory utilization:    {runtime['gpu_memory_utilization']}")
        print(f"Max num batched tokens:    {raw_info.get('max_num_batched_tokens', '(vLLM default)')}")
        print(f"Max num seqs:              {runtime.get('max_num_seqs', '(vLLM default)')}")
        print(f"Data type:                 {runtime.get('dtype', 'auto')}")
        print(f"Enforce eager:             {as_bool(raw_info.get('enforce_eager', False))}")
        print(f"Trust remote code:         {as_bool(raw_info.get('trust_remote_code', False))}")
        print(f"Enable prefix caching:     {as_bool(raw_info.get('enable_prefix_caching', False))}")
        print(f"Attention backend:         {raw_info.get('attention_backend', '(default)')}")
        print(f"Generation config:         {raw_info.get('generation_config', '(model default)')}")
        print(f"Auto tool choice:          {as_bool(raw_info.get('enable_auto_tool_choice', False))}")
        print(f"Tool call parser:          {raw_info.get('tool_call_parser', '(none)')}")
        print(f"Quantization:              {raw_info.get('quantization', '(none)')}")
        print(f"KV cache dtype:            {raw_info.get('kv_cache_dtype', '(auto)')}")
        print()
        print("Docker run command:")
        print("  " + " ".join(redact_command(command)))

    def start_vllm(self, model: ModelDeployment, options: StartOptions) -> None:
        """Start a Dockerized vLLM backend.

        Handles checking for existing containers, reusing or recreating them,
        port conflict detection, and building and executing the docker run
        command.

        Safety:
        - Checks for existing containers before creating new ones
        - Respects --recreate flag (explicit user intent required)
        - Default behavior: reuse existing containers (no data loss)
        - Validates port availability before binding
        - Respects dry_run flag throughout
        """
        try:
            model_dir = self._find_model_dir(model, dry_run=options.dry_run)
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)
        container_name = f"vllm-{docker_safe_name(model.name)}-{options.port}"

        if model.api_base_host and model.api_base_host != self.config.vllm_host:
            print("WARNING: api_base host does not match VLLM_HOST from .env")
            print(f"  api_base host: {model.api_base_host}")
            print(f"  VLLM_HOST:     {self.config.vllm_host}")
            print()

        existing_state = self._docker_container_state(container_name)

        if existing_state is not None:
            print()
            print("=== Existing container found ===")
            print(f"Container: {container_name}")
            print(f"State:     {existing_state}")
            print()

            if options.recreate:
                print("--recreate was supplied.")

                if existing_state == "running":
                    self._docker_stop_container(container_name, dry_run=options.dry_run)

                self._docker_remove_container(container_name, dry_run=options.dry_run)

            elif existing_state == "running":
                print(f"Container already running: {container_name}")
                print("No duplicate vLLM instance will be started.")
                self._print_container_reuse_notice(container_name, model.name, options.port)
                return

            elif options.reuse_existing:
                print(f"Container exists but is not running: {container_name}")
                print("Reusing it by starting the existing container.")

                if not options.dry_run and (
                    port_in_use(options.port, self.config.vllm_bind_host)
                    or self._docker_port_in_use(options.port)
                ):
                    print(f"ERROR: Port {options.port} is already in use on {self.config.vllm_bind_host}")
                    sys.exit(1)

                self._docker_start_container(container_name, dry_run=options.dry_run)
                self._print_container_reuse_notice(container_name, model.name, options.port)
                return

            else:
                print(f"ERROR: Container already exists but is not running: {container_name}")
                print(f"Use --recreate or docker start {container_name}")
                sys.exit(1)

        if not options.dry_run and (
            port_in_use(options.port, self.config.vllm_bind_host)
            or self._docker_port_in_use(options.port)
        ):
            print(f"ERROR: Port {options.port} is already in use on {self.config.vllm_bind_host}")
            sys.exit(1)

        command = self._build_vllm_command(model, model_dir, options.port, options.runtime)
        self._print_vllm_start_summary(model, model_dir, options.port, options.runtime, command)

        if not options.dry_run:
            start_background_process(command)

    def stop_vllm(self, model_name: str, *, dry_run: bool = False) -> None:
        """Stop running vLLM containers for one model.

        Safety:
        - Only stops containers matching vllm-{model_name}- pattern
        - Does not remove containers (can be restarted)
        - Respects dry_run flag
        """
        containers = self._find_running_vllm_containers(model_name)

        if not containers:
            print(f"No running vLLM containers found for model: {model_name}")
            return

        print(f"\n=== Stopping vLLM model: {model_name} ===")

        for container in containers:
            print(f"  {container}")

        if dry_run:
            print("\nDry run only. No containers stopped.")
            return

        result = run_command(["docker", "stop"] + containers, capture=True)

        if result.returncode != 0:
            print("ERROR: Failed to stop container(s)")
            print(result.stderr.strip())
            sys.exit(1)

        print("\nStopped:")
        print(result.stdout.strip())

    def stop_all_vllm(self, *, dry_run: bool = False) -> None:
        """Stop all running vLLM containers for configured models.

        Safety:
        - Only affects containers for models in configuration
        - Does not stop unrelated Docker containers
        - Respects dry_run flag
        """
        containers: list[str] = []

        for model_name in self.models:
            containers.extend(self._find_running_vllm_containers(model_name))

        unique = sorted(set(containers))

        if not unique:
            print("No running vLLM containers found for configured models.")
            return

        print("\n=== Stopping all configured vLLM containers ===")

        for container in unique:
            print(f"  {container}")

        if dry_run:
            print("\nDry run only. No containers stopped.")
            return

        result = run_command(["docker", "stop"] + unique, capture=True)

        if result.returncode != 0:
            print("ERROR: Failed to stop container(s)")
            print(result.stderr.strip())
            sys.exit(1)

        print("\nStopped:")
        print(result.stdout.strip())

    # ------------------------------------------------------------------
    # LiteLLM lifecycle
    # ------------------------------------------------------------------

    def start_litellm(self, *, dry_run: bool = False, follow_log: bool = False) -> None:
        """Start LiteLLM proxy.

        When *follow_log* is True the process runs in the foreground and its
        output is streamed through the log formatter so stack traces are
        printed with real newlines.  Otherwise it is detached as a background
        process.

        Safety:
        - Checks if port is already in use before starting
        - Respects dry_run flag
        """
        if port_in_use(self.config.litellm_port, self.config.litellm_bind_host):
            print(
                f"LiteLLM port {self.config.litellm_port} appears to be in use on "
                f"{self.config.litellm_bind_host}"
            )
            return

        command = (
            f"set -a && "
            f"source {self.config.env_file} && "
            f"set +a && "
            f"export LITELLM_LOG=INFO && "
            f"source {self.config.litellm_venv_activate} && "
            f"{self.config.litellm_bin} --config {self.config.litellm_config} "
            f"--host {self.config.litellm_bind_host} "
            f"--port {self.config.litellm_port}"
        )

        print(
            f"\n=== Starting LiteLLM router on "
            f"{self.config.litellm_bind_host}:{self.config.litellm_port} ==="
        )
        print("Command:", command)

        if not dry_run:
            if follow_log:
                _stream_formatted_logs(["bash", "-lc", command])
            else:
                start_background_process(["bash", "-lc", command])

    def stop_litellm(self, *, dry_run: bool = False) -> None:
        """Stop LiteLLM proxy processes.

        Finds LiteLLM processes by matching command patterns and sends
        SIGTERM signal to stop them gracefully.

        Safety:
        - Uses specific pattern matching to avoid wrong processes
        - Shows PIDs before killing
        - Uses SIGTERM (graceful shutdown) not SIGKILL
        - Respects dry_run flag
        - Includes self-PID protection
        """
        patterns = [
            f"{self.config.litellm_bin} --config {self.config.litellm_config}",
            f"litellm --config {self.config.litellm_config}",
            self.config.litellm_bin,
        ]

        pids: list[str] = []

        for pattern in patterns:
            result = run_command(["pgrep", "-f", pattern], capture=True)

            if result.returncode in (0, 1):
                for line in result.stdout.splitlines():
                    candidate = line.strip()

                    if candidate and candidate.isdigit():
                        pids.append(candidate)
            else:
                print("ERROR: Failed to query LiteLLM processes")
                print(result.stderr.strip())
                sys.exit(1)

        unique_pids = sorted(set(pids))

        if not unique_pids:
            print("No running LiteLLM process found.")
            return

        my_pid = str(os.getpid())

        if my_pid in unique_pids:
            unique_pids.remove(my_pid)
            print(f"Warning: Skipping self-PID {my_pid}")

        if not unique_pids:
            print("No running LiteLLM processes found (excluding self).")
            return

        print("\n=== Stopping LiteLLM ===")

        for pid in unique_pids:
            print(f"  {pid}")

        if dry_run:
            print("\nDry run only. LiteLLM was not stopped.")
            return

        result = run_command(["kill", "-TERM"] + unique_pids, capture=True)

        if result.returncode != 0:
            print("ERROR: Failed to stop LiteLLM")
            print(result.stderr.strip())
            sys.exit(1)

        print("\nLiteLLM stop signal sent.")

    # ------------------------------------------------------------------
    # CLI handlers
    # ------------------------------------------------------------------

    def _resolve_runtime_settings(
        self,
        args: argparse.Namespace,
        model: ModelDeployment,
    ) -> dict[str, int | float | str | None]:
        """Resolve runtime settings from config.yaml plus explicit CLI overrides.

        Combines default settings from model configuration with any CLI overrides
        specified by the user. Validates that settings are within acceptable ranges.
        """
        context_length = model.context_length
        max_input_tokens = model.max_input_tokens
        max_output_tokens = model.max_output_tokens
        gpu_memory_utilization = model.gpu_memory_utilization
        max_num_seqs = model.max_num_seqs
        dtype = model.dtype

        if args.context_length is not None:
            context_length = args.context_length

        if args.max_input_tokens is not None:
            max_input_tokens = args.max_input_tokens

        if args.max_output_tokens is not None:
            max_output_tokens = args.max_output_tokens

        if args.gpu_memory_utilization is not None:
            gpu_memory_utilization = args.gpu_memory_utilization

        if args.max_num_seqs is not None:
            max_num_seqs = args.max_num_seqs

        if args.dtype is not None:
            dtype = args.dtype

        if context_length < 1:
            print("ERROR: context_length must be >= 1")
            sys.exit(1)

        if max_input_tokens < 1:
            print("ERROR: max_input_tokens must be >= 1")
            sys.exit(1)

        if max_output_tokens < 1:
            print("ERROR: max_output_tokens must be >= 1")
            sys.exit(1)

        if max_input_tokens + max_output_tokens > context_length:
            print("ERROR: max_input_tokens + max_output_tokens exceeds context_length")
            print(f"  max_input_tokens:  {max_input_tokens}")
            print(f"  max_output_tokens: {max_output_tokens}")
            print(f"  context_length:    {context_length}")
            sys.exit(1)

        if gpu_memory_utilization <= 0 or gpu_memory_utilization > 1:
            print("ERROR: gpu_memory_utilization must be > 0 and <= 1")
            sys.exit(1)

        if max_num_seqs is not None and max_num_seqs < 1:
            print("ERROR: max_num_seqs must be >= 1")
            sys.exit(1)

        return {
            "context_length": context_length,
            "max_input_tokens": max_input_tokens,
            "max_output_tokens": max_output_tokens,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_num_seqs": max_num_seqs,
            "dtype": dtype,
        }

    def handle_logs(self, args: argparse.Namespace) -> bool:
        """Handle log-related commands. Return True if handled."""
        if not args.show_log and not args.show_log_path and not args.clean_log:
            return False

        # If --start is specified with --show-log, let run() handle it after starting
        if args.start and args.show_log:
            return False

        if not args.model:
            print("ERROR: --show-log/--show-log-path/--clean-log requires MODEL_NAME or all")
            sys.exit(1)

        target = args.model.lower()

        if args.clean_log:
            self.clean_vllm_logs(target, dry_run=args.dry_run)
            return True

        if args.show_log_path:
            self.show_vllm_log_paths(target)
            return True

        if target == "all" and args.show_log:
            self.show_vllm_log_paths(target)
            return True

        resolved = self._resolve_model_name(args.model)

        if resolved is None:
            print(f"ERROR: Unknown model '{args.model}'")
            print(self.generate_help())
            sys.exit(1)

        if not self._find_all_vllm_containers(resolved):
            print(f"No container found for model '{target}'.")
            print()
            print("Start the model first:")
            print(f"  harinezumigel-llm-stack {target} --start")
            print()
            print("Or start and show logs:")
            print(f"  harinezumigel-llm-stack {target} --start --show-log")
            sys.exit(1)

        self.show_vllm_logs(target, tail=args.tail, follow=args.follow, show_log_path=False)
        return True

    def handle_ps(self, args: argparse.Namespace) -> bool:
        """Handle container status display. Return True if handled."""
        if not args.ps:
            return False

        if not args.model:
            print("ERROR: --ps requires MODEL_NAME or all")
            sys.exit(1)

        target = args.model.lower()

        if target == "all":
            self.show_all_container_status()
            return True

        resolved = self._resolve_model_name(args.model)

        if resolved is None:
            print(f"ERROR: Unknown model '{args.model}'")
            print(self.generate_help())
            sys.exit(1)

        self.show_container_status(resolved)
        return True

    def handle_stop(self, args: argparse.Namespace) -> bool:
        """Handle stop commands. Return True if handled."""
        if not args.stop:
            return False

        if not args.model:
            print("ERROR: --stop requires MODEL_NAME, all, litellm")
            sys.exit(1)

        target = args.model.lower()

        if target == "all":
            self.stop_all_vllm(dry_run=args.dry_run)
            return True

        if target in ("litellm", "llm-lite", "lite-llm"):
            self.stop_litellm(dry_run=args.dry_run)
            return True

        resolved = self._resolve_model_name(args.model)

        if resolved is None:
            print(f"ERROR: Unknown model '{args.model}'")
            print(self.generate_help())
            sys.exit(1)

        self.stop_vllm(resolved, dry_run=args.dry_run)
        return True

    def run(self, args: argparse.Namespace) -> None:
        """Dispatch a parsed argument namespace to the appropriate operation.

        Handles litellm commands before loading model config, then loads
        models and dispatches to the appropriate handler.

        Safety: All destructive operations require explicit flags and respect
        dry-run mode. Default behavior is conservative (reuse, don't recreate).
        """
        # Handle litellm target before loading model config (config not needed)
        if args.model and args.model.lower() in ("litellm", "llm-lite", "lite-llm"):
            if args.start:
                self.start_litellm(dry_run=args.dry_run, follow_log=args.show_log)
                return

            if args.stop:
                self.stop_litellm(dry_run=args.dry_run)
                return

            print("ERROR: Specify an action for litellm (--start or --stop)")
            sys.exit(1)

        self.load_models()

        if args.help:
            print(self.generate_help())
            return

        if args.list:
            self.list_models()
            return

        if self.handle_logs(args):
            return

        if self.handle_ps(args):
            return

        if self.handle_stop(args):
            return

        if not args.model:
            print(self.generate_help())
            sys.exit(1)

        # Model specified — require an action
        if not args.start and not args.show_log and not args.show_log_path:
            print(f"ERROR: Specify an action for model '{args.model}'")
            print()
            print("Available actions:")
            print("  --start              Start the container")
            print("  --start --show-log   Start and show logs")
            print("  --show-log           View logs")
            print("  --show-log --follow  Follow logs")
            print("  --stop               Stop the container")
            sys.exit(1)

        if args.recreate and not args.start:
            print("ERROR: --recreate can only be used together with --start")
            sys.exit(1)

        resolved = self._resolve_model_name(args.model)

        if resolved is None:
            print(f"ERROR: Unknown model '{args.model}'")
            print(self.generate_help())
            sys.exit(1)

        model = self.models[resolved]
        runtime = self._resolve_runtime_settings(args, model)

        if args.auto_port:
            port = self._find_free_port()
        elif args.port is not None:
            port = args.port
        else:
            port = model.api_base_port

            if port is None:
                print(f"ERROR: No port found in LiteLLM config for model {model.name}")
                print("Use --port PORT or add api_base with a port to LiteLLM config.")
                sys.exit(1)

        if port < 1 or port > 65535:
            print(f"ERROR: Invalid port: {port}")
            sys.exit(1)

        if args.start:
            self.start_vllm(
                model,
                StartOptions(
                    port=port,
                    runtime=runtime,
                    dry_run=args.dry_run,
                    reuse_existing=not args.no_reuse_existing,
                    recreate=args.recreate,
                ),
            )

            if args.show_log and not args.dry_run:
                container_name = f"vllm-{docker_safe_name(model.name)}-{port}"

                print()
                print("Waiting for container to be ready...")

                for _ in range(10):
                    if self._docker_container_state(container_name) == "running":
                        break
                    time.sleep(1)

                print()
                self.show_container_logs(
                    container_name,
                    tail=args.tail,
                    follow=True,
                    show_log_path=False,
                )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        add_help=False,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("model", nargs="?")
    parser.add_argument("--port", type=int)
    parser.add_argument("--auto-port", action="store_true")
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--max-input-tokens", type=int)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--dtype", type=str)

    parser.add_argument("--start", action="store_true", help="Start the model container")
    parser.add_argument("--stop", action="store_true", help="Stop the model container")
    parser.add_argument("--ps", action="store_true", help="Show container status")
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--no-reuse-existing", action="store_true")

    parser.add_argument("--show-log", action="store_true", help="Show container logs (or paths for 'all')")
    parser.add_argument("--show-log-path", action="store_true", help="Show Docker log file paths")
    parser.add_argument("--clean-log", action="store_true", help="Clean/truncate container log files")
    parser.add_argument("--follow", "-f", action="store_true", help="Follow log output (use with --show-log)")
    parser.add_argument("--tail", default="200")

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--help", action="store_true")

    return parser.parse_args()


def main() -> None:
    """Program entry point."""
    LLMStack(APP).run(parse_args())


if __name__ == "__main__":
    main()
