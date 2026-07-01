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

    # List configured models
    harinezumigel-llm-stack --list

    # Start a model (reuses existing container if available)
    harinezumigel-llm-stack mistral_7b

    # Start with explicit port
    harinezumigel-llm-stack qwen-coder --port 8003

    # Recreate container with new settings
    harinezumigel-llm-stack qwen2_5_32b_awq --recreate --max-num-seqs 1

    # Preview changes without executing
    harinezumigel-llm-stack mistral_7b --dry-run

    # Start LiteLLM proxy
    harinezumigel-llm-stack --litellm

    # View logs
    harinezumigel-llm-stack qwen-coder --logs --follow

    # Stop containers
    harinezumigel-llm-stack --stop qwen-coder
    harinezumigel-llm-stack --stop all
The script does not hardcode model ports, bind hosts, Docker image names, or
model paths. All values come from .env and config.yaml files.

See README.md for configuration details.

Author: harinezumigel-llm-stack contributors
License: MIT
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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
# Generic helpers
# ---------------------------------------------------------------------------
# Utility functions for string manipulation, type conversion, network checks,
# and subprocess execution.


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


def canonical(name: str) -> str:
    """Normalize model names for matching.

    Converts to lowercase and replaces dots/hyphens with underscores
    for fuzzy matching of model names.
    """
    name = name.replace(".", "_").replace("-", "_")
    name = re.sub(r"_+", "_", name)
    return name.lower()


def docker_safe_name(name: str) -> str:
    """Make a Docker-safe container-name fragment."""
    name = name.lower()
    name = re.sub(r"[^a-z0-9_.-]+", "-", name)
    name = re.sub(r"-+", "-", name)
    return name.strip("-")


def compact_name(name: str) -> str:
    """Remove punctuation for fuzzy model-dir matching."""
    return re.sub(r"[^a-zA-Z0-9]", "", name).lower()


def model_tokens(name: str) -> list[str]:
    """Tokenize a model name for fuzzy model-dir matching."""
    parts = re.split(r"[^a-zA-Z0-9]+", name.lower())
    return [part for part in parts if part and len(part) >= 2]


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


def parse_api_base(api_base: str) -> tuple[str, int]:
    """Extract host and port from an api_base URL."""
    resolved_api_base = resolve_env_value(api_base)
    parsed = urlparse(resolved_api_base)

    host = parsed.hostname or APP.vllm_host

    if parsed.port:
        port = parsed.port
    elif parsed.scheme == "https":
        port = 443
    else:
        port = 80

    return host, port


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


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
# Functions and classes for loading and managing model deployment
# configuration from LiteLLM's config.yaml.


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


@dataclass(frozen=True)
class StartOptions:
    """Options for starting or recreating a vLLM backend."""

    port: int
    runtime: dict[str, int | float | None]
    dry_run: bool
    reuse_existing: bool
    recreate: bool


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


def load_litellm_models() -> dict[str, ModelDeployment]:
    """Load model deployments from config.yaml.

    Parses the LiteLLM configuration file and extracts model deployment
    information including API endpoints, context lengths, and resource
    limits. Returns a dictionary keyed by model name.

    Raises:
        RuntimeError: If config file not found or required fields missing
    """
    config_path = Path(APP.litellm_config)

    if not config_path.exists():
        raise RuntimeError(f"LiteLLM config not found: {APP.litellm_config}")

    with config_path.open("r", encoding="utf-8") as file_handle:
        config = yaml.safe_load(file_handle) or {}

    deployments: dict[str, ModelDeployment] = {}

    for entry in config.get("model_list", []):
        model_name = entry.get("model_name")
        params = entry.get("litellm_params", {})
        model_info = entry.get("model_info", {})

        if not model_name:
            continue

        api_base = resolve_env_value(params.get("api_base", ""))
        api_key = resolve_env_value(params.get("api_key", ""))
        backend_model = resolve_env_value(params.get("model", ""))

        if api_base:
            api_base_host, api_base_port = parse_api_base(api_base)
        else:
            api_base_host, api_base_port = APP.vllm_host, None

        deployments[model_name] = ModelDeployment(
            name=model_name,
            backend_model=backend_model,
            api_base=api_base,
            api_key=api_key,
            api_base_host=api_base_host,
            api_base_port=api_base_port,
            model_info=model_info,
            context_length=require_model_int(model_name, model_info, "context_length"),
            max_input_tokens=require_model_int(
                model_name,
                model_info,
                "max_input_tokens",
            ),
            max_output_tokens=require_model_int(
                model_name,
                model_info,
                "max_output_tokens",
            ),
            gpu_memory_utilization=require_model_float(
                model_name,
                model_info,
                "gpu_memory_utilization",
            ),
            max_num_seqs=as_int(model_info.get("max_num_seqs"), None),
        )

    return deployments


def resolve_model_name(
    user_input: str,
    models: dict[str, ModelDeployment],
) -> str | None:
    """Resolve user input to a configured model name."""
    canon_input = canonical(user_input)

    for model_name in models:
        if canonical(model_name) == canon_input:
            return model_name

    return None


def find_model_dir(model: ModelDeployment) -> Path:
    """Find the local model directory for a deployment.

    Searches for the model directory in MODEL_ROOT using:
    1. Explicit model_info.model_dir if configured
    2. Exact directory name match (fuzzy)
    3. Directory name contains all model name tokens

    Returns:
        Path to the model directory

    Raises:
        RuntimeError: If model directory cannot be found

    Safety: This function only reads directories, never modifies them.
    """
    explicit_model_dir = model.model_info.get("model_dir")

    if explicit_model_dir:
        explicit_path = Path(APP.model_root) / explicit_model_dir

        if explicit_path.is_dir():
            return explicit_path

        raise RuntimeError(
            f"Configured model_dir does not exist for {model.name}: {explicit_path}"
        )

    model_root = Path(APP.model_root)

    if not model_root.is_dir():
        raise RuntimeError(f"Model root does not exist: {APP.model_root}")

    target_compact = compact_name(model.name)
    target_tokens = model_tokens(model.name)
    entries = [entry for entry in model_root.iterdir() if entry.is_dir()]

    for entry in entries:
        cleaned = compact_name(entry.name)
        if cleaned.startswith(target_compact):
            return entry

    for entry in entries:
        cleaned = compact_name(entry.name)
        if target_compact.startswith(cleaned):
            return entry

    for entry in entries:
        entry_lower = entry.name.lower()
        if all(token in entry_lower for token in target_tokens):
            return entry

    raise RuntimeError(f"No matching model directory found for {model.name}")


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------
# Functions for interacting with Docker: inspecting, starting, stopping,
# and removing containers. All destructive operations respect dry_run flags.


def docker_container_state(name: str) -> str | None:
    """Return Docker container state or None if not found.

    Possible states: created, running, paused, restarting, removing,
    exited, dead
    """
    result = run_command(
        ["docker", "inspect", "-f", "{{.State.Status}}", name],
        capture=True,
    )

    if result.returncode != 0:
        return None

    return result.stdout.strip() or None


def docker_port_in_use(port: int, host: str) -> bool:
    """Return True if a Docker container already publishes the port.

    Checks running Docker containers for port mappings to detect
    port conflicts before starting new containers.
    """
    result = run_command(["docker", "ps", "--format", "{{.Ports}}"], capture=True)
    ports_output = result.stdout

    return (
        f":{port}->" in ports_output
        or f"{host}:{port}->" in ports_output
        or f"0.0.0.0:{port}->" in ports_output
    )


def docker_list_container_names(*, all_containers: bool) -> list[str]:
    """List Docker container names."""
    if all_containers:
        command = ["docker", "ps", "-a", "--format", "{{.Names}}"]
    else:
        command = ["docker", "ps", "--format", "{{.Names}}"]

    result = run_command(command, capture=True)

    if result.returncode != 0:
        print("ERROR: Failed to query Docker containers")
        print(result.stderr.strip())
        sys.exit(1)

    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def docker_start_container(name: str, *, dry_run: bool = False) -> None:
    """Start an existing Docker container.

    Safety: Only starts containers, does not modify or remove them.
    Respects dry_run flag.
    """
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


def docker_stop_container(name: str, *, dry_run: bool = False) -> None:
    """Stop a Docker container.

    Sends SIGTERM to the container's main process for graceful shutdown.

    Safety: Only stops the specified container. Does not remove it.
    Respects dry_run flag.
    """
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


def docker_remove_container(
    name: str,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> None:
    """Remove a Docker container.

    Args:
        name: Container name to remove
        dry_run: If True, only show what would be done
        force: If True, force removal of running containers (not used in practice)

    Safety: Only removes the specified container. Container must be stopped
    first unless force=True. In this codebase, force is never used - containers
    are always stopped before removal. Respects dry_run flag.
    """
    command = ["docker", "rm"]

    if force:
        command.append("-f")

    command.append(name)

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


def docker_container_log_path(name: str) -> str | None:
    """Return Docker container log file path."""
    result = run_command(
        ["docker", "inspect", "-f", "{{.LogPath}}", name],
        capture=True,
    )

    if result.returncode != 0:
        return None

    return result.stdout.strip() or None


def find_all_vllm_containers_for_model(model_name: str) -> list[str]:
    """Find all vLLM container names for a model.

    Searches for containers matching the pattern: vllm-{model_name}-{port}
    Includes both running and stopped containers.
    """
    name_prefix = f"vllm-{docker_safe_name(model_name)}-"
    all_names = docker_list_container_names(all_containers=True)

    return [name for name in all_names if name.startswith(name_prefix)]


def find_running_vllm_containers_for_model(model_name: str) -> list[str]:
    """Find running vLLM container names for a model.

    Searches for running containers matching: vllm-{model_name}-{port}
    Only includes containers in 'running' state.
    """
    name_prefix = f"vllm-{docker_safe_name(model_name)}-"
    running_names = docker_list_container_names(all_containers=False)

    return [name for name in running_names if name.startswith(name_prefix)]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
# Functions for formatting and displaying information to the user.


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


def print_container_reuse_notice(container_name: str, model_name: str, port: int) -> None:
    """Print a notice for reused containers."""
    print()
    print("=== Reusing existing vLLM container ===")
    print(f"Container: {container_name}")
    print(f"Model:     {model_name}")
    print(f"Endpoint:  http://{APP.vllm_bind_host}:{port}/v1")
    print()
    print("Important:")
    print("  Existing containers keep original Docker run settings.")
    print("  Use --recreate to rebuild the container with current settings.")
    print()


def list_models(models: dict[str, ModelDeployment]) -> None:
    """Print configured model deployments."""
    print("\nModels from LiteLLM config:")
    print("---------------------------")

    for model in models.values():
        raw_info = model.model_info
        backend_served_model = backend_model_without_provider(model.backend_model)

        print(f"{model.name}")
        print(f"  api_base:                 {model.api_base}")
        print(f"  api_base_host:            {model.api_base_host}")
        print(f"  api_base_port:            {model.api_base_port}")
        print(f"  vllm_host:                {APP.vllm_host}")
        print(f"  vllm_bind_host:           {APP.vllm_bind_host}")
        print(f"  backend_model:            {model.backend_model}")
        print(f"  backend_served_model:     {backend_served_model}")
        print(f"  context_length:           {model.context_length}")
        print(f"  max_input_tokens:         {model.max_input_tokens}")
        print(f"  max_output_tokens:        {model.max_output_tokens}")
        print(f"  gpu_memory_utilization:   {model.gpu_memory_utilization}")
        print(f"  max_num_seqs:             {model.max_num_seqs or '(vLLM default)'}")

        interesting_keys = [
            "model_dir",
            "max_num_batched_tokens",
            "max_num_seqs",
            "attention_backend",
            "enable_prefix_caching",
            "enforce_eager",
            "enable_auto_tool_choice",
            "tool_call_parser",
            "trust_remote_code",
        ]

        for key in interesting_keys:
            if key in raw_info:
                print(f"  {key:<25} {raw_info.get(key)}")

        if model.backend_model and backend_served_model != model.name:
            print("  WARNING: backend served model does not match model_name")
            print(f"  Recommended: model: openai/{model.name}")

        print()


def generate_help(models: dict[str, ModelDeployment]) -> str:
    """Generate dynamic help text."""
    model_lines = []

    for model in models.values():
        extras = []

        for key in [
            "model_dir",
            "max_num_batched_tokens",
            "max_num_seqs",
            "attention_backend",
            "enable_prefix_caching",
            "enforce_eager",
            "enable_auto_tool_choice",
            "tool_call_parser",
        ]:
            if key in model.model_info:
                extras.append(f"{key}={model.model_info.get(key)}")

        model_lines.append(
            "    "
            f"{model.name:<22} "
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
  {APP.litellm_config}

Env:
  {APP.env_file}

Important:
  Runtime values come from .env and config.yaml.
  --profile is accepted only for legacy compatibility and does not override config.

Examples:
  harinezumigel-llm-stack --list

  harinezumigel-llm-stack mistral_7b --dry-run
  harinezumigel-llm-stack mistral_7b --recreate

  harinezumigel-llm-stack mistral_7b --max-num-seqs 1 --dry-run
  harinezumigel-llm-stack mistral_7b --context-length 32768 --max-input-tokens 28672 --max-output-tokens 4096 --max-num-seqs 1 --dry-run

  harinezumigel-llm-stack qwen-coder --dry-run
  harinezumigel-llm-stack qwen-coder --recreate

  harinezumigel-llm-stack qwen2_5_32b_awq --dry-run
  harinezumigel-llm-stack qwen2_5_32b_awq --recreate

  harinezumigel-llm-stack --litellm
  harinezumigel-llm-stack --stop-litellm

  harinezumigel-llm-stack --stop qwen-coder
  harinezumigel-llm-stack qwen-coder --logs --follow

Current model values from LiteLLM config:
{model_list}
"""


# ---------------------------------------------------------------------------
# Log operations
# ---------------------------------------------------------------------------
# Functions for viewing Docker container logs and log file paths.


def show_container_logs(
    container_name: str,
    *,
    tail: str = "200",
    follow: bool = False,
    show_log_path: bool = False,
) -> None:
    """Show Docker logs for one container."""
    state = docker_container_state(container_name)

    print()
    print("=== Docker logs ===")
    print(f"Container: {container_name}")
    print(f"State:     {state or 'unknown'}")

    if show_log_path:
        log_path = docker_container_log_path(container_name)
        print(f"Log path:  {log_path or '(not available)'}")

    print()

    command = ["docker", "logs", "--tail", str(tail)]

    if follow:
        command.append("-f")

    command.append(container_name)

    result = run_command(command, capture=False)

    if result.returncode != 0:
        print(f"ERROR: Failed to read logs for container: {container_name}")
        sys.exit(result.returncode)


def show_vllm_logs(
    models: dict[str, ModelDeployment],
    target: str,
    *,
    tail: str,
    follow: bool,
    show_log_path: bool,
) -> None:
    """Show logs for one model or all models."""
    if target == "all":
        containers: list[str] = []

        for model_name in models:
            containers.extend(find_all_vllm_containers_for_model(model_name))

        containers = sorted(set(containers))
    else:
        resolved = resolve_model_name(target, models)

        if resolved is None:
            print(f"ERROR: Unknown model '{target}'")
            print(generate_help(models))
            sys.exit(1)

        containers = find_all_vllm_containers_for_model(resolved)

    if not containers:
        print("No vLLM containers found.")
        return

    if follow and len(containers) > 1:
        print("ERROR: --follow can only be used when exactly one container matches.")

        for container in containers:
            print(f"  {container}")

        sys.exit(1)

    for container in containers:
        show_container_logs(
            container,
            tail=tail,
            follow=follow,
            show_log_path=show_log_path,
        )


def show_vllm_log_paths(models: dict[str, ModelDeployment], target: str) -> None:
    """Show Docker log file paths."""
    if target == "all":
        containers: list[str] = []

        for model_name in models:
            containers.extend(find_all_vllm_containers_for_model(model_name))

        containers = sorted(set(containers))
    else:
        resolved = resolve_model_name(target, models)

        if resolved is None:
            print(f"ERROR: Unknown model '{target}'")
            print(generate_help(models))
            sys.exit(1)

        containers = find_all_vllm_containers_for_model(resolved)

    if not containers:
        print("No vLLM containers found.")
        return

    print()
    print("=== Docker log paths ===")

    for container in containers:
        state = docker_container_state(container)
        path = docker_container_log_path(container)

        print(f"{container}")
        print(f"  state:    {state or 'unknown'}")
        print(f"  log_path: {path or '(not available)'}")
        print()


# ---------------------------------------------------------------------------
# vLLM lifecycle
# ---------------------------------------------------------------------------
# Core functions for managing vLLM Docker container lifecycle:
# starting, stopping, and configuring model backends.


def find_free_port() -> int:
    """Find a free vLLM port in the configured auto-port range.

    Scans VLLM_AUTO_PORT_START to VLLM_AUTO_PORT_END for an available
    port that is not in use by the system or Docker.

    Returns:
        First available port number

    Raises:
        RuntimeError: If no free ports available in range
    """
    for port in range(APP.vllm_auto_port_start, APP.vllm_auto_port_end + 1):
        if (
            not port_in_use(port, APP.vllm_bind_host)
            and not docker_port_in_use(port, APP.vllm_bind_host)
        ):
            return port

    raise RuntimeError(
        "No free ports available in range "
        f"{APP.vllm_auto_port_start}-{APP.vllm_auto_port_end}"
    )


def build_vllm_command(
    model: ModelDeployment,
    model_dir: Path,
    port: int,
    runtime: dict[str, int | float | None],
) -> list[str]:
    """Build the Docker command for a vLLM backend.

    Constructs a complete 'docker run' command with all necessary flags,
    volumes, and vLLM parameters based on the model configuration.

    Args:
        model: Model deployment configuration
        model_dir: Path to local model files
        port: Host port to bind
        runtime: Runtime settings (context_length, gpu_memory_utilization, etc.)

    Returns:
        Complete command as list of strings
    """
    raw_info = model.model_info
    model_dir_name = model_dir.name

    command = [
        "docker",
        "run",
        "--name",
        f"vllm-{docker_safe_name(model.name)}-{port}",
        "--runtime=nvidia",
        "--gpus=all",
        "--ipc=host",
        "-p",
        f"{APP.vllm_bind_host}:{port}:{APP.vllm_container_port}",
        "-v",
        APP.vllm_model_volume,
        "-v",
        APP.vllm_cache_volume,
        APP.vllm_docker_image,
        "python3",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        f"/models/{model_dir_name}",
        "--served-model-name",
        model.name,
        "--dtype",
        "auto",
        "--max-model-len",
        str(runtime["context_length"]),
        "--gpu-memory-utilization",
        str(runtime["gpu_memory_utilization"]),
        "--host",
        "0.0.0.0",
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

    if as_bool(raw_info.get("enable_auto_tool_choice", False)):
        command.append("--enable-auto-tool-choice")

    if raw_info.get("tool_call_parser"):
        command.extend(["--tool-call-parser", str(raw_info["tool_call_parser"])])

    api_key = model.api_key or os.environ.get("LOCAL_VLLM_API_KEY", "")

    if api_key:
        command.extend(["--api-key", api_key])

    return command


def print_vllm_start_summary(
    model: ModelDeployment,
    model_dir: Path,
    port: int,
    runtime: dict[str, int | float | None],
    command: list[str],
) -> None:
    """Print a readable vLLM startup summary."""
    raw_info = model.model_info
    api_key = model.api_key or os.environ.get("LOCAL_VLLM_API_KEY", "")

    print("API key resolved:          yes" if api_key else "API key resolved:          no")

    if api_key:
        print(f"API key length:            {len(api_key)}")
        print(f"API key sha256:            {hashlib.sha256(api_key.encode()).hexdigest()}")

    print()
    print(f"=== Starting vLLM model: {model.name} ===")
    print(f"Container:                vllm-{docker_safe_name(model.name)}-{port}")
    print(f"Model directory:           {model_dir}")
    print(f"Directory basename:        {model_dir.name}")
    print(f"Served model name:         {model.name}")
    print(
        "Bind:                      "
        f"{APP.vllm_bind_host}:{port} -> "
        f"container:{APP.vllm_container_port}"
    )
    print(f"Context length:            {runtime['context_length']}")
    print(f"Max input tokens:          {runtime['max_input_tokens']}")
    print(f"Max output tokens:         {runtime['max_output_tokens']}")
    print(f"GPU memory utilization:    {runtime['gpu_memory_utilization']}")
    print(
        "Max num batched tokens:    "
        f"{raw_info.get('max_num_batched_tokens', '(vLLM default)')}"
    )
    print(
        "Max num seqs:              "
        f"{runtime.get('max_num_seqs', '(vLLM default)')}"
    )
    print(f"Enforce eager:             {as_bool(raw_info.get('enforce_eager', False))}")
    print(f"Trust remote code:         {as_bool(raw_info.get('trust_remote_code', False))}")
    print(
        "Enable prefix caching:     "
        f"{as_bool(raw_info.get('enable_prefix_caching', False))}"
    )
    print(f"Attention backend:         {raw_info.get('attention_backend', '(default)')}")
    print(
        "Auto tool choice:          "
        f"{as_bool(raw_info.get('enable_auto_tool_choice', False))}"
    )
    print(f"Tool call parser:          {raw_info.get('tool_call_parser', '(none)')}")
    print("Command:", " ".join(redact_command(command)))


def start_vllm(model: ModelDeployment, options: StartOptions) -> None:
    """Start a Dockerized vLLM backend.

    Main orchestration function for starting vLLM containers. Handles:
    - Checking for existing containers
    - Reusing or recreating containers based on options
    - Port conflict detection
    - Building and executing docker run command

    Args:
        model: Model deployment configuration
        options: Start options including port, runtime overrides, and flags

    Safety:
    - Checks for existing containers before creating new ones
    - Respects --recreate flag (explicit user intent required)
    - Default behavior: reuse existing containers (no data loss)
    - Validates port availability before binding
    - Respects dry_run flag throughout
    """
    model_dir = find_model_dir(model)
    container_name = f"vllm-{docker_safe_name(model.name)}-{options.port}"

    if model.api_base_host and model.api_base_host != APP.vllm_host:
        print("WARNING: api_base host does not match VLLM_HOST from .env")
        print(f"  api_base host: {model.api_base_host}")
        print(f"  VLLM_HOST:     {APP.vllm_host}")
        print()

    existing_state = docker_container_state(container_name)

    if existing_state is not None:
        print()
        print("=== Existing container found ===")
        print(f"Container: {container_name}")
        print(f"State:     {existing_state}")
        print()

        if options.recreate:
            print("--recreate was supplied.")

            if existing_state == "running":
                docker_stop_container(container_name, dry_run=options.dry_run)

            docker_remove_container(container_name, dry_run=options.dry_run)

        elif existing_state == "running":
            print(f"Container already running: {container_name}")
            print("No duplicate vLLM instance will be started.")
            print_container_reuse_notice(container_name, model.name, options.port)
            return

        elif options.reuse_existing:
            print(f"Container exists but is not running: {container_name}")
            print("Reusing it by starting the existing container.")

            if not options.dry_run and (
                port_in_use(options.port, APP.vllm_bind_host)
                or docker_port_in_use(options.port, APP.vllm_bind_host)
            ):
                print(
                    f"ERROR: Port {options.port} is already in use on "
                    f"{APP.vllm_bind_host}"
                )
                sys.exit(1)

            docker_start_container(container_name, dry_run=options.dry_run)
            print_container_reuse_notice(container_name, model.name, options.port)
            return

        else:
            print(f"ERROR: Container already exists but is not running: {container_name}")
            print(f"Use --recreate or docker start {container_name}")
            sys.exit(1)

    if not options.dry_run and (
        port_in_use(options.port, APP.vllm_bind_host)
        or docker_port_in_use(options.port, APP.vllm_bind_host)
    ):
        print(f"ERROR: Port {options.port} is already in use on {APP.vllm_bind_host}")
        sys.exit(1)

    command = build_vllm_command(model, model_dir, options.port, options.runtime)
    print_vllm_start_summary(model, model_dir, options.port, options.runtime, command)

    if not options.dry_run:
        start_background_process(command)


def stop_vllm(model_name: str, *, dry_run: bool = False) -> None:
    """Stop running vLLM containers for one model.

    Finds and stops all running containers matching the model name.
    Containers are stopped gracefully (SIGTERM).

    Safety:
    - Only stops containers matching vllm-{model_name}- pattern
    - Does not remove containers (can be restarted)
    - Respects dry_run flag
    """
    containers = find_running_vllm_containers_for_model(model_name)

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


def stop_all_vllm(models: dict[str, ModelDeployment], *, dry_run: bool = False) -> None:
    """Stop all running vLLM containers for configured models.

    Finds and stops all vLLM containers for models defined in config.yaml.

    Safety:
    - Only affects containers for models in configuration
    - Does not stop unrelated Docker containers
    - Respects dry_run flag
    """
    containers: list[str] = []

    for model_name in models:
        containers.extend(find_running_vllm_containers_for_model(model_name))

    unique_containers = sorted(set(containers))

    if not unique_containers:
        print("No running vLLM containers found for configured models.")
        return

    print("\n=== Stopping all configured vLLM containers ===")

    for container in unique_containers:
        print(f"  {container}")

    if dry_run:
        print("\nDry run only. No containers stopped.")
        return

    result = run_command(["docker", "stop"] + unique_containers, capture=True)

    if result.returncode != 0:
        print("ERROR: Failed to stop container(s)")
        print(result.stderr.strip())
        sys.exit(1)

    print("\nStopped:")
    print(result.stdout.strip())


# ---------------------------------------------------------------------------
# LiteLLM lifecycle
# ---------------------------------------------------------------------------
# Functions for managing the LiteLLM proxy server process.


def start_litellm(*, dry_run: bool = False) -> None:
    """Start LiteLLM proxy.

    Starts the LiteLLM proxy server as a background process using the
    configured virtual environment and settings.

    Safety:
    - Checks if port is already in use before starting
    - Respects dry_run flag
    """
    if port_in_use(APP.litellm_port, APP.litellm_bind_host):
        print(
            f"LiteLLM port {APP.litellm_port} appears to be in use on "
            f"{APP.litellm_bind_host}"
        )
        return

    command = (
        f"set -a && "
        f"source {APP.env_file} && "
        f"set +a && "
        f"source {APP.litellm_venv_activate} && "
        f"{APP.litellm_bin} --config {APP.litellm_config} "
        f"--host {APP.litellm_bind_host} "
        f"--port {APP.litellm_port}"
    )

    print(
        f"\n=== Starting LiteLLM router on "
        f"{APP.litellm_bind_host}:{APP.litellm_port} ==="
    )
    print("Command:", command)

    if not dry_run:
        start_background_process(["bash", "-lc", command])


def stop_litellm(*, dry_run: bool = False) -> None:
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
        f"{APP.litellm_bin} --config {APP.litellm_config}",
        f"litellm --config {APP.litellm_config}",
        APP.litellm_bin,
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

    # Safety check: don't kill our own process
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

    # Use explicit SIGTERM for graceful shutdown
    result = run_command(["kill", "-TERM"] + unique_pids, capture=True)

    if result.returncode != 0:
        print("ERROR: Failed to stop LiteLLM")
        print(result.stderr.strip())
        sys.exit(1)

    print("\nLiteLLM stop signal sent.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
# Command-line interface: argument parsing, validation, and main entry point.


def resolve_runtime_settings(
    args: argparse.Namespace,
    model: ModelDeployment,
) -> dict[str, int | float | None]:
    """Resolve runtime settings from config.yaml plus explicit CLI overrides.

    Combines default settings from model configuration with any CLI overrides
    specified by the user. Validates that settings are within acceptable ranges.

    Args:
        args: Parsed command-line arguments
        model: Model deployment with default settings

    Returns:
        Dictionary of resolved runtime settings

    Raises:
        SystemExit: If settings are invalid or inconsistent
    """
    context_length = model.context_length
    max_input_tokens = model.max_input_tokens
    max_output_tokens = model.max_output_tokens
    gpu_memory_utilization = model.gpu_memory_utilization
    max_num_seqs = model.max_num_seqs

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
    }


def parse_args(models: dict[str, ModelDeployment] | None = None) -> argparse.Namespace:
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

    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--no-reuse-existing", action="store_true")

    parser.add_argument("--litellm", action="store_true")
    parser.add_argument("--stop-litellm", action="store_true")
    parser.add_argument("--stop", action="store_true")

    parser.add_argument("--logs", action="store_true")
    parser.add_argument("--follow", "-f", action="store_true")
    parser.add_argument("--tail", default="200")
    parser.add_argument("--show-log-path", action="store_true")
    parser.add_argument("--log-path", action="store_true")

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--help", action="store_true")

    args = parser.parse_args()

    if args.help:
        if models is None:
            print("Use --list after config is valid, or run with a model name.")
        else:
            print(generate_help(models))

        sys.exit(0)

    return args


def handle_logs(args: argparse.Namespace, models: dict[str, ModelDeployment]) -> bool:
    """Handle log-related commands. Return True if handled."""
    if not args.logs and not args.log_path:
        return False

    if not args.model:
        print("ERROR: --logs/--log-path requires MODEL_NAME or all")
        sys.exit(1)

    target = args.model.lower()

    if args.logs:
        show_vllm_logs(
            models,
            target,
            tail=args.tail,
            follow=args.follow,
            show_log_path=args.show_log_path,
        )
        return True

    show_vllm_log_paths(models, target)
    return True


def handle_stop(args: argparse.Namespace, models: dict[str, ModelDeployment]) -> bool:
    """Handle stop commands. Return True if handled."""
    if not args.stop:
        return False

    if not args.model:
        print("ERROR: --stop requires MODEL_NAME, all, litellm, or llmlite")
        sys.exit(1)

    target = args.model.lower()

    if target == "all":
        stop_all_vllm(models, dry_run=args.dry_run)
        return True

    if target in ("litellm", "llmlite", "llm-lite", "lite-llm"):
        stop_litellm(dry_run=args.dry_run)
        return True

    resolved = resolve_model_name(args.model, models)

    if resolved is None:
        print(f"ERROR: Unknown model '{args.model}'")
        print(generate_help(models))
        sys.exit(1)

    stop_vllm(resolved, dry_run=args.dry_run)
    return True


def main() -> None:
    """Program entrypoint.

    Main orchestration function that:
    1. Parses command-line arguments
    2. Loads configuration from .env and config.yaml
    3. Dispatches to appropriate handlers based on command
    4. Executes requested operations (start/stop/logs/list)

    Safety: All destructive operations require explicit flags and respect
    dry-run mode. Default behavior is conservative (reuse, don't recreate).
    """
    early_args = parse_args(models=None)

    if early_args.stop_litellm:
        stop_litellm(dry_run=early_args.dry_run)
        return

    if early_args.litellm:
        start_litellm(dry_run=early_args.dry_run)
        return

    models = load_litellm_models()
    args = early_args

    if args.list:
        list_models(models)
        return

    if handle_logs(args, models):
        return

    if handle_stop(args, models):
        return

    if not args.model:
        print(generate_help(models))
        sys.exit(1)

    resolved = resolve_model_name(args.model, models)

    if resolved is None:
        print(f"ERROR: Unknown model '{args.model}'")
        print(generate_help(models))
        sys.exit(1)

    model = models[resolved]
    runtime = resolve_runtime_settings(args, model)

    if args.auto_port:
        port = find_free_port()
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

    start_vllm(
        model,
        StartOptions(
            port=port,
            runtime=runtime,
            dry_run=args.dry_run,
            reuse_existing=not args.no_reuse_existing,
            recreate=args.recreate,
        ),
    )


if __name__ == "__main__":
    main()
