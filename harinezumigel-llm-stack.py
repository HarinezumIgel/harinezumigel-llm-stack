#!/usr/bin/env python3
# pylint: disable=too-many-lines,too-few-public-methods
"""harinezumigel-llm-stack - Unified LiteLLM + vLLM launcher.

This script provides a unified interface for managing LiteLLM proxy servers and
vLLM model backends running in Docker containers.

DISCLAIMER: This project is not affiliated with LLMStack (llmstack.ai / Promptly)

## Overview

The script reads configuration from two sources:
- Deployment settings: /opt/litellm/.env (or HLS_ENV_FILE)
- Model runtime settings: config.yaml under INSTALLED_BASE (/opt/litellm)

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

    # Start a model and stream logs (stays attached)
    harinezumigel-llm-stack mistral_7b --start --stream-log

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
    harinezumigel-llm-stack litellm --show-log-path
    harinezumigel-llm-stack all --show-log-path
    harinezumigel-llm-stack all --show-log  # shows vLLM + LiteLLM log paths

    # Clean log files (truncate to free space)
    harinezumigel-llm-stack qwen-coder --clean-log
    harinezumigel-llm-stack litellm --clean-log
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
import datetime
import errno
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import ParseResult, urlparse

import yaml

__version__ = "1.4.0/63 10/19/2026"

_RED = "\033[31m"
_RESET = "\033[0m"
LITELLM_TARGETS = ("litellm", "llm-lite", "lite-llm")
VLLM_CONTAINER_PREFIX = "vllm-"
DOCKER_LOG_ROOT = Path("/var/lib/docker")
INSTALLED_BASE = Path("/opt/litellm")

if not INSTALLED_BASE.is_absolute():
    print(f"ERROR: INSTALLED_BASE must be an absolute path: {INSTALLED_BASE}")
    sys.exit(1)

if not INSTALLED_BASE.is_dir():
    print(f"ERROR: LiteLLM deployment directory does not exist: {INSTALLED_BASE}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Environment loading
# ---------------------------------------------------------------------------
# These functions safely load and parse environment variables from .env files
# and provide type-safe access with validation.


def _strip_env_comment(value: str) -> str:
    """Remove whitespace-separated comments outside quoted env values."""
    quote: str | None = None
    escaped = False

    for index, char in enumerate(value):
        if escaped:
            escaped = False
        elif char == "\\" and quote != "'":
            escaped = True
        elif quote is not None:
            if char == quote:
                quote = None
        elif char in ("'", '"'):
            quote = char
        elif char == "#" and (index == 0 or value[index - 1].isspace()):
            return value[:index].rstrip()

    return value


def load_env_file(path: str) -> None:
    """Load a shell-style KEY=value env file into os.environ.

    Supported forms:
      KEY=value
      export KEY=value
      KEY=${OTHER_KEY}
      KEY=$OTHER_KEY
      KEY=value # inline comment (outside quotes)

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
            value = _strip_env_comment(value.strip()).strip()
            if len(value) >= 2 and value[0] in ("'", '"') and value[-1] == value[0]:
                value = value[1:-1]

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
load_env_file(os.environ.get("HLS_ENV_FILE", str(INSTALLED_BASE / ".env")))


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
            env_file=os.environ.get("HLS_ENV_FILE", str(INSTALLED_BASE / ".env")),
            litellm_config=str(INSTALLED_BASE / "config.yaml"),
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
        return value.lower() in ("true", "1", "yes", "on")

    return bool(value)


def docker_ipc_arg(value: Any) -> str | None:
    """Normalize model_info.ipc to a Docker --ipc flag.

    Accepted forms:
    - host
    - --ipc=host
    - ipc=host
    """
    if value is None:
        return None

    ipc_value = str(value).strip()

    if not ipc_value:
        return None

    if ipc_value.startswith("--ipc="):
        return ipc_value

    if ipc_value.startswith("ipc="):
        ipc_value = ipc_value.split("=", 1)[1].strip()
        if not ipc_value:
            return None

    return f"--ipc={ipc_value}"


def format_sampling_parameters(
    override_gen_config: dict[str, Any], litellm_params: dict[str, Any]
) -> list[str]:
    """Format sampling parameters with source labels.

    Args:
        override_gen_config: Parameters from model_info.override_generation_config
        litellm_params: Parameters from litellm_params

    Returns:
        List of formatted strings like "temperature=0.7(litellm)"
    """
    sampling_parts: list[str] = []
    for param_name in [
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
    ]:
        override_val = override_gen_config.get(param_name)
        litellm_val = litellm_params.get(param_name)
        final_val = override_val if override_val is not None else litellm_val
        if final_val is not None:
            source = "model-spec" if override_val is not None else "litellm"
            sampling_parts.append(f"{param_name}={final_val}({source})")
    return sampling_parts


def extract_nvidia_driver_mismatch(message: str) -> tuple[str, str] | None:
    """Extract required/detected driver versions from NVIDIA runtime errors.

    Typical message pattern:
    "This container was built for NVIDIA Driver Release 595.58 or later,
    but version 580.178.04 was detected"
    """
    match = re.search(
        r"NVIDIA Driver Release\s+([0-9]+(?:\.[0-9]+){1,2})\s+or later,\s+"
        r"but version\s+([0-9]+(?:\.[0-9]+){1,2})\s+was detected",
        message,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    required = match.group(1)
    detected = match.group(2)
    return required, detected


def port_in_use(port: int, host: str) -> bool:
    """Return True if host:port is already in use.

    Attempts to bind to the port to check availability.
    Other bind errors are reported explicitly and stop the launcher.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        sock.bind((host, port))
        return False
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            return True
        print(f"ERROR: Could not check port {port} on {host}: {exc}")
        sys.exit(1)
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


def _format_human_log_line(obj_dict: dict[str, Any]) -> str:
    """Render a JSON log object in a compact human-readable form."""
    timestamp = str(obj_dict.get("timestamp", "-"))
    level = str(obj_dict.get("level", "LOG")).upper()
    component = str(obj_dict.get("component", "-"))
    logger = str(obj_dict.get("logger", "-"))
    message = str(obj_dict.get("message", "")).strip()

    lines = [f"[{timestamp}] {level:<7} {component} ({logger})"]

    if message:
        lines.append(message)

    known_keys = {
        "timestamp",
        "level",
        "component",
        "logger",
        "message",
        "stacktrace",
    }
    extra_keys = [key for key in obj_dict if key not in known_keys]

    if extra_keys:
        shown_parts: list[str] = []

        for key in extra_keys[:5]:
            value = obj_dict.get(key)
            if isinstance(value, (dict, list)):
                rendered = json.dumps(value, ensure_ascii=False)
            else:
                rendered = str(value)

            rendered = rendered.replace("\n", "\\n")
            if len(rendered) > 100:
                rendered = rendered[:97] + "..."

            shown_parts.append(f"{key}={rendered}")

        lines.append("  " + " | ".join(shown_parts))

        remaining = len(extra_keys) - 5
        if remaining > 0:
            lines.append(f"  ... +{remaining} more fields")

    stacktrace = obj_dict.get("stacktrace")
    if stacktrace:
        lines.append("stacktrace:")
        lines.append(str(stacktrace))

    return "\n".join(lines)


def _format_log_line(line: str, *, pretty_debug: bool = False) -> str:
    """Expand escaped newlines in a JSON log line's stacktrace field.

    LiteLLM json_logs mode serialises multi-line stack traces as a single JSON
    string with \\n escape sequences.  This function detects that pattern and
    emits the stacktrace separately so it is readable in the terminal.
    """
    try:
        obj = json.loads(line)
        if not isinstance(obj, dict):
            return line
        obj_dict = cast(dict[str, Any], obj)

        if pretty_debug:
            return _format_human_log_line(obj_dict)

        stacktrace = obj_dict.get("stacktrace")
        if stacktrace:
            without_trace = {k: v for k, v in obj_dict.items() if k != "stacktrace"}
            return json.dumps(without_trace) + "\n" + str(stacktrace)
        return line
    except (json.JSONDecodeError, ValueError):
        return line


def _stream_formatted_logs(command: list[str], *, pretty_debug: bool = False) -> int:
    """Run *command* and stream its merged stdout/stderr with log formatting."""
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ) as proc:
        assert proc.stdout is not None
        for raw in proc.stdout:
            print(
                _format_log_line(raw.rstrip("\n"), pretty_debug=pretty_debug),
                flush=True,
            )
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
    litellm_params: dict[str, Any]
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
    description: str | None
    detail: str | None
    allow_rebuild: bool
    rebuild_key_present: bool = False


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
        self.litellm_keep_alive: float | None = None
        self.litellm_debug: bool = False
        self.litellm_detailed_debug: bool = False

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

    def _load_litellm_runtime_settings(self, cfg: dict[str, Any]) -> None:
        """Load LiteLLM startup settings from top-level litellm_settings."""
        raw_settings: Any = cfg.get("litellm_settings")

        if raw_settings is None:
            raw_settings = {}

        if not isinstance(raw_settings, dict):
            print("ERROR: litellm_settings must be a key/value mapping")
            sys.exit(1)

        settings = cast(dict[str, Any], raw_settings)
        raw_keep_alive = settings.get("keep_alive")
        keep_alive = as_float(raw_keep_alive, None)

        if raw_keep_alive is not None and keep_alive is None:
            print("ERROR: litellm_settings.keep_alive must be numeric (seconds)")
            sys.exit(1)

        if keep_alive is not None and keep_alive <= 0:
            print("ERROR: litellm_settings.keep_alive must be greater than 0")
            sys.exit(1)

        self.litellm_keep_alive = keep_alive
        self.litellm_detailed_debug = as_bool(settings.get("detailed_debug"), False)
        self.litellm_debug = as_bool(
            settings.get("debug"),
            as_bool(settings.get("set_verbose"), False),
        )

    def _load_litellm_runtime_settings_from_file(self) -> None:
        """Best-effort reload of LiteLLM runtime settings from config file."""
        config_path = Path(self.config.litellm_config)

        if not config_path.exists():
            return

        try:
            with config_path.open("r", encoding="utf-8") as fh:
                cfg: Any = yaml.safe_load(fh) or {}
        except (OSError, yaml.YAMLError):
            return

        if not isinstance(cfg, dict):
            return

        cfg_dict = cast(dict[str, Any], cfg)
        raw_settings = cfg_dict.get("litellm_settings")
        if not isinstance(raw_settings, dict):
            return

        settings = cast(dict[str, Any], raw_settings)
        self.litellm_detailed_debug = as_bool(settings.get("detailed_debug"), False)
        self.litellm_debug = as_bool(
            settings.get("debug"),
            as_bool(settings.get("set_verbose"), False),
        )

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

        self._load_litellm_runtime_settings(cfg)

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
            description = model_info.get("description")
            detail = model_info.get("detail")
            rebuild_key_present = "rebuild" in entry
            rebuild_flag = entry.get("rebuild")
            allow_rebuild = as_bool(rebuild_flag, False)

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
                litellm_params=params,
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
                description=description,
                detail=detail,
                allow_rebuild=allow_rebuild,
                rebuild_key_present=rebuild_key_present,
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

    def _is_managed_vllm_container_name(self, name: str) -> bool:
        """Return True only for container names managed by this tool."""
        return bool(name) and name.startswith(VLLM_CONTAINER_PREFIX)

    def _is_safe_path_under_root(self, path: Path, *, allowed_root: Path) -> bool:
        """Return True only for absolute paths under an allowed root.

        This check is lexical only (no stat/resolve calls) so it remains safe on
        root-owned paths such as /var/lib/docker where permission to traverse may
        be restricted for non-root users.
        """
        if not path.is_absolute():
            return False

        normalized_allowed_root = Path(os.path.normpath(str(allowed_root)))
        normalized_path = Path(os.path.normpath(str(path)))

        try:
            normalized_path.relative_to(normalized_allowed_root)
            return True
        except ValueError:
            return False

    def _is_safe_log_path(self, log_path: Path, *, allowed_root: Path) -> bool:
        """Return True only for absolute .log paths under an allowed root.

        This check is lexical only (no stat/resolve calls) so it remains safe on
        root-owned paths such as /var/lib/docker where permission to traverse may
        be restricted for non-root users.
        """
        if not self._is_safe_path_under_root(log_path, allowed_root=allowed_root):
            return False

        if log_path.suffix != ".log":
            return False

        return True

    def _is_safe_docker_log_path(self, log_path: str) -> bool:
        """Return True only for Docker-managed json-file log paths."""
        return self._is_safe_log_path(Path(log_path), allowed_root=DOCKER_LOG_ROOT)

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
        if not self._is_managed_vllm_container_name(name):
            print(f"ERROR: Refusing to remove unmanaged container: {name}")
            print(f"Allowed prefix: {VLLM_CONTAINER_PREFIX}")
            sys.exit(1)

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

    def _host_nvidia_driver_version(self) -> str | None:
        """Read host NVIDIA driver version from nvidia-smi (if available)."""
        result = run_command(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture=True,
        )

        if result.returncode != 0:
            return None

        for line in result.stdout.splitlines():
            candidate = line.strip()
            if candidate:
                return candidate

        return None

    def _preflight_check_nvidia_driver(self, *, dry_run: bool = False) -> None:
        """Run a startup GPU/image compatibility probe and warn on failure.

        Runs a lightweight GPU-enabled container probe before launching vLLM.
        This catches common NVIDIA driver/image mismatches early with a
        readable warning message.
        """
        command = [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "--entrypoint",
            "python3",
            self.config.vllm_docker_image,
            "-c",
            "import sys; sys.exit(0)",
        ]

        if dry_run:
            print("Preflight check (dry-run): skipping NVIDIA compatibility probe.")
            print("Would run:", " ".join(command))
            return

        result = run_command(command, capture=True)

        if result.returncode == 0:
            print("NVIDIA pre-run compatibility check passed.")
            return

        output_parts = [result.stdout.strip(), result.stderr.strip()]
        output = "\n".join(part for part in output_parts if part)

        print("WARNING: NVIDIA pre-run compatibility check failed.")
        print(f"Docker image: {self.config.vllm_docker_image}")

        mismatch = extract_nvidia_driver_mismatch(output)

        if mismatch:
            required, detected = mismatch
            print(f"Required NVIDIA driver: >= {required}")
            print(f"Detected NVIDIA driver: {detected}")
        else:
            host_driver = self._host_nvidia_driver_version()
            if host_driver:
                print(f"Detected NVIDIA driver: {host_driver}")

        if output:
            print()
            print("Docker/NVIDIA output:")
            print(output)

        print()
        print(
            "Fix by selecting a compatible VLLM_DOCKER_IMAGE tag "
            "or upgrading the host NVIDIA driver."
        )
        print("Startup will continue, but container launch may fail.")

    def _find_all_vllm_containers(self, model_name: str) -> list[str]:
        """Find all vLLM container names for a model (running and stopped)."""
        prefix = f"{VLLM_CONTAINER_PREFIX}{docker_safe_name(model_name)}-"
        return [n for n in self._docker_list_container_names(all_containers=True) if n.startswith(prefix)]

    def _find_running_vllm_containers(self, model_name: str) -> list[str]:
        """Find running vLLM container names for a model."""
        prefix = f"{VLLM_CONTAINER_PREFIX}{docker_safe_name(model_name)}-"
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
            "ipc",
            "quantization",
            "kv_cache_dtype",
            "generation_config",
            "override_generation_config",
            "max_num_batched_tokens",
            "max_num_seqs",
            "attention_backend",
            "attention_config",
            "compilation_config",
            "structured_outputs_config",
            "enable_prefix_caching",
            "enforce_eager",
            "enable_auto_tool_choice",
            "reasoning_config",
            "tool_call_parser",
            "reasoning_parser",
            "speculative_config",
            "trust_remote_code",
        ]

        for model in self.models.values():
            raw_info = model.model_info
            backend_served = backend_model_without_provider(model.backend_model)

            print(f"{model.name}")
            if model.alias:
                print(f"  alias:                    {model.alias}")
            if model.description:
                print(f"  description:              {model.description}")
            if model.detail:
                print(f"  detail:                   {model.detail}")
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

        aliases = [(m.alias, m.name, m.description) for m in self.models.values() if m.alias]
        if aliases:
            col_alias = max(len(a) for a, _, _ in aliases)
            col_name = max(len(n) for _, n, _ in aliases)
            print("Aliases")
            print("-------")
            for alias, model_name, description in aliases:
                desc_display = description or "(no description)"
                print(f"  {alias:<{col_alias}}  →  {model_name:<{col_name}}  {desc_display}")
            print()

    def generate_help(self) -> str:
        """Generate dynamic help text."""
        keys = [
            "model_dir", "ipc", "quantization", "kv_cache_dtype", "generation_config",
            "override_generation_config", "max_num_batched_tokens", "max_num_seqs",
            "attention_backend", "attention_config", "enable_prefix_caching", "enforce_eager",
            "compilation_config", "structured_outputs_config",
            "reasoning_config",
            "enable_auto_tool_choice", "tool_call_parser", "reasoning_parser",
            "speculative_config", "license", "upstream",
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
    harinezumigel-llm-stack mistral_7b --start --stream-log  # start and follow logs (stays attached)
    harinezumigel-llm-stack mistral_7b --start --dry-run      # preview what would be done
    harinezumigel-llm-stack mistral_7b --start --recreate     # force recreate container

  View logs:
  ----------

    harinezumigel-llm-stack mistral_7b --show-log             # show last 200 lines
    harinezumigel-llm-stack mistral_7b --stream-log           # show and follow logs
    harinezumigel-llm-stack mistral_7b --show-log --follow    # equivalent to --stream-log

  Show log paths:
  ---------------

    harinezumigel-llm-stack mistral_7b --show-log-path        # show Docker log file path
    harinezumigel-llm-stack litellm --show-log-path           # show LiteLLM log directory + latest log
    harinezumigel-llm-stack all --show-log-path               # show vLLM + LiteLLM log paths
    harinezumigel-llm-stack all --show-log                    # shows paths for all logs

  Clean logs:
  -----------

    harinezumigel-llm-stack mistral_7b --clean-log            # truncate log file for one model
    harinezumigel-llm-stack litellm --clean-log               # remove LiteLLM timestamped logs
    harinezumigel-llm-stack all --clean-log                   # truncate vLLM + clean LiteLLM logs
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
    harinezumigel-llm-stack litellm --show-log
    harinezumigel-llm-stack litellm --show-log-path
    harinezumigel-llm-stack litellm --clean-log
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
        since: str | None = None,
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

        if since is not None:
            command = ["docker", "logs", "--since", since]
        else:
            command = ["docker", "logs", "--tail", str(tail)]

        if follow:
            command.append("-f")

        command.append(container_name)

        returncode = _stream_formatted_logs(command)

        if returncode != 0:
            print(f"ERROR: Failed to read logs for container: {container_name}")
            sys.exit(returncode)

    def _litellm_logs_dir(self, *, create: bool = True) -> Path:
        """Use only the logs directory under the existing LiteLLM deployment."""
        logs_dir = INSTALLED_BASE / "logs"
        if logs_dir.is_symlink():
            print(f"ERROR: LiteLLM logs directory must not be a symlink: {logs_dir}")
            sys.exit(1)

        if create:
            try:
                logs_dir.mkdir(exist_ok=True)
            except OSError as exc:
                print(f"ERROR: Failed to create LiteLLM log directory: {logs_dir}")
                print(str(exc))
                sys.exit(1)

        return logs_dir

    def _litellm_log_path(self) -> Path | None:
        """Return the latest managed LiteLLM log path, if present."""
        logs_dir = self._litellm_logs_dir()

        if logs_dir.exists() and logs_dir.is_dir():
            paths = sorted(
                path
                for path in logs_dir.glob("litellm_*.log")
                if path.is_file() and not path.is_symlink()
            )
            if paths:
                return paths[-1]

        return None

    def _new_litellm_log_path(self, *, create: bool = True) -> Path:
        """Return the next timestamped LiteLLM log file path."""
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return self._litellm_logs_dir(create=create) / f"litellm_{stamp}.log"

    def _is_safe_litellm_log_path(self, log_path: Path) -> bool:
        """Return True only for safe LiteLLM log paths.

        Safety policy:
            - Must be an absolute path
            - Must end with .log
            - Must stay under /opt/litellm/logs
        """
        return self._is_safe_log_path(log_path, allowed_root=INSTALLED_BASE / "logs")

    def _is_safe_litellm_logs_dir(self, logs_dir: Path) -> bool:
        """Return True only for safe LiteLLM logs directories."""
        return logs_dir == INSTALLED_BASE / "logs" and not logs_dir.is_symlink()

    def _tail_file_logs(
        self,
        log_path: Path,
        *,
        tail: str,
        follow: bool,
        pretty_debug: bool = False,
    ) -> int:
        """Tail a local log file and stream output through the formatter."""
        if tail.lower() == "all":
            command = ["tail", "-n", "+1"]
        else:
            command = ["tail", "-n", str(tail)]

        if follow:
            command.append("-f")

        command.append(str(log_path))
        return _stream_formatted_logs(command, pretty_debug=pretty_debug)

    def show_litellm_logs(
        self,
        *,
        tail: str,
        follow: bool,
        show_log_path: bool,
    ) -> None:
        """Show logs for the host-process LiteLLM instance."""
        self._load_litellm_runtime_settings_from_file()
        pretty_debug = self.litellm_debug or self.litellm_detailed_debug

        logs_dir = self._litellm_logs_dir()
        log_path = self._litellm_log_path()
        state = "running" if self._find_litellm_pids() else "stopped"

        print()
        print("=== LiteLLM logs ===")
        print(f"Process:   {state}")
        print(f"Logs dir:  {logs_dir}")

        if show_log_path:
            print(f"Log path:  {log_path or '(none)'}")

        print()

        if log_path is None:
            print(f"No LiteLLM log file found in {logs_dir}")
            return

        returncode = self._tail_file_logs(
            log_path,
            tail=tail,
            follow=follow,
            pretty_debug=pretty_debug,
        )

        if returncode != 0:
            print(f"ERROR: Failed to read LiteLLM log file: {log_path}")
            sys.exit(returncode)

    def show_litellm_log_path(self) -> None:
        """Show LiteLLM logs directory and latest log file path."""
        logs_dir = self._litellm_logs_dir()
        log_path = self._litellm_log_path()
        state = "running" if self._find_litellm_pids() else "stopped"

        print()
        print("=== LiteLLM log path ===")
        print("litellm")
        print(f"  state:    {state}")
        print(f"  logs_dir: {logs_dir}")
        print(f"  log_path: {log_path or '(none)'}")
        print(f"  exists:   {'yes' if log_path is not None and log_path.exists() else 'no'}")
        print()

    def clean_litellm_logs(self, *, dry_run: bool = False) -> None:
        """Clean/remove managed LiteLLM log files."""
        logs_dir = self._litellm_logs_dir(create=False)

        print()
        print("=== Cleaning LiteLLM logs ===")
        print("litellm")
        print(f"  Logs dir: {logs_dir}")

        if not logs_dir.is_dir():
            print("  Status:   Skipped - logs directory does not exist")
            print()
            return

        if not self._is_safe_litellm_logs_dir(logs_dir):
            print(f"  Status:   ✗ Skipped — unexpected logs directory: {logs_dir}")
            print(f"            Allowed root: {INSTALLED_BASE / 'logs'}")
            print()
            return

        safe_paths = [
            path
            for path in sorted(logs_dir.glob("*.log"))
            if path.is_file()
            and not path.is_symlink()
            and self._is_safe_litellm_log_path(path)
        ]

        if dry_run:
            print(f"  Action:   Would remove {len(safe_paths)} file(s) (dry run)")
            print()
            return

        if not safe_paths:
            print("  Status:   ✗ Skipped — no managed LiteLLM log files found")
            print()
            return

        removed = 0
        failures: list[str] = []

        for path in safe_paths:
            try:
                path.unlink()
                removed += 1
            except OSError as exc:
                failures.append(f"{path}: {exc}")

        if failures:
            print(f"  Status:   ✗ Removed {removed}, failed {len(failures)}")
            for failure in failures:
                print(f"            {failure}")
        else:
            print(f"  Status:   ✓ Removed {removed} file(s)")

        print()

    def show_all_log_paths(self) -> None:
        """Show log paths for all vLLM containers and LiteLLM."""
        self.show_vllm_log_paths("all")
        self.show_litellm_log_path()

    def clean_all_logs(self, *, dry_run: bool = False) -> None:
        """Clean/truncate all vLLM logs and remove LiteLLM managed logs."""
        self.clean_vllm_logs("all", dry_run=dry_run)
        self.clean_litellm_logs(dry_run=dry_run)

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
                if not self._is_safe_docker_log_path(log_path):
                    print(f"  Status:   ✗ Skipped — unexpected log path: {log_path}")
                    print(f"            Allowed root: {DOCKER_LOG_ROOT}")
                    continue

                result = run_command(["sudo", "truncate", "-s", "0", log_path], capture=True)

                if result.returncode == 0:
                    state = self._docker_container_state(container)
                    print("  Status:   ✓ Cleaned")
                    if state == "running":
                        # docker logs caches the old EOF offset; restart resets the reader
                        print(f"{_RED}  Warning:  'docker logs' won't show new entries until the container is restarted.{_RESET}")
                        print(f"{_RED}            Run: docker restart {container}{_RESET}")
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
        ipc_arg = docker_ipc_arg(raw_info.get("ipc"))

        command = [
            "docker", "run", "-d",
            "--name", f"vllm-{docker_safe_name(model.name)}-{port}",
            "--runtime=nvidia",
            "--gpus=all",
        ]

        if ipc_arg:
            command.append(ipc_arg)

        command.extend(
            [
                "-p", f"{self.config.vllm_bind_host}:{port}:{self.config.vllm_container_port}",
                "-v", self.config.vllm_model_volume,
                "-v", self.config.vllm_cache_volume,
                "-v", "/etc/timezone:/etc/timezone:ro",
                "-v", "/etc/localtime:/etc/localtime:ro",
                "-e", "VLLM_CONFIGURE_LOGGING=1",
                self.config.vllm_docker_image,
                # Some NVIDIA images export internal VLLM_* vars that newer vLLM
                # versions don't recognize. Remove them for a clean startup.
                "env",
                "-u",
                "VLLM_VERSION",
                "-u",
                "VLLM_FLASH_ATTN_SRC_DIR",
                "-u",
                "VLLM_LOG_LEVEL",
                "python3",
                "-u",
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model", f"/models/{model_dir.name}",
                "--served-model-name", model.name,
                "--dtype", str(runtime.get("dtype", "auto")),
                "--max-model-len", str(runtime["context_length"]),
                "--gpu-memory-utilization", str(runtime["gpu_memory_utilization"]),
                "--host", "0.0.0.0",
            ]
        )

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

        attention_config = raw_info.get("attention_config")

        if attention_config:
            if not isinstance(attention_config, str):
                attention_config = json.dumps(attention_config, separators=(",", ":"))
            command.extend(["--attention-config", attention_config])

        for config_key in ("compilation_config", "structured_outputs_config"):
            config_value = raw_info.get(config_key)
            if config_value:
                if not isinstance(config_value, str):
                    config_value = json.dumps(config_value, separators=(",", ":"))
                command.extend([f"--{config_key.replace('_', '-')}", config_value])

        if raw_info.get("generation_config"):
            command.extend(["--generation-config", str(raw_info["generation_config"])])

        # Note: Sampling parameters (temperature, top_p, top_k, repetition_penalty, etc.)
        # are request-time parameters in vLLM, not server startup parameters.
        # They should be passed in API requests, not as command-line flags.
        # The parameters from config.yaml are used for display/documentation only.

        if raw_info.get("quantization"):
            command.extend(["--quantization", str(raw_info["quantization"])])

        if raw_info.get("kv_cache_dtype"):
            command.extend(["--kv-cache-dtype", str(raw_info["kv_cache_dtype"])])

        if as_bool(raw_info.get("enable_auto_tool_choice", False)):
            command.append("--enable-auto-tool-choice")

        reasoning_config_value = raw_info.get("reasoning_config")
        reasoning_parser = raw_info.get("reasoning_parser")
        reasoning_config_arg = ""

        if isinstance(reasoning_config_value, str):
            try:
                reasoning_config_value = json.loads(reasoning_config_value)
            except json.JSONDecodeError:
                print("ERROR: model_info.reasoning_config must be valid JSON.")
                sys.exit(1)

        if isinstance(reasoning_config_value, dict):
            normalized_reasoning_config: dict[str, Any] = dict(
                cast(dict[str, Any], reasoning_config_value)
            )

            if (
                "enable_reasoning" in normalized_reasoning_config
                or "enable_thinking" in normalized_reasoning_config
            ):
                print(
                    "ERROR: model_info.reasoning_config no longer accepts "
                    "enable_reasoning/enable_thinking. "
                    "Use reasoning_parser, reasoning_start_str, and/or reasoning_end_str."
                )
                sys.exit(1)

            nested_parser = normalized_reasoning_config.pop("reasoning_parser", None)
            if reasoning_parser and nested_parser and reasoning_parser != nested_parser:
                print(
                    "ERROR: model_info.reasoning_parser conflicts with "
                    "model_info.reasoning_config.reasoning_parser."
                )
                sys.exit(1)
            reasoning_parser = reasoning_parser or nested_parser

            if normalized_reasoning_config:
                reasoning_config_arg = json.dumps(
                    normalized_reasoning_config,
                    separators=(",", ":"),
                )
        elif reasoning_config_value is not None:
            print(
                "ERROR: model_info.reasoning_config must be an object or JSON string, "
                "containing an object."
            )
            sys.exit(1)

        if reasoning_config_arg:
            command.extend(["--reasoning-config", reasoning_config_arg])

        if raw_info.get("tool_call_parser"):
            command.extend(["--tool-call-parser", str(raw_info["tool_call_parser"])])

        if reasoning_parser:
            command.extend(["--reasoning-parser", str(reasoning_parser)])

        speculative_config = raw_info.get("speculative_config")

        if speculative_config:
            if not isinstance(speculative_config, str):
                speculative_config = json.dumps(speculative_config, separators=(",", ":"))
            command.extend(["--speculative-config", speculative_config])

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
        ipc_arg = docker_ipc_arg(raw_info.get("ipc"))
        ipc_mode = ipc_arg.split("=", 1)[1] if ipc_arg else "(default)"

        print(f"=== Starting vLLM model: {model.name} ===")
        print(f"Docker image:             {self.config.vllm_docker_image}")
        print(f"Container:                vllm-{docker_safe_name(model.name)}-{port}")
        print(f"Docker IPC mode:          {ipc_mode}")
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
        print(f"Attention config:          {raw_info.get('attention_config', '(none)')}")
        print(f"Compilation config:        {raw_info.get('compilation_config', '(none)')}")
        print(f"Structured outputs config: {raw_info.get('structured_outputs_config', '(none)')}")
        print(f"Generation config:         {raw_info.get('generation_config', '(model default)')}")

        # Show sampling parameters
        override_gen_config = cast(dict[str, Any], raw_info.get('override_generation_config') or {})
        litellm_params = model.litellm_params or {}
        sampling_info = format_sampling_parameters(override_gen_config, litellm_params)

        if sampling_info:
            print(f"Sampling parameters:       {', '.join(sampling_info)}")

        print(f"Auto tool choice:          {as_bool(raw_info.get('enable_auto_tool_choice', False))}")
        print(f"Reasoning config:          {raw_info.get('reasoning_config', '(none)')}")
        print(f"Tool call parser:          {raw_info.get('tool_call_parser', '(none)')}")
        print(f"Reasoning parser:          {raw_info.get('reasoning_parser', '(none)')}")
        print(f"Speculative config:        {raw_info.get('speculative_config', '(none)')}")
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
        if options.recreate and not model.allow_rebuild:
            if not model.rebuild_key_present:
                print(
                    f"ERROR: --recreate is blocked for model '{model.name}': "
                    "rebuild: true must be specified at the model entry level in config.yaml"
                )
            else:
                print(
                    f"ERROR: --recreate is blocked for model '{model.name}' "
                    "(set top-level rebuild: true in config.yaml to allow recreate)"
                )
            sys.exit(1)

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

                self._preflight_check_nvidia_driver(dry_run=options.dry_run)
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

        self._preflight_check_nvidia_driver(dry_run=options.dry_run)
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
        - Enforces a single-instance policy for this LITELLM_CONFIG
        - Checks if port is already in use before starting
        - Respects dry_run flag
        """
        existing_pids = self._find_litellm_pids()
        my_pid = str(os.getpid())

        if my_pid in existing_pids:
            existing_pids.remove(my_pid)

        if existing_pids:
            print("ERROR: Another LiteLLM instance is already running.")
            print(
                "Only one LiteLLM instance can run for this config path when "
                "started via harinezumigel-llm-stack."
            )
            print("Running LiteLLM PID(s):")

            for pid in existing_pids:
                print(f"  {pid}")

            print()
            print(
                "Stop the existing instance first "
                "(harinezumigel-llm-stack litellm --stop for this config, "
                "or stop it manually)."
            )
            sys.exit(1)

        if port_in_use(self.config.litellm_port, self.config.litellm_bind_host):
            print(
                f"LiteLLM port {self.config.litellm_port} appears to be in use on "
                f"{self.config.litellm_bind_host}"
            )
            return

        log_level = "DEBUG" if (self.litellm_debug or self.litellm_detailed_debug) else "INFO"
        keep_alive_export = (
            f"export AIOHTTP_KEEPALIVE_TIMEOUT={self.litellm_keep_alive:g} && "
            if self.litellm_keep_alive is not None
            else ""
        )
        debug_flag = (
            " --detailed_debug"
            if self.litellm_detailed_debug
            else (" --debug" if self.litellm_debug else "")
        )

        command = (
            f"set -a && "
            f"source {self.config.env_file} && "
            f"set +a && "
            f"export LITELLM_LOG={log_level} && "
            f"{keep_alive_export}"
            f"source {self.config.litellm_venv_activate} && "
            # litellm CLI does bare `from proxy_server import` which only works when
            # the proxy package directory is on PYTHONPATH
            f"export PYTHONPATH=\"$(python -c 'import litellm,os;print(os.path.dirname(litellm.__file__)+\"/proxy\")')\" && "
            f"{self.config.litellm_bin} --config {self.config.litellm_config} "
            f"--host {self.config.litellm_bind_host} "
            f"--port {self.config.litellm_port}{debug_flag}"
        )
        log_path = self._new_litellm_log_path(create=not dry_run)

        quoted_log_path = shlex.quote(str(log_path))
        detached_command = f"{command} >> {quoted_log_path} 2>&1"
        streaming_command = f"{command} 2>&1 | tee -a {quoted_log_path}"

        print(
            f"\n=== Starting LiteLLM router on "
            f"{self.config.litellm_bind_host}:{self.config.litellm_port} ==="
        )
        print(f"Config: {self.config.litellm_config}")
        print(f"Log file: {log_path}")
        print()

        # Display configured models with their sampling parameters
        print("Configured models:")
        for model_name, model in self.models.items():
            raw_info = model.model_info
            override_gen_config = cast(dict[str, Any], raw_info.get('override_generation_config') or {})
            litellm_params = model.litellm_params or {}
            sampling_parts = format_sampling_parameters(override_gen_config, litellm_params)

            alias_str = f" (alias: {model.alias})" if model.alias else ""
            sampling_str = f" [{', '.join(sampling_parts)}]" if sampling_parts else ""
            print(f"  - {model_name}{alias_str}{sampling_str}")

        print()
        print("Command:", command)

        if not dry_run:
            if follow_log:
                _stream_formatted_logs(
                    ["bash", "-lc", streaming_command],
                    pretty_debug=self.litellm_debug or self.litellm_detailed_debug,
                )
            else:
                start_background_process(["bash", "-lc", detached_command])

    def _find_litellm_pids(self) -> list[str]:
        """Find LiteLLM process IDs.

        Matches only real LiteLLM CLI processes for this LITELLM_CONFIG.

        Uses ``pgrep -fa`` so we can inspect the full argv for each candidate.
        This avoids false positives from wrapper tools (for example log viewers
        or shell commands) that merely contain "litellm" in their command line.
        """
        pids: list[str] = []
        target_config = os.path.normpath(self.config.litellm_config)
        configured_bin = os.path.normpath(self.config.litellm_bin)
        result = run_command(["pgrep", "-fa", "litellm"], capture=True)

        if result.returncode not in (0, 1):
            print("ERROR: Failed to query LiteLLM processes")
            print(result.stderr.strip())
            sys.exit(1)

        def _extract_config(argv: list[str]) -> str | None:
            """Return the --config value from argv, if present."""
            for index, token in enumerate(argv):
                if token == "--config" and index + 1 < len(argv):
                    return argv[index + 1]
                if token.startswith("--config="):
                    return token.split("=", 1)[1]
            return None

        def _is_litellm_argv(argv: list[str]) -> bool:
            """Return True only for direct LiteLLM CLI process argv forms."""
            if not argv:
                return False

            executable = Path(argv[0]).name

            if executable == "litellm":
                return True

            if os.path.normpath(argv[0]) == configured_bin:
                return True

            if executable.startswith("python"):
                if len(argv) > 1 and os.path.normpath(argv[1]) == configured_bin:
                    return True
                if len(argv) > 1 and Path(argv[1]).name == "litellm":
                    return True

                for index, token in enumerate(argv):
                    if token == "-m" and index + 1 < len(argv):
                        module_name = argv[index + 1]
                        if module_name == "litellm" or module_name.startswith("litellm."):
                            return True

            return False

        for line in result.stdout.splitlines():
            pid_and_cmd = line.strip().split(maxsplit=1)

            if len(pid_and_cmd) != 2:
                continue

            pid = pid_and_cmd[0]

            if not pid.isdigit():
                continue

            try:
                argv = shlex.split(pid_and_cmd[1])
            except ValueError:
                continue

            if not _is_litellm_argv(argv):
                continue

            config_value = _extract_config(argv)

            if config_value is None:
                continue

            if os.path.normpath(config_value) != target_config:
                continue

            pids.append(pid)

        return sorted(set(pids))

    def stop_litellm(self, *, dry_run: bool = False) -> None:
        """Stop LiteLLM proxy processes.

        Finds LiteLLM processes by matching command patterns and sends
        SIGTERM signal to stop them gracefully.

        Safety:
        - Matches only processes that include this config path
        - Shows PIDs before killing
        - Uses SIGTERM (graceful shutdown) not SIGKILL
        - Respects dry_run flag
        - Includes self-PID protection
        """
        unique_pids = self._find_litellm_pids()

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

        if target in LITELLM_TARGETS:
            if args.clean_log:
                self.clean_litellm_logs(dry_run=args.dry_run)
                return True

            if args.show_log_path:
                self.show_litellm_log_path()
                return True

            self.show_litellm_logs(tail=args.tail, follow=args.follow, show_log_path=False)
            return True

        if target == "all":
            if args.clean_log:
                self.clean_all_logs(dry_run=args.dry_run)
                return True

            if args.show_log_path or args.show_log:
                self.show_all_log_paths()
                return True

        if args.clean_log:
            self.clean_vllm_logs(target, dry_run=args.dry_run)
            return True

        if args.show_log_path:
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

        self.show_vllm_logs(resolved, tail=args.tail, follow=args.follow, show_log_path=False)
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

        if target in LITELLM_TARGETS:
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
        # Handle litellm target
        if args.model and args.model.lower() in LITELLM_TARGETS:
            if args.recreate:
                print("ERROR: --recreate is not allowed for litellm")
                sys.exit(1)

            if args.start:
                # Load models to display their sampling parameters
                self.load_models()
                self.start_litellm(dry_run=args.dry_run, follow_log=args.show_log)
                return

            if args.stop:
                if args.show_log or args.follow:
                    print("ERROR: --show-log, --follow, and --stream-log are not allowed with --stop")
                    sys.exit(1)
                self.stop_litellm(dry_run=args.dry_run)
                return

            if args.clean_log:
                self.clean_litellm_logs(dry_run=args.dry_run)
                return

            if args.show_log_path:
                self.show_litellm_log_path()
                return

            if args.show_log:
                self.show_litellm_logs(tail=args.tail, follow=args.follow, show_log_path=False)
                return

            print(
                "ERROR: Specify an action for litellm "
                "(--start, --stop, --show-log, --show-log-path, or --clean-log)"
            )
            sys.exit(1)

        self.load_models()

        if args.help:
            print(self.generate_help())
            return

        if args.list:
            self.list_models()
            return

        if args.stop and (args.show_log or args.follow):
            print("ERROR: --show-log, --follow, and --stream-log are not allowed with --stop")
            sys.exit(1)

        if self.handle_logs(args):
            return

        if self.handle_ps(args):
            return

        if args.recreate and (not args.start or not args.model):
            print("ERROR: --recreate requires --start and a model or alias")
            sys.exit(1)

        if args.recreate and args.stop:
            print("ERROR: --recreate is not allowed when --stop is given")
            sys.exit(1)

        if self.handle_stop(args):
            return

        if not args.model:
            self.list_models()
            return

        # Model specified — require an action
        if not args.start and not args.show_log and not args.show_log_path:
            print(f"ERROR: Specify an action for model '{args.model}'")
            print()
            print("Available actions:")
            print("  --start              Start the container")
            print("  --start --stream-log Start and show logs")
            print("  --show-log           View logs")
            print("  --stream-log         Follow logs")
            print("  --stop               Stop the container")
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
            log_since = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
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
                    follow=True,
                    show_log_path=False,
                    since=log_since,
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

    parser.add_argument(
        "--show-log",
        action="store_true",
        help="Show logs (for 'all': show vLLM + LiteLLM log paths)",
    )
    parser.add_argument("--stream-log", action="store_true", help="Show and follow logs (equivalent to --show-log --follow)")
    parser.add_argument(
        "--show-log-path",
        action="store_true",
        help="Show log file paths (Docker + LiteLLM)",
    )
    parser.add_argument(
        "--clean-log",
        action="store_true",
        help="Clean/truncate log files (Docker + LiteLLM)",
    )
    parser.add_argument("--follow", "-f", action="store_true", help="Follow log output (use with --show-log or use --stream-log)")
    parser.add_argument("--tail", default="200")

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--help", action="store_true")

    args = parser.parse_args()

    if args.stream_log and args.show_log:
        parser.error("--stream-log is mutually exclusive with --show-log")

    if args.stream_log and args.follow:
        parser.error("--stream-log is mutually exclusive with --follow")

    if args.stream_log:
        args.show_log = True
        args.follow = True

    return args


def main() -> None:
    """Program entry point."""
    msg = f" harinezumigel-llm-stack {__version__} "
    border = "*" * (len(msg) + 2)
    print(f"\n{border}\n*{msg}*\n{border}\n")
    LLMStack(APP).run(parse_args())


if __name__ == "__main__":
    main()
