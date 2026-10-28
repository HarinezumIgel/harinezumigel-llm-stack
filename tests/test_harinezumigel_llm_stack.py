#!/usr/bin/env python3
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
"""Test suite for harinezumigel-llm-stack.

This module tests the core functionality of the LLM stack launcher including:
- Environment variable loading and parsing
- Configuration validation
- Utility functions
- Docker container management
- Model deployment operations
"""

import errno
import io
import json
import os
import runpy
import socket
import sys
import tempfile
import importlib.util
import unittest
from pathlib import Path
from dataclasses import replace
from contextlib import redirect_stderr, redirect_stdout
from typing import Any
from unittest.mock import MagicMock, patch

# Add the source directory to Python path
PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPT_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SCRIPT_DIR))

# Import the main script as a module
# Using importlib to handle the file with dashes in the name
spec = importlib.util.spec_from_file_location(
    "harinezumigel_llm_stack", SCRIPT_DIR / "harinezumigel-llm-stack.py"
)
if spec is None or spec.loader is None:
    raise RuntimeError("Failed to load harinezumigel-llm-stack.py module spec")

harinezumigel_llm_stack = importlib.util.module_from_spec(spec)
sys.modules["harinezumigel_llm_stack"] = harinezumigel_llm_stack

# Import against the example environment, never the machine's deployment or keys.
path_is_dir = Path.is_dir
path_exists = Path.exists
path_open = Path.open
with (
    patch.dict(os.environ, {}, clear=True),
    patch.object(
        Path,
        "exists",
        autospec=True,
        side_effect=lambda path: (
            True if path == Path("/opt/litellm/config/env") else path_exists(path)
        ),
    ),
    patch.object(
        Path,
        "open",
        autospec=True,
        side_effect=lambda path, *args, **kwargs: path_open(
            (
                PROJECT_DIR / "config" / "env.example"
                if path == Path("/opt/litellm/config/env")
                else path
            ),
            *args,
            **kwargs,
        ),
    ),
    patch.object(
        Path,
        "is_dir",
        autospec=True,
        side_effect=lambda path: path == Path("/opt/litellm") or path_is_dir(path),
    ),
):
    spec.loader.exec_module(harinezumigel_llm_stack)


def safe_unlink(path: str | Path) -> None:
    """Unlink a file path only when it exists."""
    file_path = Path(path)
    if file_path.exists():
        file_path.unlink()


class TestEnvironmentLoading(unittest.TestCase):
    """Test environment variable loading and parsing."""

    def test_load_env_file_basic(self):
        """Test loading basic KEY=value pairs."""
        from harinezumigel_llm_stack import load_env_file  # type: ignore

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix="env") as f:
            f.write("TEST_VAR1=value1\n")
            f.write("TEST_VAR2=value2\n")
            f.write("# Comment line\n")
            f.write("\n")
            f.write("TEST_VAR3=value3\n")
            temp_path = f.name

        try:
            load_env_file(temp_path)
            self.assertEqual(os.environ.get("TEST_VAR1"), "value1")
            self.assertEqual(os.environ.get("TEST_VAR2"), "value2")
            self.assertEqual(os.environ.get("TEST_VAR3"), "value3")
        finally:
            safe_unlink(temp_path)
            # Clean up environment
            for key in ["TEST_VAR1", "TEST_VAR2", "TEST_VAR3"]:
                os.environ.pop(key, None)

    def test_load_env_file_with_export(self):
        """Test loading KEY=value with export prefix."""
        from harinezumigel_llm_stack import load_env_file  # type: ignore

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix="env") as f:
            f.write("export EXPORT_VAR1=exported1\n")
            f.write("EXPORT_VAR2=exported2\n")
            temp_path = f.name

        try:
            load_env_file(temp_path)
            self.assertEqual(os.environ.get("EXPORT_VAR1"), "exported1")
            self.assertEqual(os.environ.get("EXPORT_VAR2"), "exported2")
        finally:
            safe_unlink(temp_path)
            os.environ.pop("EXPORT_VAR1", None)
            os.environ.pop("EXPORT_VAR2", None)

    def test_load_env_file_with_variable_expansion(self):
        """Test variable expansion ${VAR} and $VAR."""
        from harinezumigel_llm_stack import load_env_file  # type: ignore

        os.environ["BASE_DIR"] = "/opt/test"

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix="env") as f:
            f.write("EXPANDED1=${BASE_DIR}/subdir\n")
            f.write("EXPANDED2=$BASE_DIR/another\n")
            temp_path = f.name

        try:
            load_env_file(temp_path)
            self.assertEqual(os.environ.get("EXPANDED1"), "/opt/test/subdir")
            self.assertEqual(os.environ.get("EXPANDED2"), "/opt/test/another")
        finally:
            safe_unlink(temp_path)
            os.environ.pop("BASE_DIR", None)
            os.environ.pop("EXPANDED1", None)
            os.environ.pop("EXPANDED2", None)

    def test_load_env_file_nonexistent(self):
        """Test that loading nonexistent file doesn't raise error."""
        from harinezumigel_llm_stack import load_env_file  # type: ignore

        # Should not raise
        load_env_file("/nonexistent/fileenv")

    def test_load_env_file_inline_comments(self):
        """Strip comments without damaging quoted or embedded hash characters."""
        cases = [
            ("localhost    # <YOUR HOST>", "localhost"),
            ("0.0.0.0\t# <YOUR BIND HOST>", "0.0.0.0"),
            ("8011 # backend port", "8011"),
            ('"value # literal" # comment', "value # literal"),
            ("'value # literal' # comment", "value # literal"),
            ("secret#suffix # comment", "secret#suffix"),
            ("# comment only", ""),
            ('"" # empty value', ""),
            ('"  padded # literal  " # comment', "  padded # literal  "),
            ('"say \\"hi # literal\\"" # comment', 'say \\"hi # literal\\"'),
        ]
        with patch.dict(os.environ):
            for raw_value, expected in cases:
                with self.subTest(raw_value=raw_value):
                    with tempfile.TemporaryDirectory() as directory:
                        env_path = Path(directory) / "env"
                        env_path.write_text(
                            f"TEST_INLINE={raw_value}\n", encoding="utf-8"
                        )
                        harinezumigel_llm_stack.load_env_file(str(env_path))
                    self.assertEqual(os.environ["TEST_INLINE"], expected)

    def test_env_string_required(self):
        """Test env_string with required flag."""
        from harinezumigel_llm_stack import env_string  # type: ignore

        os.environ["REQUIRED_VAR"] = "test_value"
        self.assertEqual(env_string("REQUIRED_VAR", required=True), "test_value")

        # Test missing required variable causes exit
        with self.assertRaises(SystemExit):
            env_string("MISSING_REQUIRED_VAR", required=True)

        os.environ.pop("REQUIRED_VAR", None)

    def test_env_string_with_default(self):
        """Test env_string with default value."""
        from harinezumigel_llm_stack import env_string  # type: ignore

        result = env_string("NONEXISTENT_VAR", default="default_value")
        self.assertEqual(result, "default_value")

    def test_env_int(self):
        """Test env_int parsing."""
        from harinezumigel_llm_stack import env_int  # type: ignore

        os.environ["INT_VAR"] = "42"
        self.assertEqual(env_int("INT_VAR"), 42)

        # Test invalid integer
        os.environ["BAD_INT"] = "not_a_number"
        with self.assertRaises(SystemExit):
            env_int("BAD_INT", required=True)

        os.environ.pop("INT_VAR", None)
        os.environ.pop("BAD_INT", None)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""

    def test_resolve_env_value_plain_string(self):
        """Test resolve_env_value with plain strings."""
        from harinezumigel_llm_stack import resolve_env_value  # type: ignore

        self.assertEqual(resolve_env_value("plain"), "plain")
        self.assertEqual(resolve_env_value(None), "")
        self.assertEqual(resolve_env_value(123), 123)

    def test_resolve_env_value_os_environ(self):
        """Test resolve_env_value with os.environ/ prefix."""
        from harinezumigel_llm_stack import resolve_env_value  # type: ignore

        os.environ["TEST_RESOLVE"] = "resolved_value"
        result = resolve_env_value("os.environ/TEST_RESOLVE")
        self.assertEqual(result, "resolved_value")
        os.environ.pop("TEST_RESOLVE", None)

    def test_resolve_env_value_shell_style(self):
        """Test resolve_env_value with shell-style variables."""
        from harinezumigel_llm_stack import resolve_env_value  # type: ignore

        os.environ["SHELL_VAR"] = "expanded"
        result = resolve_env_value("$SHELL_VAR/path")
        self.assertEqual(result, "expanded/path")
        os.environ.pop("SHELL_VAR", None)

    def test_docker_safe_name(self):
        """Test docker_safe_name conversion."""
        from harinezumigel_llm_stack import docker_safe_name  # type: ignore

        self.assertEqual(docker_safe_name("Model_Name"), "model_name")
        self.assertEqual(docker_safe_name("model@#$name"), "model-name")
        self.assertEqual(docker_safe_name("--multiple---dashes--"), "multiple-dashes")
        self.assertEqual(docker_safe_name("CamelCase123"), "camelcase123")

    def test_backend_model_without_provider(self):
        """Test backend_model_without_provider."""
        from harinezumigel_llm_stack import backend_model_without_provider  # type: ignore

        self.assertEqual(backend_model_without_provider("openai/gpt-4"), "gpt-4")
        self.assertEqual(backend_model_without_provider("model-name"), "model-name")
        self.assertEqual(backend_model_without_provider(""), "")

    def test_as_int(self):
        """Test as_int conversion."""
        from harinezumigel_llm_stack import as_int  # type: ignore

        self.assertEqual(as_int("42"), 42)
        self.assertEqual(as_int(42), 42)
        self.assertEqual(as_int("not_a_number", default=99), 99)
        self.assertIsNone(as_int(None))
        self.assertEqual(as_int(None, default=10), 10)

    def test_as_float(self):
        """Test as_float conversion."""
        from harinezumigel_llm_stack import as_float  # type: ignore

        self.assertEqual(as_float("3.14"), 3.14)
        self.assertEqual(as_float(3.14), 3.14)
        self.assertEqual(as_float("not_a_number", default=1.5), 1.5)
        self.assertIsNone(as_float(None))
        self.assertEqual(as_float(None, default=2.5), 2.5)

    def test_as_bool(self):
        """Test as_bool conversion."""
        from harinezumigel_llm_stack import as_bool  # type: ignore

        # Test truthy values
        self.assertTrue(as_bool(True))
        self.assertTrue(as_bool("true"))
        self.assertTrue(as_bool("True"))
        self.assertTrue(as_bool("1"))
        self.assertTrue(as_bool("yes"))
        self.assertTrue(as_bool("on"))

        # Test falsy values
        self.assertFalse(as_bool(False))
        self.assertFalse(as_bool("false"))
        self.assertFalse(as_bool("0"))
        self.assertFalse(as_bool("no"))
        self.assertFalse(as_bool("y"))  # 'y' removed from truthy set
        self.assertFalse(as_bool(""))
        self.assertFalse(as_bool(None))

    def test_docker_ipc_arg(self):
        """Test docker_ipc_arg normalization."""
        from harinezumigel_llm_stack import docker_ipc_arg  # type: ignore

        self.assertIsNone(docker_ipc_arg(None))
        self.assertIsNone(docker_ipc_arg(""))
        self.assertEqual(docker_ipc_arg("host"), "--ipc=host")
        self.assertEqual(docker_ipc_arg(" --ipc=host "), "--ipc=host")
        self.assertEqual(docker_ipc_arg("ipc=private"), "--ipc=private")

    def test_port_in_use(self):
        """Test port_in_use check."""
        from harinezumigel_llm_stack import port_in_use  # type: ignore

        # Create a temporary server to occupy a port
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_socket.bind(("127.0.0.1", 0))
        _, port = test_socket.getsockname()
        test_socket.listen(1)

        try:
            # Port should be in use
            self.assertTrue(port_in_use(port, "127.0.0.1"))
        finally:
            test_socket.close()

        # Port should now be free (after a brief moment)
        import time

        time.sleep(0.1)
        self.assertFalse(port_in_use(port, "127.0.0.1"))

    @patch("harinezumigel_llm_stack.socket.socket")
    def test_port_in_use_reports_bind_errors(self, mock_socket: Any) -> None:
        """Do not misreport invalid hosts or permission errors as occupied ports."""
        errors = [
            socket.gaierror(socket.EAI_NONAME, "Name or service not known"),
            OSError(errno.EADDRNOTAVAIL, "Cannot assign requested address"),
            PermissionError(errno.EACCES, "Permission denied"),
        ]
        for error in errors:
            with self.subTest(error=error):
                mock_socket.return_value.bind.side_effect = error
                with patch("builtins.print") as mock_print:
                    with self.assertRaises(SystemExit) as context:
                        harinezumigel_llm_stack.port_in_use(8011, "invalid-host")
                self.assertEqual(context.exception.code, 1)
                mock_print.assert_called_once_with(
                    f"ERROR: Could not check port 8011 on invalid-host: {error}"
                )
                mock_socket.return_value.close.assert_called()

    def test_redact_command(self):
        """Test redact_command for sensitive data."""
        from harinezumigel_llm_stack import redact_command  # type: ignore

        cmd = [
            "litellm",
            "--api-key",
            "secret123",
            "--token",
            "token456",
            "--other-flag",
            "value",
        ]

        redacted = redact_command(cmd)

        self.assertEqual(redacted[0], "litellm")
        self.assertEqual(redacted[1], "--api-key")
        self.assertEqual(redacted[2], "***REDACTED***")
        self.assertEqual(redacted[3], "--token")
        self.assertEqual(redacted[4], "***REDACTED***")
        self.assertEqual(redacted[5], "--other-flag")
        self.assertEqual(redacted[6], "value")

    def test_extract_nvidia_driver_mismatch(self):
        """Extract required and detected versions from NVIDIA mismatch errors."""
        from hls_preflight import extract_nvidia_driver_mismatch

        message = (
            "This container was built for NVIDIA Driver Release 595.58 or later, "
            "but version 580.178.04 was detected and compatibility mode is unavailable"
        )

        self.assertEqual(
            extract_nvidia_driver_mismatch(message),
            ("595.58", "580.178.04"),
        )

    def test_extract_nvidia_driver_mismatch_no_match(self):
        """Non-mismatch messages should return None."""
        from hls_preflight import extract_nvidia_driver_mismatch

        self.assertIsNone(extract_nvidia_driver_mismatch("unrelated docker error"))


class TestFormatLogLine(unittest.TestCase):
    """Test log formatting functions."""

    def test_format_log_line_with_stacktrace(self):
        """Test _format_log_line extracts stacktrace."""
        from harinezumigel_llm_stack import _format_log_line  # type: ignore

        log_obj = {
            "level": "ERROR",
            "message": "Test error",
            "stacktrace": "line1\\nline2\\nline3",
        }
        log_line = json.dumps(log_obj)

        result = _format_log_line(log_line)

        # Should have JSON without stacktrace + actual stacktrace
        self.assertIn('"level": "ERROR"', result)
        self.assertIn('"message": "Test error"', result)
        self.assertNotIn('"stacktrace"', result.split("\n")[0])
        self.assertIn("line1\\nline2\\nline3", result)

    def test_format_log_line_without_stacktrace(self):
        """Test _format_log_line with regular JSON."""
        from harinezumigel_llm_stack import _format_log_line  # type: ignore

        log_obj = {"level": "INFO", "message": "Normal log"}
        log_line = json.dumps(log_obj)

        result = _format_log_line(log_line)
        self.assertEqual(result, log_line)

    def test_format_log_line_non_json(self):
        """Test _format_log_line with non-JSON input."""
        from harinezumigel_llm_stack import _format_log_line  # type: ignore

        plain_text = "This is not JSON"
        result = _format_log_line(plain_text)
        self.assertEqual(result, plain_text)

    def test_format_log_line_pretty_debug_json(self):
        """Test _format_log_line pretty debug output for JSON records."""
        from harinezumigel_llm_stack import _format_log_line  # type: ignore

        log_obj = {
            "timestamp": "2026-10-18T09:00:00",
            "level": "DEBUG",
            "component": "LiteLLM",
            "logger": "proxy.py:42",
            "message": "Debug message",
            "extra": "value",
            "stacktrace": "line1\\nline2",
        }
        log_line = json.dumps(log_obj)

        result = _format_log_line(log_line, pretty_debug=True)

        self.assertIn("[2026-10-18T09:00:00] DEBUG", result)
        self.assertIn("LiteLLM (proxy.py:42)", result)
        self.assertIn("Debug message", result)
        self.assertIn("extra=value", result)
        self.assertIn("stacktrace:\nline1\\nline2", result)

    def test_format_log_line_pretty_debug_non_json(self):
        """Test _format_log_line pretty debug output for plain text records."""
        from harinezumigel_llm_stack import _format_log_line  # type: ignore

        plain_text = "plain line"
        result = _format_log_line(plain_text, pretty_debug=True)
        self.assertEqual(result, plain_text)


class TestAppConfig(unittest.TestCase):
    """Test AppConfig dataclass."""

    def setUp(self):
        """Set up test environment variables."""
        self.env_vars = {
            "LITELLM_CONFIG": "/opt/litellm/config/config.yaml",
            "MODEL_ROOT": "/models",
            "LITELLM_VENV_ACTIVATE": "/opt/litellm/venv/bin/activate",
            "LITELLM_BIN": "/opt/litellm/venv/bin/litellm",
            "LITELLM_BIND_HOST": "0.0.0.0",
            "LITELLM_PORT": "4000",
            "VLLM_HOST": "localhost",
            "VLLM_BIND_HOST": "0.0.0.0",
            "VLLM_CONTAINER_PORT": "8000",
            "VLLM_DOCKER_IMAGE": "vllm/vllm-openai:latest",
            "VLLM_MODEL_VOLUME": "/models:/models:ro",
            "VLLM_CACHE_VOLUME": "/cache:/cache",
        }
        for key, value in self.env_vars.items():
            os.environ[key] = value

    def tearDown(self):
        """Clean up test environment variables."""
        for key in self.env_vars:
            os.environ.pop(key, None)

    @patch("harinezumigel_llm_stack.load_env_file")
    def test_app_config_from_env(self, mock_load: Any) -> None:
        """Test AppConfig.from_env() loads all required fields."""
        from harinezumigel_llm_stack import AppConfig, INSTALLED_BASE  # type: ignore

        config = AppConfig.from_env()

        self.assertEqual(config.litellm_config, "/opt/litellm/config/config.yaml")
        self.assertEqual(config.model_root, "/models")
        self.assertEqual(config.env_file, str(INSTALLED_BASE / "config" / "env"))
        self.assertEqual(config.litellm_port, 4000)
        self.assertEqual(config.vllm_container_port, 8000)

    @patch("harinezumigel_llm_stack.load_env_file")
    def test_app_config_missing_required(self, mock_load: Any) -> None:
        """Test AppConfig.from_env() fails with missing required vars."""
        from harinezumigel_llm_stack import AppConfig  # type: ignore

        os.environ.pop("MODEL_ROOT")

        with self.assertRaises(SystemExit):
            AppConfig.from_env()

    def test_app_config_derives_config_path_from_installed_base(self) -> None:
        """The installed base determines the config path, not the environment."""
        with (
            patch.object(
                harinezumigel_llm_stack, "INSTALLED_BASE", Path("/test/install")
            ),
            patch.dict(os.environ, {"LITELLM_CONFIG": "/ignored/config.yaml"}),
        ):
            config = harinezumigel_llm_stack.AppConfig.from_env()
        self.assertEqual(config.litellm_config, "/test/install/config/config.yaml")


class TestModelRequirements(unittest.TestCase):
    """Test model requirement validation functions."""

    def test_require_model_int_valid(self):
        """Test require_model_int with valid integer."""
        from harinezumigel_llm_stack import require_model_int  # type: ignore

        model_info = {"max_tokens": 4096}
        result = require_model_int("test_model", model_info, "max_tokens")
        self.assertEqual(result, 4096)

    def test_require_model_int_missing(self):
        """Test require_model_int with missing key."""
        from harinezumigel_llm_stack import require_model_int  # type: ignore

        model_info: dict[str, Any] = {}

        with self.assertRaises(SystemExit):
            require_model_int("test_model", model_info, "missing_key")

    def test_require_model_float_valid(self):
        """Test require_model_float with valid float."""
        from harinezumigel_llm_stack import require_model_float  # type: ignore

        model_info = {"gpu_memory_utilization": 0.9}
        result = require_model_float("test_model", model_info, "gpu_memory_utilization")
        self.assertEqual(result, 0.9)

    def test_require_model_float_missing(self):
        """Test require_model_float with missing key."""
        from harinezumigel_llm_stack import require_model_float  # type: ignore

        model_info: dict[str, Any] = {}

        with self.assertRaises(SystemExit):
            require_model_float("test_model", model_info, "missing_key")


class TestLLMStackInit(unittest.TestCase):
    """Test LLMStack initialization and configuration parsing."""

    def setUp(self):
        """Set up test configuration."""
        self.env_vars = {
            "LITELLM_CONFIG": "/opt/litellm/config/config.yaml",
            "MODEL_ROOT": "/models",
            "LITELLM_VENV_ACTIVATE": "/opt/litellm/venv/bin/activate",
            "LITELLM_BIN": "/opt/litellm/venv/bin/litellm",
            "LITELLM_BIND_HOST": "0.0.0.0",
            "LITELLM_PORT": "4000",
            "VLLM_HOST": "localhost",
            "VLLM_BIND_HOST": "0.0.0.0",
            "VLLM_CONTAINER_PORT": "8000",
            "VLLM_DOCKER_IMAGE": "vllm/vllm-openai:latest",
            "VLLM_MODEL_VOLUME": "/models:/models:ro",
            "VLLM_CACHE_VOLUME": "/cache:/cache",
        }
        for key, value in self.env_vars.items():
            os.environ[key] = value

    def tearDown(self):
        """Clean up test environment."""
        for key in self.env_vars:
            os.environ.pop(key, None)

    @patch("harinezumigel_llm_stack.load_env_file")
    def test_llmstack_init(self, mock_load: Any) -> None:
        """Test LLMStack initialization."""
        from harinezumigel_llm_stack import AppConfig, LLMStack  # type: ignore

        config = AppConfig.from_env()
        stack = LLMStack(config)

        self.assertEqual(stack.config, config)
        self.assertIsInstance(stack.models, dict)
        self.assertEqual(len(stack.models), 0)

    @patch("harinezumigel_llm_stack.load_env_file")
    def test_parse_api_base(self, mock_load: Any) -> None:
        """Test _parse_api_base URL parsing."""
        from harinezumigel_llm_stack import AppConfig, LLMStack  # type: ignore

        config = AppConfig.from_env()
        stack = LLMStack(config)

        host, port = stack._parse_api_base("http://localhost:8001/v1")
        self.assertEqual(host, "localhost")
        self.assertEqual(port, 8001)

        host, port = stack._parse_api_base("http://127.0.0.1:9000")
        self.assertEqual(host, "127.0.0.1")
        self.assertEqual(port, 9000)

        for url, expected_port in [
            ("http://localhost/v1", 80),
            ("https://localhost/v1", 443),
        ]:
            with self.subTest(url=url):
                self.assertEqual(
                    stack._parse_api_base(url), ("localhost", expected_port)
                )

    def test_example_env_preserves_backend_host_and_port(self):
        """Host comments must not hide the backend port in expanded API URLs."""
        with patch.dict(os.environ, clear=True):
            harinezumigel_llm_stack.load_env_file(
                str(PROJECT_DIR / "config" / "env.example")
            )
            with patch("harinezumigel_llm_stack.load_env_file"):
                config = harinezumigel_llm_stack.AppConfig.from_env()
            stack = harinezumigel_llm_stack.LLMStack(config)
            api_base = os.environ["QWEN3_CODER_30B_A3B_INSTRUCT_API_BASE"]

            self.assertEqual(config.vllm_host, "127.0.0.1")
            self.assertEqual(config.vllm_bind_host, "127.0.0.1")
            self.assertEqual(api_base, "http://127.0.0.1:8011/v1")
            self.assertEqual(
                stack._parse_api_base(api_base),
                (config.vllm_host, 8011),
            )

    def test_standalone_coding_profiles_preserve_native_precision_and_context(self):
        with patch.dict(os.environ, clear=True):
            harinezumigel_llm_stack.load_env_file(
                str(PROJECT_DIR / "config" / "env.example")
            )
            config = replace(
                harinezumigel_llm_stack.AppConfig.from_env(),
                litellm_config=str(PROJECT_DIR / "config" / "config.yaml.example"),
            )
            stack = harinezumigel_llm_stack.LLMStack(config)
            stack.load_models()
            for name in ("qwen3.8-27B", "qwen3-coder-30b-a3b-instruct"):
                with self.subTest(model=name):
                    model = stack.models[name]
                    self.assertEqual(
                        (
                            model.context_length,
                            model.max_input_tokens,
                            model.max_output_tokens,
                            model.model_info["max_tokens"],
                        ),
                        (253952, 233472, 16384, 16384),
                    )
                    self.assertEqual(model.max_num_seqs, 1)
                    self.assertIsNone(model.model_info["quantization"])
                    self.assertEqual(model.model_info["kv_cache_dtype"], "auto")
                    self.assertTrue(model.model_info["enable_prefix_caching"])
                    self.assertTrue(model.model_info["enable_chunked_prefill"])
                    self.assertFalse(model.model_info["enforce_eager"])
                    self.assertEqual(model.model_info["max_num_batched_tokens"], 16384)
                    runtime: dict[str, int | float | str | None] = {
                        "context_length": model.context_length,
                        "gpu_memory_utilization": model.gpu_memory_utilization,
                        "max_num_seqs": model.max_num_seqs,
                        "dtype": model.dtype,
                    }
                    command = stack._build_vllm_command(
                        model,
                        Path("/models") / model.model_info["model_dir"],
                        model.api_base_port,
                        runtime,
                    )
                    self.assertIn("--enable-chunked-prefill", command)
                    self.assertIn("--enable-prefix-caching", command)
                    self.assertNotIn("--enforce-eager", command)
                    self.assertEqual(
                        command[command.index("--max-model-len") + 1], "253952"
                    )
                    self.assertEqual(
                        json.loads(command[command.index("--speculative-config") + 1]),
                        model.model_info["speculative_config"],
                    )
            self.assertEqual(
                stack.models["qwen3.8-27B"].model_info["speculative_config"],
                {"method": "mtp", "num_speculative_tokens": 3},
            )
            self.assertNotIn(
                "attention_backend", stack.models["qwen3.8-27B"].model_info
            )
            self.assertEqual(
                stack.models["qwen3-coder-30b-a3b-instruct"].model_info[
                    "speculative_config"
                ],
                {
                    "method": "dflash",
                    "model": "z-lab/Qwen3-Coder-30B-A3B-DFlash",
                    "num_speculative_tokens": 15,
                },
            )


class TestLLMStackDockerOperations(unittest.TestCase):
    """Test LLMStack Docker-related operations."""

    def setUp(self):
        """Set up test configuration."""
        log_root = tempfile.TemporaryDirectory()
        self.addCleanup(log_root.cleanup)
        root_patch = patch.object(
            harinezumigel_llm_stack, "INSTALLED_BASE", Path(log_root.name)
        )
        root_patch.start()
        self.addCleanup(root_patch.stop)
        self.env_vars = {
            "LITELLM_CONFIG": "/opt/litellm/config/config.yaml",
            "MODEL_ROOT": "/models",
            "LITELLM_VENV_ACTIVATE": "/opt/litellm/venv/bin/activate",
            "LITELLM_BIN": "/opt/litellm/venv/bin/litellm",
            "LITELLM_BIND_HOST": "0.0.0.0",
            "LITELLM_PORT": "4000",
            "VLLM_HOST": "localhost",
            "VLLM_BIND_HOST": "0.0.0.0",
            "VLLM_CONTAINER_PORT": "8000",
            "VLLM_DOCKER_IMAGE": "vllm/vllm-openai:latest",
            "VLLM_MODEL_VOLUME": "/models:/models:ro",
            "VLLM_CACHE_VOLUME": "/cache:/cache",
        }
        for key, value in self.env_vars.items():
            os.environ[key] = value

    def tearDown(self):
        """Clean up test environment."""
        for key in self.env_vars:
            os.environ.pop(key, None)

    @patch("harinezumigel_llm_stack.load_env_file")
    @patch("harinezumigel_llm_stack.run_command")
    def test_docker_container_state(self, mock_run: Any, mock_load: Any) -> None:
        """Test _docker_container_state."""
        from harinezumigel_llm_stack import AppConfig, LLMStack  # type: ignore

        config = AppConfig.from_env()
        stack = LLMStack(config)

        # Mock running container
        mock_run.return_value = MagicMock(stdout="running\n", returncode=0)
        state = stack._docker_container_state("vllm-test")
        self.assertEqual(state, "running")

        # Mock nonexistent container
        mock_run.return_value = MagicMock(stdout="", returncode=1)
        state = stack._docker_container_state("vllm-nonexistent")
        self.assertIsNone(state)

    @patch("harinezumigel_llm_stack.load_env_file")
    @patch("harinezumigel_llm_stack.run_command")
    def test_docker_list_container_names(self, mock_run: Any, mock_load: Any) -> None:
        """Test _docker_list_container_names."""
        from harinezumigel_llm_stack import AppConfig, LLMStack  # type: ignore

        config = AppConfig.from_env()
        stack = LLMStack(config)

        mock_run.return_value = MagicMock(
            stdout="vllm-model1\nvllm-model2\nother-container\n", returncode=0
        )

        names = stack._docker_list_container_names(all_containers=False)

        # Returns all container names (filtering happens elsewhere)
        self.assertIn("vllm-model1", names)
        self.assertIn("vllm-model2", names)
        self.assertIn("other-container", names)
        self.assertEqual(len(names), 3)

    @patch("harinezumigel_llm_stack.load_env_file")
    @patch("harinezumigel_llm_stack.run_command")
    def test_docker_port_in_use(self, mock_run: Any, mock_load: Any) -> None:
        """Test _docker_port_in_use."""
        from harinezumigel_llm_stack import AppConfig, LLMStack  # type: ignore

        config = AppConfig.from_env()
        stack = LLMStack(config)

        # Mock docker ps output with port format
        mock_run.return_value = MagicMock(
            stdout="0.0.0.0:8001->8000/tcp\n0.0.0.0:8002->8000/tcp\n", returncode=0
        )
        self.assertTrue(stack._docker_port_in_use(8001))
        self.assertTrue(stack._docker_port_in_use(8002))
        self.assertFalse(stack._docker_port_in_use(8003))

    @patch("harinezumigel_llm_stack.load_env_file")
    def test_build_vllm_command_includes_ipc_when_configured(
        self, mock_load: Any
    ) -> None:
        """model_info.ipc should add a Docker --ipc flag."""
        from harinezumigel_llm_stack import AppConfig, LLMStack, ModelDeployment  # type: ignore

        config = AppConfig.from_env()
        stack = LLMStack(config)

        model = ModelDeployment(
            name="test_model",
            backend_model="openai/test_model",
            api_base="http://localhost:8001/v1",
            api_key="sk-test",
            api_base_host="localhost",
            api_base_port=8001,
            litellm_params={},
            model_info={"ipc": "host"},
            context_length=4096,
            max_input_tokens=3000,
            max_output_tokens=1000,
            gpu_memory_utilization=0.9,
            max_num_seqs=None,
            dtype="auto",
            license=None,
            upstream=None,
            alias=None,
            description=None,
            detail=None,
            allow_rebuild=True,
        )

        runtime: dict[str, int | float | str | None] = {
            "context_length": 4096,
            "max_input_tokens": 3000,
            "max_output_tokens": 1000,
            "gpu_memory_utilization": 0.9,
            "max_num_seqs": None,
            "dtype": "auto",
        }
        command = stack._build_vllm_command(
            model,
            Path("/models/test-model"),
            8001,
            runtime,
        )

        self.assertIn("--ipc=host", command)

    @patch("harinezumigel_llm_stack.load_env_file")
    def test_build_vllm_command_chunked_prefill(self, mock_load: Any) -> None:
        config = harinezumigel_llm_stack.AppConfig.from_env()
        stack = harinezumigel_llm_stack.LLMStack(config)
        model = harinezumigel_llm_stack.ModelDeployment(
            name="test_model",
            backend_model="openai/test_model",
            api_base="http://localhost:8001/v1",
            api_key="sk-test",
            api_base_host="localhost",
            api_base_port=8001,
            litellm_params={},
            model_info={},
            context_length=253952,
            max_input_tokens=233472,
            max_output_tokens=16384,
            gpu_memory_utilization=0.9,
            max_num_seqs=1,
            dtype="auto",
            license=None,
            upstream=None,
            alias=None,
            description=None,
            detail=None,
            allow_rebuild=True,
        )
        runtime: dict[str, int | float | str | None] = {
            "context_length": model.context_length,
            "gpu_memory_utilization": model.gpu_memory_utilization,
            "max_num_seqs": model.max_num_seqs,
            "dtype": model.dtype,
        }
        for setting in (None, True, False, "true", "false"):
            with self.subTest(setting=setting):
                model.model_info["enable_chunked_prefill"] = setting
                command = stack._build_vllm_command(
                    model, Path("/models/test-model"), 8001, runtime
                )
                self.assertEqual(
                    "--enable-chunked-prefill" in command,
                    setting in (True, "true"),
                )
                self.assertEqual(
                    "--no-enable-chunked-prefill" in command,
                    setting in (False, "false"),
                )
                self.assertEqual(
                    command[command.index("--max-model-len") + 1], "253952"
                )

    @patch("harinezumigel_llm_stack.load_env_file")
    def test_build_vllm_command_omits_ipc_when_not_configured(
        self, mock_load: Any
    ) -> None:
        """No model_info.ipc should omit Docker --ipc flag."""
        from harinezumigel_llm_stack import AppConfig, LLMStack, ModelDeployment  # type: ignore

        config = AppConfig.from_env()
        stack = LLMStack(config)

        model = ModelDeployment(
            name="test_model",
            backend_model="openai/test_model",
            api_base="http://localhost:8001/v1",
            api_key="sk-test",
            api_base_host="localhost",
            api_base_port=8001,
            litellm_params={},
            model_info={},
            context_length=4096,
            max_input_tokens=3000,
            max_output_tokens=1000,
            gpu_memory_utilization=0.9,
            max_num_seqs=None,
            dtype="auto",
            license=None,
            upstream=None,
            alias=None,
            description=None,
            detail=None,
            allow_rebuild=True,
        )

        runtime: dict[str, int | float | str | None] = {
            "context_length": 4096,
            "max_input_tokens": 3000,
            "max_output_tokens": 1000,
            "gpu_memory_utilization": 0.9,
            "max_num_seqs": None,
            "dtype": "auto",
        }
        command = stack._build_vllm_command(
            model,
            Path("/models/test-model"),
            8001,
            runtime,
        )

        self.assertFalse(any(part.startswith("--ipc=") for part in command))

    @patch("harinezumigel_llm_stack.load_env_file")
    def test_build_vllm_command_includes_reasoning_config_flag(
        self, mock_load: Any
    ) -> None:
        """Normalize parser settings and preserve additional reasoning options."""
        from harinezumigel_llm_stack import AppConfig, LLMStack, ModelDeployment  # type: ignore

        config = AppConfig.from_env()
        stack = LLMStack(config)

        model = ModelDeployment(
            name="test_model",
            backend_model="openai/test_model",
            api_base="http://localhost:8001/v1",
            api_key="sk-test",
            api_base_host="localhost",
            api_base_port=8001,
            litellm_params={},
            model_info={"reasoning_config": {"reasoning_parser": "qwen3"}},
            context_length=4096,
            max_input_tokens=3000,
            max_output_tokens=1000,
            gpu_memory_utilization=0.9,
            max_num_seqs=None,
            dtype="auto",
            license=None,
            upstream=None,
            alias=None,
            description=None,
            detail=None,
            allow_rebuild=True,
        )

        runtime: dict[str, int | float | str | None] = {
            "context_length": 4096,
            "max_input_tokens": 3000,
            "max_output_tokens": 1000,
            "gpu_memory_utilization": 0.9,
            "max_num_seqs": None,
            "dtype": "auto",
        }
        for as_json in (False, True):
            for top_level_parser in (None, "qwen3", "deepseek_r1"):
                for extra_options in ({}, {"reasoning_end_str": "</think>"}):
                    with self.subTest(
                        as_json=as_json,
                        top_level_parser=top_level_parser,
                        extra_options=extra_options,
                    ):
                        reasoning_options = {
                            "reasoning_parser": "qwen3",
                            **extra_options,
                        }
                        model.model_info = {
                            "reasoning_config": (
                                json.dumps(reasoning_options)
                                if as_json
                                else reasoning_options
                            ),
                            "reasoning_parser": top_level_parser,
                        }
                        if top_level_parser == "deepseek_r1":
                            with self.assertRaises(SystemExit):
                                stack._build_vllm_command(
                                    model, Path("/models/test-model"), 8001, runtime
                                )
                            continue

                        command = stack._build_vllm_command(
                            model, Path("/models/test-model"), 8001, runtime
                        )
                        self.assertEqual(command.count("--reasoning-parser"), 1)
                        parser_index = command.index("--reasoning-parser")
                        self.assertEqual(command[parser_index + 1], "qwen3")
                        if extra_options:
                            config_index = command.index("--reasoning-config")
                            self.assertEqual(
                                json.loads(command[config_index + 1]), extra_options
                            )
                        else:
                            self.assertNotIn("--reasoning-config", command)
                        self.assertNotIn("--enable-reasoning", command)

    @patch("harinezumigel_llm_stack.load_env_file")
    def test_build_vllm_command_rejects_legacy_reasoning_config_enable_flag(
        self,
        mock_load: Any,
    ) -> None:
        """Legacy reasoning_config.enable_reasoning should fail fast."""
        from harinezumigel_llm_stack import AppConfig, LLMStack, ModelDeployment  # type: ignore

        config = AppConfig.from_env()
        stack = LLMStack(config)

        model = ModelDeployment(
            name="test_model",
            backend_model="openai/test_model",
            api_base="http://localhost:8001/v1",
            api_key="sk-test",
            api_base_host="localhost",
            api_base_port=8001,
            litellm_params={},
            model_info={
                "reasoning_config": {"enable_reasoning": True},
                "reasoning_parser": "qwen3",
            },
            context_length=4096,
            max_input_tokens=3000,
            max_output_tokens=1000,
            gpu_memory_utilization=0.9,
            max_num_seqs=None,
            dtype="auto",
            license=None,
            upstream=None,
            alias=None,
            description=None,
            detail=None,
            allow_rebuild=True,
        )

        runtime: dict[str, int | float | str | None] = {
            "context_length": 4096,
            "max_input_tokens": 3000,
            "max_output_tokens": 1000,
            "gpu_memory_utilization": 0.9,
            "max_num_seqs": None,
            "dtype": "auto",
        }

        with self.assertRaises(SystemExit):
            stack._build_vllm_command(
                model,
                Path("/models/test-model"),
                8001,
                runtime,
            )

    @patch("harinezumigel_llm_stack.load_env_file")
    def test_build_vllm_command_includes_attention_config_flag(
        self, mock_load: Any
    ) -> None:
        """model_info.attention_config should add --attention-config."""
        from harinezumigel_llm_stack import AppConfig, LLMStack, ModelDeployment  # type: ignore

        config = AppConfig.from_env()
        stack = LLMStack(config)

        model = ModelDeployment(
            name="test_model",
            backend_model="openai/test_model",
            api_base="http://localhost:8001/v1",
            api_key="sk-test",
            api_base_host="localhost",
            api_base_port=8001,
            litellm_params={},
            model_info={"attention_config": {"backend": "flashinfer"}},
            context_length=4096,
            max_input_tokens=3000,
            max_output_tokens=1000,
            gpu_memory_utilization=0.9,
            max_num_seqs=None,
            dtype="auto",
            license=None,
            upstream=None,
            alias=None,
            description=None,
            detail=None,
            allow_rebuild=True,
        )

        runtime: dict[str, int | float | str | None] = {
            "context_length": 4096,
            "max_input_tokens": 3000,
            "max_output_tokens": 1000,
            "gpu_memory_utilization": 0.9,
            "max_num_seqs": None,
            "dtype": "auto",
        }
        command = stack._build_vllm_command(
            model,
            Path("/models/test-model"),
            8001,
            runtime,
        )

        self.assertIn("--attention-config", command)
        attention_config_index = command.index("--attention-config")
        self.assertEqual(
            command[attention_config_index + 1], '{"backend":"flashinfer"}'
        )

        for config_key, config_value in (
            ("compilation_config", {"mode": "max-autotune"}),
            ("structured_outputs_config", {"backend": "xgrammar"}),
        ):
            flag = f"--{config_key.replace('_', '-')}"
            self.assertNotIn(flag, command)
            for value in (config_value, json.dumps(config_value)):
                with self.subTest(config_key=config_key, value=value):
                    model.model_info = {config_key: value}
                    configured_command = stack._build_vllm_command(
                        model, Path("/models/test-model"), 8001, runtime
                    )
                    self.assertEqual(configured_command.count(flag), 1)
                    self.assertEqual(
                        json.loads(
                            configured_command[configured_command.index(flag) + 1]
                        ),
                        config_value,
                    )

    @patch("harinezumigel_llm_stack.load_env_file")
    def test_build_vllm_command_omits_reasoning_config_flag_by_default(
        self,
        mock_load: Any,
    ) -> None:
        """Without model_info.reasoning_config, --reasoning-config should be omitted."""
        from harinezumigel_llm_stack import AppConfig, LLMStack, ModelDeployment  # type: ignore

        config = AppConfig.from_env()
        stack = LLMStack(config)

        model = ModelDeployment(
            name="test_model",
            backend_model="openai/test_model",
            api_base="http://localhost:8001/v1",
            api_key="sk-test",
            api_base_host="localhost",
            api_base_port=8001,
            litellm_params={},
            model_info={},
            context_length=4096,
            max_input_tokens=3000,
            max_output_tokens=1000,
            gpu_memory_utilization=0.9,
            max_num_seqs=None,
            dtype="auto",
            license=None,
            upstream=None,
            alias=None,
            description=None,
            detail=None,
            allow_rebuild=True,
        )

        runtime: dict[str, int | float | str | None] = {
            "context_length": 4096,
            "max_input_tokens": 3000,
            "max_output_tokens": 1000,
            "gpu_memory_utilization": 0.9,
            "max_num_seqs": None,
            "dtype": "auto",
        }
        command = stack._build_vllm_command(
            model,
            Path("/models/test-model"),
            8001,
            runtime,
        )

        self.assertNotIn("--reasoning-config", command)

    @patch("harinezumigel_llm_stack.load_env_file")
    @patch("harinezumigel_llm_stack.run_command")
    def test_preflight_nvidia_driver_check_success(
        self, mock_run: Any, mock_load: Any
    ) -> None:
        """Preflight should pass when docker GPU probe succeeds."""
        from harinezumigel_llm_stack import AppConfig, LLMStack  # type: ignore

        config = AppConfig.from_env()
        stack = LLMStack(config)

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        stack.preflight_checks.check_nvidia_driver(dry_run=False)

        mock_run.assert_called_once_with(
            [
                "docker",
                "run",
                "--rm",
                "--gpus",
                "all",
                "--entrypoint",
                "python3",
                config.vllm_docker_image,
                "-c",
                "import sys; sys.exit(0)",
            ],
            capture=True,
        )

    @patch("harinezumigel_llm_stack.load_env_file")
    @patch("harinezumigel_llm_stack.run_command")
    def test_preflight_nvidia_driver_check_mismatch_aborts(
        self,
        mock_run: Any,
        mock_load: Any,
    ) -> None:
        """A confirmed driver/image mismatch must stop startup."""
        from harinezumigel_llm_stack import AppConfig, LLMStack  # type: ignore

        config = AppConfig.from_env()
        stack = LLMStack(config)

        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr=(
                "This container was built for NVIDIA Driver Release 595.58 or later, "
                "but version 580.178.04 was detected"
            ),
        )

        for dry_run in (False, True):
            with (
                redirect_stdout(io.StringIO()) as output,
                self.assertRaises(SystemExit) as error,
            ):
                stack.preflight_checks.check_nvidia_driver(dry_run=dry_run)

            self.assertEqual(error.exception.code, 1)
            self.assertIn("Required NVIDIA driver: >= 595.58", output.getvalue())
            self.assertIn("Startup aborted", output.getvalue())

        self.assertEqual(mock_run.call_count, 2)

    @patch("harinezumigel_llm_stack.load_env_file")
    @patch("harinezumigel_llm_stack.run_command")
    def test_preflight_nvidia_driver_check_dry_run(
        self, mock_run: Any, mock_load: Any
    ) -> None:
        """Dry-run mode executes the NVIDIA compatibility probe."""
        from harinezumigel_llm_stack import AppConfig, LLMStack  # type: ignore

        config = AppConfig.from_env()
        stack = LLMStack(config)
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        with redirect_stdout(io.StringIO()) as output:
            stack.preflight_checks.check_nvidia_driver(dry_run=True)

        mock_run.assert_called_once_with(
            [
                "docker",
                "run",
                "--rm",
                "--gpus",
                "all",
                "--entrypoint",
                "python3",
                config.vllm_docker_image,
                "-c",
                "import sys; sys.exit(0)",
            ],
            capture=True,
        )
        self.assertIn("compatibility check (dry-run) passed", output.getvalue())

    @patch("harinezumigel_llm_stack.load_env_file")
    @patch("harinezumigel_llm_stack.run_command")
    def test_preflight_nvidia_driver_check_unrelated_failure_warns(
        self,
        mock_run: Any,
        mock_load: Any,
    ) -> None:
        """A non-mismatch probe failure warns but does not block startup."""
        from harinezumigel_llm_stack import AppConfig, LLMStack  # type: ignore

        stack = LLMStack(AppConfig.from_env())
        mock_run.side_effect = [
            MagicMock(returncode=1, stdout="", stderr="registry unavailable"),
            MagicMock(returncode=0, stdout="580.1", stderr=""),
        ]

        with redirect_stdout(io.StringIO()) as output:
            stack.preflight_checks.check_nvidia_driver(dry_run=True)

        self.assertIn("registry unavailable", output.getvalue())
        self.assertIn("startup will continue", output.getvalue())

    @patch("harinezumigel_llm_stack.load_env_file")
    @patch("harinezumigel_llm_stack.run_command")
    def test_preflight_gpu_memory_check_passes_on_dry_run(
        self,
        mock_run: Any,
        mock_load: Any,
    ) -> None:
        """VRAM is checked during dry runs using the configured fraction."""
        from harinezumigel_llm_stack import AppConfig, LLMStack  # type: ignore

        config = AppConfig.from_env()
        stack = LLMStack(config)
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="0, 8192, 16384\n1, 4096, 8192\n",
            stderr="",
        )

        with redirect_stdout(io.StringIO()) as output:
            stack.preflight_checks.check_gpu_memory(["test_model"], 0.5, dry_run=True)

        mock_run.assert_called_once_with(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture=True,
        )
        self.assertIn(
            "GPU VRAM preflight (dry-run) passed for model 'test_model'",
            output.getvalue(),
        )
        self.assertIn("Available: 8192 MiB (8.00 GiB)", output.getvalue())

    @patch("harinezumigel_llm_stack.load_env_file")
    @patch("harinezumigel_llm_stack.run_command")
    def test_preflight_gpu_memory_check_rejects_insufficient_memory(
        self,
        mock_run: Any,
        mock_load: Any,
    ) -> None:
        """Startup stops when free VRAM is below the configured budget."""
        from harinezumigel_llm_stack import AppConfig, LLMStack  # type: ignore

        config = AppConfig.from_env()
        stack = LLMStack(config)
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="0, 4096, 16384\n",
            stderr="",
        )

        with (
            redirect_stdout(io.StringIO()) as output,
            self.assertRaises(SystemExit) as error,
        ):
            stack.preflight_checks.check_gpu_memory(["test_model"], 0.5)

        self.assertEqual(error.exception.code, 1)
        self.assertIn("Insufficient GPU VRAM", output.getvalue())
        self.assertIn("for model 'test_model'", output.getvalue())
        self.assertIn("Available: 4096 MiB (4.00 GiB)", output.getvalue())
        self.assertIn("required: 8192 MiB", output.getvalue())

    @patch("harinezumigel_llm_stack.load_env_file")
    @patch("harinezumigel_llm_stack.run_command")
    def test_preflight_gpu_memory_check_names_multiple_models(
        self,
        mock_run: Any,
        mock_load: Any,
    ) -> None:
        """Group VRAM failures identify the count and names of selected models."""
        from harinezumigel_llm_stack import AppConfig, LLMStack  # type: ignore

        stack = LLMStack(AppConfig.from_env())
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="0, 4096, 16384\n",
            stderr="",
        )

        with redirect_stdout(io.StringIO()) as output, self.assertRaises(SystemExit):
            stack.preflight_checks.check_gpu_memory(["model_a", "model_b"], 0.6)

        self.assertIn("for 2 selected models (model_a, model_b)", output.getvalue())
        self.assertIn("Available: 4096 MiB (4.00 GiB)", output.getvalue())

    @patch("harinezumigel_llm_stack.load_env_file")
    @patch("harinezumigel_llm_stack.run_command")
    def test_clean_vllm_logs_attempts_truncate_when_path_is_safe(
        self,
        mock_run: Any,
        mock_load: Any,
    ) -> None:
        """Docker log cleanup should delegate missing-file handling to sudo truncate."""
        from harinezumigel_llm_stack import AppConfig, LLMStack  # type: ignore

        config = AppConfig.from_env()
        stack = LLMStack(config)
        docker_log_path = "/var/lib/docker/containers/test/test-json.log"
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="No such file"
        )

        with patch.object(stack, "_resolve_model_name", return_value="test_model"):
            with patch.object(
                stack,
                "_find_all_vllm_containers",
                return_value=["vllm-test_model-8001"],
            ):
                with patch.object(
                    stack,
                    "_docker_container_log_path",
                    return_value=docker_log_path,
                ):
                    stack.clean_vllm_logs("test_model", dry_run=False)

        mock_run.assert_called_once_with(
            ["sudo", "truncate", "-s", "0", docker_log_path],
            capture=True,
        )

    @patch("harinezumigel_llm_stack.load_env_file")
    @patch("harinezumigel_llm_stack.run_command")
    def test_clean_vllm_logs_skips_unexpected_log_path(
        self,
        mock_run: Any,
        mock_load: Any,
    ) -> None:
        """Docker log clean should skip paths outside Docker log root."""
        from harinezumigel_llm_stack import AppConfig, LLMStack  # type: ignore

        config = AppConfig.from_env()
        stack = LLMStack(config)

        with patch.object(stack, "_resolve_model_name", return_value="test_model"):
            with patch.object(
                stack,
                "_find_all_vllm_containers",
                return_value=["vllm-test_model-8001"],
            ):
                with patch.object(
                    stack,
                    "_docker_container_log_path",
                    return_value="/tmp/unmanaged.log",
                ):
                    stack.clean_vllm_logs("test_model", dry_run=False)

        mock_run.assert_not_called()

    @patch("harinezumigel_llm_stack.load_env_file")
    @patch("harinezumigel_llm_stack.run_command")
    def test_docker_remove_container_rejects_unmanaged_name(
        self,
        mock_run: Any,
        mock_load: Any,
    ) -> None:
        """Container removal should reject names outside managed vLLM prefix."""
        from harinezumigel_llm_stack import AppConfig, LLMStack  # type: ignore

        config = AppConfig.from_env()
        stack = LLMStack(config)

        with self.assertRaises(SystemExit):
            stack._docker_remove_container("postgres", dry_run=False)

        mock_run.assert_not_called()

    @patch("harinezumigel_llm_stack.load_env_file")
    @patch("harinezumigel_llm_stack.run_command")
    def test_clean_litellm_logs_skips_unexpected_log_path(
        self,
        mock_run: Any,
        mock_load: Any,
    ) -> None:
        """A configured path must not change the fixed LiteLLM logs directory."""
        from harinezumigel_llm_stack import AppConfig, LLMStack  # type: ignore

        os.environ["LITELLM_LOG_FILE"] = "relative.log"

        try:
            config = AppConfig.from_env()
            stack = LLMStack(config)

            stack.clean_litellm_logs(dry_run=False)

            mock_run.assert_not_called()
            self.assertEqual(
                stack._litellm_logs_dir(),
                harinezumigel_llm_stack.INSTALLED_BASE / "logs",
            )
        finally:
            os.environ.pop("LITELLM_LOG_FILE", None)

    def test_startup_missing_deployment_aborts(self) -> None:
        """Missing installation aborts before environment loading or dispatch."""
        with (
            patch.object(Path, "is_dir", return_value=False),
            patch.object(Path, "open") as mock_open,
            patch.object(Path, "mkdir") as mock_mkdir,
            patch.dict(os.environ, {}, clear=True),
            patch("builtins.print") as mock_print,
        ):
            with self.assertRaises(SystemExit) as caught:
                runpy.run_path(str(SCRIPT_DIR / "harinezumigel-llm-stack.py"))
            self.assertEqual(caught.exception.code, 1)
            mock_print.assert_called_once_with(
                "ERROR: LiteLLM deployment directory does not exist: /opt/litellm"
            )
            mock_open.assert_not_called()
            mock_mkdir.assert_not_called()

    def test_startup_non_absolute_base_aborts(self) -> None:
        """Non-absolute bases abort before filesystem checks or environment loading."""
        with (
            patch.object(Path, "is_absolute", return_value=False),
            patch.object(Path, "is_dir") as mock_is_dir,
            patch.object(Path, "open") as mock_open,
            patch.object(Path, "mkdir") as mock_mkdir,
            patch("builtins.print") as mock_print,
        ):
            with self.assertRaises(SystemExit) as caught:
                runpy.run_path(str(SCRIPT_DIR / "harinezumigel-llm-stack.py"))
            self.assertEqual(caught.exception.code, 1)
            mock_print.assert_called_once_with(
                "ERROR: INSTALLED_BASE must be an absolute path: /opt/litellm"
            )
            mock_is_dir.assert_not_called()
            mock_open.assert_not_called()
            mock_mkdir.assert_not_called()

    def test_litellm_logs_directory_created(self) -> None:
        """Log access creates only the deployment's logs subdirectory."""
        stack = harinezumigel_llm_stack.LLMStack(
            harinezumigel_llm_stack.AppConfig.from_env()
        )
        logs_dir = harinezumigel_llm_stack.INSTALLED_BASE / "logs"
        self.assertFalse(logs_dir.exists())
        self.assertEqual(stack._litellm_logs_dir(), logs_dir)
        self.assertTrue(logs_dir.is_dir())

    def test_clean_litellm_logs_missing_directory_is_not_created(self) -> None:
        """Cleaning absent logs leaves the directory absent."""
        stack = harinezumigel_llm_stack.LLMStack(
            harinezumigel_llm_stack.AppConfig.from_env()
        )
        stack.clean_litellm_logs()
        self.assertFalse((harinezumigel_llm_stack.INSTALLED_BASE / "logs").exists())

    def test_clean_litellm_logs_empty_directory_reports_cleaned(self) -> None:
        stack = harinezumigel_llm_stack.LLMStack(
            harinezumigel_llm_stack.AppConfig.from_env()
        )
        stack._litellm_logs_dir()
        with patch("builtins.print") as mock_print:
            stack.clean_litellm_logs()
        mock_print.assert_any_call("  Status:   ✓ Cleaned")

    def test_clean_litellm_logs_dry_run_preserves_files(self) -> None:
        """Dry-run cleanup must leave existing logs untouched."""
        stack = harinezumigel_llm_stack.LLMStack(
            harinezumigel_llm_stack.AppConfig.from_env()
        )
        log_path = stack._litellm_logs_dir() / "litellm.log"
        log_path.write_text("keep", encoding="utf-8")
        stack.clean_litellm_logs(dry_run=True)
        self.assertEqual(log_path.read_text(encoding="utf-8"), "keep")

    @patch("harinezumigel_llm_stack.load_env_file")
    def test_show_litellm_logs_enables_pretty_format_on_debug(
        self, mock_load: Any
    ) -> None:
        """show_litellm_logs should use pretty formatting when debug is enabled."""
        import tempfile
        import yaml  # type: ignore

        from harinezumigel_llm_stack import AppConfig, LLMStack  # type: ignore

        config_data = {
            "litellm_settings": {
                "debug": True,
                "detailed_debug": False,
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_config = f.name

        old_config = os.environ.get("LITELLM_CONFIG")

        try:
            config = replace(AppConfig.from_env(), litellm_config=temp_config)
            stack = LLMStack(config)

            fake_log_path = Path(tempfile.gettempdir()) / "litellm-debug-test.log"

            with patch.object(stack, "_find_litellm_pids", return_value=[]):
                with patch.object(
                    stack, "_litellm_log_path", return_value=fake_log_path
                ):
                    with patch.object(
                        stack, "_tail_file_logs", return_value=0
                    ) as mock_tail:
                        stack.show_litellm_logs(
                            tail="10", follow=False, show_log_path=False
                        )

            mock_tail.assert_called_once_with(
                fake_log_path,
                tail="10",
                follow=False,
                pretty_debug=True,
            )
        finally:
            if old_config is None:
                os.environ.pop("LITELLM_CONFIG", None)
            else:
                os.environ["LITELLM_CONFIG"] = old_config
            safe_unlink(temp_config)

    @patch("harinezumigel_llm_stack.load_env_file")
    def test_show_litellm_logs_enables_pretty_format_on_detailed_debug(
        self,
        mock_load: Any,
    ) -> None:
        """show_litellm_logs should use pretty formatting when detailed_debug is enabled."""
        import tempfile
        import yaml  # type: ignore

        from harinezumigel_llm_stack import AppConfig, LLMStack  # type: ignore

        config_data = {
            "litellm_settings": {
                "debug": False,
                "detailed_debug": True,
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_config = f.name

        old_config = os.environ.get("LITELLM_CONFIG")

        try:
            config = replace(AppConfig.from_env(), litellm_config=temp_config)
            stack = LLMStack(config)

            fake_log_path = (
                Path(tempfile.gettempdir()) / "litellm-detailed-debug-test.log"
            )

            with patch.object(stack, "_find_litellm_pids", return_value=[]):
                with patch.object(
                    stack, "_litellm_log_path", return_value=fake_log_path
                ):
                    with patch.object(
                        stack, "_tail_file_logs", return_value=0
                    ) as mock_tail:
                        stack.show_litellm_logs(
                            tail="10", follow=False, show_log_path=False
                        )

            mock_tail.assert_called_once_with(
                fake_log_path,
                tail="10",
                follow=False,
                pretty_debug=True,
            )
        finally:
            if old_config is None:
                os.environ.pop("LITELLM_CONFIG", None)
            else:
                os.environ["LITELLM_CONFIG"] = old_config
            safe_unlink(temp_config)

    @patch("harinezumigel_llm_stack.load_env_file")
    def test_handle_logs_all_show_log_includes_litellm_paths(
        self, mock_load: Any
    ) -> None:
        """all --show-log should show vLLM and LiteLLM log paths."""
        import argparse

        from harinezumigel_llm_stack import AppConfig, LLMStack  # type: ignore

        config = AppConfig.from_env()
        stack = LLMStack(config)

        args = argparse.Namespace(
            model="all",
            start=False,
            show_log=True,
            show_log_path=False,
            clean_log=False,
            dry_run=False,
            follow=False,
            tail="200",
        )

        with patch.object(stack, "show_all_log_paths") as mock_show_all_log_paths:
            handled = stack.handle_logs(args)

        self.assertTrue(handled)
        mock_show_all_log_paths.assert_called_once_with()

    @patch("harinezumigel_llm_stack.load_env_file")
    def test_handle_logs_litellm_clean_calls_litellm_clean(
        self, mock_load: Any
    ) -> None:
        """litellm --clean-log should route to clean_litellm_logs."""
        import argparse

        from harinezumigel_llm_stack import AppConfig, LLMStack  # type: ignore

        config = AppConfig.from_env()
        stack = LLMStack(config)

        args = argparse.Namespace(
            model="litellm",
            start=False,
            show_log=False,
            show_log_path=False,
            clean_log=True,
            dry_run=True,
            follow=False,
            tail="200",
        )

        with patch.object(stack, "clean_litellm_logs") as mock_clean_litellm_logs:
            handled = stack.handle_logs(args)

        self.assertTrue(handled)
        mock_clean_litellm_logs.assert_called_once_with(dry_run=True)


class TestLLMStackModelResolution(unittest.TestCase):
    """Test model name resolution and lookup."""

    def setUp(self):
        """Set up test configuration."""
        self.env_vars = {
            "LITELLM_CONFIG": "/opt/litellm/config/config.yaml",
            "MODEL_ROOT": "/models",
            "LITELLM_VENV_ACTIVATE": "/opt/litellm/venv/bin/activate",
            "LITELLM_BIN": "/opt/litellm/venv/bin/litellm",
            "LITELLM_BIND_HOST": "0.0.0.0",
            "LITELLM_PORT": "4000",
            "VLLM_HOST": "localhost",
            "VLLM_BIND_HOST": "0.0.0.0",
            "VLLM_CONTAINER_PORT": "8000",
            "VLLM_DOCKER_IMAGE": "vllm/vllm-openai:latest",
            "VLLM_MODEL_VOLUME": "/models:/models:ro",
            "VLLM_CACHE_VOLUME": "/cache:/cache",
        }
        for key, value in self.env_vars.items():
            os.environ[key] = value

    def tearDown(self):
        """Clean up test environment."""
        for key in self.env_vars:
            os.environ.pop(key, None)

    @patch("harinezumigel_llm_stack.load_env_file")
    def test_resolve_model_name_exact(self, mock_load: Any) -> None:
        """Test _resolve_model_name with exact match."""
        from harinezumigel_llm_stack import AppConfig, LLMStack, ModelDeployment  # type: ignore

        config = AppConfig.from_env()
        stack = LLMStack(config)

        # Add a test model
        test_model = ModelDeployment(
            name="test_model",
            backend_model="test/model",
            api_base="http://localhost:8001",
            api_key="test",
            api_base_host="localhost",
            api_base_port=8001,
            litellm_params={},
            model_info={},
            context_length=4096,
            max_input_tokens=3000,
            max_output_tokens=1000,
            gpu_memory_utilization=0.9,
            max_num_seqs=256,
            dtype="auto",
            license=None,
            upstream=None,
            alias=None,
            description=None,
            detail=None,
            allow_rebuild=True,
        )
        stack.models["test_model"] = test_model

        resolved = stack._resolve_model_name("test_model")
        self.assertEqual(resolved, "test_model")

    @patch("harinezumigel_llm_stack.load_env_file")
    def test_resolve_model_name_alias(self, mock_load: Any) -> None:
        """Test _resolve_model_name with alias."""
        from harinezumigel_llm_stack import AppConfig, LLMStack, ModelDeployment  # type: ignore

        config = AppConfig.from_env()
        stack = LLMStack(config)

        # Add test model with alias
        test_model1 = ModelDeployment(
            name="mistral_7b_instruct",
            backend_model="mistralai/Mistral-7B-Instruct-v0.2",
            api_base="http://localhost:8001",
            api_key="test",
            api_base_host="localhost",
            api_base_port=8001,
            litellm_params={},
            model_info={},
            context_length=4096,
            max_input_tokens=3000,
            max_output_tokens=1000,
            gpu_memory_utilization=0.9,
            max_num_seqs=256,
            dtype="auto",
            license=None,
            upstream=None,
            alias="mistral",
            description=None,
            detail=None,
            allow_rebuild=True,
        )
        stack.models["mistral_7b_instruct"] = test_model1

        # Exact match by alias should work
        resolved = stack._resolve_model_name("mistral")
        self.assertEqual(resolved, "mistral_7b_instruct")

        # Non-matching partial should return None
        resolved = stack._resolve_model_name("mist")
        self.assertIsNone(resolved)


class TestFormatSamplingParameters(unittest.TestCase):
    """Test format_sampling_parameters utility."""

    def test_backend_defaults_are_labeled(self):
        """Backend defaults are distinct from proxy defaults."""
        from harinezumigel_llm_stack import format_sampling_parameters  # type: ignore

        result = format_sampling_parameters(
            {"temperature": 0.5, "top_p": 0.9},
            {"top_k": 20},
        )
        self.assertIn("temperature=0.5(backend default)", result)
        self.assertIn("top_p=0.9(backend default)", result)
        self.assertIn("top_k=20(proxy default)", result)

    def test_litellm_params_fallback(self):
        """Proxy-only defaults remain visible."""
        from harinezumigel_llm_stack import format_sampling_parameters  # type: ignore

        result = format_sampling_parameters(
            {},
            {"temperature": 0.2, "top_k": 20},
        )
        self.assertIn("temperature=0.2(proxy default)", result)
        self.assertIn("top_k=20(proxy default)", result)

    def test_zero_defaults_are_displayed(self):
        result = harinezumigel_llm_stack.format_sampling_parameters(
            {"temperature": 0.0}, {"presence_penalty": 0.0}
        )
        self.assertIn("temperature=0.0(backend default)", result)
        self.assertIn("presence_penalty=0.0(proxy default)", result)

    def test_empty_params_returns_empty_list(self):
        """No params configured returns empty list."""
        from harinezumigel_llm_stack import format_sampling_parameters  # type: ignore

        result = format_sampling_parameters({}, {})
        self.assertEqual(result, [])


class TestSafetyGuards(unittest.TestCase):
    """Test that destructive operations are properly guarded."""

    def setUp(self):
        log_root = tempfile.TemporaryDirectory()
        self.addCleanup(log_root.cleanup)
        root_patch = patch.object(
            harinezumigel_llm_stack, "INSTALLED_BASE", Path(log_root.name)
        )
        root_patch.start()
        self.addCleanup(root_patch.stop)
        self.env_vars = {
            "LITELLM_CONFIG": "/opt/litellm/config/config.yaml",
            "MODEL_ROOT": "/models",
            "LITELLM_VENV_ACTIVATE": "/opt/litellm/venv/bin/activate",
            "LITELLM_BIN": "/opt/litellm/venv/bin/litellm",
            "LITELLM_BIND_HOST": "0.0.0.0",
            "LITELLM_PORT": "4000",
            "VLLM_HOST": "localhost",
            "VLLM_BIND_HOST": "0.0.0.0",
            "VLLM_CONTAINER_PORT": "8000",
            "VLLM_DOCKER_IMAGE": "vllm/vllm-openai:latest",
            "VLLM_MODEL_VOLUME": "/models:/models:ro",
            "VLLM_CACHE_VOLUME": "/cache:/cache",
        }
        for key, value in self.env_vars.items():
            os.environ[key] = value

    def tearDown(self):
        for key in self.env_vars:
            os.environ.pop(key, None)

    @patch("harinezumigel_llm_stack.load_env_file")
    def test_duplicate_alias_exits(self, mock_load: Any) -> None:
        """Duplicate aliases in config cause sys.exit(1)."""
        import tempfile
        import yaml  # type: ignore
        from harinezumigel_llm_stack import AppConfig, LLMStack  # type: ignore

        config_data = {
            "model_list": [
                {
                    "model_name": "model_a",
                    "litellm_params": {
                        "model": "openai/model_a",
                        "api_base": "http://localhost:8001/v1",
                        "api_key": "sk-test",
                    },
                    "model_info": {
                        "alias": "myalias",
                        "context_length": 4096,
                        "max_input_tokens": 3000,
                        "max_output_tokens": 512,
                        "gpu_memory_utilization": 0.9,
                    },
                },
                {
                    "model_name": "model_b",
                    "litellm_params": {
                        "model": "openai/model_b",
                        "api_base": "http://localhost:8002/v1",
                        "api_key": "sk-test",
                    },
                    "model_info": {
                        "alias": "myalias",  # duplicate
                        "context_length": 4096,
                        "max_input_tokens": 3000,
                        "max_output_tokens": 512,
                        "gpu_memory_utilization": 0.9,
                    },
                },
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_config = f.name

        try:
            config = replace(AppConfig.from_env(), litellm_config=temp_config)
            stack = LLMStack(config)
            with self.assertRaises(SystemExit):
                stack.load_models()
        finally:
            safe_unlink(temp_config)

    @patch("harinezumigel_llm_stack.load_env_file")
    def test_load_models_reads_litellm_runtime_settings(self, mock_load: Any) -> None:
        """load_models should parse keep_alive and debug flags from litellm_settings."""
        import tempfile
        import yaml  # type: ignore
        from harinezumigel_llm_stack import AppConfig, LLMStack  # type: ignore

        config_data = {
            "model_list": [
                {
                    "model_name": "model_a",
                    "litellm_params": {
                        "model": "openai/model_a",
                        "api_base": "http://localhost:8001/v1",
                        "api_key": "sk-test",
                    },
                    "model_info": {
                        "context_length": 4096,
                        "max_input_tokens": 3000,
                        "max_output_tokens": 512,
                        "gpu_memory_utilization": 0.9,
                    },
                }
            ],
            "litellm_settings": {
                "keep_alive": 30,
                "debug": True,
                "detailed_debug": False,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_config = f.name

        try:
            config = replace(AppConfig.from_env(), litellm_config=temp_config)
            stack = LLMStack(config)
            stack.load_models()

            self.assertEqual(stack.litellm_keep_alive, 30.0)
            self.assertTrue(stack.litellm_debug)
            self.assertFalse(stack.litellm_detailed_debug)
        finally:
            safe_unlink(temp_config)

    @patch("harinezumigel_llm_stack.load_env_file")
    def test_recreate_requires_start(self, mock_load: Any) -> None:
        """--recreate without --start must exit with an error."""
        import argparse
        import tempfile
        import yaml  # type: ignore
        from harinezumigel_llm_stack import AppConfig, LLMStack  # type: ignore

        config_data = {
            "model_list": [
                {
                    "model_name": "my_model",
                    "litellm_params": {
                        "model": "openai/my_model",
                        "api_base": "http://localhost:8001/v1",
                        "api_key": "sk-test",
                    },
                    "model_info": {
                        "context_length": 4096,
                        "max_input_tokens": 3000,
                        "max_output_tokens": 512,
                        "gpu_memory_utilization": 0.9,
                    },
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_config = f.name

        try:
            config = replace(AppConfig.from_env(), litellm_config=temp_config)
            stack = LLMStack(config)

            args = argparse.Namespace(
                model="my_model",
                start=False,
                stop=False,
                recreate=True,
                no_reuse_existing=False,
                dry_run=False,
                show_log=False,
                show_log_path=False,
                clean_log=False,
                follow=False,
                tail="200",
                ps=False,
                list=False,
                help=False,
                port=None,
                auto_port=False,
                context_length=None,
                max_input_tokens=None,
                max_output_tokens=None,
                gpu_memory_utilization=None,
                max_num_seqs=None,
                dtype=None,
            )

            with self.assertRaises(SystemExit):
                stack.run(args)
        finally:
            safe_unlink(temp_config)

    @patch("harinezumigel_llm_stack.load_env_file")
    def test_recreate_not_allowed_with_stop(self, mock_load: Any) -> None:
        """--recreate with --stop must exit with an error."""
        import argparse
        import tempfile
        import yaml  # type: ignore
        from harinezumigel_llm_stack import AppConfig, LLMStack  # type: ignore

        config_data = {
            "model_list": [
                {
                    "model_name": "my_model",
                    "litellm_params": {
                        "model": "openai/my_model",
                        "api_base": "http://localhost:8001/v1",
                        "api_key": "sk-test",
                    },
                    "model_info": {
                        "context_length": 4096,
                        "max_input_tokens": 3000,
                        "max_output_tokens": 512,
                        "gpu_memory_utilization": 0.9,
                    },
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_config = f.name

        try:
            config = replace(AppConfig.from_env(), litellm_config=temp_config)
            stack = LLMStack(config)

            args = argparse.Namespace(
                model="my_model",
                start=False,
                stop=True,
                recreate=True,
                no_reuse_existing=False,
                dry_run=False,
                show_log=False,
                show_log_path=False,
                clean_log=False,
                follow=False,
                tail="200",
                ps=False,
                list=False,
                help=False,
                port=None,
                auto_port=False,
                context_length=None,
                max_input_tokens=None,
                max_output_tokens=None,
                gpu_memory_utilization=None,
                max_num_seqs=None,
                dtype=None,
            )

            with self.assertRaises(SystemExit):
                stack.run(args)
        finally:
            safe_unlink(temp_config)

    @patch("harinezumigel_llm_stack.load_env_file")
    def test_recreate_not_allowed_for_litellm(self, mock_load: Any) -> None:
        """--recreate is never allowed for the litellm target."""
        import argparse
        import tempfile
        import yaml  # type: ignore
        from harinezumigel_llm_stack import AppConfig, LLMStack  # type: ignore

        config_data = {"model_list": []}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_config = f.name

        try:
            config = replace(AppConfig.from_env(), litellm_config=temp_config)
            stack = LLMStack(config)

            for model_name in ("litellm", "llm-lite", "lite-llm"):
                for start, stop in ((True, False), (False, True), (False, False)):
                    args = argparse.Namespace(
                        model=model_name,
                        start=start,
                        stop=stop,
                        recreate=True,
                        no_reuse_existing=False,
                        dry_run=False,
                        show_log=False,
                        show_log_path=False,
                        clean_log=False,
                        follow=False,
                        tail="200",
                        ps=False,
                        list=False,
                        help=False,
                        port=None,
                        auto_port=False,
                        context_length=None,
                        max_input_tokens=None,
                        max_output_tokens=None,
                        gpu_memory_utilization=None,
                        max_num_seqs=None,
                        dtype=None,
                    )
                    with self.assertRaises(SystemExit):
                        stack.run(args)
        finally:
            safe_unlink(temp_config)

    @patch("harinezumigel_llm_stack.load_env_file")
    def test_litellm_show_log_path_is_supported(self, mock_load: Any) -> None:
        """litellm target should support --show-log-path."""
        import argparse
        import tempfile
        import yaml  # type: ignore
        from harinezumigel_llm_stack import AppConfig, LLMStack  # type: ignore

        config_data = {"model_list": []}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_config = f.name

        try:
            config = replace(AppConfig.from_env(), litellm_config=temp_config)
            stack = LLMStack(config)

            args = argparse.Namespace(
                model="litellm",
                start=False,
                stop=False,
                recreate=False,
                no_reuse_existing=False,
                dry_run=False,
                show_log=False,
                show_log_path=True,
                clean_log=False,
                follow=False,
                tail="200",
                ps=False,
                list=False,
                help=False,
                port=None,
                auto_port=False,
                context_length=None,
                max_input_tokens=None,
                max_output_tokens=None,
                gpu_memory_utilization=None,
                max_num_seqs=None,
                dtype=None,
            )

            with patch.object(
                stack, "show_litellm_log_path"
            ) as mock_show_litellm_log_path:
                stack.run(args)

            mock_show_litellm_log_path.assert_called_once_with()
        finally:
            safe_unlink(temp_config)

    @patch("harinezumigel_llm_stack.load_env_file")
    def test_litellm_clean_log_is_supported(self, mock_load: Any) -> None:
        """litellm target should support --clean-log."""
        import argparse
        import tempfile
        import yaml  # type: ignore
        from harinezumigel_llm_stack import AppConfig, LLMStack  # type: ignore

        config_data = {"model_list": []}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_config = f.name

        try:
            config = replace(AppConfig.from_env(), litellm_config=temp_config)
            stack = LLMStack(config)

            args = argparse.Namespace(
                model="litellm",
                start=False,
                stop=False,
                recreate=False,
                no_reuse_existing=False,
                dry_run=True,
                show_log=False,
                show_log_path=False,
                clean_log=True,
                follow=False,
                tail="200",
                ps=False,
                list=False,
                help=False,
                port=None,
                auto_port=False,
                context_length=None,
                max_input_tokens=None,
                max_output_tokens=None,
                gpu_memory_utilization=None,
                max_num_seqs=None,
                dtype=None,
            )

            with patch.object(stack, "clean_litellm_logs") as mock_clean_litellm_logs:
                stack.run(args)

            mock_clean_litellm_logs.assert_called_once_with(dry_run=True)
        finally:
            safe_unlink(temp_config)

    @patch("harinezumigel_llm_stack.load_env_file")
    def test_rebuild_false_blocks_recreate(self, mock_load: Any) -> None:
        """rebuild: false in config.yaml must block --recreate for that model."""
        import argparse
        import tempfile
        import yaml  # type: ignore
        from harinezumigel_llm_stack import AppConfig, LLMStack  # type: ignore

        config_data = {
            "model_list": [
                {
                    "model_name": "protected_model",
                    "rebuild": False,
                    "litellm_params": {
                        "model": "openai/protected_model",
                        "api_base": "http://localhost:8001/v1",
                        "api_key": "sk-test",
                    },
                    "model_info": {
                        "context_length": 4096,
                        "max_input_tokens": 3000,
                        "max_output_tokens": 512,
                        "gpu_memory_utilization": 0.9,
                    },
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_config = f.name

        try:
            config = replace(AppConfig.from_env(), litellm_config=temp_config)
            stack = LLMStack(config)

            args = argparse.Namespace(
                model="protected_model",
                start=True,
                stop=False,
                recreate=True,
                no_reuse_existing=False,
                dry_run=False,
                show_log=False,
                show_log_path=False,
                clean_log=False,
                follow=False,
                tail="200",
                ps=False,
                list=False,
                help=False,
                port=8001,
                auto_port=False,
                context_length=None,
                max_input_tokens=None,
                max_output_tokens=None,
                gpu_memory_utilization=None,
                max_num_seqs=None,
                dtype=None,
            )

            with self.assertRaises(SystemExit):
                stack.run(args)
        finally:
            safe_unlink(temp_config)

    @patch("harinezumigel_llm_stack.load_env_file")
    @patch("harinezumigel_llm_stack.run_command")
    def test_stop_litellm_uses_config_scoped_patterns(
        self, mock_run: Any, mock_load: Any
    ) -> None:
        """stop_litellm must search only for processes using this config path."""
        from harinezumigel_llm_stack import AppConfig, LLMStack  # type: ignore

        config = AppConfig.from_env()
        stack = LLMStack(config)

        # pgrep returns 1 when no matching process exists.
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")

        stack.stop_litellm(dry_run=True)

        self.assertEqual(mock_run.call_count, 1)
        commands = [call.args[0] for call in mock_run.call_args_list]
        self.assertEqual(
            commands,
            [
                [
                    "pgrep",
                    "-fa",
                    "litellm",
                ],
            ],
        )

    @patch("harinezumigel_llm_stack.load_env_file")
    @patch("harinezumigel_llm_stack.run_command")
    def test_find_litellm_pids_filters_non_process_matches(
        self,
        mock_run: Any,
        mock_load: Any,
    ) -> None:
        """Only true LiteLLM process argv entries should be matched."""
        from harinezumigel_llm_stack import AppConfig, LLMStack  # type: ignore

        config = AppConfig.from_env()
        stack = LLMStack(config)

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=(
                f"1001 logview --name litellm --config {config.litellm_config}\n"
                f"1002 bash -lc litellm --config {config.litellm_config}\n"
                f"1003 {config.litellm_bin} --config {config.litellm_config} --port 4000\n"
                f"1004 python3 {config.litellm_bin} --config={config.litellm_config} --port 4001\n"
                "1005 litellm --config /tmp/other-config.yaml --port 4010\n"
            ),
            stderr="",
        )

        self.assertEqual(stack._find_litellm_pids(), ["1003", "1004"])

    @patch("harinezumigel_llm_stack.load_env_file")
    @patch("harinezumigel_llm_stack.start_background_process")
    @patch("harinezumigel_llm_stack.port_in_use")
    @patch("harinezumigel_llm_stack.run_command")
    def test_start_litellm_blocks_when_config_scoped_instance_exists(
        self,
        mock_run: Any,
        mock_port_in_use: Any,
        mock_start_background: Any,
        mock_load: Any,
    ) -> None:
        """start_litellm must refuse startup when this config already runs."""
        import io
        from contextlib import redirect_stdout

        from harinezumigel_llm_stack import AppConfig, LLMStack  # type: ignore

        config = AppConfig.from_env()
        stack = LLMStack(config)

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=f"1234 {config.litellm_bin} --config {config.litellm_config} --port 4000\n",
            stderr="",
        )
        mock_port_in_use.return_value = False

        output = io.StringIO()

        with self.assertRaises(SystemExit):
            with redirect_stdout(output):
                stack.start_litellm(dry_run=False, follow_log=False)

        self.assertIn(
            "Only one LiteLLM instance can run for this config path",
            output.getvalue(),
        )
        commands = [call.args[0] for call in mock_run.call_args_list]
        self.assertEqual(
            commands,
            [
                [
                    "pgrep",
                    "-fa",
                    "litellm",
                ],
            ],
        )
        mock_port_in_use.assert_not_called()
        mock_start_background.assert_not_called()

    @patch("harinezumigel_llm_stack.load_env_file")
    @patch("harinezumigel_llm_stack.start_background_process")
    @patch("harinezumigel_llm_stack.port_in_use")
    @patch("harinezumigel_llm_stack.run_command")
    def test_start_litellm_background_redirects_to_log_file(
        self,
        mock_run: Any,
        mock_port_in_use: Any,
        mock_start_background: Any,
        mock_load: Any,
    ) -> None:
        """Detached startup writes timestamped logs under the installed base."""
        import shlex
        import tempfile

        from harinezumigel_llm_stack import AppConfig, LLMStack  # type: ignore

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(
                harinezumigel_llm_stack, "INSTALLED_BASE", Path(temp_dir)
            ):
                config = AppConfig.from_env()
                stack = LLMStack(config)

                # pgrep returns 1 when no matching process exists.
                mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
                mock_port_in_use.return_value = False

                stack.start_litellm(dry_run=False, follow_log=False)

                mock_start_background.assert_called_once()
                command = mock_start_background.call_args.args[0]

                self.assertEqual(command[0:2], ["bash", "-lc"])
                self.assertIn(
                    f">> {shlex.quote(str(Path(temp_dir) / 'logs'))}/litellm_",
                    command[2],
                )
                self.assertIn(".log 2>&1", command[2])

    @patch("harinezumigel_llm_stack.load_env_file")
    def test_clean_litellm_logs_removes_managed_files(self, mock_load: Any) -> None:
        """clean_litellm_logs should remove timestamped logs in managed directory."""
        import tempfile

        from harinezumigel_llm_stack import AppConfig, LLMStack  # type: ignore

        with tempfile.TemporaryDirectory() as temp_dir:
            logs_dir = Path(temp_dir) / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)

            first = logs_dir / "litellm_20261018_101010.log"
            second = logs_dir / "litellm_20261018_101011.log"
            legacy = Path(temp_dir) / "litellm.log"

            first.write_text("first", encoding="utf-8")
            second.write_text("second", encoding="utf-8")
            legacy.write_text("legacy", encoding="utf-8")

            with patch.object(
                harinezumigel_llm_stack, "INSTALLED_BASE", Path(temp_dir)
            ):
                config = AppConfig.from_env()
                stack = LLMStack(config)
                stack.clean_litellm_logs(dry_run=False)

                self.assertFalse(first.exists())
                self.assertFalse(second.exists())
                self.assertTrue(legacy.exists())

    @patch("harinezumigel_llm_stack.load_env_file")
    @patch("harinezumigel_llm_stack.start_background_process")
    @patch("harinezumigel_llm_stack.port_in_use")
    @patch("harinezumigel_llm_stack.run_command")
    def test_start_litellm_applies_keep_alive_and_debug_settings(
        self,
        mock_run: Any,
        mock_port_in_use: Any,
        mock_start_background: Any,
        mock_load: Any,
    ) -> None:
        """start_litellm should apply keep_alive and debug settings to the command."""
        import tempfile
        import yaml  # type: ignore

        from harinezumigel_llm_stack import AppConfig, LLMStack  # type: ignore

        config_data = {
            "model_list": [
                {
                    "model_name": "model_a",
                    "litellm_params": {
                        "model": "openai/model_a",
                        "api_base": "http://localhost:8001/v1",
                        "api_key": "sk-test",
                    },
                    "model_info": {
                        "context_length": 4096,
                        "max_input_tokens": 3000,
                        "max_output_tokens": 512,
                        "gpu_memory_utilization": 0.9,
                    },
                }
            ],
            "litellm_settings": {
                "keep_alive": 45,
                "debug": True,
                "detailed_debug": False,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_config = f.name

        try:
            config = replace(AppConfig.from_env(), litellm_config=temp_config)
            stack = LLMStack(config)
            stack.load_models()

            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
            mock_port_in_use.return_value = False

            stack.start_litellm(dry_run=False, follow_log=False)

            mock_start_background.assert_called_once()
            command = mock_start_background.call_args.args[0][2]

            self.assertIn("export LITELLM_LOG=DEBUG", command)
            self.assertIn("export AIOHTTP_KEEPALIVE_TIMEOUT=45", command)
            self.assertIn(" --debug", command)
        finally:
            safe_unlink(temp_config)

    @patch("harinezumigel_llm_stack.load_env_file")
    def test_top_level_rebuild_false_blocks_recreate_alias(
        self, mock_load: Any
    ) -> None:
        """Top-level rebuild: false must block --recreate when starting by alias."""
        import argparse
        import tempfile
        import yaml  # type: ignore
        from harinezumigel_llm_stack import AppConfig, LLMStack  # type: ignore

        config_data = {
            "model_list": [
                {
                    "model_name": "mistral_7b",
                    "rebuild": False,
                    "litellm_params": {
                        "model": "openai/mistral_7b",
                        "api_base": "http://localhost:8001/v1",
                        "api_key": "sk-test",
                    },
                    "model_info": {
                        "alias": "mistral",
                        "context_length": 4096,
                        "max_input_tokens": 3000,
                        "max_output_tokens": 512,
                        "gpu_memory_utilization": 0.9,
                    },
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_config = f.name

        try:
            config = replace(AppConfig.from_env(), litellm_config=temp_config)
            stack = LLMStack(config)

            args = argparse.Namespace(
                model="mistral",
                start=True,
                stop=False,
                recreate=True,
                no_reuse_existing=False,
                dry_run=False,
                show_log=False,
                show_log_path=False,
                clean_log=False,
                follow=False,
                tail="200",
                ps=False,
                list=False,
                help=False,
                port=8001,
                auto_port=False,
                context_length=None,
                max_input_tokens=None,
                max_output_tokens=None,
                gpu_memory_utilization=None,
                max_num_seqs=None,
                dtype=None,
            )

            with self.assertRaises(SystemExit):
                stack.run(args)
        finally:
            safe_unlink(temp_config)

    @patch("harinezumigel_llm_stack.load_env_file")
    def test_missing_top_level_rebuild_blocks_recreate_with_message(
        self, mock_load: Any
    ) -> None:
        """Missing top-level rebuild must block --recreate even if model_info.rebuild is true."""
        import argparse
        import io
        import tempfile
        from contextlib import redirect_stdout

        import yaml  # type: ignore
        from harinezumigel_llm_stack import AppConfig, LLMStack  # type: ignore

        config_data = {
            "model_list": [
                {
                    "model_name": "protected_model",
                    "litellm_params": {
                        "model": "openai/protected_model",
                        "api_base": "http://localhost:8001/v1",
                        "api_key": "sk-test",
                    },
                    "model_info": {
                        "rebuild": True,
                        "context_length": 4096,
                        "max_input_tokens": 3000,
                        "max_output_tokens": 512,
                        "gpu_memory_utilization": 0.9,
                    },
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_config = f.name

        try:
            config = replace(AppConfig.from_env(), litellm_config=temp_config)
            stack = LLMStack(config)

            args = argparse.Namespace(
                model="protected_model",
                start=True,
                stop=False,
                recreate=True,
                no_reuse_existing=False,
                dry_run=False,
                show_log=False,
                show_log_path=False,
                clean_log=False,
                follow=False,
                tail="200",
                ps=False,
                list=False,
                help=False,
                port=8001,
                auto_port=False,
                context_length=None,
                max_input_tokens=None,
                max_output_tokens=None,
                gpu_memory_utilization=None,
                max_num_seqs=None,
                dtype=None,
            )

            output = io.StringIO()

            with self.assertRaises(SystemExit):
                with redirect_stdout(output):
                    stack.run(args)

            self.assertIn("rebuild: true must be specified", output.getvalue())
        finally:
            safe_unlink(temp_config)


class TestCLIProgramName(unittest.TestCase):
    """Keep user-facing commands consistent with the invoked filename."""

    def setUp(self) -> None:
        self.stack = harinezumigel_llm_stack.LLMStack(harinezumigel_llm_stack.APP)

    def test_help_examples_are_aligned(self) -> None:
        for invoked in ("llm", "harinezumigel-llm-stack"):
            with self.subTest(invoked=invoked), patch.object(sys, "argv", [invoked]):
                text = self.stack.generate_help()
                examples = text.split("Examples:\n", 1)[1]
                commands = [
                    line
                    for line in examples.splitlines()
                    if line.lstrip().startswith(f"{invoked} ")
                ]
                self.assertTrue(commands)
                self.assertTrue(
                    all(line.startswith(f"    {invoked} ") for line in commands)
                )
                columns = {line.index("#") for line in commands if "#" in line}
                self.assertEqual(columns, {4 + len(invoked) + 1 + 42 + 1})
                self.assertIn("  Batch startup:\n  --------------\n\n", examples)
                self.assertIn(f"{invoked} snowflake-embedder --start", examples)
                self.assertIn(f"{invoked} mmarco-reranker --start", examples)
                self.assertIn("# embedding encoder alias (Snowflake)", examples)
                self.assertIn("# cross-encoder alias (mMARCO)", examples)
                self.assertIn(
                    f"{invoked} --group rag --start --startup-timeout 600", examples
                )

    def test_help_does_not_dump_model_values(self) -> None:
        self.stack.models = {"not-for-help": MagicMock()}
        text = self.stack.generate_help()
        self.assertNotIn("Current model values", text)
        self.assertNotIn("not-for-help", text)
        self.assertNotIn("gpu_mem=", text)
        self.assertIn("Run without arguments to show detailed model settings.", text)

    def test_terminal_color_respects_output_and_environment(self) -> None:
        cases = [
            (True, {"TERM": "xterm-256color"}, True),
            (True, {"TERM": "screen"}, True),
            (True, {"TERM": "xterm", "NO_COLOR": ""}, False),
            (True, {"TERM": "dumb"}, False),
            (True, {}, False),
            (False, {"TERM": "xterm-256color"}, False),
        ]
        for tty, env, colored in cases:
            with self.subTest(tty=tty, env=env):
                with patch.dict(os.environ, env, clear=True):
                    with patch.object(sys.stdout, "isatty", return_value=tty):
                        text = harinezumigel_llm_stack.terminal_style(
                            "example", accent=True
                        )
                self.assertEqual("\033[" in text, colored)
                if colored:
                    self.assertEqual(text, "\033[36mexample\033[0m")
                else:
                    self.assertEqual(text, "example")

    def test_colored_help_preserves_plain_layout(self) -> None:
        import re

        with patch.object(sys, "argv", ["llm"]):
            with patch.dict(os.environ, {"TERM": "xterm"}, clear=True):
                with patch.object(sys.stdout, "isatty", return_value=False):
                    plain = self.stack.generate_help()
                with patch.object(sys.stdout, "isatty", return_value=True):
                    colored = self.stack.generate_help()
        self.assertIn("\033[1m  Batch startup:\033[0m", colored)
        self.assertIn("\033[36m    llm --group rag --start", colored)
        self.assertEqual(re.sub(r"\033\[[0-9;]*m", "", colored), plain)
        self.assertNotIn("\033[37m", colored)
        self.assertNotIn("\033[30m", colored)

    def test_help_uses_invoked_filename(self) -> None:
        for invoked, expected in [
            ("harinezumigel-llm-stack", "harinezumigel-llm-stack"),
            ("./short.py", "short.py"),
            ("/tmp/bin/llm", "llm"),
        ]:
            with self.subTest(invoked=invoked), patch.object(sys, "argv", [invoked]):
                text = self.stack.generate_help()
                self.assertIn(f"{expected} — Unified LiteLLM", text)
                commands = [
                    line.strip()
                    for line in text.splitlines()
                    if " --" in line and not line.lstrip().startswith("-")
                ]
                self.assertGreaterEqual(len(commands), 29)
                for command in commands:
                    self.assertTrue(command.startswith(f"{expected} "), command)
                if expected != "harinezumigel-llm-stack":
                    self.assertNotIn("harinezumigel-llm-stack", text)

    def test_parser_error_uses_invoked_filename(self) -> None:
        output = io.StringIO()
        with patch.object(sys, "argv", ["/tmp/bin/llm", "--invalid-option"]):
            with redirect_stderr(output), self.assertRaises(SystemExit) as error:
                harinezumigel_llm_stack.parse_args()
        self.assertEqual(error.exception.code, 2)
        self.assertIn("usage: llm ", output.getvalue())
        self.assertIn("llm: error:", output.getvalue())
        self.assertNotIn("harinezumigel-llm-stack", output.getvalue())

    def test_banner_uses_invoked_filename(self) -> None:
        output = io.StringIO()
        with patch.object(sys, "argv", ["llm", "--help"]):
            with patch.object(harinezumigel_llm_stack.LLMStack, "run") as run:
                with redirect_stdout(output):
                    harinezumigel_llm_stack.main()
        self.assertIn(f" llm {harinezumigel_llm_stack.__version__} ", output.getvalue())
        self.assertNotIn("harinezumigel-llm-stack", output.getvalue())
        self.assertTrue(run.call_args.args[0].help)

    def test_missing_container_status_uses_invoked_filename(self) -> None:
        output = io.StringIO()
        with patch.object(sys, "argv", ["llm"]):
            with patch.object(self.stack, "_find_all_vllm_containers", return_value=[]):
                with redirect_stdout(output):
                    self.stack.show_container_status("coder")
        self.assertIn("Start with: llm coder --start", output.getvalue())
        self.assertNotIn("harinezumigel-llm-stack", output.getvalue())

    def test_missing_container_logs_use_invoked_filename(self) -> None:
        output = io.StringIO()
        with patch.object(sys, "argv", ["llm", "coder", "--show-log"]):
            args = harinezumigel_llm_stack.parse_args()
            with patch.object(self.stack, "_resolve_model_name", return_value="coder"):
                with patch.object(
                    self.stack, "_find_all_vllm_containers", return_value=[]
                ):
                    with (
                        redirect_stdout(output),
                        self.assertRaises(SystemExit) as error,
                    ):
                        self.stack.handle_logs(args)
        self.assertEqual(error.exception.code, 1)
        self.assertIn("  llm coder --start\n", output.getvalue())
        self.assertIn("  llm coder --start --show-log\n", output.getvalue())
        self.assertNotIn("harinezumigel-llm-stack", output.getvalue())

    def test_existing_litellm_instance_hint_uses_invoked_filename(self) -> None:
        output = io.StringIO()
        with patch.object(sys, "argv", ["llm"]):
            with patch.object(
                self.stack, "_find_litellm_pids", return_value=["other-pid"]
            ):
                with redirect_stdout(output), self.assertRaises(SystemExit) as error:
                    self.stack.start_litellm()
        self.assertEqual(error.exception.code, 1)
        self.assertIn("started via llm.", output.getvalue())
        self.assertIn("(llm litellm --stop for this config,", output.getvalue())
        self.assertNotIn("harinezumigel-llm-stack", output.getvalue())


class TestCLIParseArgs(unittest.TestCase):
    """Test CLI argument parsing behavior."""

    def test_stream_log_sets_show_log_and_follow(self) -> None:
        """--stream-log should normalize to --show-log + --follow."""
        from harinezumigel_llm_stack import parse_args  # type: ignore

        with patch.object(
            sys, "argv", ["harinezumigel-llm-stack", "mistral_7b", "--stream-log"]
        ):
            args = parse_args()

        self.assertTrue(args.stream_log)
        self.assertTrue(args.show_log)
        self.assertTrue(args.follow)

    def test_stream_log_conflicts_with_show_log(self) -> None:
        """--stream-log and --show-log cannot be used together."""
        from harinezumigel_llm_stack import parse_args  # type: ignore

        with patch.object(
            sys,
            "argv",
            ["harinezumigel-llm-stack", "mistral_7b", "--stream-log", "--show-log"],
        ):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_stream_log_conflicts_with_follow(self) -> None:
        """--stream-log and --follow cannot be used together."""
        from harinezumigel_llm_stack import parse_args  # type: ignore

        with patch.object(
            sys,
            "argv",
            ["harinezumigel-llm-stack", "mistral_7b", "--stream-log", "--follow"],
        ):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_show_log_with_follow_still_valid(self) -> None:
        """Existing --show-log --follow behavior should remain valid."""
        from harinezumigel_llm_stack import parse_args  # type: ignore

        with patch.object(
            sys,
            "argv",
            ["harinezumigel-llm-stack", "mistral_7b", "--show-log", "--follow"],
        ):
            args = parse_args()

        self.assertFalse(args.stream_log)
        self.assertTrue(args.show_log)
        self.assertTrue(args.follow)


class TestRunCommand(unittest.TestCase):
    """Test run_command utility."""

    def test_run_command_success(self):
        """Test run_command with successful execution."""
        from harinezumigel_llm_stack import run_command  # type: ignore

        result = run_command(["echo", "test"], capture=True, check=False)

        self.assertEqual(result.returncode, 0)
        self.assertIn("test", result.stdout)

    def test_run_command_failure(self):
        """Test run_command with failed execution."""
        from harinezumigel_llm_stack import run_command  # type: ignore

        result = run_command(["false"], capture=True, check=False)
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
