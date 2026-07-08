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

import json
import os
import socket
import sys
import tempfile
import importlib.util
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# Add the script directory to Python path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

# Import the main script as a module
# Using importlib to handle the file with dashes in the name
spec = importlib.util.spec_from_file_location(
    "harinezumigel_llm_stack",
    SCRIPT_DIR / "harinezumigel-llm-stack.py"
)
if spec is None or spec.loader is None:
    raise RuntimeError("Failed to load harinezumigel-llm-stack.py module spec")

harinezumigel_llm_stack = importlib.util.module_from_spec(spec)
sys.modules["harinezumigel_llm_stack"] = harinezumigel_llm_stack
spec.loader.exec_module(harinezumigel_llm_stack)


class TestEnvironmentLoading(unittest.TestCase):
    """Test environment variable loading and parsing."""

    def test_load_env_file_basic(self):
        """Test loading basic KEY=value pairs."""
        from harinezumigel_llm_stack import load_env_file # type: ignore

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".env") as f:
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
            os.unlink(temp_path)
            # Clean up environment
            for key in ["TEST_VAR1", "TEST_VAR2", "TEST_VAR3"]:
                os.environ.pop(key, None)

    def test_load_env_file_with_export(self):
        """Test loading KEY=value with export prefix."""
        from harinezumigel_llm_stack import load_env_file # type: ignore

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".env") as f:
            f.write("export EXPORT_VAR1=exported1\n")
            f.write("EXPORT_VAR2=exported2\n")
            temp_path = f.name

        try:
            load_env_file(temp_path)
            self.assertEqual(os.environ.get("EXPORT_VAR1"), "exported1")
            self.assertEqual(os.environ.get("EXPORT_VAR2"), "exported2")
        finally:
            os.unlink(temp_path)
            os.environ.pop("EXPORT_VAR1", None)
            os.environ.pop("EXPORT_VAR2", None)

    def test_load_env_file_with_variable_expansion(self):
        """Test variable expansion ${VAR} and $VAR."""
        from harinezumigel_llm_stack import load_env_file # type: ignore

        os.environ["BASE_DIR"] = "/opt/test"

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".env") as f:
            f.write("EXPANDED1=${BASE_DIR}/subdir\n")
            f.write("EXPANDED2=$BASE_DIR/another\n")
            temp_path = f.name

        try:
            load_env_file(temp_path)
            self.assertEqual(os.environ.get("EXPANDED1"), "/opt/test/subdir")
            self.assertEqual(os.environ.get("EXPANDED2"), "/opt/test/another")
        finally:
            os.unlink(temp_path)
            os.environ.pop("BASE_DIR", None)
            os.environ.pop("EXPANDED1", None)
            os.environ.pop("EXPANDED2", None)

    def test_load_env_file_nonexistent(self):
        """Test that loading nonexistent file doesn't raise error."""
        from harinezumigel_llm_stack import load_env_file # type: ignore

        # Should not raise
        load_env_file("/nonexistent/file.env")

    def test_env_string_required(self):
        """Test env_string with required flag."""
        from harinezumigel_llm_stack import env_string # type: ignore

        os.environ["REQUIRED_VAR"] = "test_value"
        self.assertEqual(env_string("REQUIRED_VAR", required=True), "test_value")

        # Test missing required variable causes exit
        with self.assertRaises(SystemExit):
            env_string("MISSING_REQUIRED_VAR", required=True)

        os.environ.pop("REQUIRED_VAR", None)

    def test_env_string_with_default(self):
        """Test env_string with default value."""
        from harinezumigel_llm_stack import env_string # type: ignore

        result = env_string("NONEXISTENT_VAR", default="default_value")
        self.assertEqual(result, "default_value")

    def test_env_int(self):
        """Test env_int parsing."""
        from harinezumigel_llm_stack import env_int # type: ignore

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
        from harinezumigel_llm_stack import resolve_env_value # type: ignore

        self.assertEqual(resolve_env_value("plain"), "plain")
        self.assertEqual(resolve_env_value(None), "")
        self.assertEqual(resolve_env_value(123), 123)

    def test_resolve_env_value_os_environ(self):
        """Test resolve_env_value with os.environ/ prefix."""
        from harinezumigel_llm_stack import resolve_env_value # type: ignore

        os.environ["TEST_RESOLVE"] = "resolved_value"
        result = resolve_env_value("os.environ/TEST_RESOLVE")
        self.assertEqual(result, "resolved_value")
        os.environ.pop("TEST_RESOLVE", None)

    def test_resolve_env_value_shell_style(self):
        """Test resolve_env_value with shell-style variables."""
        from harinezumigel_llm_stack import resolve_env_value # type: ignore

        os.environ["SHELL_VAR"] = "expanded"
        result = resolve_env_value("$SHELL_VAR/path")
        self.assertEqual(result, "expanded/path")
        os.environ.pop("SHELL_VAR", None)

    def test_docker_safe_name(self):
        """Test docker_safe_name conversion."""
        from harinezumigel_llm_stack import docker_safe_name # type: ignore

        self.assertEqual(docker_safe_name("Model_Name"), "model_name")
        self.assertEqual(docker_safe_name("model@#$name"), "model-name")
        self.assertEqual(docker_safe_name("--multiple---dashes--"), "multiple-dashes")
        self.assertEqual(docker_safe_name("CamelCase123"), "camelcase123")

    def test_backend_model_without_provider(self):
        """Test backend_model_without_provider."""
        from harinezumigel_llm_stack import backend_model_without_provider # type: ignore

        self.assertEqual(backend_model_without_provider("openai/gpt-4"), "gpt-4")
        self.assertEqual(backend_model_without_provider("model-name"), "model-name")
        self.assertEqual(backend_model_without_provider(""), "")

    def test_as_int(self):
        """Test as_int conversion."""
        from harinezumigel_llm_stack import as_int # type: ignore

        self.assertEqual(as_int("42"), 42)
        self.assertEqual(as_int(42), 42)
        self.assertEqual(as_int("not_a_number", default=99), 99)
        self.assertIsNone(as_int(None))
        self.assertEqual(as_int(None, default=10), 10)

    def test_as_float(self):
        """Test as_float conversion."""
        from harinezumigel_llm_stack import as_float # type: ignore

        self.assertEqual(as_float("3.14"), 3.14)
        self.assertEqual(as_float(3.14), 3.14)
        self.assertEqual(as_float("not_a_number", default=1.5), 1.5)
        self.assertIsNone(as_float(None))
        self.assertEqual(as_float(None, default=2.5), 2.5)

    def test_as_bool(self):
        """Test as_bool conversion."""
        from harinezumigel_llm_stack import as_bool # type: ignore

        # Test truthy values
        self.assertTrue(as_bool(True))
        self.assertTrue(as_bool("true"))
        self.assertTrue(as_bool("True"))
        self.assertTrue(as_bool("1"))
        self.assertTrue(as_bool("yes"))
        self.assertTrue(as_bool("y"))
        self.assertTrue(as_bool("on"))

        # Test falsy values
        self.assertFalse(as_bool(False))
        self.assertFalse(as_bool("false"))
        self.assertFalse(as_bool("0"))
        self.assertFalse(as_bool("no"))
        self.assertFalse(as_bool(""))
        self.assertFalse(as_bool(None))

    def test_port_in_use(self):
        """Test port_in_use check."""
        from harinezumigel_llm_stack import port_in_use # type: ignore

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

    def test_redact_command(self):
        """Test redact_command for sensitive data."""
        from harinezumigel_llm_stack import redact_command # type: ignore

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


class TestFormatLogLine(unittest.TestCase):
    """Test log formatting functions."""

    def test_format_log_line_with_stacktrace(self):
        """Test _format_log_line extracts stacktrace."""
        from harinezumigel_llm_stack import _format_log_line # type: ignore

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
        from harinezumigel_llm_stack import _format_log_line # type: ignore

        log_obj = {"level": "INFO", "message": "Normal log"}
        log_line = json.dumps(log_obj)

        result = _format_log_line(log_line)
        self.assertEqual(result, log_line)

    def test_format_log_line_non_json(self):
        """Test _format_log_line with non-JSON input."""
        from harinezumigel_llm_stack import _format_log_line # type: ignore

        plain_text = "This is not JSON"
        result = _format_log_line(plain_text)
        self.assertEqual(result, plain_text)


class TestAppConfig(unittest.TestCase):
    """Test AppConfig dataclass."""

    def setUp(self):
        """Set up test environment variables."""
        self.env_vars = {
            "LITELLM_CONFIG": "/opt/litellm/config.yaml",
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
            "VLLM_AUTO_PORT_START": "8001",
            "VLLM_AUTO_PORT_END": "8010",
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
        from harinezumigel_llm_stack import AppConfig # type: ignore

        config = AppConfig.from_env()

        self.assertEqual(config.litellm_config, "/opt/litellm/config.yaml")
        self.assertEqual(config.model_root, "/models")
        self.assertEqual(config.litellm_port, 4000)
        self.assertEqual(config.vllm_container_port, 8000)
        self.assertEqual(config.vllm_auto_port_start, 8001)
        self.assertEqual(config.vllm_auto_port_end, 8010)

    @patch("harinezumigel_llm_stack.load_env_file")
    def test_app_config_missing_required(self, mock_load: Any) -> None:
        """Test AppConfig.from_env() fails with missing required vars."""
        from harinezumigel_llm_stack import AppConfig # type: ignore

        os.environ.pop("LITELLM_CONFIG")

        with self.assertRaises(SystemExit):
            AppConfig.from_env()


class TestModelRequirements(unittest.TestCase):
    """Test model requirement validation functions."""

    def test_require_model_int_valid(self):
        """Test require_model_int with valid integer."""
        from harinezumigel_llm_stack import require_model_int # type: ignore

        model_info = {"max_tokens": 4096}
        result = require_model_int("test_model", model_info, "max_tokens")
        self.assertEqual(result, 4096)

    def test_require_model_int_missing(self):
        """Test require_model_int with missing key."""
        from harinezumigel_llm_stack import require_model_int # type: ignore

        model_info: dict[str, Any] = {}

        with self.assertRaises(SystemExit):
            require_model_int("test_model", model_info, "missing_key")

    def test_require_model_float_valid(self):
        """Test require_model_float with valid float."""
        from harinezumigel_llm_stack import require_model_float # type: ignore

        model_info = {"gpu_memory_utilization": 0.9}
        result = require_model_float("test_model", model_info, "gpu_memory_utilization")
        self.assertEqual(result, 0.9)

    def test_require_model_float_missing(self):
        """Test require_model_float with missing key."""
        from harinezumigel_llm_stack import require_model_float # type: ignore

        model_info: dict[str, Any] = {}

        with self.assertRaises(SystemExit):
            require_model_float("test_model", model_info, "missing_key")


class TestLLMStackInit(unittest.TestCase):
    """Test LLMStack initialization and configuration parsing."""

    def setUp(self):
        """Set up test configuration."""
        self.env_vars = {
            "LITELLM_CONFIG": "/opt/litellm/config.yaml",
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
        from harinezumigel_llm_stack import AppConfig, LLMStack # type: ignore

        config = AppConfig.from_env()
        stack = LLMStack(config)

        self.assertEqual(stack.config, config)
        self.assertIsInstance(stack.models, dict)
        self.assertEqual(len(stack.models), 0)

    @patch("harinezumigel_llm_stack.load_env_file")
    def test_parse_api_base(self, mock_load: Any) -> None:
        """Test _parse_api_base URL parsing."""
        from harinezumigel_llm_stack import AppConfig, LLMStack # type: ignore

        config = AppConfig.from_env()
        stack = LLMStack(config)

        host, port = stack._parse_api_base("http://localhost:8001/v1")
        self.assertEqual(host, "localhost")
        self.assertEqual(port, 8001)

        host, port = stack._parse_api_base("http://127.0.0.1:9000")
        self.assertEqual(host, "127.0.0.1")
        self.assertEqual(port, 9000)


class TestLLMStackDockerOperations(unittest.TestCase):
    """Test LLMStack Docker-related operations."""

    def setUp(self):
        """Set up test configuration."""
        self.env_vars = {
            "LITELLM_CONFIG": "/opt/litellm/config.yaml",
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
            "VLLM_AUTO_PORT_START": "8001",
            "VLLM_AUTO_PORT_END": "8010",
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
        from harinezumigel_llm_stack import AppConfig, LLMStack # type: ignore

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
        from harinezumigel_llm_stack import AppConfig, LLMStack # type: ignore

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
    @patch("harinezumigel_llm_stack.port_in_use")
    def test_find_free_port(self, mock_port_in_use: Any, mock_load: Any) -> None:
        """Test _find_free_port."""
        from harinezumigel_llm_stack import AppConfig, LLMStack # type: ignore

        config = AppConfig.from_env()
        stack = LLMStack(config)

        # Mock first port in use, second port free
        mock_port_in_use.side_effect = [True, False]

        port = stack._find_free_port()
        self.assertEqual(port, 8002)  # Second port in range

    @patch("harinezumigel_llm_stack.load_env_file")
    @patch("harinezumigel_llm_stack.port_in_use")
    def test_find_free_port_all_busy(self, mock_port_in_use: Any, mock_load: Any) -> None:
        """Test _find_free_port when all ports are busy."""
        from harinezumigel_llm_stack import AppConfig, LLMStack # type: ignore

        config = AppConfig.from_env()
        stack = LLMStack(config)

        # Mock all ports in use
        mock_port_in_use.return_value = True

        with self.assertRaises(RuntimeError) as context:
            stack._find_free_port()

        self.assertIn("No free ports available", str(context.exception))

    @patch("harinezumigel_llm_stack.load_env_file")
    @patch("harinezumigel_llm_stack.run_command")
    def test_docker_port_in_use(self, mock_run: Any, mock_load: Any) -> None:
        """Test _docker_port_in_use."""
        from harinezumigel_llm_stack import AppConfig, LLMStack # type: ignore

        config = AppConfig.from_env()
        stack = LLMStack(config)

        # Mock docker ps output with port format
        mock_run.return_value = MagicMock(
            stdout="0.0.0.0:8001->8000/tcp\n0.0.0.0:8002->8000/tcp\n", returncode=0
        )
        self.assertTrue(stack._docker_port_in_use(8001))
        self.assertTrue(stack._docker_port_in_use(8002))
        self.assertFalse(stack._docker_port_in_use(8003))


class TestLLMStackModelResolution(unittest.TestCase):
    """Test model name resolution and lookup."""

    def setUp(self):
        """Set up test configuration."""
        self.env_vars = {
            "LITELLM_CONFIG": "/opt/litellm/config.yaml",
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
        from harinezumigel_llm_stack import AppConfig, LLMStack, ModelDeployment # type: ignore

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
        )
        stack.models["test_model"] = test_model

        resolved = stack._resolve_model_name("test_model")
        self.assertEqual(resolved, "test_model")

    @patch("harinezumigel_llm_stack.load_env_file")
    def test_resolve_model_name_alias(self, mock_load: Any) -> None:
        """Test _resolve_model_name with alias."""
        from harinezumigel_llm_stack import AppConfig, LLMStack, ModelDeployment # type: ignore

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
        )
        stack.models["mistral_7b_instruct"] = test_model1

        # Exact match by alias should work
        resolved = stack._resolve_model_name("mistral")
        self.assertEqual(resolved, "mistral_7b_instruct")

        # Non-matching partial should return None
        resolved = stack._resolve_model_name("mist")
        self.assertIsNone(resolved)


class TestRunCommand(unittest.TestCase):
    """Test run_command utility."""

    def test_run_command_success(self):
        """Test run_command with successful execution."""
        from harinezumigel_llm_stack import run_command # type: ignore

        result = run_command(["echo", "test"], capture=True, check=False)

        self.assertEqual(result.returncode, 0)
        self.assertIn("test", result.stdout)

    def test_run_command_failure(self):
        """Test run_command with failed execution."""
        from harinezumigel_llm_stack import run_command # type: ignore

        result = run_command(["false"], capture=True, check=False)
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
