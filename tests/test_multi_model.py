#!/usr/bin/env python3
"""Multi-model selection, startup orchestration, and follower lifecycle tests."""

import io
import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
from typing import Any, cast

import yaml

from test_harinezumigel_llm_stack import harinezumigel_llm_stack as app
import hls_startup
import hls_terminal


class TestMultiModel(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.config_path = self.root / "config.yaml"
        self.data: dict[str, Any] = {
            "model_groups": {"rag": ["first", "second", "model_a"]},
            "model_list": [
                {
                    "model_name": name,
                    "rebuild": True,
                    "litellm_params": {
                        "model": f"openai/{name}",
                        "api_base": f"http://localhost:{port}/v1",
                        "api_key": "sk-test-secret",
                    },
                    "model_info": {
                        "alias": alias,
                        "model_dir": name,
                        "context_length": 1024,
                        "max_input_tokens": 768,
                        "max_output_tokens": 128,
                        "gpu_memory_utilization": 0.3,
                        "max_num_seqs": 1,
                    },
                }
                for name, alias, port in [
                    ("model_a", "first", 8101),
                    ("model_b", "second", 8102),
                ]
            ],
        }
        for name in ("model_a", "model_b"):
            (self.root / name).mkdir()
        self.write_config()
        self.stack = app.LLMStack(
            replace(
                app.APP,
                litellm_config=str(self.config_path),
                model_root=str(self.root),
                vllm_bind_host="0.0.0.0",
            )
        )
        self.stack.load_models()
        self.output = io.StringIO()
        self.addCleanup(patch.stopall)
        patch.object(app, "check_litellm_environment", return_value="1.106.2").start()
        patch.object(self.stack, "_docker_container_state", return_value=None).start()
        patch.object(
            self.stack, "_find_running_vllm_containers", return_value=[]
        ).start()
        patch.object(self.stack, "_docker_port_in_use", return_value=False).start()
        patch.object(app, "port_in_use", return_value=False).start()
        patch.object(self.stack, "_build_vllm_command", return_value=[]).start()
        self.vram_check = patch.object(
            self.stack.preflight_checks, "check_gpu_memory"
        ).start()
        self.start = patch.object(self.stack, "start_vllm").start()
        self.start_litellm = patch.object(self.stack, "start_litellm").start()
        self.stop_litellm = patch.object(self.stack, "stop_litellm").start()
        patch.object(self.stack, "_find_litellm_pids", return_value=[]).start()
        self.monitor = patch.object(hls_startup.StartupMonitor, "wait").start()
        self.started_at = patch.object(
            hls_startup,
            "container_started_at",
            return_value="2026-10-26T15:06:24.359526098Z",
        ).start()
        patch.object(app, "container_started_at", self.started_at).start()

    def write_config(self):
        self.config_path.write_text(yaml.safe_dump(self.data), encoding="utf-8")

    def args(self, *args: str) -> argparse.Namespace:
        with patch.object(sys, "argv", ["llm", *args]):
            return app.parse_args()

    def run_command(self, *args: str) -> None:
        with redirect_stdout(self.output):
            self.stack.run(self.args(*args))

    def test_multiple_positionals_and_intermixed_options(self):
        args = self.args("first", "--start", "second")
        self.assertEqual(args.models, ["first", "second"])
        self.assertEqual(args.model, "first")

    def test_list_shows_only_aliases_and_groups(self):
        self.run_command("--list")
        text = self.output.getvalue()
        self.assertIn("Aliases\n-------", text)
        self.assertIn("first", text)
        self.assertIn("model_a", text)
        self.assertIn("Model groups\n------------", text)
        self.assertIn("first, second, model_a", text)
        self.assertNotIn("context_length", text)
        self.assertNotIn("api_base", text)
        self.start.assert_not_called()

    def test_empty_aliases_and_groups_are_explicit(self):
        self.stack.models = {}
        self.stack.model_groups = {}
        with redirect_stdout(self.output):
            self.stack.list_aliases_and_groups()
        self.assertEqual(self.output.getvalue().count("(none configured)"), 2)

    def test_group_and_positionals_are_exclusive(self):
        with self.assertRaises(SystemExit) as error:
            self.args("first", "--group", "rag", "--start")
        self.assertEqual(error.exception.code, 2)

    def test_group_starts_in_order_and_deduplicates(self):
        self.run_command("--group", "rag", "--start")
        self.assertEqual(
            [call.args[0].name for call in self.start.call_args_list],
            ["model_a", "model_b"],
        )
        self.assertEqual(self.monitor.call_count, 2)
        self.assertEqual(
            self.monitor.call_args_list[0].args[0].health_url,
            "http://127.0.0.1:8101/health",
        )
        self.assertIn("ready", self.output.getvalue())

    def test_group_can_start_litellm_in_order(self):
        self.data["model_groups"]["stack"] = ["litellm", "first", "second"]
        self.write_config()
        events: list[tuple[str, str]] = []

        def _start_litellm(**_: Any) -> None:
            events.append(("start", "litellm"))

        def _start(model: Any, _: Any) -> None:
            events.append(("start", model.name))

        def _ready(backend: hls_startup.BackendHandle, **_: Any) -> None:
            events.append(("ready", backend.container))

        self.start_litellm.side_effect = _start_litellm
        self.start.side_effect = _start
        self.monitor.side_effect = _ready

        self.run_command("--group", "stack", "--start")

        self.assertEqual(
            events,
            [
                ("start", "litellm"),
                ("start", "model_a"),
                ("ready", "vllm-model_a-8101"),
                ("start", "model_b"),
                ("ready", "vllm-model_b-8102"),
            ],
        )

    def test_recreate_applies_to_vllm_and_restarts_litellm(self):
        self.data["model_groups"]["stack"] = ["litellm", "first"]
        self.write_config()
        events: list[str] = []

        def _stop(**_: Any) -> None:
            events.append("stop")

        def _start(**_: Any) -> None:
            events.append("start")

        self.stop_litellm.side_effect = _stop
        self.start_litellm.side_effect = _start
        with patch.object(self.stack, "_find_litellm_pids", side_effect=[["123"], []]):
            self.run_command("--group", "stack", "--start", "--recreate")

        self.assertEqual(events, ["stop", "start"])
        self.assertTrue(self.start.call_args.args[1].recreate)
        self.stop_litellm.assert_called_once_with(dry_run=False)
        self.start_litellm.assert_called_once_with(dry_run=False)

    def test_litellm_recreate_dry_run_reports_stop_and_start_without_actions(self):
        self.data["model_groups"]["proxy"] = ["litellm"]
        self.write_config()

        self.run_command("--group", "proxy", "--start", "--recreate", "--dry-run")

        self.stop_litellm.assert_called_once_with(dry_run=True)
        self.start_litellm.assert_not_called()
        self.assertIn("LiteLLM would be started after the stop", self.output.getvalue())

    def test_litellm_only_group_starts_without_readiness_monitor(self):
        self.data["model_groups"]["proxy"] = ["litellm"]
        self.write_config()

        self.run_command("--group", "proxy", "--start")

        self.start_litellm.assert_called_once_with(dry_run=False)
        self.start.assert_not_called()
        self.monitor.assert_not_called()
        self.assertIn("started", self.output.getvalue())
        self.assertIn("no readiness check", self.output.getvalue())

    def test_group_start_reuses_matching_litellm(self):
        self.data["model_groups"]["proxy"] = ["litellm"]
        self.write_config()
        with patch.object(self.stack, "_find_litellm_pids", return_value=["123"]):
            self.run_command("--group", "proxy", "--start")

        self.start_litellm.assert_not_called()
        self.stop_litellm.assert_not_called()
        self.assertIn("Reusing running LiteLLM", self.output.getvalue())
        self.assertIn("configuration is not reloaded", self.output.getvalue())

    def test_group_dry_run_reuses_matching_litellm_without_changes(self):
        with patch.object(self.stack, "_find_litellm_pids", return_value=["123"]):
            with redirect_stdout(self.output):
                self.stack._start_litellm_for_batch(dry_run=True, recreate=False)
        self.start_litellm.assert_not_called()
        self.stop_litellm.assert_not_called()
        self.assertIn("Reusing running LiteLLM", self.output.getvalue())

    def test_group_start_does_not_reuse_self_pid(self):
        with patch.object(
            self.stack, "_find_litellm_pids", return_value=[str(os.getpid())]
        ):
            self.stack._start_litellm_for_batch(dry_run=False, recreate=False)
        self.start_litellm.assert_called_once_with(dry_run=False)

    def test_group_start_fails_on_unrelated_port_use(self):
        self.data["model_groups"]["proxy"] = ["litellm"]
        self.write_config()
        with patch.object(app, "port_in_use", return_value=True):
            with patch.object(self.stack, "show_litellm_logs"):
                with self.assertRaises(SystemExit) as error:
                    self.run_command("--group", "proxy", "--start")
        self.assertEqual(error.exception.code, 1)
        self.assertIn("no matching LiteLLM process", self.output.getvalue())
        self.start_litellm.assert_not_called()
        self.stop_litellm.assert_not_called()

    def test_litellm_restart_timeout_does_not_start_another_process(self):
        with patch.object(self.stack, "_find_litellm_pids", return_value=["123"]):
            with patch.object(app.time, "monotonic", side_effect=[0, 11]):
                with self.assertRaisesRegex(
                    hls_startup.StartupError, "did not stop within 10 seconds"
                ):
                    self.stack._start_litellm_for_batch(dry_run=False, recreate=True)
        self.stop_litellm.assert_called_once_with(dry_run=False)
        self.start_litellm.assert_not_called()

    def test_starts_wait_sequentially(self):
        events: list[tuple[str, str]] = []

        def _start(model: Any, _: Any) -> None:
            events.append(("start", model.name))

        def _wait(backend: hls_startup.BackendHandle, **_: Any) -> None:
            events.append(("wait", backend.container))

        self.start.side_effect = _start
        self.monitor.side_effect = _wait
        self.run_command("first", "second", "--start")
        self.assertEqual(
            [kind for kind, _ in events], ["start", "wait", "start", "wait"]
        )

    def test_all_stop_includes_litellm_before_backends(self):
        events: list[str] = []

        def _stop_litellm(**_: Any) -> None:
            events.append("litellm")

        def _stop_backends(**_: Any) -> None:
            events.append("vllm")

        self.stop_litellm.side_effect = _stop_litellm
        with patch.object(self.stack, "stop_all_vllm") as stop_backends:
            stop_backends.side_effect = _stop_backends
            self.run_command("all", "--stop")
        self.assertEqual(events, ["litellm", "vllm"])
        self.stop_litellm.assert_called_once_with(dry_run=False)
        stop_backends.assert_called_once_with(dry_run=False)

    def test_all_stop_dry_run_previews_both_services(self):
        with patch.object(self.stack, "stop_all_vllm") as stop_backends:
            self.run_command("all", "--stop", "--dry-run")
        self.stop_litellm.assert_called_once_with(dry_run=True)
        stop_backends.assert_called_once_with(dry_run=True)

    def test_single_model_stop_does_not_stop_litellm(self):
        with patch.object(self.stack, "stop_vllm") as stop_backend:
            self.run_command("first", "--stop")
        stop_backend.assert_called_once_with("model_a", dry_run=False)
        self.stop_litellm.assert_not_called()

    def test_ready_banner_is_separated_and_bordered_without_color(self):
        with patch.dict("os.environ", {"NO_COLOR": "1"}):
            self.run_command("--group", "rag", "--start")
        for label, port in (("first", 8101), ("second", 8102)):
            message = f"READY: {label} - port {port}"
            border = "=" * (len(message) + 4)
            self.assertIn(
                f"\n{border}\n| {message} |\n{border}\n\n", self.output.getvalue()
            )

    def test_unknown_last_target_has_no_side_effects(self):
        with self.assertRaises(SystemExit):
            self.run_command("first", "missing", "--start")
        self.start.assert_not_called()

    def test_missing_last_directory_has_no_side_effects(self):
        (self.root / "model_b").rmdir()
        with self.assertRaises(SystemExit):
            self.run_command("--group", "rag", "--start")
        self.start.assert_not_called()

    def test_recreate_permissions_validated_before_first_start(self):
        self.data["model_list"][1]["rebuild"] = False
        self.write_config()
        with patch.object(self.stack, "_docker_stop_container") as stop:
            with self.assertRaises(SystemExit):
                self.run_command("--group", "rag", "--start", "--recreate")
        stop.assert_not_called()
        self.start.assert_not_called()

    def test_recreate_stop_failure_prevents_memory_check_and_launch(self):
        with (
            patch.object(
                self.stack,
                "_find_running_vllm_containers",
                return_value=["vllm-model_a-8101"],
            ),
            patch.object(
                self.stack, "_docker_stop_container", side_effect=SystemExit(1)
            ),
        ):
            with self.assertRaises(SystemExit):
                self.run_command("--group", "rag", "--start", "--recreate")
        self.vram_check.assert_not_called()
        self.start.assert_not_called()

    def test_recreation_dry_run_individual_check_uses_capacity_and_checks_driver(self):
        model = self.stack.models["model_a"]
        options = app.StartOptions(
            port=8101,
            runtime=self.stack._resolve_runtime_settings(model),
            dry_run=True,
            reuse_existing=True,
            recreate=True,
            memory_capacity_only=True,
        )
        with (
            patch.object(self.stack.preflight_checks, "check_nvidia_driver") as driver,
            patch.object(app, "run_command") as run,
            redirect_stdout(self.output),
        ):
            app.LLMStack.start_vllm(self.stack, model, options)
        self.vram_check.assert_called_once_with(
            ["model_a"], 0.3, dry_run=True, capacity_only=True
        )
        driver.assert_called_once_with(dry_run=True)
        run.assert_not_called()

    def test_group_recreate_stops_all_selected_containers_before_memory_and_launch(
        self,
    ):
        events: list[tuple[str, Any]] = []

        def _containers(name: str) -> list[str]:
            return [f"vllm-{name}-8101", f"vllm-{name}-8999"]

        def _stop(name: str, **_: Any) -> None:
            events.append(("stop", name))

        def _memory(*args: Any, **_: Any) -> None:
            events.append(("memory", args))

        def _start(model: Any, _: Any) -> None:
            events.append(("start", model.name))

        with (
            patch.object(
                self.stack, "_find_running_vllm_containers", side_effect=_containers
            ),
            patch.object(
                self.stack,
                "_docker_stop_container",
                side_effect=_stop,
            ),
        ):
            self.vram_check.side_effect = _memory
            self.start.side_effect = _start
            self.run_command("--group", "rag", "--start", "--recreate")
        self.assertEqual(
            [event[0] for event in events],
            ["stop", "stop", "stop", "stop", "memory", "start", "start"],
        )
        self.vram_check.assert_called_once_with(
            ["model_a", "model_b"], 0.6, dry_run=False
        )
        self.stop_litellm.assert_not_called()

    def test_recreate_failed_combined_memory_leaves_selected_backends_stopped(self):
        with (
            patch.object(
                self.stack,
                "_find_running_vllm_containers",
                return_value=["vllm-model_a-8101"],
            ),
            patch.object(self.stack, "_docker_stop_container") as stop,
        ):
            self.vram_check.side_effect = SystemExit(1)
            with self.assertRaises(SystemExit):
                self.run_command("--group", "rag", "--start", "--recreate")
        self.assertTrue(stop.called)
        self.start.assert_not_called()

    def test_recreate_dry_run_uses_capacity_only_without_stopping_containers(self):
        with (
            patch.object(
                self.stack,
                "_find_running_vllm_containers",
                side_effect=[["vllm-model_a-8101"], []],
            ),
            patch.object(app, "run_command") as run,
        ):
            self.run_command("--group", "rag", "--start", "--recreate", "--dry-run")
        run.assert_not_called()
        self.vram_check.assert_called_once_with(
            ["model_a", "model_b"], 0.6, dry_run=True, capacity_only=True
        )
        self.assertTrue(
            all(call.args[1].memory_capacity_only for call in self.start.call_args_list)
        )
        self.assertIn("Container was not stopped", self.output.getvalue())

    def test_conflicting_ports_have_no_side_effects(self):
        self.data["model_list"][1]["litellm_params"][
            "api_base"
        ] = "http://localhost:8101/v1"
        self.write_config()
        with self.assertRaises(SystemExit):
            self.run_command("--group", "rag", "--start")
        self.start.assert_not_called()

    def test_port_conflict_has_no_side_effects(self):
        with patch.object(app, "port_in_use", side_effect=[False, True]):
            with self.assertRaises(SystemExit):
                self.run_command("--group", "rag", "--start")
        self.start.assert_not_called()

    def test_removed_deployment_overrides_are_not_cli_options(self):
        for option in [
            ["--port", "8105"],
            ["--auto-port"],
            ["--context-length", "4096"],
            ["--max-input-tokens", "3072"],
            ["--max-output-tokens", "1024"],
            ["--gpu-memory-utilization", "0.5"],
            ["--max-num-seqs", "1"],
            ["--dtype", "float16"],
        ]:
            with self.subTest(option=option), self.assertRaises(SystemExit) as error:
                self.args("first", "--start", *option)
            self.assertEqual(error.exception.code, 2)

    def test_operational_batch_flags_remain_supported(self):
        for options in [
            ["--start", "--stream-log"],
            ["--show-log", "--follow"],
            ["--start", "--stop"],
            ["--stop", "--recreate"],
            ["--ps", "--quiet"],
        ]:
            with self.subTest(options=options), self.assertRaises(SystemExit):
                self.run_command("first", "second", *options)
        self.start.assert_not_called()

    def test_reserved_targets_are_rejected(self):
        with self.assertRaises(SystemExit):
            self.run_command("first", "all", "--stop")
        self.start.assert_not_called()

    def test_group_ps_reports_litellm_status(self):
        self.data["model_groups"]["proxy"] = ["litellm"]
        self.write_config()
        with patch.object(self.stack, "_find_litellm_pids", return_value=["123"]):
            self.run_command("--group", "proxy", "--ps")
        self.assertIn("LiteLLM: running", self.output.getvalue())
        self.assertIn("PID(s): 123", self.output.getvalue())

    def test_dry_run_does_not_monitor(self):
        self.run_command("--group", "rag", "--start", "--dry-run")
        self.assertTrue(all(call.args[1].dry_run for call in self.start.call_args_list))
        self.monitor.assert_not_called()

    def test_multi_model_start_checks_combined_vram_before_launch(self):
        self.run_command("--group", "rag", "--start", "--dry-run")
        self.vram_check.assert_called_once_with(
            ["model_a", "model_b"],
            0.6,
            dry_run=True,
        )
        self.assertEqual(self.start.call_count, 2)

    def test_multi_model_vram_failure_prevents_launch(self):
        self.vram_check.side_effect = SystemExit(1)
        with self.assertRaises(SystemExit):
            self.run_command("--group", "rag", "--start", "--dry-run")
        self.start.assert_not_called()

    def test_failure_stops_remaining_and_reports_logs_even_when_quiet(self):
        self.monitor.side_effect = hls_startup.StartupError("readiness timeout")
        with patch.object(
            self.stack, "_docker_container_state", return_value="running"
        ):
            with patch.object(self.stack, "show_container_logs") as logs:
                with self.assertRaises(SystemExit) as error:
                    self.run_command("--group", "rag", "--start", "--quiet")
        self.assertEqual(error.exception.code, 1)
        self.assertEqual(self.start.call_count, 1)
        self.assertEqual(logs.call_args.args[0], "vllm-model_a-8101")
        self.assertEqual(logs.call_args.kwargs["tail"], "100")
        self.assertIn("sk-test-secret", logs.call_args.kwargs["redact_values"])
        self.assertIn("not attempted", self.output.getvalue())
        self.assertIn("--stream-log", self.output.getvalue())

    def test_failure_logs_exclude_previous_container_runs(self):
        self.monitor.side_effect = hls_startup.StartupError("readiness timeout")
        with (
            patch.object(self.stack, "_docker_container_state", return_value="running"),
            patch.object(
                app,
                "container_started_at",
                return_value="2026-10-26T15:06:24.359526098Z",
            ),
            patch.object(self.stack, "show_container_logs") as logs,
        ):
            with self.assertRaises(SystemExit):
                self.run_command("--group", "rag", "--start")
        self.assertEqual(
            logs.call_args.kwargs["since"], "2026-10-26T15:06:24.359526098Z"
        )

    def test_launch_failure_reports_summary(self):
        self.start.side_effect = SystemExit(1)
        with self.assertRaises(SystemExit) as error:
            self.run_command("--group", "rag", "--start")
        self.assertEqual(error.exception.code, 1)
        self.assertEqual(self.start.call_count, 1)
        self.assertIn("failed", self.output.getvalue())

    def test_interrupt_leaves_containers_and_returns_130(self):
        self.monitor.side_effect = KeyboardInterrupt
        with self.assertRaises(SystemExit) as error:
            self.run_command("--group", "rag", "--start")
        self.assertEqual(error.exception.code, 130)
        self.assertEqual(self.start.call_count, 1)
        self.assertIn("left running", self.output.getvalue())

    def test_reused_backend_filters_logs_since_group_start(self):
        with patch.object(
            self.stack, "_docker_container_state", return_value="running"
        ):
            self.run_command("--group", "rag", "--start")
        self.started_at.assert_not_called()
        for call in self.monitor.call_args_list:
            self.assertRegex(
                call.kwargs["since"],
                r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+00:00$",
            )

    def test_restarted_backend_filters_logs_since_docker_start(self):
        with patch.object(self.stack, "_docker_container_state", return_value="exited"):
            self.run_command("--group", "rag", "--start")
        self.assertEqual(self.started_at.call_count, 2)
        self.assertTrue(
            all(
                call.kwargs["since"] == "2026-10-26T15:06:24.359526098Z"
                for call in self.monitor.call_args_list
            )
        )

    def test_single_monitored_restart_filters_historical_logs(self):
        with patch.object(self.stack, "_docker_container_state", return_value="exited"):
            self.run_command("first", "--start", "--startup-logs")
        self.assertIsNotNone(self.monitor.call_args.kwargs["since"])

    def test_startup_formatter_redacts_secret(self):
        with patch.object(hls_startup, "StartupMonitor") as monitor:
            self.run_command("--group", "rag", "--start")
        formatter = monitor.call_args.args[0]
        self.assertEqual(formatter("key=sk-test-secret"), "key=***REDACTED***")

    def test_nonstart_actions_operate_on_every_selected_model(self):
        for flag, method in [
            ("--stop", "handle_stop"),
            ("--ps", "handle_ps"),
            ("--show-log", "handle_logs"),
            ("--show-log-path", "handle_logs"),
        ]:
            with self.subTest(flag=flag), patch.object(self.stack, method) as handler:
                self.run_command("--group", "rag", flag)
                self.assertEqual(
                    [c.args[0].model for c in handler.call_args_list],
                    ["model_a", "model_b"],
                )

    def test_single_model_default_does_not_wait(self):
        self.run_command("first", "--start")
        self.monitor.assert_not_called()
        self.start.assert_called_once()
        self.assertIn("add --stream-log", self.output.getvalue())
        self.assertIn(
            f"{app.program_name()} first --stream-log", self.output.getvalue()
        )
        self.assertNotIn("group/batch startup", self.output.getvalue())

    def test_log_view_defaults_to_full_history(self):
        self.assertEqual(self.args("first", "--show-log").tail, "all")
        self.assertEqual(self.args("first", "--show-log", "--tail", "50").tail, "50")

    def test_stream_filters_at_latest_container_start(self):
        with (
            patch.object(app, "run_command") as inspect,
            patch.object(app, "_stream_formatted_logs", return_value=0) as stream,
            redirect_stdout(self.output),
        ):
            inspect.return_value = subprocess.CompletedProcess(
                [], 0, "2026-10-26T15:00:00.123456789Z\n", ""
            )
            self.stack.show_container_logs("vllm-test-8001", tail="all", follow=True)
        self.assertEqual(
            stream.call_args.args[0],
            [
                "docker",
                "logs",
                "--tail",
                "all",
                "--since",
                "2026-10-26T15:00:00.123456789Z",
                "-f",
                "vllm-test-8001",
            ],
        )

    def test_container_lookup_excludes_longer_model_names(self):
        names = [
            "vllm-qwen3.5-8009",
            "vllm-qwen3.5-8010",
            "vllm-qwen3.5-122b-a10b-fp8-8011",
            "vllm-qwen3.5-",
        ]
        with patch.object(
            self.stack, "_docker_list_container_names", return_value=names
        ):
            for find in (
                app.LLMStack._find_all_vllm_containers,
                app.LLMStack._find_running_vllm_containers,
            ):
                self.assertEqual(find(self.stack, "qwen3.5"), names[:2])

    def test_stop_waits_for_litellm_process_exit(self):
        with (
            patch.object(
                self.stack, "_find_litellm_pids", side_effect=[["123"], ["123"], []]
            ),
            patch.object(app, "run_command") as command,
            patch.object(app.time, "sleep") as sleep,
            redirect_stdout(self.output),
        ):
            command.return_value.returncode = 0
            app.LLMStack.stop_litellm(self.stack)
        command.assert_called_once_with(["kill", "-TERM", "123"], capture=True)
        sleep.assert_called_once_with(0.1)
        self.assertIn("Verified stopped: LiteLLM", self.output.getvalue())

    def test_stop_verification_timeout_is_an_error(self):
        with (
            patch.object(app.time, "monotonic", side_effect=[0, 11]),
            redirect_stdout(self.output),
            self.assertRaises(SystemExit) as error,
        ):
            self.stack._wait_for_stop("test backend", lambda: ["vllm-stuck"])
        self.assertEqual(error.exception.code, 1)
        self.assertIn("still running: vllm-stuck", self.output.getvalue())

    def test_stop_model_verifies_all_matching_containers(self):
        with (
            patch.object(
                self.stack,
                "_find_running_vllm_containers",
                side_effect=[["vllm-model_a-8101"], []],
            ) as find,
            patch.object(app, "run_command") as command,
            redirect_stdout(self.output),
        ):
            command.return_value.returncode = 0
            self.stack.stop_vllm("model_a")
        self.assertEqual(find.call_count, 2)
        self.assertIn("Verified stopped: vLLM model model_a", self.output.getvalue())

    def test_stop_dry_run_does_not_wait_or_signal(self):
        with (
            patch.object(self.stack, "_find_litellm_pids", return_value=["123"]),
            patch.object(
                self.stack,
                "_find_running_vllm_containers",
                return_value=["vllm-model_a-8101"],
            ),
            patch.object(self.stack, "_wait_for_stop") as wait,
            patch.object(app, "run_command") as command,
            redirect_stdout(self.output),
        ):
            app.LLMStack.stop_litellm(self.stack, dry_run=True)
            self.stack.stop_vllm("model_a", dry_run=True)
        wait.assert_not_called()
        command.assert_not_called()

    def test_stop_all_verifies_each_configured_model(self):
        with (
            patch.object(
                self.stack,
                "_find_running_vllm_containers",
                side_effect=[["vllm-model_a-8101"], ["vllm-model_b-8102"], [], []],
            ) as find,
            patch.object(app, "run_command") as command,
            redirect_stdout(self.output),
        ):
            command.return_value.returncode = 0
            self.stack.stop_all_vllm()
        self.assertEqual(find.call_count, 4)
        self.assertIn("Verified stopped: all configured", self.output.getvalue())

    def test_container_stop_verification_query_failure_is_not_success(self):
        with (
            patch.object(app, "run_command") as command,
            redirect_stdout(self.output),
            self.assertRaises(SystemExit),
        ):
            command.side_effect = [
                subprocess.CompletedProcess([], 0, "vllm-test", ""),
                subprocess.CompletedProcess([], 1, "", "Docker unavailable"),
            ]
            self.stack._docker_stop_container("vllm-test")
        self.assertIn("Failed to query Docker", self.output.getvalue())
        self.assertNotIn("Verified stopped", self.output.getvalue())

    def test_group_start_log_hint_uses_separate_model_commands(self):
        self.run_command("--group", "rag", "--start")
        text = self.output.getvalue()
        self.assertIn("after group/batch startup", text)
        self.assertIn(f"{app.program_name()} first --stream-log", text)
        self.assertIn(f"{app.program_name()} second --stream-log", text)
        self.assertNotIn(f"{app.program_name()} litellm --stream-log", text)
        self.assertNotIn("add --stream-log", text)

    def test_start_log_hints_include_selected_litellm_in_order(self):
        self.data["model_groups"]["stack"] = ["first", "lite-llm", "second"]
        self.data["model_groups"]["proxy"] = ["litellm"]
        self.write_config()
        selections = [
            (("--group", "stack"), ["first", "second", "litellm"]),
            (("first", "llm-lite", "second"), ["first", "second", "litellm"]),
            (("--group", "proxy"), ["litellm"]),
        ]
        for selection, targets in selections:
            for pids in ([], ["123"]):
                with (
                    self.subTest(selection=selection, pids=pids),
                    patch.object(self.stack, "_find_litellm_pids", return_value=pids),
                ):
                    self.output.seek(0)
                    self.output.truncate()
                    self.run_command(*selection, "--start")
                    commands = [
                        line.strip()
                        for line in self.output.getvalue().splitlines()
                        if line.endswith(" --stream-log")
                    ]
                    self.assertEqual(
                        commands,
                        [
                            f"{app.program_name()} {target} --stream-log"
                            for target in targets
                        ],
                    )

    def test_startup_display_groups_modes_without_reordering_launches(self):
        self.data["model_groups"]["stack"] = ["first", "litellm", "second", "first"]
        self.data["model_list"][0]["model_info"]["mode"] = "embedding"
        self.data["model_list"][1]["model_info"]["mode"] = "rerank"
        self.write_config()
        self.run_command("--group", "stack", "--start")
        self.assertEqual(
            [call.args[0].name for call in self.start.call_args_list],
            ["model_a", "model_b"],
        )
        text = self.output.getvalue()
        summary = text.split("=== Startup summary ===", 1)[1]
        for category, target in (
            ("Embeddings", "first"),
            ("Proxy", "LiteLLM"),
            ("Reranking", "second"),
        ):
            self.assertIn(f"\n{category}\n{target}", summary)
        self.assertIn(f"\nEmbeddings\n  {app.program_name()} first --stream-log", text)
        self.assertIn(f"\nProxy\n  {app.program_name()} litellm --stream-log", text)
        self.assertIn(f"\nReranking\n  {app.program_name()} second --stream-log", text)

    def test_startup_display_mode_defaults_and_unknown_modes(self):
        model = self.stack.models["model_a"]
        self.assertEqual(model.display_mode(), "inference")
        for mode, expected in (
            ("inference", "inference"),
            ("chat", "inference"),
            (" embedding ", "embedding"),
            ("audio", "audio"),
        ):
            with self.subTest(mode=mode):
                model.model_info["mode"] = mode
                self.assertEqual(model.display_mode(), expected)

    def test_startup_category_headings_use_configured_accent(self):
        for color, code in (("cyan", "36"), ("blue", "34")):
            with (
                self.subTest(color=color),
                patch.dict(
                    os.environ,
                    {"TERM": "xterm", "HLS_ACCENT_COLOR": color},
                    clear=True,
                ),
                patch.object(self.output, "isatty", return_value=True),
            ):
                self.output.seek(0)
                self.output.truncate()
                self.run_command("--group", "rag", "--start")
                self.assertEqual(
                    self.output.getvalue().count(f"\033[{code}mInference\033[0m"),
                    2,
                )

    def test_invalid_display_mode_fails_before_launch(self):
        for mode in (None, "", 42, []):
            with self.subTest(mode=mode):
                self.data["model_list"][0]["model_info"]["mode"] = mode
                self.write_config()
                with self.assertRaises(SystemExit):
                    self.run_command("--group", "rag", "--start")
                self.start.assert_not_called()
                self.start_litellm.assert_not_called()
                self.assertIn("model_info.mode", self.output.getvalue())

    def test_start_log_hints_are_suppressed_for_dry_runs(self):
        self.data["model_groups"]["stack"] = ["first", "litellm"]
        self.data["model_groups"]["proxy"] = ["litellm"]
        self.write_config()
        selections: list[tuple[str, ...]] = [
            ("first",),
            ("--group", "rag"),
            ("--group", "stack"),
            ("--group", "proxy"),
            ("first", "litellm"),
        ]
        for selection in selections:
            with self.subTest(selection=selection):
                self.output.seek(0)
                self.output.truncate()
                self.run_command(*selection, "--start", "--dry-run")
                self.assertNotIn("--stream-log", self.output.getvalue())

    def test_single_model_streaming_has_no_additional_log_hint(self):
        with patch.object(
            self.stack, "_docker_container_state", return_value="running"
        ):
            with patch.object(self.stack, "show_container_logs") as logs:
                self.run_command("first", "--start", "--stream-log")
        logs.assert_called_once()
        self.assertNotIn("add --stream-log", self.output.getvalue())

    def test_single_model_startup_logs_opt_in(self):
        self.run_command("first", "--start", "--startup-logs")
        self.monitor.assert_called_once()

    def test_single_model_explicit_timeout_enables_monitoring(self):
        self.run_command("first", "--start", "--startup-timeout", "12")
        self.assertEqual(self.monitor.call_args.kwargs["timeout"], 12)

    def test_unknown_group_has_no_side_effects(self):
        for name in ("missing", ""):
            with self.subTest(name=name), self.assertRaises(SystemExit):
                self.run_command("--group", name, "--start")
        self.start.assert_not_called()

    def test_recreate_without_start_explains_requirement(self):
        with self.assertRaises(SystemExit) as error:
            self.run_command("--group", "rag", "--recreate")
        self.assertEqual(error.exception.code, 1)
        self.assertIn("--recreate requires --start", self.output.getvalue())
        self.assertIn("--start --recreate", self.output.getvalue())
        self.start.assert_not_called()
        self.monitor.assert_not_called()

    def test_group_without_action_explains_recreation(self):
        with self.assertRaises(SystemExit) as error:
            self.run_command("--group", "rag")
        self.assertEqual(error.exception.code, 1)
        self.assertIn("Select exactly one batch action", self.output.getvalue())
        self.assertIn("--start --recreate", self.output.getvalue())
        self.start.assert_not_called()

    def test_litellm_model_summary_is_aligned_and_command_is_labeled(self):
        first = self.stack.models["model_a"]
        second = self.stack.models["model_b"]
        first.alias = "a"
        second.alias = "longer-alias"
        first.model_info["override_generation_config"] = {"temperature": 0.0}
        second.model_info["override_generation_config"] = {"temperature": 0.7}
        self.stack.models = {"short": first, "longer-model-name": second}
        with redirect_stdout(self.output):
            app.LLMStack.start_litellm(self.stack, dry_run=True)
        text = self.output.getvalue()
        rows = [line for line in text.splitlines() if line.startswith("  - ")]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].index("(alias:"), rows[1].index("(alias:"))
        self.assertEqual(rows[0].index("[temperature"), rows[1].index("[temperature"))
        self.assertIn("LiteLLM command: set -a", text)

    def test_docker_launch_is_synchronous_and_reports_failure(self):
        model = self.stack.models["model_a"]
        options = app.StartOptions(
            port=8101,
            runtime=self.stack._resolve_runtime_settings(model),
            dry_run=False,
            reuse_existing=True,
            recreate=False,
        )
        with patch.object(
            self.stack.preflight_checks, "check_gpu_memory"
        ) as vram_check:
            with patch.object(self.stack.preflight_checks, "check_nvidia_driver"):
                with patch.object(app, "run_command") as run:
                    run.return_value = subprocess.CompletedProcess(
                        ["docker"],
                        125,
                        "",
                        "Docker launch rejected",
                    )
                    with (
                        redirect_stdout(self.output),
                        self.assertRaises(SystemExit) as error,
                    ):
                        # Call the real method, bypassing the fixture's launch mock.
                        app.LLMStack.start_vllm(self.stack, model, options)
        self.assertEqual(error.exception.code, 125)
        self.assertIn("Docker launch rejected", self.output.getvalue())
        vram_check.assert_called_once_with(["model_a"], 0.3, dry_run=False)

    def test_dry_run_performs_vram_preflight(self):
        model = self.stack.models["model_a"]
        options = app.StartOptions(
            port=8101,
            runtime=self.stack._resolve_runtime_settings(model),
            dry_run=True,
            reuse_existing=True,
            recreate=False,
        )
        with patch.object(
            self.stack.preflight_checks, "check_gpu_memory"
        ) as vram_check:
            with (
                patch.object(self.stack.preflight_checks, "check_nvidia_driver"),
                redirect_stdout(self.output),
            ):
                app.LLMStack.start_vllm(self.stack, model, options)
        vram_check.assert_called_once_with(["model_a"], 0.3, dry_run=True)

    def test_new_and_stopped_backends_share_preflight_before_launch(self):
        model = self.stack.models["model_a"]
        for state in (None, "exited"):
            for dry_run in (False, True):
                with self.subTest(state=state, dry_run=dry_run):
                    options = app.StartOptions(
                        port=8101,
                        runtime=self.stack._resolve_runtime_settings(model),
                        dry_run=dry_run,
                        reuse_existing=True,
                        recreate=False,
                    )
                    events: list[str] = []

                    def _memory(*_: Any, **__: Any) -> None:
                        events.append("memory")

                    def _driver(**_: Any) -> None:
                        events.append("driver")

                    with (
                        patch.object(
                            self.stack, "_docker_container_state", return_value=state
                        ),
                        patch.object(
                            self.stack.preflight_checks,
                            "check_gpu_memory",
                            side_effect=_memory,
                        ),
                        patch.object(
                            self.stack.preflight_checks,
                            "check_nvidia_driver",
                            side_effect=_driver,
                        ),
                        patch.object(
                            self.stack, "_docker_start_container"
                        ) as start_existing,
                        patch.object(app, "run_command") as launch,
                        redirect_stdout(self.output),
                    ):
                        launch.return_value = subprocess.CompletedProcess(
                            ["docker"], 0, "container-id", ""
                        )
                        app.LLMStack.start_vllm(self.stack, model, options)
                        self.assertEqual(events, ["memory", "driver"])
                        if state is None:
                            start_existing.assert_not_called()
                            self.assertEqual(launch.call_count, int(not dry_run))
                        else:
                            start_existing.assert_called_once_with(
                                "vllm-model_a-8101", dry_run=dry_run
                            )
                            launch.assert_not_called()

    def test_failed_preflight_blocks_new_and_stopped_backends(self):
        model = self.stack.models["model_a"]
        options = app.StartOptions(
            port=8101,
            runtime=self.stack._resolve_runtime_settings(model),
            dry_run=False,
            reuse_existing=True,
            recreate=False,
        )
        for state in (None, "exited"):
            with (
                self.subTest(state=state),
                patch.object(self.stack, "_docker_container_state", return_value=state),
                patch.object(
                    self.stack.preflight_checks,
                    "check_gpu_memory",
                    side_effect=SystemExit(1),
                ),
                patch.object(self.stack, "_docker_start_container") as start_existing,
                patch.object(app, "run_command") as launch,
                redirect_stdout(self.output),
            ):
                with self.assertRaises(SystemExit):
                    app.LLMStack.start_vllm(self.stack, model, options)
                start_existing.assert_not_called()
                launch.assert_not_called()

    def test_timeout_validation(self):
        for value in ("0", "-1", "nan", "inf"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                self.args("--group", "rag", "--start", "--startup-timeout", value)

    def test_generation_overrides_are_forwarded_as_json(self):
        model = self.stack.models["model_a"]
        runtime = self.stack._resolve_runtime_settings(model)
        for generation_config in ("auto", "vllm"):
            with self.subTest(generation_config=generation_config):
                model.model_info["generation_config"] = generation_config
                overrides = {"temperature": 0.0, "top_p": 0.9, "presence_penalty": 0.0}
                model.model_info["override_generation_config"] = overrides
                command = app.LLMStack._build_vllm_command(
                    self.stack, model, self.root / model.name, 8101, runtime
                )
                self.assertEqual(
                    json.loads(
                        command[command.index("--override-generation-config") + 1]
                    ),
                    overrides,
                )
                self.assertEqual(
                    command[command.index("--generation-config") + 1], generation_config
                )

    def test_generation_overrides_are_optional_and_validated(self):
        model = self.stack.models["model_a"]
        runtime = self.stack._resolve_runtime_settings(model)
        command = app.LLMStack._build_vllm_command(
            self.stack, model, self.root / model.name, 8101, runtime
        )
        self.assertNotIn("--override-generation-config", command)
        self.assertEqual(command[command.index("--generation-config") + 1], "vllm")
        invalid_values: list[Any] = [
            "not a mapping",
            [],
            {"temperature": float("nan")},
            {"temperature": float("inf")},
            {"temperature": {1, 2}},
        ]
        for invalid in invalid_values:
            with self.subTest(invalid=invalid):
                model.model_info["override_generation_config"] = invalid
                with redirect_stdout(self.output), self.assertRaises(SystemExit):
                    app.LLMStack._build_vllm_command(
                        self.stack, model, self.root / model.name, 8101, runtime
                    )

    def test_explicit_generation_config_path_is_preserved(self):
        model = self.stack.models["model_a"]
        model.model_info["generation_config"] = "/model/generation-defaults"
        runtime = self.stack._resolve_runtime_settings(model)
        command = app.LLMStack._build_vllm_command(
            self.stack, model, self.root / model.name, 8101, runtime
        )
        self.assertEqual(
            command[command.index("--generation-config") + 1],
            "/model/generation-defaults",
        )

    def test_invalid_generation_settings_rejected_while_loading(self):
        invalid_settings: list[dict[str, Any]] = [
            {"generation_config": None},
            {"generation_config": ""},
            {"generation_config": 123},
            {"override_generation_config": []},
            {"override_generation_config": {"temperature": float("nan")}},
        ]
        for settings in invalid_settings:
            with self.subTest(settings=settings):
                self.data["model_list"][0]["model_info"].update(settings)
                self.write_config()
                output = io.StringIO()
                with redirect_stdout(output), self.assertRaises(SystemExit) as error:
                    self.stack.load_models()
                self.assertEqual(error.exception.code, 1)
                self.assertIn("model_a", output.getvalue())
                self.data["model_list"][0]["model_info"].pop("generation_config", None)
                self.data["model_list"][0]["model_info"].pop(
                    "override_generation_config", None
                )

    def test_duplicate_defaults_rejected_before_start_and_dry_run(self):
        for proxy_value in (0.0, 0.7, None):
            for dry_run in (False, True):
                with self.subTest(proxy_value=proxy_value, dry_run=dry_run):
                    entry = self.data["model_list"][0]
                    entry["litellm_params"]["temperature"] = proxy_value
                    entry["model_info"]["override_generation_config"] = {
                        "temperature": 0.0
                    }
                    self.write_config()
                    args = ["first", "--start"]
                    if dry_run:
                        args.append("--dry-run")
                    with self.assertRaises(SystemExit) as error:
                        self.run_command(*args)
                    self.assertEqual(error.exception.code, 1)
                    self.assertIn("model_a", self.output.getvalue())
                    self.assertIn("temperature", self.output.getvalue())
                    self.assertIn("litellm_params", self.output.getvalue())
                    self.assertIn(
                        "model_info.override_generation_config", self.output.getvalue()
                    )
                    self.start.assert_not_called()
                    self.start_litellm.assert_not_called()
                    self.vram_check.assert_not_called()

    def test_proxy_only_defaults_remain_supported(self):
        self.data["model_list"][0]["litellm_params"]["temperature"] = 0.2
        self.write_config()
        self.stack.load_models()
        model = self.stack.models["model_a"]
        self.assertEqual(model.litellm_params["temperature"], 0.2)
        self.assertEqual(model.generation_settings(), ("vllm", None))

    def test_runtime_settings_come_only_from_yaml(self):
        self.data["model_list"][0]["model_info"].update(
            context_length=2048,
            max_input_tokens=1536,
            max_output_tokens=512,
            gpu_memory_utilization=0.42,
            max_num_seqs=3,
            dtype="float16",
        )
        self.write_config()
        self.stack.load_models()
        model = self.stack.models["model_a"]
        runtime = self.stack._resolve_runtime_settings(model)
        self.assertEqual(
            runtime,
            {
                "context_length": 2048,
                "max_input_tokens": 1536,
                "max_output_tokens": 512,
                "gpu_memory_utilization": 0.42,
                "max_num_seqs": 3,
                "dtype": "float16",
            },
        )

    def test_generation_defaults_shown_in_listing_and_startup_summary(self):
        model = self.stack.models["model_a"]
        model.model_info["override_generation_config"] = {"temperature": 0.0}
        model.litellm_params["top_p"] = 0.9
        runtime = self.stack._resolve_runtime_settings(model)
        for generation_config in ("vllm", "auto"):
            if generation_config == "auto":
                model.model_info["generation_config"] = "auto"
            for render, source_label in (
                (self.stack.list_models, "generation_config:        "),
                (
                    lambda: self.stack._print_vllm_start_summary(
                        model, self.root / model.name, 8101, runtime, []
                    ),
                    "Generation config:         ",
                ),
            ):
                output = io.StringIO()
                with redirect_stdout(output):
                    render()
                text = output.getvalue()
                self.assertIn(f"{source_label}{generation_config}", text)
                self.assertIn("temperature=0.0(backend default)", text)
                self.assertIn("top_p=0.9(proxy default)", text)
                self.assertNotIn("(model default)", text)
                self.assertNotIn("(model-spec)", text)

    def test_empty_generation_overrides_are_forwarded(self):
        model = self.stack.models["model_a"]
        model.model_info["override_generation_config"] = {}
        runtime = self.stack._resolve_runtime_settings(model)
        command = app.LLMStack._build_vllm_command(
            self.stack, model, self.root / model.name, 8101, runtime
        )
        self.assertEqual(
            command[command.index("--override-generation-config") + 1], "{}"
        )

    def test_example_and_readme_use_consistent_backend_defaults(self):
        root = Path(__file__).resolve().parents[1]
        example = cast(
            dict[str, Any],
            yaml.safe_load(
                (root / "config/config.yaml.example").read_text(encoding="utf-8")
            ),
        )
        expected: dict[str, dict[str, float | int]] = {
            "bge-m3": {},
            "bge-reranker-v2-m3": {},
            "Snowflake/snowflake-arctic-embed-l-v2.0": {},
            "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1": {},
            "llama_guard3_8b": {"temperature": 0.0, "top_p": 1.0},
            "llama_3_1_8b": {"temperature": 0.7, "top_p": 0.9},
            "mistral_7b": {"temperature": 0.7, "top_p": 0.9},
            "qwen2_5_32b_awq": {"temperature": 0.3, "top_p": 0.9, "top_k": 20},
            "Qwen3-Coder-Next-AWQ": {"temperature": 0.2, "top_p": 0.9, "top_k": 20},
            "qwen3.6-27b": {"temperature": 0.3, "top_p": 0.9, "top_k": 20},
            "deepseek-coder-6.7b-instruct": {
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": 20,
            },
            "qwen2_5_1_5b_instruct": {"temperature": 0.0, "top_p": 1.0},
            "qwen2_5_14b_instruct": {"temperature": 0.0, "top_p": 1.0},
            "qwen3.5": {
                "temperature": 0.2,
                "top_p": 0.95,
                "top_k": -1,
                "presence_penalty": 0.0,
                "repetition_penalty": 1.0,
            },
            "qwen3.8-27B": {"temperature": 0.2, "top_p": 0.95},
            "qwen3-coder-30b-a3b-instruct": {"temperature": 0.2, "top_p": 0.95},
        }
        sampling_keys = {
            "temperature",
            "top_p",
            "top_k",
            "presence_penalty",
            "frequency_penalty",
            "repetition_penalty",
        }
        for entry in example["model_list"]:
            with self.subTest(model=entry["model_name"]):
                info = entry["model_info"]
                if info.get("mode") in ("embedding", "rerank"):
                    self.assertEqual(expected[entry["model_name"]], {})
                    self.assertNotIn("override_generation_config", info)
                    self.assertFalse(
                        sampling_keys.intersection(entry["litellm_params"])
                    )
                    continue
                self.assertEqual(info["generation_config"], "vllm")
                self.assertEqual(
                    info["override_generation_config"], expected[entry["model_name"]]
                )
                self.assertFalse(sampling_keys.intersection(entry["litellm_params"]))
                if "top_k" in info["override_generation_config"]:
                    self.assertIsInstance(
                        info["override_generation_config"]["top_k"], int
                    )

        example_stack = app.LLMStack(
            replace(app.APP, litellm_config=str(root / "config/config.yaml.example"))
        )
        example_stack.load_models()
        self.assertEqual(len(example_stack.models), len(expected))

        readme = (root / "README.md").read_text(encoding="utf-8")
        for block in readme.split("```yaml\n")[1:]:
            sample = cast(dict[str, Any], yaml.safe_load(block.split("```", 1)[0]))
            if "model_list" in sample:
                mistral = sample["model_list"][0]
                self.assertEqual(mistral["model_name"], "mistral_7b")
                self.assertEqual(mistral["model_info"]["generation_config"], "vllm")
                self.assertEqual(
                    mistral["model_info"]["override_generation_config"],
                    expected["mistral_7b"],
                )
                self.assertFalse(sampling_keys.intersection(mistral["litellm_params"]))
                break
        else:
            self.fail("README Mistral YAML sample not found")

    def test_example_rag_groups_route_pooling_models(self):
        root = Path(__file__).resolve().parents[1]
        profiles = {
            "bge-m3": ("embedding", "bge-m3", 8014, 8192),
            "bge-reranker-v2-m3": ("rerank", "bge-reranker-v2-m3", 8015, 8192),
            "Snowflake/snowflake-arctic-embed-l-v2.0": (
                "embedding",
                "snowflake-arctic-embed-l-v2.0",
                8016,
                8192,
            ),
            "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1": (
                "rerank",
                "mmarco-mMiniLMv2-L12-H384-v1",
                8017,
                512,
            ),
        }
        with patch.dict(os.environ, clear=True):
            app.load_env_file(str(root / "config/env.example"))
            stack = app.LLMStack(
                replace(
                    app.APP, litellm_config=str(root / "config/config.yaml.example")
                )
            )
            stack.load_models()
            common = ["mistral_7b", "llama_guard3_8b", "qwen2_5_14b_instruct"]
            for group, pair in (
                ("rag", ["bge-m3", "bge-reranker-v2-m3"]),
                (
                    "rag_old",
                    [
                        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
                        "Snowflake/snowflake-arctic-embed-l-v2.0",
                    ],
                ),
            ):
                with self.subTest(group=group):
                    selected = stack._select_models(self.args("--group", group))
                    self.assertEqual(selected, common + pair + ["litellm"])
                    self.assertAlmostEqual(
                        sum(
                            stack.models[name].gpu_memory_utilization
                            for name in selected[:-1]
                        ),
                        0.87,
                    )

            for name, (mode, directory, port, context) in profiles.items():
                with self.subTest(model=name):
                    model = stack.models[name]
                    self.assertEqual(model.model_info["mode"], mode)
                    self.assertEqual(model.model_info["model_dir"], directory)
                    self.assertEqual(model.api_base, f"http://127.0.0.1:{port}/v1")
                    self.assertEqual(model.backend_model, f"hosted_vllm/{name}")
                    self.assertEqual(model.context_length, context)
                    runtime = stack._resolve_runtime_settings(model)
                    command = stack._build_vllm_command(
                        model, Path("/opt/models") / directory, port, runtime
                    )
                    self.assertEqual(
                        command[command.index("--served-model-name") + 1], name
                    )
                    self.assertEqual(
                        command[command.index("--model") + 1], f"/models/{directory}"
                    )
                    self.assertEqual(
                        command[command.index("-p") + 1], f"127.0.0.1:{port}:8000"
                    )
                    self.assertTrue(model.allow_rebuild)

    def test_invalid_groups_rejected(self):
        invalid_groups: list[Any] = [
            None,
            [],
            {"bad": []},
            {"bad": ["unknown"]},
            {"bad": ["all"]},
            {"bad": [123]},
            {"bad": "first"},
        ]
        for groups in invalid_groups:
            with self.subTest(groups=groups):
                self.data["model_groups"] = groups
                self.write_config()
                with self.assertRaises(SystemExit):
                    self.stack.load_models()

    def test_litellm_is_valid_group_member(self):
        self.data["model_groups"] = {"proxy": ["litellm"]}
        self.write_config()

        self.stack.load_models()

        self.assertEqual(self.stack.model_groups, {"proxy": ["litellm"]})


class TestStartupMonitor(unittest.TestCase):
    def setUp(self):
        self.backend = hls_startup.BackendHandle(
            "test-container", "http://localhost/health"
        )
        self.output = io.StringIO()
        self.addCleanup(patch.stopall)
        self.state = patch.object(
            hls_startup, "container_state", return_value=("running", 0)
        ).start()
        self.health = patch.object(
            hls_startup, "health_ready", return_value=False
        ).start()
        self.monitor = hls_startup.StartupMonitor(lambda line: line)

    def test_container_started_at_uses_docker_inspect_timestamp(self):
        timestamp = "2026-10-26T15:06:24.359526098Z"
        with patch.object(hls_startup.subprocess, "run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, timestamp + "\n", "")
            self.assertEqual(
                hls_startup.container_started_at("test-container"), timestamp
            )
        run.assert_called_once_with(
            [
                "docker",
                "inspect",
                "--format",
                "{{.State.StartedAt}}",
                "test-container",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_log_follower_filters_since_current_container_start(self):
        timestamp = "2026-10-26T15:06:24.359526098Z"
        popen = subprocess.Popen
        command: list[str] = []

        def _spawn(
            args: list[str], *, stdout: int, stderr: int
        ) -> subprocess.Popen[bytes]:
            command.extend(args)
            return popen(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                stdout=stdout,
                stderr=stderr,
            )

        self.health.side_effect = [False, True]
        with patch.object(hls_startup.subprocess, "Popen", side_effect=_spawn):
            self.monitor.wait(
                self.backend,
                timeout=3,
                show_logs=True,
                since=timestamp,
            )

        self.assertEqual(
            command,
            [
                "docker",
                "logs",
                "--timestamps",
                "--tail",
                "200",
                "--follow",
                "--since",
                timestamp,
                "test-container",
            ],
        )

    def test_replayed_history_is_filtered_from_displayed_output(self):
        popen = subprocess.Popen

        def _spawn(*_: object, stdout: int, stderr: int) -> subprocess.Popen[bytes]:
            return popen(
                [
                    sys.executable,
                    "-u",
                    "-c",
                    "import sys; "
                    "print('2026-10-26T14:00:00.000000000Z PREVIOUS_BOOT'); "
                    "print('2026-10-26T15:06:24.359526098Z CURRENT_BOOT'); "
                    "print('2026-10-26T15:06:25.000000000Z CURRENT_STDERR', "
                    "file=sys.stderr); sys.stdin.read()",
                ],
                stdin=subprocess.PIPE,
                stdout=stdout,
                stderr=stderr,
            )

        self.health.side_effect = [False, False, True]
        with (
            patch.object(hls_startup.subprocess, "Popen", side_effect=_spawn),
            redirect_stdout(self.output),
        ):
            self.monitor.wait(
                self.backend,
                timeout=3,
                show_logs=True,
                since="2026-10-26T15:06:24.359526098Z",
            )
        text = self.output.getvalue()
        self.assertNotIn("PREVIOUS_BOOT", text)
        self.assertIn("CURRENT_BOOT", text)
        self.assertIn("CURRENT_STDERR", text)
        self.assertNotIn("2026-10-26T", text)

    def test_already_healthy_never_opens_log_follower(self):
        self.health.return_value = True
        with patch.object(hls_startup.subprocess, "Popen") as popen:
            self.monitor.wait(self.backend, timeout=1, show_logs=True)
        popen.assert_not_called()

    def test_exited_container_reports_exit_code(self):
        self.state.return_value = ("exited", 137)
        with self.assertRaisesRegex(hls_startup.StartupError, "exit code 137"):
            self.monitor.wait(self.backend, timeout=1, show_logs=False)

    def test_timeout_reports_container_still_running(self):
        with self.assertRaisesRegex(hls_startup.StartupError, "still running"):
            self.monitor.wait(self.backend, timeout=0.01, show_logs=False)

    def test_silent_follower_cannot_block_health_and_is_reaped(self):
        popen = subprocess.Popen
        child: subprocess.Popen[bytes] | None = None

        def _spawn(*_: object, stdout: int, stderr: int) -> subprocess.Popen[bytes]:
            nonlocal child
            child = popen(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                stdout=stdout,
                stderr=stderr,
            )
            return child

        self.health.side_effect = [False, True]
        with patch.object(hls_startup.subprocess, "Popen", side_effect=_spawn):
            self.monitor.wait(self.backend, timeout=3, show_logs=True)
        assert child is not None
        assert child.stdout is not None
        self.assertIsNotNone(child.poll())
        self.assertTrue(child.stdout.closed)

    def test_partial_utf8_line_is_flushed_and_follower_reaped(self):
        popen = subprocess.Popen
        child: subprocess.Popen[bytes] | None = None

        def _spawn(*_: object, stdout: int, stderr: int) -> subprocess.Popen[bytes]:
            nonlocal child
            child = popen(
                [
                    sys.executable,
                    "-u",
                    "-c",
                    "import os,time; os.write(1,b'loading \\xe2'); "
                    "time.sleep(0.05); os.write(1,b'\\x82\\xac'); "
                    "time.sleep(30)",
                ],
                stdout=stdout,
                stderr=stderr,
            )
            return child

        self.health.side_effect = [False, False, True]
        with patch.object(hls_startup.subprocess, "Popen", side_effect=_spawn):
            with redirect_stdout(self.output):
                self.monitor.wait(self.backend, timeout=3, show_logs=True)
        self.assertIn("loading \u20ac", self.output.getvalue())
        assert child is not None
        self.assertIsNotNone(child.poll())

    def test_interrupt_reaps_follower(self):
        popen = subprocess.Popen
        child: subprocess.Popen[bytes] | None = None

        def _spawn(*_: object, stdout: int, stderr: int) -> subprocess.Popen[bytes]:
            nonlocal child
            child = popen(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                stdout=stdout,
                stderr=stderr,
            )
            return child

        self.health.side_effect = [False, KeyboardInterrupt]
        with patch.object(hls_startup.subprocess, "Popen", side_effect=_spawn):
            with self.assertRaises(KeyboardInterrupt):
                self.monitor.wait(self.backend, timeout=3, show_logs=True)
        assert child is not None
        assert child.stdout is not None
        self.assertIsNotNone(child.poll())
        self.assertTrue(child.stdout.closed)

    def test_timeout_reaps_follower(self):
        popen = subprocess.Popen
        child: subprocess.Popen[bytes] | None = None

        def _spawn(*_: object, stdout: int, stderr: int) -> subprocess.Popen[bytes]:
            nonlocal child
            child = popen(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                stdout=stdout,
                stderr=stderr,
            )
            return child

        with patch.object(hls_startup.subprocess, "Popen", side_effect=_spawn):
            with self.assertRaisesRegex(hls_startup.StartupError, "Readiness timeout"):
                self.monitor.wait(self.backend, timeout=0.05, show_logs=True)
        assert child is not None
        assert child.stdout is not None
        self.assertIsNotNone(child.poll())
        self.assertTrue(child.stdout.closed)

    def test_follower_failure_is_reported(self):
        popen = subprocess.Popen

        def _spawn(*_: object, stdout: int, stderr: int) -> subprocess.Popen[bytes]:
            return popen(
                [
                    sys.executable,
                    "-c",
                    "print('cannot read logs'); raise SystemExit(7)",
                ],
                stdout=stdout,
                stderr=stderr,
            )

        with patch.object(hls_startup.subprocess, "Popen", side_effect=_spawn):
            with redirect_stdout(self.output):
                with self.assertRaisesRegex(hls_startup.StartupError, "exit code 7"):
                    self.monitor.wait(self.backend, timeout=3, show_logs=True)
        self.assertIn("cannot read logs", self.output.getvalue())

    def test_inspect_errors_are_not_treated_as_missing_containers(self):
        with patch.object(hls_startup.subprocess, "run") as run:
            run.return_value = subprocess.CompletedProcess(
                [], 1, "", "daemon unavailable"
            )
            with self.assertRaisesRegex(hls_startup.StartupError, "daemon unavailable"):
                self.original_container_state("test")

    original_container_state = staticmethod(hls_startup.container_state)


class TestHealthProbe(unittest.TestCase):
    def test_real_health_endpoint_ignores_proxy_and_handles_unready(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200 if self.path == "/health" else 503)
                self.end_headers()

            def log_message(self, format: str, *args: Any) -> None:
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                base = f"http://127.0.0.1:{server.server_port}"
                with patch.dict(
                    "os.environ",
                    {
                        "http_proxy": "http://127.0.0.1:1",
                        "no_proxy": "",
                    },
                ):
                    self.assertTrue(hls_startup.health_ready(base + "/health"))
                    self.assertFalse(hls_startup.health_ready(base + "/unready"))
            finally:
                server.shutdown()
                thread.join()


class TestTerminalOutput(unittest.TestCase):
    def test_lifecycle_colors_preserve_plain_text(self):
        import re

        lines = [
            ("=== [1/3] Starting mistral ===", "\033[1m"),
            ("READY: mistral - port 8001", "\033[32m"),
            ("ERROR: startup failed", "\033[1;31m"),
            ("WARNING: memory pressure", "\033[1;33m"),
            ("GPU VRAM preflight passed for model 'a'.", "\033[32m"),
            ("Shared system memory preflight (dry-run) passed.", "\033[32m"),
            ("ERROR: Insufficient Shared system memory for model 'a'.", "\033[1;31m"),
            ("NVIDIA pre-run compatibility check (dry-run) passed.", "\033[32m"),
            ("NVIDIA pre-run compatibility check failed.", "\033[1;31m"),
            ("mistral                  ready            port=8001", "\033[32m"),
            (
                "litellm                  started          no readiness check",
                "\033[32m",
            ),
            (
                "litellm                  failed           no readiness check",
                "\033[1;31m",
            ),
            ("Model directory: /models/mistral", "\033[1m"),
        ]
        with patch.dict("os.environ", {"TERM": "tmux-256color"}, clear=True):
            with patch.object(sys.stdout, "isatty", return_value=True):
                for text, color in lines:
                    with self.subTest(text=text):
                        styled = hls_terminal.style_status_line(text)
                        self.assertIn(color, styled)
                        self.assertEqual(re.sub(r"\033\[[0-9;]*m", "", styled), text)

    def test_nonterminal_and_no_color_are_plain(self):
        with patch.dict("os.environ", {"TERM": "tmux-256color", "NO_COLOR": ""}):
            with patch.object(sys.stdout, "isatty", return_value=True):
                self.assertEqual(
                    hls_terminal.style_status_line("READY: model"), "READY: model"
                )

    def test_outcome_colors_are_configurable_without_changing_accents(self):
        with (
            patch.dict(
                os.environ,
                {
                    "TERM": "xterm",
                    "HLS_SUCCESS_COLOR": "bright-green",
                    "HLS_FAILURE_COLOR": "magenta",
                },
                clear=True,
            ),
            patch.object(sys.stdout, "isatty", return_value=True),
        ):
            self.assertIn("\033[92m", hls_terminal.style_status_line("READY: model"))
            self.assertIn("\033[1;35m", hls_terminal.style_status_line("ERROR: failed"))
            self.assertIn(
                "\033[1;92m", hls_terminal.terminal_style("ready", code="1;32")
            )
            self.assertIn(
                "\033[36m", hls_terminal.terminal_style("command", accent=True)
            )
            self.assertIn(
                "\033[1;33m", hls_terminal.style_status_line("WARNING: caution")
            )

    def test_invalid_outcome_color_is_reported(self):
        with (
            patch.dict(
                os.environ,
                {"TERM": "xterm", "HLS_SUCCESS_COLOR": "invalid"},
                clear=True,
            ),
            patch.object(sys.stdout, "isatty", return_value=True),
            self.assertRaisesRegex(ValueError, "HLS_SUCCESS_COLOR must be one of"),
        ):
            hls_terminal.style_status_line("READY: model")

    def test_warning_accent_and_heading_colors_are_configurable(self):
        with (
            patch.dict(
                os.environ,
                {
                    "TERM": "xterm",
                    "HLS_WARNING_COLOR": "bright-yellow",
                    "HLS_ACCENT_COLOR": "blue",
                    "HLS_HEADING_COLOR": "white",
                },
                clear=True,
            ),
            patch.object(sys.stdout, "isatty", return_value=True),
        ):
            self.assertIn(
                "\033[1;93m", hls_terminal.style_status_line("WARNING: caution")
            )
            self.assertIn(
                "\033[34m", hls_terminal.terminal_style("command", accent=True)
            )
            self.assertIn(
                "\033[34m", hls_terminal.style_status_line("Follow logs: llm model")
            )
            self.assertIn(
                "\033[1;37m", hls_terminal.style_status_line("=== Startup summary ===")
            )
            self.assertIn(
                "\033[1;37m", hls_terminal.style_status_line("Model directory: /models")
            )

    def test_print_to_explicit_file_is_plain(self):
        stream = io.StringIO()
        with patch.dict("os.environ", {"TERM": "tmux-256color"}, clear=True):
            with patch.object(sys.stdout, "isatty", return_value=True):
                hls_terminal.terminal_print("ERROR: failed", file=stream)
        self.assertEqual(stream.getvalue(), "ERROR: failed\n")


if __name__ == "__main__":
    unittest.main()
