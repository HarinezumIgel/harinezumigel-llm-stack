"""Memory preflight regression tests without Docker or GPU side effects."""

import io
import subprocess
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

from test_harinezumigel_llm_stack import harinezumigel_llm_stack as app


class TestSharedMemoryPreflight(unittest.TestCase):
    def setUp(self):
        self.runner = Mock(
            side_effect=[
                subprocess.CompletedProcess([], 0, "0, [N/A], [N/A]\n", ""),
                subprocess.CompletedProcess([], 0, "NVIDIA GB10\n", ""),
            ]
        )
        self.checks = app.PreflightChecks("unused-image", self.runner)
        self.output = io.StringIO()

    def test_shared_memory_threshold_and_model_labels(self):
        for dry_run in (False, True):
            for names in (["one"], ["one", "two"]):
                for available, success in ((524288, True), (524287, False)):
                    with self.subTest(
                        dry_run=dry_run, names=names, available=available
                    ):
                        self.setUp()
                        with (
                            patch(
                                "hls_preflight.Path.read_text",
                                return_value=(
                                    "MemTotal: 1048576 kB\n"
                                    f"MemAvailable: {available} kB\n"
                                ),
                            ),
                            redirect_stdout(self.output),
                        ):
                            if success:
                                self.checks.check_gpu_memory(
                                    names, 0.5, dry_run=dry_run
                                )
                            else:
                                with self.assertRaises(SystemExit) as error:
                                    self.checks.check_gpu_memory(
                                        names, 0.5, dry_run=dry_run
                                    )
                                self.assertEqual(error.exception.code, 1)
                        text = self.output.getvalue()
                        self.assertIn("Shared system memory", text)
                        self.assertIn("required: 512 MiB (0.50 GiB)", text)
                        self.assertIn("50% of 1024 MiB (1.00 GiB)", text)
                        self.assertIn("Available:", text)
                        self.assertIn(
                            "model 'one'" if len(names) == 1 else "2 selected models",
                            text,
                        )
                        self.assertEqual("(dry-run)" in text, dry_run)

    def test_invalid_shared_memory_fails_explicitly(self):
        for data in (
            "",
            "MemTotal: 1048576 kB\n",
            "MemTotal: 0 kB\nMemAvailable: 0 kB\n",
            "MemTotal: 100 kB\nMemAvailable: 101 kB\n",
            "MemTotal: 100 kB\nMemAvailable: -1 kB\n",
            "MemTotal: 100 MB\nMemAvailable: 50 kB\n",
            "MemTotal: invalid kB\nMemAvailable: 50 kB\n",
        ):
            with self.subTest(data=data):
                self.setUp()
                with (
                    patch("hls_preflight.Path.read_text", return_value=data),
                    redirect_stdout(self.output),
                    self.assertRaises(SystemExit),
                ):
                    self.checks.check_gpu_memory(["one"], 0.5)
                self.assertIn("ERROR:", self.output.getvalue())

    def test_unreadable_meminfo_fails_explicitly(self):
        with (
            patch("hls_preflight.Path.read_text", side_effect=OSError("unreadable")),
            redirect_stdout(self.output),
            self.assertRaises(SystemExit),
        ):
            self.checks.check_gpu_memory(["one"], 0.5)
        self.assertIn("unreadable", self.output.getvalue())

    def test_capacity_only_preview_does_not_claim_available_memory_passed(self):
        with (
            patch(
                "hls_preflight.Path.read_text",
                return_value="MemTotal: 1048576 kB\nMemAvailable: 1 kB\n",
            ),
            redirect_stdout(self.output),
        ):
            self.checks.check_gpu_memory(
                ["one", "two"], 0.8, dry_run=True, capacity_only=True
            )
        text = self.output.getvalue()
        self.assertIn("capacity check (dry-run)", text)
        self.assertIn("cannot be verified", text)
        self.assertNotIn("passed", text)

    def test_capacity_only_rejects_budget_exceeding_total(self):
        with (
            patch(
                "hls_preflight.Path.read_text",
                return_value="MemTotal: 1048576 kB\nMemAvailable: 1 kB\n",
            ),
            redirect_stdout(self.output),
            self.assertRaises(SystemExit),
        ):
            self.checks.check_gpu_memory(
                ["one", "two"], 1.01, dry_run=True, capacity_only=True
            )

    def test_other_gpu_or_failed_identity_query_cannot_use_fallback(self):
        for name, code in (("NVIDIA RTX 4090", 0), ("", 1)):
            with self.subTest(name=name, code=code):
                self.setUp()
                self.runner.side_effect = [
                    subprocess.CompletedProcess([], 0, "0, N/A, N/A\n", ""),
                    subprocess.CompletedProcess([], code, name, "query detail"),
                ]
                with (
                    patch("hls_preflight.Path.read_text") as read_memory,
                    redirect_stdout(self.output),
                    self.assertRaises(SystemExit),
                ):
                    self.checks.check_gpu_memory(["one"], 0.5)
                read_memory.assert_not_called()
                self.assertIn("requires NVIDIA GB10", self.output.getvalue())
