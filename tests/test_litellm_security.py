#!/usr/bin/env python3
"""Verify LiteLLM incident protections without executing installed packages."""

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from test_harinezumigel_llm_stack import harinezumigel_llm_stack as app
from hls_litellm_security import (
    SAFE_VERSION,
    SAFE_WHEELS,
    LiteLLMSecurityError,
    check_litellm_environment,
    safe_requirement,
)

SAFE_REQUIREMENT = safe_requirement()
SAFE_WHEEL_SHA256 = SAFE_REQUIREMENT.split("#sha256=", 1)[1]
PROJECT_DIR = Path(__file__).resolve().parent.parent
CHECKER = PROJECT_DIR / "src" / "hls_litellm_security.py"
INSTALLER = (PROJECT_DIR / "install.sh").read_text(encoding="utf-8")
DEPENDENCY_FUNCTION = (
    "pyyaml_requirements() {"
    + INSTALLER.split("pyyaml_requirements() {", 1)[1].split("\n}", 1)[0]
    + "\n}\n"
    + "show_platform_hint() {"
    + INSTALLER.split("show_platform_hint() {", 1)[1].split("\n}", 1)[0]
    + "\n}\n"
    + "install_litellm_proxy_deps() ("
    + INSTALLER.split("install_litellm_proxy_deps() (", 1)[1].split("\n)", 1)[0]
    + "\n)\n"
)


class TestLiteLLMSecurity(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.venv = Path(directory.name) / "venv"
        self.site = self.venv / "lib" / "python3.12" / "site-packages"
        self.site.mkdir(parents=True)
        (self.venv / "pyvenv.cfg").write_text(
            "include-system-site-packages = false\n", encoding="utf-8"
        )
        self.bin = self.venv / "bin"
        self.bin.mkdir()
        self.pip_calls = self.venv / "pip-calls"
        self.sentinel = self.venv / "executed"
        self.stack = app.LLMStack(
            replace(
                app.APP,
                litellm_venv_activate=str(self.bin / "activate"),
                litellm_bin=str(self.bin / "litellm"),
            )
        )

    def metadata(self, version: str) -> Path:
        metadata = self.site / f"litellm-{version}.dist-info" / "METADATA"
        metadata.parent.mkdir()
        metadata.write_text(f"Name: litellm\nVersion: {version}\n", encoding="utf-8")
        return metadata

    def run_installer(
        self, mode: str = "upgrade", **env: str
    ) -> subprocess.CompletedProcess[str]:
        command = (
            "set -e\n"
            'SCRIPT_DIR="$1"\n'
            + DEPENDENCY_FUNCTION
            + 'install_litellm_proxy_deps "$2" "$3"\n'
        )
        return subprocess.run(
            [
                "bash",
                "-c",
                command,
                "test-install",
                str(PROJECT_DIR),
                str(self.venv),
                mode,
            ],
            env={**os.environ, **env},
            capture_output=True,
            text=True,
        )

    def fake_pip(self) -> None:
        pip = self.bin / "pip"
        pip.write_text(
            '#!/bin/bash\nroot="$(dirname "$0")/.."\n'
            'printf "%s\\n" "$*" >> "$root/pip-calls"\n'
            'if [[ "$*" == *"${TEST_PIP_FAILURE:-never-fail}"* ]]; then exit 1; fi\n'
            'if [[ "$*" == *" download "* ]]; then\n'
            "  while [[ $# -gt 0 ]]; do\n"
            '    if [[ "$1" == "--dest" ]]; then\n'
            '      touch "$2/PyYAML-6.0.3-py3-none-any.whl"\n'
            "      break\n"
            "    fi\n"
            "    shift\n"
            "  done\n"
            "fi\n"
            'if [[ "$*" == *" install "* ]]; then\n'
            f'  version="${{TEST_INSTALLED_VERSION:-{SAFE_VERSION}}}"\n'
            '  info="$root/lib/python3.12/site-packages/litellm-$version.dist-info"\n'
            '  mkdir -p "$info"\n'
            '  printf "Name: litellm\\nVersion: %s\\n" "$version" > "$info/METADATA"\n'
            "fi\n",
            encoding="utf-8",
        )
        pip.chmod(0o755)
        for name in ("python", "litellm"):
            executable = self.bin / name
            executable.write_text(
                '#!/bin/bash\ntouch "$(dirname "$0")/../executed"\nexit 99\n',
                encoding="utf-8",
            )
            executable.chmod(0o755)

    def test_blocks_compromised_versions_and_normalized_variants(self) -> None:
        for version in (
            "1.107.0",
            "1.107.0+local",
            "1.107.00",
            "1.107.0.post1",
            "1.82.7",
            "1.82.8",
        ):
            with self.subTest(version=version):
                metadata = self.metadata(version)
                with self.assertRaisesRegex(
                    LiteLLMSecurityError, "SECURITY WARNING: Compromised LiteLLM"
                ) as error:
                    check_litellm_environment(self.venv, allow_missing=True)
                self.assertIn("rotate all exposed", str(error.exception))
                self.assertIn("upgrade alone does not", str(error.exception))
                metadata.unlink()
                metadata.parent.rmdir()

    def test_allows_unaffected_versions_and_enforces_pin_when_requested(self) -> None:
        for version in ("1.106.1", SAFE_VERSION, "1.106.4"):
            with self.subTest(version=version):
                metadata = self.metadata(version)
                self.assertEqual(check_litellm_environment(self.venv), version)
                if version == SAFE_VERSION:
                    check_litellm_environment(self.venv, require_pinned=True)
                else:
                    with self.assertRaisesRegex(
                        LiteLLMSecurityError, "Expected pinned"
                    ):
                        check_litellm_environment(self.venv, require_pinned=True)
                metadata.unlink()
                metadata.parent.rmdir()

    def test_residual_pth_blocks_even_with_clean_or_missing_metadata(self) -> None:
        marker = self.site / "cook.pth"
        marker.write_text(
            f"import pathlib; pathlib.Path({str(self.sentinel)!r}).touch()\n",
            encoding="utf-8",
        )
        for installed in (False, True):
            with self.subTest(installed=installed):
                if installed:
                    self.metadata(SAFE_VERSION)
                with self.assertRaisesRegex(
                    LiteLLMSecurityError, "compromise indicator"
                ):
                    check_litellm_environment(self.venv, allow_missing=True)
                result = subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        "-S",
                        str(CHECKER),
                        "check",
                        str(self.venv),
                        "--allow-missing",
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 1)
                self.assertIn("compromise indicator", result.stderr)
                self.assertFalse(self.sentinel.exists())

    def test_october_hooks_and_bytecode_are_blocked_after_reinstall(self) -> None:
        self.metadata(SAFE_VERSION)
        for name in (
            "cook.py",
            "cook.pyc",
            "sitecustomize.py",
            "sitecustomize.pyc",
            "__pycache__/cook.cpython-312.pyc",
            "__pycache__/sitecustomize.cpython-312.pyc",
            "litellm_init.pth",
        ):
            with self.subTest(name=name):
                marker = self.site / name
                marker.parent.mkdir(exist_ok=True)
                marker.touch()
                with self.assertRaisesRegex(
                    LiteLLMSecurityError, "compromise indicator"
                ):
                    check_litellm_environment(self.venv)
                self.fake_pip()
                result = self.run_installer()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("compromise indicator", result.stderr)
                self.assertFalse(self.pip_calls.exists())
                with (
                    patch.object(self.stack, "_find_litellm_pids") as find,
                    redirect_stdout(io.StringIO()),
                    self.assertRaises(SystemExit),
                ):
                    self.stack.start_litellm(dry_run=True)
                find.assert_not_called()
                self.assertFalse(self.sentinel.exists())
                marker.unlink()
        package = self.site / "requests"
        package.mkdir()
        (package / "cookies.py").touch()
        self.assertEqual(check_litellm_environment(self.venv), SAFE_VERSION)

    def test_platform_wheels_have_explicit_hashes_and_no_unsafe_fallback(self) -> None:
        for (libc, machine), (path, digest) in SAFE_WHEELS.items():
            with (
                self.subTest(libc=libc, machine=machine),
                patch(
                    "hls_litellm_security.platform.libc_ver", return_value=(libc, "")
                ),
                patch("hls_litellm_security.platform.machine", return_value=machine),
                patch("hls_litellm_security.sys.platform", "linux"),
            ):
                self.assertEqual(
                    safe_requirement(),
                    f"litellm[proxy] @ https://files.pythonhosted.org/packages/{path}"
                    f"#sha256={digest}",
                )
                self.assertEqual(len(digest), 64)
        with (
            patch("hls_litellm_security.platform.machine", return_value="unknown"),
            self.assertRaisesRegex(LiteLLMSecurityError, "No reviewed LiteLLM wheel"),
        ):
            safe_requirement()

    def test_missing_ambiguous_and_invalid_metadata_fail_closed(self) -> None:
        missing = self.venv / "missing"
        self.assertIsNone(check_litellm_environment(missing, allow_missing=True))
        with self.assertRaisesRegex(LiteLLMSecurityError, "environment is missing"):
            check_litellm_environment(missing, allow_missing=True, require_pinned=True)
        with self.assertRaisesRegex(LiteLLMSecurityError, "exactly one"):
            check_litellm_environment(self.venv)
        self.assertIsNone(check_litellm_environment(self.venv, allow_missing=True))
        package = self.site / "litellm"
        package.mkdir()
        with self.assertRaisesRegex(LiteLLMSecurityError, "exactly one"):
            check_litellm_environment(self.venv, allow_missing=True)
        metadata = self.metadata(SAFE_VERSION)
        metadata.write_text("Name: litellm\n", encoding="utf-8")
        with self.assertRaisesRegex(LiteLLMSecurityError, "Cannot verify"):
            check_litellm_environment(self.venv)
        metadata.write_text(
            f"Name: litellm\nVersion: {SAFE_VERSION}\n", encoding="utf-8"
        )
        self.metadata("1.82.6")
        with self.assertRaisesRegex(LiteLLMSecurityError, "exactly one"):
            check_litellm_environment(self.venv)

    def test_system_site_packages_and_unreadable_metadata_are_rejected(self) -> None:
        metadata = self.metadata(SAFE_VERSION)
        (self.venv / "pyvenv.cfg").write_text(
            "include-system-site-packages = true\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(LiteLLMSecurityError, "inherited system packages"):
            check_litellm_environment(self.venv)
        metadata.unlink()
        with self.assertRaisesRegex(LiteLLMSecurityError, "Cannot inspect"):
            check_litellm_environment(self.venv)

    def test_lib64_and_multiple_python_versions_are_checked(self) -> None:
        self.metadata(SAFE_VERSION)
        other_site = self.venv / "lib64" / "python3.11" / "site-packages"
        other_site.mkdir(parents=True)
        (other_site / "cook.pth").touch()
        with self.assertRaisesRegex(LiteLLMSecurityError, "compromise indicator"):
            check_litellm_environment(self.venv)

    def test_start_stream_dry_run_and_reuse_block_before_execution(self) -> None:
        self.metadata("1.107.0")
        for action in ("start", "stream", "dry-run", "reuse", "recreate"):
            with (
                self.subTest(action=action),
                patch.object(app, "start_background_process") as background,
                patch.object(app, "_stream_formatted_logs") as stream,
                patch.object(self.stack, "_find_litellm_pids", return_value=["123"]),
                patch.object(self.stack, "stop_litellm") as stop,
                redirect_stdout(io.StringIO()) as output,
                self.assertRaises(SystemExit) as error,
            ):
                if action in ("reuse", "recreate"):
                    self.stack._start_litellm_for_batch(
                        dry_run=False, recreate=action == "recreate"
                    )
                else:
                    self.stack.start_litellm(
                        dry_run=action == "dry-run", follow_log=action == "stream"
                    )
            self.assertEqual(error.exception.code, 1)
            self.assertIn("Compromised LiteLLM 1.107.0", output.getvalue())
            background.assert_not_called()
            stream.assert_not_called()
            stop.assert_not_called()

    def test_mismatched_executable_is_rejected(self) -> None:
        self.metadata(SAFE_VERSION)
        self.stack.config = replace(self.stack.config, litellm_bin="/other/bin/litellm")
        with redirect_stdout(io.StringIO()) as output, self.assertRaises(SystemExit):
            self.stack._check_litellm_security()
        self.assertIn("unchecked executable", output.getvalue())

    def test_startup_record_is_written_before_background_or_streamed_launch(
        self,
    ) -> None:
        self.metadata(SAFE_VERSION)
        log_path = self.venv.parent / "proxy.log"
        for follow_log in (False, True):
            with (
                self.subTest(follow_log=follow_log),
                patch.object(self.stack, "_find_litellm_pids", return_value=[]),
                patch.object(app, "port_in_use", return_value=False),
                patch.object(
                    self.stack, "_new_litellm_log_path", return_value=log_path
                ),
                patch.object(app, "start_background_process") as background,
                patch.object(app, "_stream_formatted_logs") as stream,
                redirect_stdout(io.StringIO()) as output,
            ):

                def _verify_record(*args: object, **kwargs: object) -> None:
                    self.assertIn(f"LiteLLM version: {SAFE_VERSION}", output.getvalue())
                    record = json.loads(log_path.read_text(encoding="utf-8"))
                    self.assertEqual(record["event"], "litellm_start_attempt")
                    self.assertEqual(record["litellm_version"], SAFE_VERSION)
                    self.assertEqual(record["launcher_version"], app.__version__)
                    timestamp = datetime.fromisoformat(record["timestamp"])
                    self.assertEqual(timestamp.utcoffset(), timedelta(0))

                background.side_effect = _verify_record
                stream.side_effect = _verify_record
                self.stack.start_litellm(follow_log=follow_log)
                (stream if follow_log else background).assert_called_once()
                (background if follow_log else stream).assert_not_called()
                log_path.unlink()

    def test_startup_record_appends_without_overwriting_existing_output(self) -> None:
        self.metadata(SAFE_VERSION)
        log_path = self.venv.parent / "proxy.log"
        log_path.write_text("existing output\n", encoding="utf-8")
        with (
            patch.object(self.stack, "_find_litellm_pids", return_value=[]),
            patch.object(app, "port_in_use", return_value=False),
            patch.object(self.stack, "_new_litellm_log_path", return_value=log_path),
            patch.object(app, "start_background_process"),
            redirect_stdout(io.StringIO()),
        ):
            self.stack.start_litellm()
        lines = log_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0], "existing output")
        self.assertEqual(json.loads(lines[1])["litellm_version"], SAFE_VERSION)

    def test_unwritable_startup_log_aborts_before_execution(self) -> None:
        self.metadata(SAFE_VERSION)
        with (
            patch.object(self.stack, "_find_litellm_pids", return_value=[]),
            patch.object(app, "port_in_use", return_value=False),
            patch.object(self.stack, "_new_litellm_log_path", return_value=self.venv),
            patch.object(app, "start_background_process") as background,
            patch.object(app, "_stream_formatted_logs") as stream,
            redirect_stdout(io.StringIO()) as output,
            self.assertRaises(SystemExit),
        ):
            self.stack.start_litellm()
        self.assertIn("Cannot write LiteLLM startup record", output.getvalue())
        background.assert_not_called()
        stream.assert_not_called()

    def test_safe_dry_run_uses_the_checked_interpreter(self) -> None:
        self.metadata(SAFE_VERSION)
        log_path = self.venv.parent / "dry-run.log"
        with (
            patch.object(self.stack, "_find_litellm_pids", return_value=[]),
            patch.object(app, "port_in_use", return_value=False),
            patch.object(app, "start_background_process") as background,
            patch.object(self.stack, "_new_litellm_log_path", return_value=log_path),
            redirect_stdout(io.StringIO()) as output,
        ):
            self.stack.start_litellm(dry_run=True)
        self.assertIn(f"LiteLLM version: {SAFE_VERSION}", output.getvalue())
        self.assertIn(f"{self.bin}/python -I -c", output.getvalue())
        background.assert_not_called()
        self.assertFalse(log_path.exists())

    def test_batch_dry_run_prints_version_on_recreate_and_reuse(self) -> None:
        self.metadata(SAFE_VERSION)
        log_path = self.venv.parent / "batch-dry-run.log"
        for recreate in (False, True):
            with (
                self.subTest(recreate=recreate),
                patch.object(self.stack, "_find_litellm_pids", return_value=["123"]),
                patch.object(self.stack, "start_litellm") as start,
                patch.object(app, "run_command") as run,
                patch.object(
                    self.stack, "_new_litellm_log_path", return_value=log_path
                ) as log,
                redirect_stdout(io.StringIO()) as output,
            ):
                self.stack._start_litellm_for_batch(dry_run=True, recreate=recreate)
                self.assertIn(f"LiteLLM version: {SAFE_VERSION}", output.getvalue())
                if not recreate:
                    self.assertIn(
                        "running process version not verified", output.getvalue()
                    )
                start.assert_not_called()
                run.assert_not_called()
                log.assert_not_called()
                self.assertFalse(log_path.exists())

    def test_installer_never_executes_compromised_environment(self) -> None:
        self.fake_pip()
        metadata = self.metadata("1.107.0")
        for marker in (False, True):
            with self.subTest(marker=marker):
                if marker:
                    metadata.unlink()
                    metadata.parent.rmdir()
                    (self.site / "cook.pth").touch()
                result = self.run_installer()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("SECURITY WARNING", result.stderr)
                self.assertFalse(self.pip_calls.exists())
                self.assertFalse(self.sentinel.exists())

    def test_installer_pins_hash_and_forces_download_on_both_paths(self) -> None:
        self.fake_pip()
        for mode in ("install", "upgrade"):
            with self.subTest(mode=mode):
                result = self.run_installer(mode)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(
                    f"LiteLLM {SAFE_VERSION} installed successfully", result.stdout
                )
                calls = self.pip_calls.read_text(encoding="utf-8")
                self.assertIn(SAFE_REQUIREMENT, calls)
                self.assertIn("#sha256=" + SAFE_WHEEL_SHA256, calls)
                self.assertIn("--no-cache-dir", calls)
                self.assertIn("--force-reinstall", calls)
                self.assertIn("--index-url https://pypi.org/simple", calls)
                self.assertIn("--isolated", calls)
                self.assertIn("--require-hashes", calls)
                self.assertIn("--only-binary=:all:", calls)
                self.assertIn("-r /dev/fd/", calls)
                self.assertNotIn("pyyaml==6.0.3", calls)
                self.assertEqual(calls.count(" install "), 1)
                lines = calls.splitlines()
                self.assertIn(" download ", lines[0])
                self.assertIn("--require-hashes", lines[0])
                self.assertIn("--no-deps", lines[0])
                self.assertIn(" install ", lines[1])
                wheel = lines[1].split("prisma ", 1)[1]
                self.assertTrue(wheel.endswith("/PyYAML-6.0.3-py3-none-any.whl"))
                self.assertFalse(Path(wheel).parent.exists())
                self.assertIn("check", calls)
                self.assertFalse(self.sentinel.exists())
                self.pip_calls.unlink()

    def test_installer_prints_pinned_requirement_from_shared_policy(self) -> None:
        result = subprocess.run(
            [sys.executable, "-I", "-S", str(CHECKER), "requirement"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(result.stdout.strip(), SAFE_REQUIREMENT)

    def test_installer_checks_before_nonisolated_python_or_pip(self) -> None:
        precheck = INSTALLER.index('python3 -I -S "$SCRIPT_DIR/src/')
        self.assertLess(precheck, INSTALLER.index("PYTHON_VERSION="))
        self.assertLess(precheck, INSTALLER.index('if python3 -c "import yaml"'))

    def test_installer_fails_on_download_dependency_or_version_failure(self) -> None:
        self.fake_pip()
        for env in (
            {"TEST_PIP_FAILURE": "download"},
            {"TEST_PIP_FAILURE": "install"},
            {"TEST_PIP_FAILURE": "check"},
            {"TEST_INSTALLED_VERSION": "1.107.0"},
        ):
            with self.subTest(env=env):
                for metadata in self.site.glob("litellm-*.dist-info/METADATA"):
                    metadata.unlink()
                    metadata.parent.rmdir()
                result = self.run_installer(**env)
                self.assertNotEqual(result.returncode, 0)
                if "TEST_INSTALLED_VERSION" in env:
                    self.assertIn("SECURITY WARNING", result.stderr)
                    self.assertNotIn("Platform compatibility:", result.stdout)
                else:
                    self.assertIn("Platform compatibility:", result.stdout)
                    self.assertIn("do not bypass verification", result.stdout)
                self.assertNotIn("dependency check passed", result.stdout)
                calls = self.pip_calls.read_text(encoding="utf-8").splitlines()
                wheel_dir = Path(calls[0].split("--dest ", 1)[1].split(" -r ", 1)[0])
                self.assertFalse(wheel_dir.exists())
                if env.get("TEST_PIP_FAILURE") == "download":
                    self.assertFalse(any(" install " in line for line in calls))
                self.assertFalse(self.sentinel.exists())
                self.pip_calls.unlink()


if __name__ == "__main__":
    unittest.main()
