#!/usr/bin/env python3
"""Exercise installer configuration migration and backups without installing."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PROJECT_DIR = Path(__file__).resolve().parent.parent
INSTALLER = (PROJECT_DIR / "install.sh").read_text(encoding="utf-8")
CONFIG_SETUP = INSTALLER.split('CONFIG_DIR="$INSTALL_DIR/config"', 1)[1].split(
    "# Create symlink for easy access", 1
)[0]
BACKUP_FUNCTION = INSTALLER.split("backup_config() {", 1)[1].split("\n}", 1)[0]
VENV_SETUP = INSTALLER.split("CREATE_VENV=false", 1)[1].split("# Summary", 1)[0]
RUNTIME_MODULES = INSTALLER.split("RUNTIME_MODULES=(", 1)[1].split("\n)", 1)[0]
COMMAND_SETUP = INSTALLER.split("# Create symlink for easy access", 1)[1].split(
    "# Check if ~/.local/bin is in PATH", 1
)[0]

sys.path.insert(0, str(PROJECT_DIR / "src"))
from fix_settings import link_repository_launcher


class TestInstallerConfig(unittest.TestCase):
    def test_embedded_pyyaml_has_exact_version_and_hashes(self) -> None:
        import re

        function = (
            "pyyaml_requirements() {"
            + INSTALLER.split("pyyaml_requirements() {", 1)[1].split("\n}", 1)[0]
            + "\n}\n"
        )
        result = subprocess.run(
            ["bash", "-c", function + "pyyaml_requirements"],
            check=True,
            capture_output=True,
            text=True,
        )
        content = result.stdout
        self.assertEqual(content.splitlines()[0], "pyyaml==6.0.3 \\")
        hashes = re.findall(r"--hash=sha256:([0-9a-f]{64})", content)
        self.assertEqual(len(hashes), 72)
        self.assertEqual(len(set(hashes)), 72)
        self.assertEqual(INSTALLER.count("-r <(pyyaml_requirements)"), 2)
        self.assertNotIn("requirements-launcher.lock", INSTALLER)
        self.assertNotIn("requirements-proxy.lock", INSTALLER)

    def test_backend_image_example_is_digest_pinned(self) -> None:
        lines = (
            (PROJECT_DIR / "config" / "env.example")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        images = [
            line.split("=", 1)[1]
            for line in lines
            if line.startswith("VLLM_DOCKER_IMAGE=")
        ]
        self.assertEqual(len(images), 1)
        self.assertRegex(
            images[0],
            r"^nvcr\.io/nvidia/vllm:26\.05\.post1-py3@sha256:[0-9a-f]{64}$",
        )

    def test_command_uses_installed_runtime_not_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            home = root / "home"
            command = (
                "set -e\n"
                'INSTALL_DIR="$1/installed"\n'
                'HOME="$1/home"\n'
                'SCRIPT_DIR="$2"\n'
                "RUNTIME_MODULES=(" + RUNTIME_MODULES + "\n)\n" + COMMAND_SETUP
            )
            link = home / ".local" / "bin" / "harinezumigel-llm-stack"
            for _ in range(2):
                subprocess.run(
                    [
                        "bash",
                        "-c",
                        command,
                        "test-runtime",
                        directory,
                        str(PROJECT_DIR),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                installed = root / "installed" / "src"
                launcher = installed / "harinezumigel-llm-stack.py"
                self.assertEqual(link.resolve(), launcher)
                self.assertTrue(launcher.stat().st_mode & 0o100)
                for name in RUNTIME_MODULES.split():
                    self.assertEqual(
                        (installed / name).read_bytes(),
                        (PROJECT_DIR / "src" / name).read_bytes(),
                    )
                self.assertFalse((installed / "fix_settings.py").exists())

    def test_local_helper_retargets_link_and_preserves_regular_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "repository" / "launcher.py"
            source.parent.mkdir()
            source.touch()
            link = root / "bin" / "launcher"
            link_repository_launcher(link, source)
            self.assertEqual(link.resolve(), source)
            link.unlink()
            link.symlink_to(root / "missing-installed-launcher")
            link_repository_launcher(link, source)
            self.assertEqual(link.resolve(), source)
            link.unlink()
            link.write_text("keep me", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                link_repository_launcher(link, source)
            self.assertEqual(link.read_text(encoding="utf-8"), "keep me")

    def test_virtual_environment_choices(self) -> None:
        cases = [
            (True, "n n", False, False),
            (True, "n y", True, True),
            (True, "y", True, False),
            (False, "n", False, False),
            (False, "y", True, True),
        ]
        for existing, answers, success, created in cases:
            with self.subTest(existing=existing, answers=answers):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    config = root / "config"
                    config.mkdir()
                    env = config / "env"
                    original = "LITELLM_VENV_ACTIVATE=/old\nLITELLM_BIN=/old-bin\n"
                    env.write_text(original, encoding="utf-8")
                    if existing:
                        (root / "venv").mkdir()
                        (root / "venv" / "original").touch()
                    command = (
                        "set -e\n"
                        'INSTALL_DIR="$1"\n'
                        'CONFIG_DIR="$1/config"\n'
                        'VENV_PATH="$1/venv"\n'
                        "choices=($2)\n"
                        'read() { REPLY=${choices[0]}; choices=("${choices[@]:1}"); }\n'
                        'backup_config() { cp "$1" "$1.saved"; }\n'
                        'python3() { mkdir -p "$VENV_PATH"; touch "$VENV_PATH/created"; }\n'
                        'install_litellm_proxy_deps() { echo "$2" >> "$INSTALL_DIR/deps"; }\n'
                        "CREATE_VENV=false\n" + VENV_SETUP
                    )
                    result = subprocess.run(
                        ["bash", "-c", command, "test-venv", directory, answers],
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(result.returncode == 0, success, result.stderr)
                    self.assertEqual((root / "venv" / "created").exists(), created)
                    if success:
                        self.assertEqual(
                            env.read_text(encoding="utf-8"),
                            f"LITELLM_VENV_ACTIVATE={root}/venv/bin/activate\n"
                            f"LITELLM_BIN={root}/venv/bin/litellm\n",
                        )
                        self.assertEqual(
                            (config / "env.saved").read_text(encoding="utf-8"), original
                        )
                    else:
                        self.assertIn("Aborted", result.stdout)
                        self.assertEqual(env.read_text(encoding="utf-8"), original)
                        self.assertFalse((root / "deps").exists())
                    if existing and created:
                        backups = list(root.glob("venv_backup_*/venv/original"))
                        self.assertEqual(len(backups), 1)

    def test_missing_or_empty_env_is_initialized_when_copy_is_declined(self) -> None:
        for empty in (False, True):
            with self.subTest(empty=empty):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    config = root / "config"
                    config.mkdir()
                    env = config / "env"
                    if empty:
                        env.touch()
                    command = (
                        "set -e\n"
                        'INSTALL_DIR="$1"\n'
                        'SCRIPT_DIR="$2"\n'
                        'CONFIG_DIR="$INSTALL_DIR/config"\n'
                        "read() { REPLY=n; }\n" + CONFIG_SETUP
                    )
                    subprocess.run(
                        [
                            "bash",
                            "-c",
                            command,
                            "test-initialization",
                            directory,
                            str(PROJECT_DIR),
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(
                        env.read_bytes(),
                        (PROJECT_DIR / "config" / "env.example").read_bytes(),
                    )
                    self.assertIn("MODEL_ROOT=", env.read_text(encoding="utf-8"))

    def test_replacing_configs_saves_originals_in_old_configs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config"
            config.mkdir()
            env = config / "env"
            yaml = config / "config.yaml"
            env.write_text("MODEL_ROOT=/original\n", encoding="utf-8")
            yaml.write_text("model_list: []\n", encoding="utf-8")
            originals = {path.name: path.read_bytes() for path in (env, yaml)}
            command = (
                "set -e\n"
                'INSTALL_DIR="$1"\n'
                'SCRIPT_DIR="$2"\n'
                'CONFIG_DIR="$INSTALL_DIR/config"\n'
                "read() { REPLY=y; }\n" + CONFIG_SETUP
            )
            subprocess.run(
                [
                    "bash",
                    "-c",
                    command,
                    "test-replacement",
                    directory,
                    str(PROJECT_DIR),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            for path in (env, yaml):
                self.assertEqual(
                    path.read_bytes(),
                    (PROJECT_DIR / "config" / f"{path.name}.example").read_bytes(),
                )
                backups = list((root / "old_configs").glob(f"{path.name}_*.bak"))
                self.assertEqual(len(backups), 1)
                self.assertEqual(backups[0].read_bytes(), originals[path.name])
            self.assertEqual(
                sorted(path.name for path in config.iterdir()), ["config.yaml", "env"]
            )

    def test_backups_are_private_unique_and_outside_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config"
            config.mkdir()
            env = config / "env"
            yaml = config / "config.yaml"
            env.write_text("MODEL_ROOT=/test/models\n", encoding="utf-8")
            yaml.write_text("model_list: []\n", encoding="utf-8")
            subprocess.run(
                [
                    "bash",
                    "-c",
                    "set -e\n"
                    'BACKUP_DIR="$1/old_configs"\n'
                    "backup_config() {" + BACKUP_FUNCTION + "\n}\n"
                    'backup_config "$1/config/env"\n'
                    'backup_config "$1/config/env"\n'
                    'backup_config "$1/config/config.yaml"\n',
                    "test-backups",
                    str(root),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            backups = list((root / "old_configs").glob("*.bak"))
            self.assertEqual(len(backups), 2)
            self.assertEqual(len(list(config.glob("*.bak"))), 0)
            for backup in backups:
                expected = yaml if backup.name.startswith("config.yaml_") else env
                self.assertEqual(backup.read_bytes(), expected.read_bytes())
                self.assertEqual(backup.stat().st_mode & 0o777, 0o600)

    def test_legacy_configs_are_backed_up_not_migrated(self) -> None:
        for legacy_name in ("config/.env", "env", ".env"):
            with self.subTest(legacy_name=legacy_name):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    (root / "config").mkdir()
                    legacy = root / legacy_name
                    legacy.write_text("MODEL_ROOT=/legacy\n", encoding="utf-8")
                    (root / "config.yaml").write_text(
                        "model_list: []\n", encoding="utf-8"
                    )
                    command = (
                        "set -e\n"
                        'INSTALL_DIR="$1"\n'
                        'SCRIPT_DIR="$2"\n'
                        'CONFIG_DIR="$INSTALL_DIR/config"\n'
                        'BACKUP_DIR="$INSTALL_DIR/old_configs"\n'
                        "CONFIG_BACKUPS=()\n"
                        "BACKED_UP_SOURCES=()\n"
                        "read() { REPLY=n; }\n"
                        + CONFIG_SETUP
                        + "\nreport_config_backups\n"
                    )
                    result = subprocess.run(
                        [
                            "bash",
                            "-c",
                            command,
                            "test-migration",
                            directory,
                            str(PROJECT_DIR),
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    env = root / "config" / "env"
                    self.assertEqual(
                        env.read_bytes(),
                        (PROJECT_DIR / "config" / "env.example").read_bytes(),
                    )
                    self.assertTrue(legacy.exists())
                    self.assertEqual(
                        (root / "config" / "config.yaml").read_bytes(),
                        (PROJECT_DIR / "config" / "config.yaml.example").read_bytes(),
                    )
                    backups = list((root / "old_configs").glob("*.bak"))
                    self.assertEqual(len(backups), 2)
                    for backup in backups:
                        if backup.name.startswith("config.yaml_"):
                            self.assertEqual(
                                backup.read_text(encoding="utf-8"), "model_list: []\n"
                            )
                        else:
                            self.assertEqual(backup.read_bytes(), legacy.read_bytes())
                        self.assertIn(str(backup), result.stdout)

    def test_first_install_creates_examples_without_backup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config"
            config.mkdir()
            command = (
                "set -e\n"
                'INSTALL_DIR="$1"\n'
                'SCRIPT_DIR="$2"\n'
                'CONFIG_DIR="$INSTALL_DIR/config"\n'
                'BACKUP_DIR="$INSTALL_DIR/old_configs"\n'
                "CONFIG_BACKUPS=()\n"
                "BACKED_UP_SOURCES=()\n"
                "read() { REPLY=n; }\n" + CONFIG_SETUP + "\nreport_config_backups\n"
            )
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    command,
                    "test-first-install",
                    directory,
                    str(PROJECT_DIR),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                (config / "env").read_bytes(),
                (PROJECT_DIR / "config" / "env.example").read_bytes(),
            )
            self.assertEqual(
                (config / "config.yaml").read_bytes(),
                (PROJECT_DIR / "config" / "config.yaml.example").read_bytes(),
            )
            self.assertFalse((root / "old_configs").exists())
            self.assertIn(
                "no existing configuration files needed backing up", result.stdout
            )


if __name__ == "__main__":
    unittest.main()
