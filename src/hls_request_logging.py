"""Per-start request logging control for vLLM containers."""

from __future__ import annotations

import tempfile
from pathlib import Path

from hls_preflight import CommandRunner
from hls_terminal import terminal_print as print


class RequestLogging:
    """Control vLLM request logging across container starts."""

    LABEL = "io.hls.request-logging-control"
    FILE = "/.hls-request-logging"
    STARTUP = """\
mode="$1"
shift
if [ -f /.hls-request-logging ]; then
    IFS= read -r mode < /.hls-request-logging || exit 1
fi
case "$mode" in
    on) set -- "$@" --enable-log-requests ;;
    off) ;;
    *) echo "Invalid request logging setting" >&2; exit 1 ;;
esac
exec "$@"
"""

    def __init__(self, command_runner: CommandRunner) -> None:
        self._run_command = command_runner

    def validate(self, name: str, state: str, enabled: bool) -> None:
        """Check reuse is safe and the running server has the requested setting."""
        result = self._run_command(
            [
                "docker",
                "inspect",
                "--format",
                '{{index .Config.Labels "io.hls.request-logging-control"}}',
                name,
            ]
        )
        if result.returncode:
            raise RuntimeError(f"Cannot inspect {name}: {result.stderr.strip()}")
        controlled = result.stdout.strip() == "1"
        if state == "running":
            result = self._run_command(
                [
                    "docker",
                    "exec",
                    name,
                    "python3",
                    "-c",
                    (
                        "from pathlib import Path; "
                        "a = Path('/proc/1/cmdline').read_bytes().split(b'\\0'); "
                        "assert b'vllm.entrypoints.openai.api_server' in a; "
                        "print('on' if b'--enable-log-requests' in a else 'off')"
                    ),
                ]
            )
            if result.returncode or result.stdout.strip() not in ("on", "off"):
                raise RuntimeError(
                    f"Cannot determine active request logging for {name}: "
                    f"{result.stderr.strip()}"
                )
            actual = result.stdout.strip() == "on"
            if actual != enabled:
                raise RuntimeError(
                    f"{name} is already running with request logging "
                    f"{'on' if actual else 'off'}; stop it before changing the setting"
                    + (
                        " (legacy containers also require --recreate)."
                        if not controlled
                        else "."
                    )
                )
        elif not controlled:
            raise RuntimeError(
                f"{name} predates per-start request logging; "
                "use --start --recreate once to upgrade it"
            )

    def set_mode(self, name: str, enabled: bool, *, dry_run: bool) -> None:
        """Set the next start's mode in a stopped, managed container."""
        print(f"Request logging for {name}: {'on' if enabled else 'off'}")
        if dry_run:
            print("Dry run only. Container settings were not changed.")
            return
        with tempfile.TemporaryDirectory(prefix="hls-request-logging-") as directory:
            path = Path(directory) / "mode"
            path.write_text("on\n" if enabled else "off\n", encoding="ascii")
            result = self._run_command(
                ["docker", "cp", str(path), f"{name}:{self.FILE}"]
            )
        if result.returncode:
            raise RuntimeError(
                f"Cannot set request logging for {name}: {result.stderr.strip()}"
            )
