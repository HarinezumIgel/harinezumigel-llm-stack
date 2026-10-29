"""Bounded Docker startup logs and HTTP readiness monitoring."""

from __future__ import annotations

import codecs
import builtins
import datetime
import json
import os
import selectors
import shlex
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

from hls_terminal import terminal_print as print, terminal_style


class StartupError(RuntimeError):
    """A backend or its startup log follower failed."""


@dataclass(frozen=True)
class BackendHandle:
    """The actual container and published endpoint being monitored."""

    container: str
    health_url: str


@dataclass(frozen=True)
class StartupJob:
    """An explicitly planned launch, independent of the stack/CLI implementation."""

    name: str
    label: str
    port: int | None
    backend: BackendHandle | None
    launch: Callable[[], None]
    reused: bool
    secrets: tuple[str, ...]
    monitor: bool = True
    mode: str = "inference"


def group_startup_jobs(jobs: list[StartupJob]) -> dict[str, list[StartupJob]]:
    """Group display rows by mode, retaining first-seen category and target order."""
    groups: dict[str, list[StartupJob]] = {}
    labels = {
        "inference": "Inference",
        "embedding": "Embeddings",
        "rerank": "Reranking",
        "proxy": "Proxy",
    }
    for job in jobs:
        label = labels.get(job.mode, job.mode)
        groups.setdefault(label, []).append(job)
    return groups


def run_startup(
    jobs: list[StartupJob],
    *,
    timeout: float,
    quiet: bool,
    dry_run: bool,
    program: str,
    format_line: Callable[[str], str],
    diagnose: Callable[[StartupJob], None],
) -> int:
    """Execute a validated sequence, retaining backends on failure or interruption."""
    results = ["not attempted"] * len(jobs)
    exit_code = 0
    for index, job in enumerate(jobs):
        print(f"\n=== [{index + 1}/{len(jobs)}] Starting {job.label} ===", flush=True)
        try:
            job.launch()
            if dry_run:
                results[index] = "dry-run"
                continue
            if not job.monitor:
                results[index] = "started"
                continue
            if job.backend is None or job.port is None:
                raise StartupError(f"Missing readiness endpoint for {job.name}")
            since = (
                datetime.datetime.now(datetime.timezone.utc).isoformat(
                    timespec="microseconds"
                )
                if job.reused
                else container_started_at(job.backend.container)
            )

            def sanitized_line(line: str) -> str:
                for secret in job.secrets:
                    if secret:
                        line = line.replace(secret, "***REDACTED***")
                return format_line(line)

            StartupMonitor(sanitized_line).wait(
                job.backend,
                timeout=timeout,
                show_logs=not quiet,
                since=since,
            )
            results[index] = "ready"
            message = f"READY: {job.label} - port {job.port}"
            border = "=" * (len(message) + 4)
            banner = f"{border}\n| {message} |\n{border}"
            print(f"\n{terminal_style(banner, code='1;32')}\n", flush=True)
        except KeyboardInterrupt:
            results[index] = "interrupted"
            print("\nStartup interrupted; started containers are left running.")
            exit_code = 130
            break
        except (StartupError, OSError, SystemExit) as exc:
            results[index] = "failed"
            print(f"ERROR: Startup failed for {job.name}: {exc}")
            try:
                diagnose(job)
            except (StartupError, OSError, SystemExit) as log_exc:
                print(f"ERROR: Cannot display failure logs: {log_exc}")
            print(f"Follow logs: {program} {shlex.quote(job.name)} --stream-log")
            print("Startup stopped; previously started containers are left running.")
            exit_code = 1
            break
    print("\n=== Startup summary ===")
    results_by_name = {job.name: result for job, result in zip(jobs, results)}
    for category, members in group_startup_jobs(jobs).items():
        print("\n" + terminal_style(category, accent=True))
        for job in members:
            result = results_by_name[job.name]
            port = f"port={job.port}" if job.port is not None else "no readiness check"
            print(f"{job.label:<24} {result:<16} {port}")
    return exit_code


def container_started_at(container: str) -> str:
    """Return Docker's start timestamp for filtering logs to the current run."""
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.StartedAt}}", container],
        capture_output=True,
        text=True,
        check=False,
    )
    started_at = result.stdout.strip()
    if result.returncode or not started_at:
        detail = result.stderr.strip()
        suffix = f": {detail}" if detail else ""
        raise StartupError(f"Cannot determine start time for {container}{suffix}")
    return started_at


def container_state(container: str) -> tuple[str, int]:
    """Read container state, surfacing Docker errors rather than treating them as absence."""
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{json .State}}", container],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise StartupError(f"Cannot inspect {container}: {result.stderr.strip()}")
    try:
        state = json.loads(result.stdout)
        return str(state["Status"]), int(state["ExitCode"])
    except (ValueError, KeyError, TypeError) as exc:
        raise StartupError(f"Invalid Docker state for {container}") from exc


def health_ready(url: str, timeout: float = 2.0) -> bool:
    """Probe a direct local endpoint without inheriting HTTP proxy settings."""
    try:
        with build_opener(ProxyHandler({})).open(
            Request(url),
            timeout=timeout,
        ) as response:
            return response.status == 200
    except (URLError, TimeoutError, ConnectionError):
        return False


class StartupMonitor:
    """Follow one backend's logs until healthy, failed, interrupted, or timed out."""

    def __init__(self, format_line: Callable[[str], str]) -> None:
        self.format_line = format_line

    def wait(
        self,
        backend: BackendHandle,
        *,
        timeout: float,
        show_logs: bool,
        since: str | None = None,
    ) -> None:
        """Wait for real readiness and always close/reap the log follower."""
        deadline = time.monotonic() + timeout
        follower: subprocess.Popen[bytes] | None = None
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        pending = ""
        cutoff = datetime.datetime.fromisoformat(since) if since is not None else None

        def _print_line(line: str) -> None:
            timestamp, separator, message = line.partition(" ")
            if separator:
                try:
                    recorded_at = datetime.datetime.fromisoformat(timestamp)
                except ValueError:
                    pass
                else:
                    if recorded_at.tzinfo is not None:
                        if cutoff is not None and recorded_at < cutoff:
                            return
                        line = message
            builtins.print(self.format_line(line.rstrip("\r")), flush=True)

        def emit(chunk: bytes, *, final: bool = False) -> None:
            nonlocal pending
            pending += decoder.decode(chunk, final=final)
            while "\n" in pending:
                line, pending = pending.split("\n", 1)
                _print_line(line)
            if final and pending:
                _print_line(pending)
                pending = ""

        with selectors.DefaultSelector() as selector:
            try:
                while True:
                    status, exit_code = container_state(backend.container)
                    if status not in ("running", "created", "restarting"):
                        raise StartupError(
                            f"{backend.container} is {status} (exit code {exit_code})"
                        )
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise StartupError(
                            f"Readiness timeout after {timeout:g}s; "
                            f"{backend.container} is still {status}"
                        )
                    if status == "running" and health_ready(
                        backend.health_url,
                        min(2.0, remaining),
                    ):
                        break
                    if show_logs and follower is None:
                        command = [
                            "docker",
                            "logs",
                            "--timestamps",
                            "--tail",
                            "200",
                            "--follow",
                        ]
                        if since is not None:
                            command.extend(["--since", since])
                        command.append(backend.container)
                        follower = subprocess.Popen(
                            command,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                        )
                        assert follower.stdout is not None
                        os.set_blocking(follower.stdout.fileno(), False)
                        selector.register(follower.stdout, selectors.EVENT_READ)
                    if follower is None:
                        time.sleep(min(0.5, max(0, deadline - time.monotonic())))
                    else:
                        for key, _ in selector.select(
                            min(0.5, max(0, deadline - time.monotonic()))
                        ):
                            chunk = os.read(key.fd, 65536)
                            if chunk:
                                emit(chunk)
                            else:
                                selector.unregister(key.fileobj)
                        if follower.poll() is not None:
                            # Check container state first on the next iteration so an
                            # exited backend is reported instead of just a closed pipe.
                            status, exit_code = container_state(backend.container)
                            if status not in ("running", "created", "restarting"):
                                raise StartupError(
                                    f"{backend.container} is {status} (exit code {exit_code})"
                                )
                            raise StartupError(
                                f"Startup log follower ended unexpectedly "
                                f"(exit code {follower.returncode})"
                            )
            finally:
                if follower is not None:
                    try:
                        if follower.poll() is None:
                            follower.terminate()
                        try:
                            follower.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            follower.kill()
                            follower.wait()
                        assert follower.stdout is not None
                        while True:
                            chunk = follower.stdout.read(65536)
                            if not chunk:
                                break
                            emit(chunk)
                        emit(b"", final=True)
                    finally:
                        if follower.stdout is not None:
                            follower.stdout.close()
