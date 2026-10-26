"""GPU and NVIDIA driver startup preflight checks."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Protocol

from hls_terminal import terminal_print as print


class CommandRunner(Protocol):
    """Callable interface used to run checked host and Docker commands."""

    def __call__(
        self,
        command: list[str],
        *,
        capture: bool = True,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]: ...


def extract_nvidia_driver_mismatch(message: str) -> tuple[str, str] | None:
    """Extract required/detected driver versions from NVIDIA runtime errors."""
    match = re.search(
        r"NVIDIA Driver Release\s+([0-9]+(?:\.[0-9]+){1,2})\s+or later,\s+"
        r"but version\s+([0-9]+(?:\.[0-9]+){1,2})\s+was detected",
        message,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    return match.group(1), match.group(2)


class PreflightChecks:
    """Run GPU capacity and Docker/NVIDIA compatibility checks before startup."""

    def __init__(self, image: str, command_runner: CommandRunner) -> None:
        self.image = image
        self._run_command = command_runner

    def check_gpu_memory(
        self,
        model_names: list[str],
        gpu_memory_utilization: float,
        *,
        dry_run: bool = False,
        capacity_only: bool = False,
    ) -> None:
        """Check GPU 0's budget, using shared system memory on GB10."""
        if capacity_only and not dry_run:
            raise ValueError("Capacity-only memory checks require dry-run.")
        target = (
            f"model '{model_names[0]}'"
            if len(model_names) == 1
            else f"{len(model_names)} selected models ({', '.join(model_names)})"
        )
        command = [
            "nvidia-smi",
            "--query-gpu=index,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ]
        result = self._run_command(command, capture=True)
        if result.returncode != 0:
            print(f"ERROR: Could not check available GPU VRAM for {target}.")
            if result.stderr.strip():
                print(result.stderr.strip())
            sys.exit(1)

        gpu_zero: tuple[float, float] | None = None
        memory_name = "GPU VRAM"
        try:
            for line in result.stdout.splitlines():
                fields = [field.strip() for field in line.split(",")]
                if len(fields) != 3:
                    raise ValueError("unexpected nvidia-smi output")
                index = int(fields[0])
                if index != 0:
                    continue
                if any(value in ("N/A", "[N/A]") for value in fields[1:]):
                    gpu_zero = self._gb10_shared_memory()
                    memory_name = "Shared system memory"
                    continue
                free_mb, total_mb = (int(field) for field in fields[1:])
                if total_mb <= 0 or free_mb < 0:
                    raise ValueError("invalid GPU memory values")
                if index == 0:
                    gpu_zero = (free_mb, total_mb)
        except ValueError:
            print("ERROR: Could not parse GPU VRAM reported by nvidia-smi.")
            sys.exit(1)

        if gpu_zero is None:
            print("ERROR: nvidia-smi did not report GPU 0.")
            sys.exit(1)

        free_mb, total_mb = gpu_zero
        required_mb = total_mb * gpu_memory_utilization
        label = " (dry-run)" if dry_run else ""
        memory_details = (
            f"Available: {free_mb:g} MiB ({free_mb / 1024:.2f} GiB); "
            f"required: {required_mb:.0f} MiB ({required_mb / 1024:.2f} GiB) "
            f"({gpu_memory_utilization:.0%} of "
            f"{total_mb:g} MiB ({total_mb / 1024:.2f} GiB))."
        )
        if (total_mb if capacity_only else free_mb) < required_mb:
            capacity_label = " total capacity" if capacity_only else ""
            print(
                f"ERROR: Insufficient {memory_name}{capacity_label} "
                f"for {target}{label}."
            )
            print(f"  GPU 0: {memory_details}")
            sys.exit(1)

        if capacity_only:
            print(
                f"WARNING: {memory_name} capacity check (dry-run) for {target}. "
                f"GPU 0: {memory_details} Containers remain running; available "
                "memory after stopping them cannot be verified. Startup will "
                "check the combined budget after stopping selected backends."
            )
            return

        print(
            f"{memory_name} preflight{label} passed for {target}. "
            f"GPU 0: {memory_details}"
        )

    def _gb10_shared_memory(self) -> tuple[float, float]:
        """Read GB10 shared memory using the same counters as DGX Dashboard."""
        result = self._run_command(
            ["nvidia-smi", "--id=0", "--query-gpu=name", "--format=csv,noheader"],
            capture=True,
        )
        if result.returncode != 0 or result.stdout.strip() != "NVIDIA GB10":
            print(
                "ERROR: GPU 0 memory counters are unavailable; "
                "shared system memory fallback requires NVIDIA GB10."
            )
            if result.stderr.strip():
                print(result.stderr.strip())
            sys.exit(1)

        try:
            values: dict[str, int] = {}
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                fields = line.split()
                if fields and fields[0] in ("MemTotal:", "MemAvailable:"):
                    if len(fields) != 3 or fields[2] != "kB":
                        raise ValueError(f"unexpected memory field: {line}")
                    values[fields[0]] = int(fields[1])
            total_kib = values["MemTotal:"]
            available_kib = values["MemAvailable:"]
            if total_kib <= 0 or not 0 <= available_kib <= total_kib:
                raise ValueError("invalid MemTotal or MemAvailable")
        except (OSError, KeyError, ValueError) as exc:
            print(f"ERROR: Could not read GB10 shared system memory: {exc}")
            sys.exit(1)
        return available_kib / 1024, total_kib / 1024

    def check_nvidia_driver(self, *, dry_run: bool = False) -> None:
        """Probe GPU/image compatibility and fail on confirmed driver mismatch."""
        command = [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "--entrypoint",
            "python3",
            self.image,
            "-c",
            "import sys; sys.exit(0)",
        ]
        result = self._run_command(command, capture=True)
        if result.returncode == 0:
            label = " (dry-run)" if dry_run else ""
            print(f"NVIDIA pre-run compatibility check{label} passed.")
            return

        output_parts = [result.stdout.strip(), result.stderr.strip()]
        output = "\n".join(part for part in output_parts if part)

        print("NVIDIA pre-run compatibility check failed.")
        print(f"Docker image: {self.image}")

        mismatch = extract_nvidia_driver_mismatch(output)
        if mismatch:
            required, detected = mismatch
            print(f"Required NVIDIA driver: >= {required}")
            print(f"Detected NVIDIA driver: {detected}")
            print(
                "ERROR: NVIDIA driver is incompatible with the configured vLLM image."
            )
            print(
                "Startup aborted; select a compatible VLLM_DOCKER_IMAGE "
                "or upgrade the driver."
            )
            sys.exit(1)

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
        print("Could not confirm a driver mismatch; startup will continue.")

    def _host_nvidia_driver_version(self) -> str | None:
        """Read host NVIDIA driver version from nvidia-smi (if available)."""
        result = self._run_command(
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
