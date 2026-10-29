"""Inspect LiteLLM environments without importing or executing their packages."""

from __future__ import annotations

import argparse
import platform
import re
import sys
from email.parser import Parser
from pathlib import Path

SAFE_VERSION = "1.106.2"
# PyPI artifact identities for the release recommended in the October advisory.
SAFE_WHEELS = {
    ("glibc", "aarch64"): (
        "cf/a8/f21d7fd8f7ffa6ebacefbedfb6268430e2b55ce9b2f33a36ac0295ccd500/"
        "litellm-1.106.2-cp310-abi3-manylinux_2_28_aarch64.whl",
        "34bfdc26c2c075726eda59ec0675e2a8a67b6e59a03980def7ef7651b849141a",
    ),
    ("glibc", "x86_64"): (
        "6b/30/4a38c22181ef7169b6c044b8294f2e5b06236aeeccfbe57c2f0e85a51bfe/"
        "litellm-1.106.2-cp310-abi3-manylinux_2_28_x86_64.whl",
        "cb1a2a1e55a4a8350ae2167525dc0d06f59b078cd685fb3e0f5419f02da5420e",
    ),
    ("musl", "aarch64"): (
        "a0/76/e7ca210ad574f5321c162f2b27f8674c260e0857323a9991c4d3765ce8f1/"
        "litellm-1.106.2-cp310-abi3-musllinux_1_2_aarch64.whl",
        "ce18971e2bb2aa58ef8efae1cd12da1248e1c2e5ae3cf5903bfb9d563bb34d9d",
    ),
    ("musl", "x86_64"): (
        "22/03/95518dfb58e29c45ab40a7bc3d1d76509d09011172c380145fc9e900a214/"
        "litellm-1.106.2-cp310-abi3-musllinux_1_2_x86_64.whl",
        "32e567ee53c812e36f79158ec280c8850973f84a9e5fdef50d31ca09306d148c",
    ),
}
ADVISORY_URL = "https://docs.litellm.ai/blog/security-incident-oct-21-2026"
INCIDENT_RESPONSE = (
    "Do not run this environment's Python, pip, or LiteLLM. Stop affected "
    "services, isolate and investigate the host, and rotate all exposed API "
    "keys, SSH keys, cloud credentials, and other secrets from a clean machine. "
    "Rebuild the environment on a trusted host; an upgrade alone does not "
    f"remediate credential theft or persistence. Advisory: {ADVISORY_URL}"
)


class LiteLLMSecurityError(RuntimeError):
    """The configured environment cannot safely be used."""


def safe_requirement() -> str:
    """Select a hash-pinned Linux wheel without falling back to another release."""
    libc = platform.libc_ver()[0]
    if not libc and any(Path("/lib").glob("ld-musl-*.so.1")):
        libc = "musl"
    key = (libc, platform.machine())
    if sys.platform != "linux" or key not in SAFE_WHEELS:
        raise LiteLLMSecurityError(
            f"No reviewed LiteLLM wheel for {sys.platform}/{key}; "
            "supported platforms are Linux glibc/musl on aarch64/x86_64."
        )
    path, digest = SAFE_WHEELS[key]
    return (
        f"litellm[proxy] @ https://files.pythonhosted.org/packages/{path}"
        f"#sha256={digest}"
    )


def check_litellm_environment(
    venv: Path, *, allow_missing: bool = False, require_pinned: bool = False
) -> str | None:
    """Read metadata and known compromise indicators without starting Python."""
    try:
        if not venv.exists():
            if allow_missing and not require_pinned:
                return None
            raise LiteLLMSecurityError(f"LiteLLM environment is missing: {venv}")

        sites: set[Path] = set()
        for library in (venv / "lib", venv / "lib64"):
            if library.is_dir():
                for directory in library.iterdir():
                    site = directory / "site-packages"
                    if directory.name.startswith("python") and site.is_dir():
                        sites.add(site.resolve())

        versions: list[str] = []
        package_present = False
        for site in sorted(sites):
            package_present = package_present or (site / "litellm").exists()
            markers = [
                site / name
                for name in (
                    "cook.pth",
                    "cook.py",
                    "cook.pyc",
                    "sitecustomize.py",
                    "sitecustomize.pyc",
                    "litellm_init.pth",
                )
            ]
            cache = site / "__pycache__"
            if cache.is_dir():
                markers.extend(cache.glob("cook.*.pyc"))
                markers.extend(cache.glob("sitecustomize.*.pyc"))
            for marker in markers:
                if marker.exists() or marker.is_symlink():
                    raise LiteLLMSecurityError(
                        f"SECURITY WARNING: Potential LiteLLM compromise indicator "
                        f"found: {marker}. Custom sitecustomize files can also be "
                        "legitimate; investigate without executing them. "
                        + INCIDENT_RESPONSE
                    )
            for entry in site.iterdir():
                if not (
                    entry.name.lower().startswith("litellm-")
                    and entry.name.endswith(".dist-info")
                ):
                    continue
                metadata = Parser().parsestr(
                    (entry / "METADATA").read_text(encoding="utf-8")
                )
                version = metadata.get("Version", "").strip()
                match = re.fullmatch(
                    r"(?:0!)?v?(\d+)\.(\d+)\.(\d+)([a-zA-Z0-9.+_-]*)", version
                )
                if metadata.get("Name", "").lower() != "litellm" or match is None:
                    raise LiteLLMSecurityError(
                        f"Cannot verify LiteLLM metadata in {entry}; refusing to run."
                    )
                release = tuple(int(match[index]) for index in (1, 2, 3))
                if release in ((1, 107, 0), (1, 82, 7), (1, 82, 8)):
                    raise LiteLLMSecurityError(
                        f"SECURITY WARNING: Compromised LiteLLM {version} in {venv}. "
                        + INCIDENT_RESPONSE
                    )
                versions.append(version)

        config = (venv / "pyvenv.cfg").read_text(encoding="utf-8")
        if re.search(r"(?im)^\s*include-system-site-packages\s*=\s*true\s*$", config):
            raise LiteLLMSecurityError(
                f"Cannot verify inherited system packages in {venv}; "
                "recreate it without --system-site-packages."
            )
        if (
            not versions
            and not package_present
            and allow_missing
            and not require_pinned
        ):
            return None
        if len(versions) != 1:
            raise LiteLLMSecurityError(
                f"Expected exactly one LiteLLM installation in {venv}; "
                f"found {len(versions)}. Recreate the environment."
            )
        if require_pinned and versions[0] != SAFE_VERSION:
            raise LiteLLMSecurityError(
                f"Expected pinned LiteLLM {SAFE_VERSION}, found {versions[0]} in {venv}."
            )
        return versions[0]
    except (OSError, UnicodeError) as error:
        raise LiteLLMSecurityError(
            f"Cannot inspect LiteLLM environment {venv}: {error}"
        ) from error


def main() -> int:
    """Provide an isolated, stdlib-only checker for the shell installer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "requirement"))
    parser.add_argument("venv", nargs="?", type=Path)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--require-pinned", action="store_true")
    args = parser.parse_args()
    try:
        if args.action == "requirement":
            print(safe_requirement())
            return 0
        if args.venv is None:
            parser.error("check requires a virtual environment path")
        version = check_litellm_environment(
            args.venv,
            allow_missing=args.allow_missing,
            require_pinned=args.require_pinned,
        )
        if version is not None:
            print(version)
    except LiteLLMSecurityError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
