# LiteLLM October 2026 Incident

See:

- [Upstream advisory](https://docs.litellm.ai/blog/security-incident-oct-21-2026)
- [Upstream investigation](https://docs.litellm.ai/blog/security-incident-oct-21-2026-attack-chain)

Upstream's [October incident advisory](https://docs.litellm.ai/blog/security-incident-oct-21-2026)
identifies **1.107.0** as compromised on **both PyPI and official Docker
registries**. The [canonical investigation](https://docs.litellm.ai/blog/security-incident-oct-21-2026-attack-chain)
supersedes preliminary explanations of the attack chain. This is a separate
incident from the March compromise; March indicators alone cannot assess it.

### Exposure windows

All times below are **October 21, 2026, UTC**. Central Europe was on **CEST
(UTC+2)**, not CET.

| Artifact | Reported availability | Central European time |
|---|---|---|
| PyPI `litellm==1.107.0`, all seven wheels | 11:39:51 to 13:23 | 13:39:51 to 15:23 CEST |
| Docker Hub affected tags | 11:46:56 to 14:20 | 13:46:56 to 16:20 CEST |
| GHCR affected tags | 11:46:56; unavailable by 13:51 | 13:46:56; unavailable by 15:51 CEST |

Affected Docker tags were `v1.107.0`, `latest`, `main-latest`, and `main-stable`.
The compromised main image index was
`sha256:1bbcb1443bc962e45f5aad53fe3d2c30600e0379715fb6c664cdd3ac126c339e`;
the advisory also lists separate database and non-root image indexes.
An official image or signature alone was not sufficient protection.
Review installation, CI/build, image-pull, and deployment logs for these windows.

**Installation time alone cannot establish safety.** Cached wheels, mirrors,
retained virtual environments, and previously built images could deliver or
run the compromised versions later. Reinstallation may remove the original
version evidence; filesystem timestamps alone do not prove which version ran.
Confirmed installation or execution of **1.107.0** requires incident
response regardless of date. The automatic checks therefore block these
versions without applying a time cutoff.

### Proxy startup audit records (since 1.5.2)

Starting with **harinezumigel-llm-stack 1.5.2**, proxy launches append a JSON
`litellm_start_attempt` record before executing
the proxy, with the filesystem-checked `litellm_version`, separate
`launcher_version`, and an ISO 8601 UTC `timestamp` (`+00:00`). This applies to
both background and streamed starts; dry-runs and reuse do not append records.
Failure to write the record aborts startup. It records an attempt, not confirmed
readiness. Search for `litellm_start_attempt` or `litellm_version` to find it.

The launcher also prints `LiteLLM version: <installed version>` in its startup
banner, including `litellm --start --dry-run`. It reads the version from the
checked environment's package metadata, without invoking `litellm --version`
or importing LiteLLM. A dry-run prints the version but does not execute the
proxy or create a startup log record. The installer separately prints the
installed version after its post-installation check.
Batch/group dry-runs print the installed version even when previewing recreation
or reuse; the reuse output explicitly does not attest to the running process's
version.

Logs may have been truncated by `--clean-log`, deleted, or altered, and hooks
could execute before the proxy reached application startup. Neither missing
entries nor a later unaffected run excludes earlier compromise.
Preserve logs before cleanup and redact secrets before sharing them.

### Installation and runtime protections

The installer pins **1.106.2**, the explicit fallback recommended in the October
advisory. It selects a reviewed Linux aarch64/x86_64 glibc/musl wheel from
`files.pythonhosted.org` with its corresponding pinned SHA-256.
Pip verifies the hash and reinstalls the wheel without using its download cache,
including when reusing an existing environment. The pin and artifact identity
are maintained in [hls_litellm_security.py](../src/hls_litellm_security.py);
new releases require an explicit reviewed update.

Other proxy dependencies, including Prisma, are resolved by pip according to
the target environment and LiteLLM's requirements, not a host-specific lock.
They are not individually hash-pinned. PyYAML **6.0.3** remains explicitly
version- and hash-pinned in the shared `pyyaml_requirements` definition in
[install.sh](../install.sh), which lists hashes for all non-yanked PyPI wheels
for that release.
The installer downloads PyYAML for the proxy using
`--require-hashes --only-binary=:all: --no-deps`
into a private temporary directory before installing any proxy packages.
It then installs that verified local wheel alongside the hash-pinned LiteLLM
wheel in one pip installation and runs `pip check`. A failed hash check aborts
before installation; there is no initial unverified PyYAML installation.
The launcher pip fallback uses the same embedded definition. Apt/dnf PyYAML remains OS-managed.

The installer requires Python 3.10+; the pinned LiteLLM wheel selector supports
Linux glibc/musl arm64/amd64, not arbitrary Unix hosts. Compatible dependencies
and wheels are still necessary. A missing PyYAML wheel or hash mismatch aborts
installation; there is no unhashed fallback. Review version and artifact hash
updates together and validate dependency resolution before publishing them.
Hashes verify artifact identity, not the integrity of the host or freedom from
malware or unrelated vulnerabilities. These pins require ongoing security review.

Before executing an existing environment's pip, the installer checks package
metadata and startup-hook indicators using isolated Python with site
initialization disabled (`-I -S`). October indicators include `cook.pth`,
`cook.py`, `sitecustomize.py`, standalone `cook.pyc`/`sitecustomize.pyc`
directly in the environment's site-packages, and matching tagged bytecode in
its `__pycache__`. Custom `sitecustomize`
files can be legitimate; the checker conservatively blocks them for investigation,
not as proof of infection. It does not flag distro-owned `sitecustomize.py`
outside the inspected environment's site-packages.
Historical March releases **1.82.7/1.82.8** and `litellm_init.pth` remain blocked
as separate known compromises.
Proxy startup (including dry-runs and streaming) and batch reuse/restart perform
the same filesystem-only check.
Compromised versions, residual indicators, unreadable/ambiguous metadata, and
environments inheriting system site-packages are rejected. Unaffected existing
versions remain usable by the launcher; only the installer enforces the pin.
These checks do not prove an existing environment or running process is clean.

### Forensic investigation and remediation

This document focuses on project-specific protections, audit records,
version checks, and installation behavior.

For indicators of compromise (IOCs), forensic investigation procedures,
network indicators, host analysis, credential-exposure assessment,
incident response, and remediation guidance, consult the upstream
resources:

- LiteLLM October 2026 incident advisory
- LiteLLM October 2026 canonical investigation

The upstream incident documentation is the authoritative source for
investigation and recovery guidance and may be updated as new information
becomes available.

The protections implemented by harinezumigel-llm-stack are intended to
help prevent installation or reuse of known-compromised LiteLLM releases
and to provide additional audit information. They do not constitute a
complete forensic assessment of a host, environment, container image,
or deployment.