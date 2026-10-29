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

### Other evidence to investigate

- `/tmp/remote_code_loader.lock` and a detached Python process holding it,
  even after reinstalling the package. Inspect without executing Python.
- `litellm-1.107.0.dist-info` in retained environments, cached artifacts, and images.
- Historical DNS/proxy/firewall logs, using the October payload indicators
  from [BerriAI/litellm#47826](https://github.com/BerriAI/litellm/issues/47826).
  Do not contact suspected exfiltration or command-and-control destinations.
- The canonical investigation lists `69.48.229.143` and `199.91.221.39`
  as attacker source IPs used against upstream infrastructure. These are
  not documented there as customer malware exfiltration destinations.
- On AWS, CloudTrail `GetCallerIdentity` calls around installation and
  subsequent use of the role's credentials outside your infrastructure.

Missing files, no current network connections, or no retained logs cannot
exclude earlier exfiltration. The March metadata IP `169.254.169.254` and
March domains are not a complete October detection checklist.

### If a compromised version or indicator is detected

1. Do not activate the environment or run its Python, pip, or LiteLLM.
   The October `cook.pth` and `sitecustomize.py` hooks can execute at Python
   startup without importing LiteLLM. Run inspection tools only from a trusted
   interpreter/environment.
2. Stop affected services, isolate the host, and investigate for persistence.
   Preserve evidence when needed for incident response.
3. From a clean machine, rotate all potentially exposed API keys, SSH keys,
   cloud credentials and instance/task IAM roles, database passwords (including
   `DATABASE_URL`), Kubernetes tokens, and other secrets. Include every provider
   key in the proxy database, LiteLLM master and virtual keys, and secrets
   reachable through other host processes.
4. Rebuild the environment on a trusted host and rerun the installer.
   Do not simply upgrade or reactivate a retained environment backup:
   replacing the package does not undo credential theft or host compromise.

The installer intentionally aborts rather than running pip inside a compromised
environment. It does not automatically delete evidence, stop services, or rotate
credentials.
