# Security Policy

## Supported Versions

Only the latest release receives security fixes.

## LiteLLM October 2026 Incident

For detailed discussion of the LiteLLM October 2026 compromise,
forensic indicators, audit records, and project-specific mitigations,
see:
- [Project-specific mitigations and incident response](security_incidents/litellm-oct-2026.md)


Here are the LiteLLM ressources related to this security incident:

- [Upstream advisory](https://docs.litellm.ai/blog/security-incident-oct-21-2026)
- [Upstream investigation](https://docs.litellm.ai/blog/security-incident-oct-21-2026-attack-chain)


## NVIDIA backend image pinning

The example `VLLM_DOCKER_IMAGE` selects
`nvcr.io/nvidia/vllm:26.05.post1-py3@sha256:94e21552f644e0c1627464ba89d2f7a4ce7442e196f72afa0bb5d7fba23cbb03`.
This is the multi-platform manifest index, covering Linux amd64 and arm64.
The tag is descriptive; the digest determines the content Docker selects.
Both backend launches and the compatibility probe use this exact reference.

The digest was compared with NVIDIA's registry and the local image's
`RepoDigests`. To inspect the release without starting it:

```bash
docker buildx imagetools inspect nvcr.io/nvidia/vllm:26.05.post1-py3
docker image inspect nvcr.io/nvidia/vllm:26.05.post1-py3 --format '{{json .RepoDigests}}'
```

Docker verifies downloaded manifests and layers against their content digests;
no separate manual image-file hash or custom download verifier is needed.
Digest pinning prevents tag drift, but does not establish that the pinned
image is free of malware or vulnerabilities, or authenticate its publisher
independently of the registry. Review NVIDIA advisories and image provenance
before approving an updated digest, and retain host-driver compatibility.

Existing deployments must update `VLLM_DOCKER_IMAGE` explicitly; updating the
repository example alone does not change installed configuration.
Existing containers keep their original image when reused. When changing to
a different approved image, explicitly recreate the affected containers.
Selecting the digest of the already deployed image does not require restarting
them merely to record the pin.

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Report vulnerabilities privately via one of these channels:

- **GitHub Security Advisories**: [Report a vulnerability](https://github.com/Harinezumigel/harinezumigel-llm-stack/security/advisories/new)
- **Email**: Open a private contact via the repository owner's GitHub profile

### What to include

- A description of the vulnerability and its potential impact
- Steps to reproduce or a proof-of-concept
- The version or commit you tested against
- Any suggested fixes, if you have them

### What to expect

- Acknowledgement within a few days
- A fix or mitigation will be worked on as soon as possible
- You will be credited in the release notes (unless you prefer to remain anonymous)

## Scope

This tool manages local Docker containers and LiteLLM processes.
Security issues in scope include:

- Command injection via model names or configuration values
- Privilege escalation via `sudo` calls
- Unintended file modification or deletion
- Credential or API key exposure in logs or output

Issues in third-party dependencies (LiteLLM, vLLM, Docker) should be reported
to their respective upstream projects.
