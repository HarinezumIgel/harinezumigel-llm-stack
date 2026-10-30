# Security Policy

## Supported Versions

Only the latest release receives security fixes.

## Network Architecture & Security Isolation

`HarinezumIgel` uses a hybrid deployment model: **LiteLLM runs directly on the Docker host**, while **vLLM backends run inside isolated Docker containers** on the same machine.

This design ensures that only the LiteLLM proxy port is exposed to your network, keeping the internal vLLM backend ports strictly bound to loopback (`127.0.0.1`).

### Architecture Overview

```text
[ Client / Copilot ] (e.g., LAN / External)
         │
         ▼ (Port specified by LITELLM_PORT, e.g., 4000)
┌────────────────────────────────────────────────────────────┐
│ HOST SYSTEM                                                │
│                                                            │
│  [ LiteLLM Proxy ] (Binds to LITELLM_BIND_HOST)            │
│         │                                                  │
│         │ (Internal loopback routing: http://127.0.0.1)    │
│         ▼                                                  │
│  [ Docker Engine ]                                         │
│         │                                                  │
│         └──────► [ vLLM Container ] (Port 8000)            │
│                  (Binds ONLY to VLLM_BIND_HOST=127.0.0.1)  │
└────────────────────────────────────────────────────────────┘
```

### Security Rationale & Configuration Guidelines

1. **Expose Only LiteLLM:**
   Clients (such as IDE extensions or client applications) connect exclusively to the LiteLLM proxy (`LITELLM_BIND_HOST` and `LITELLM_PORT`). They never communicate directly with vLLM.

2. **Isolate vLLM Backends:**
   * `VLLM_BIND_HOST` and `VLLM_HOST` **should remain set to `127.0.0.1`** when both LiteLLM and Docker reside on the same target host.
   * This prevents backend vLLM ports from leaking onto your LAN or public network interfaces via Docker port publishing.
   * **Warning:** Do not pair a LAN-facing `VLLM_HOST` with a loopback `VLLM_BIND_HOST`, as this will result in connection-refused errors.

3. **Access Control:**
   * Bind `LITELLM_BIND_HOST` carefully: use `127.0.0.1` for local-only access, a specific LAN IP (e.g., `192.168.100.50`) for trusted network access, or `0.0.0.0` with extreme caution.
   * Always secure your LiteLLM instance with a strong master key (`LITELLM_MASTER_KEY`) when opening it up beyond localhost.

### Firewall Rules (`ufw`)

Since vLLM stays bound to `127.0.0.1`, you only need to manage firewall access for the LiteLLM proxy port (`4000` by default).

#### Standard UFW Setup

* **Allow access from a specific LAN IP (Recommended):**
  Restrict proxy access exclusively to trusted machines (e.g., your workstation at `192.168.100.10`):
  ```bash
  sudo ufw allow from 192.168.100.10 to any port 4000 proto tcp comment 'Allow LiteLLM from Dev Workstation'
  ```

* **Allow access from an entire LAN subnet:**
  ```bash
  sudo ufw allow from 192.168.100.0/24 to any port 4000 proto tcp comment 'Allow LiteLLM from LAN'
  ```

* **Allow access from anywhere (Use with caution):**
  Only recommended if LiteLLM is protected by a strong `LITELLM_MASTER_KEY` or additional authentication layers:
  ```bash
  sudo ufw allow 4000/tcp comment 'Allow public LiteLLM access'
  ```

#### Critical Security Note on Docker & UFW

> **❗ Docker iptables Bypass:**
> By default, Docker manages its own `iptables` rules and published container ports may bypass expected `ufw` filtering. As a result, a service exposed with a generic port mapping such as `-p 8000:8000` can become reachable even when `ufw` is configured to deny inbound access.
>
> This setup mitigates that risk by enforcing `VLLM_BIND_HOST=127.0.0.1` in `.env`. Docker therefore publishes all vLLM ports **exclusively on the loopback interface** (for example `127.0.0.1:8001->8000/tcp`), preventing external network access to the model endpoints through Docker's port publishing mechanism.
>
> Docker still creates its normal `iptables` rules, but because the published ports are bound only to `127.0.0.1`, they are accessible only from the local host and are not exposed on external network interfaces.
>
> Always ensure `VLLM_BIND_HOST` is explicitly set to `127.0.0.1` when containerizing backend models that should remain locally accessible only. For this deployment, where LiteLLM and vLLM run on the same host, `VLLM_HOST` is also intentionally set to `127.0.0.1`. `VLLM_BIND_HOST` restricts Docker port publication to the loopback interface, while `VLLM_HOST` ensures LiteLLM connects to the backend models through that same local interface. Changing one without considering the other may break backend connectivity or unintentionally alter the network exposure model.

> After deployment, verify with `docker ps` that all published ports appear as `127.0.0.1:<port>` rather than `0.0.0.0:<port>` or `<host-ip>:<port>`.

## LiteLLM October 2026 Incident

For detailed discussion of the LiteLLM October 2026 compromise,
forensic indicators, audit records, and project-specific mitigations,
see:
- [Project-specific mitigations and incident response](security_incidents/litellm-oct-2026.md)


Here are the LiteLLM resources related to this security incident:

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
