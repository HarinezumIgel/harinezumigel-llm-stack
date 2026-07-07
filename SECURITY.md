# Security Policy

## Supported Versions

Only the latest release receives security fixes.

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
