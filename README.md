# harinezumigel-llm-stack

![Hedgehog mascot](https://raw.githubusercontent.com/HarinezumIgel/RAG-LCC/main/docs/assets/AI_Igel.png)

Manage a host-based [LiteLLM](https://github.com/BerriAI/litellm) proxy and
[vLLM](https://github.com/vllm-project/vllm) backends in NVIDIA GPU-enabled Docker
containers through one Python CLI. Clients use an OpenAI-compatible API; the
launcher handles model selection, container reuse, startup monitoring, and logs.

This is not limited to NVIDIA DGX Spark: your host architecture, GPU, and driver
must support the selected NVIDIA NGC vLLM image.

## Components at a glance

```text
Client (Copilot, RAG app, ...)
               |
               | OpenAI-compatible HTTP API
               v
+---------------------------------------------------------+
| Target host: 192.168.100.50                             |
|                                                         |
| +-----------------------------------------------------+ |
| | LiteLLM proxy (host process, LAN port 4000)         | |
| +------------+----------------------------------------+ |
|              |                                          |
|              | Local backend API requests               |
|              v                                          |
| Docker host ports bound to 127.0.0.1                    |
|   8003 -> [vLLM: Guard]                                 |
|   8001 -> [vLLM: Mistral]                               |
|   8013 -> [vLLM: Judge]                                 |
|   8005 -> [vLLM: Coder]  (alternative workload)         |
|              |                                          |
|       NVIDIA GPU / shared memory                        |
|       Model files mounted from /opt/models              |
|                                                         |
| harinezumigel-llm-stack CLI                             |
|   reads config/env + config/config.yaml                 |
|   manages Docker backends and the host LiteLLM process  |
+---------------------------------------------------------+
```

The CLI is the control tool, not part of the inference request path. LiteLLM
routes requests to the selected backend model; it does not automatically run a
guard/inference/judge pipeline. That orchestration belongs to the client app.
The backends above illustrate configured choices, not models that must all run
together. Named groups select which services to manage.

## Documentation website

The GitHub Pages landing page is maintained in [docs/index.md](docs/index.md).
After pushing these files, open the repository's **Settings > Pages**, select
**Deploy from a branch**, choose **main** and **/docs**, then click **Save**.
Once the Pages deployment completes, the website is available at
<https://harinezumigel.github.io/harinezumigel-llm-stack/>.
The README remains the full installation and operating guide.

> ❗ **Important** Provided **as is**, without warranty. Read the [disclaimer](#disclaimer)
> before use and validate before production deployment.
> This project is not affiliated with LLMStack (llmstack.ai / Promptly).
> LiteLLM serves an admin UI at `/ui`; do not assume API authentication hides
> that route. Restrict network access or block it at a reverse proxy.

## Example: Develop local RAG pipeline and use local coding LLM

Suppose your RAG project needs three models:

| Role | CLI alias in the example config | Model |
|---|---|---|
| Prompt-check / safety | `guard` | Llama Guard 3 8B |
| Answer generation / inference | `mistral` | Mistral 7B Instruct v0.3 |
| Grounding and answer-quality judge | `judge` | Qwen2.5-14B-Instruct |
| Separate coding assistant | `coder` | Qwen3-Coder-Next-AWQ |

Your hardware can run the RAG trio **or** the coding model, but not both.
Use ordered `model_groups` to name these selections, then explicitly stop one
workload before starting the other. **Starting a group does not stop another
group or any unselected model.** Groups are selections, not resource schedulers,
mutually exclusive profiles, or GPU-priority settings.

For a taste of the workflow, give each workload a name:

```yaml
model_groups:
  rag: [mistral, guard, judge, litellm]
  coding: [coder, litellm]
```

Then create and start the RAG backend containers
together, and switch to coding when needed:

**`--start` creates a model's container if it does not exist, then starts it.**
If the container already exists, it is reused: stopped containers are started,
and running containers are left running.

```bash
# Preview creation, then create and start missing RAG backend containers.
# LiteLLM runs as a host process, not a container.
harinezumigel-llm-stack --group rag --start --dry-run
harinezumigel-llm-stack --group rag --start

# After stopping client traffic, swap the RAG workload for the coding model.
# The coding backend container is created on its first start.
harinezumigel-llm-stack --group rag --stop &&
  harinezumigel-llm-stack --group coding --start
```

These commands assume installation, downloaded model files, and configured
groups. The [installation guide](#install-from-any-checkout-location) gets you
started; the [config.yaml guide](#configyaml-guide-models-groups-and-runtime-settings)
explains models, groups, and runtime settings, while the
[env guide](#env-guide-urls-ports-secrets-and-host-settings) explains URLs, ports,
secrets, and host settings. The
[running and switching workloads](#run-and-switch-workloads) covers status,
switching back, and the shared proxy. The
[Mistral 7B example](#a-realistic-mistral-entry) illustrates how a group member
is configured; it is not an instruction to edit or replace that model's entry.

> ❗ **Important**
> The launcher does not download models. Download the **complete model files**
> to `/opt/models` first, including weights, tokenizer, and configuration files.

Use the directory names in the
[model-files example](#env-guide-urls-ports-secrets-and-host-settings).

## Install from any checkout location

### Requirements

- Unix host, Python 3.10+, and [PyYAML](https://pyyaml.org/).
- Docker with NVIDIA Container Toolkit and working GPU passthrough
  (`--gpus all`); `nvidia-smi` available on the host.
- An NVIDIA GPU, driver, and CPU architecture compatible with your chosen
  `VLLM_DOCKER_IMAGE`. NGC access / registry authentication may be required.
- Complete model files downloaded under `MODEL_ROOT`, with any upstream access
  requirements and licenses satisfied.
- LiteLLM installed in its own virtual environment (the installer sets this up).

Choose any checkout directory; no specific checkout location
is required:

```bash
git clone https://github.com/Harinezumigel/harinezumigel-llm-stack.git "$HOME/harinezumigel-llm-stack"
cd "$HOME/harinezumigel-llm-stack"
./install.sh
```

If already cloned elsewhere, run `./install.sh` from that repository root.
All repository-relative commands below assume that root, whatever its location.
The installer copies the launcher and runtime helper modules into
`/opt/litellm/src` and links `~/.local/bin/harinezumigel-llm-stack` to
`/opt/litellm/src/harinezumigel-llm-stack.py`. The installed command does not
depend on the checkout location. Rerun the installer to update the installed
runtime after updating the repository.

The `install` command used below is a standard Linux file-copying tool, not a
package manager. `install -m 644` copies files with owner read/write and
everyone else read-only permissions. The launcher is then made executable
with `chmod +x`; its helper modules only need to be readable.

The interactive installer checks prerequisites, installs PyYAML, creates
`/opt/litellm`, offers to copy the configuration examples, and installs/upgrades
`litellm[proxy]` in `/opt/litellm/venv` with a final `pip check`.
Add `export PATH="$HOME/.local/bin:$PATH"` to your shell profile if needed.

Deployment files are separate from the checkout:

| Path | Purpose |
|---|---|
| `/opt/litellm/src` | Installed launcher and runtime helper modules |
| `/opt/litellm/config/env` | Paths, image, bind addresses, ports, keys |
| `/opt/litellm/config/config.yaml` | Models, aliases, groups, LiteLLM settings |
| `/opt/litellm/venv` | LiteLLM proxy virtual environment |
| `/opt/litellm/logs` | Timestamped proxy logs |
| `/opt/litellm/old_configs` | Timestamped configuration backups |

Missing or empty configuration files are initialized from the current examples.
Before replacing existing configuration files, the installer makes uniquely
named backups with owner-only permissions and reports their exact paths at the
end. On a fresh install, it reports that no previous configuration files needed
backing up. Legacy `config/.env`, root-level `env` / `.env`, and root-level
`config.yaml` are backed up rather than migrated as active configuration; the
originals remain in place. Review the backups and manually carry over any
host-specific values into the current examples.
Replacing an existing virtual environment retains it under
`/opt/litellm/venv_backup_*/venv`. Declining both reuse and reinstallation aborts.

### Manual alternative

From your checkout root, on Debian/Ubuntu:

```bash
sudo apt install python3-yaml python3-venv
sudo mkdir -p /opt/litellm
sudo chown "$USER:$(id -gn)" /opt/litellm
mkdir -p /opt/litellm/config
# Fresh installation only: do not overwrite existing deployment settings.
cp config/env.example /opt/litellm/config/env
cp config/config.yaml.example /opt/litellm/config/config.yaml
python3 -m venv /opt/litellm/venv
/opt/litellm/venv/bin/pip install 'litellm[proxy]'
/opt/litellm/venv/bin/pip check
mkdir -p "$HOME/.local/bin"
mkdir -p /opt/litellm/src
install -m 644 src/harinezumigel-llm-stack.py src/hls_*.py /opt/litellm/src/
chmod +x /opt/litellm/src/harinezumigel-llm-stack.py
ln -s /opt/litellm/src/harinezumigel-llm-stack.py "$HOME/.local/bin/harinezumigel-llm-stack"
export PATH="$HOME/.local/bin:$PATH"
```

On Fedora/RHEL, PyYAML is available as `python3-pyyaml`; alternatively install
PyYAML in the Python environment used to run the launcher. The launcher and
LiteLLM proxy can use different Python environments.

## Configure the deployment

Start with [config/env.example](config/env.example) and
[config/config.yaml.example](config/config.yaml.example), copied to the paths
above. Review them before starting anything; the catalog includes alternative
models, not a recommendation to run them all.

- **`config.yaml` defines the models:** their names, aliases, groups, and runtime
  settings. See the [config.yaml guide](#configyaml-guide-models-groups-and-runtime-settings).
- **`env` defines deployment values:** URLs, ports, secrets, host paths, and
  Docker settings. See the [env guide](#env-guide-urls-ports-secrets-and-host-settings).
  Model entries reference these environment variables rather than embedding secrets.

### env guide: URLs, ports, secrets, and host settings

Edit `/opt/litellm/config/env`:

- Set `MODEL_ROOT` to the parent directory containing your model directories
  and `VLLM_MODEL_VOLUME` to mount
  that same directory at `/models` (example: `/opt/models:/models`).
  Set `VLLM_CACHE_VOLUME` to a persistent host cache.
- Check `LITELLM_BIN` and `LITELLM_VENV_ACTIVATE` point to your proxy environment.
- Choose a `VLLM_DOCKER_IMAGE` compatible with your host; check `nvidia-smi`
  and the image's requirements rather than assuming the example tag fits.
- Replace **both** placeholder keys: `LITELLM_MASTER_KEY` authenticates clients
  to the proxy; `LOCAL_VLLM_API_KEY` authenticates proxy-to-backend requests.
- The example exposes LiteLLM on the target host's LAN IP and keeps vLLM
  backends private on loopback. Replace the example LAN IP with your host's IP.
  For host-local clients only, also set `LITELLM_BIND_HOST=127.0.0.1`.

**Example: Copilot connects over the LAN; LiteLLM and vLLM run on the same
target host (`192.168.100.50`).**

```dotenv
LITELLM_BIND_HOST=192.168.100.50
LITELLM_PORT=4000
VLLM_HOST=127.0.0.1
VLLM_BIND_HOST=127.0.0.1

MISTRAL_7B_PORT=8001
MISTRAL_7B_API_BASE=http://${VLLM_HOST}:${MISTRAL_7B_PORT}/v1
```

Configure Copilot's API base URL as `http://192.168.100.50:4000/v1` and use
your `LITELLM_MASTER_KEY` for authentication. The request path is:

```text
Copilot -> 192.168.100.50:4000 (LiteLLM) -> 127.0.0.1:8001 (Mistral)
                                      -> 127.0.0.1:8003 (Guard)
```

`LITELLM_BIND_HOST` is the supported setting; **`LITELLM_HOST` is not used**.
Binding LiteLLM to its LAN IP is sufficient for LAN clients; `0.0.0.0` is not
required. Allow only intended clients through the firewall, and restrict access
to LiteLLM's `/ui` admin route.

**Bind address and routing address serve different purposes:**

- `VLLM_BIND_HOST` selects the host interface on which Docker publishes backend
  ports. `VLLM_HOST` supplies the address LiteLLM connects to through the
  `*_API_BASE` URLs.
- If only the host-based LiteLLM proxy needs backend access, prefer
  `VLLM_BIND_HOST=127.0.0.1` and `VLLM_HOST=127.0.0.1`. Clients can access the
  proxy without direct network access to the backends.
- If direct backend access is needed on a specific network interface, use that
  interface's host IP for both values, for example:

  ```bash
  VLLM_BIND_HOST=192.168.100.50
  VLLM_HOST=192.168.100.50
  ```

  This limits the bind to that address, rather than all interfaces, but still
  permits network access through it subject to firewall rules and authentication.
  Do not leave `VLLM_HOST=localhost` with this restricted bind: loopback is a
  different address, so LiteLLM cannot reach the published backend port there.
- `VLLM_BIND_HOST=0.0.0.0` listens on all IPv4 interfaces, including loopback;
  it is not needed for the same-host setup above. Restrict the bind to the
  interface actually needed instead of exposing every interface.

Changing `VLLM_BIND_HOST` requires recreating affected backend containers.
Changing `LITELLM_BIND_HOST`, `VLLM_HOST`, or the backend URLs requires restarting
LiteLLM. Existing containers retain their original bind until recreated.
If your backends already bind to `127.0.0.1` and only `VLLM_HOST` was wrong,
correct it and restart only LiteLLM; no backend recreation is needed:

```bash
harinezumigel-llm-stack litellm --stop &&
  harinezumigel-llm-stack litellm --start
```

Each backend needs a distinct port and a matching `*_API_BASE`. The example uses
8003 for Guard, 8013 for the 14B judge, and 8005 for `coder`; LiteLLM uses 4000.
The launcher expands env references. Whitespace-separated inline comments are
supported; quote values containing a literal whitespace-separated `#`.

Every model needs an explicit `model_info.model_dir`, matching a directory
under `MODEL_ROOT`, for example:

```text
/opt/models/
├── mistral-7b-instruct-v0.3/
├── llama-guard3-8b/
├── Qwen2.5-14B-Instruct/
└── Qwen3-Coder-Next-AWQ/
```

Download each model's weights, tokenizer, and configuration files from its
upstream repository into the matching directory under `/opt/models` (the
example `MODEL_ROOT`).

> ❗ **Important**
> Complete any upstream access or license acceptance requirements before downloading.

If you choose a different root, update both
`MODEL_ROOT` and `VLLM_MODEL_VOLUME` to match.

For example, download Mistral into the directory used by its supplied entry:

```bash
hf download mistralai/Mistral-7B-Instruct-v0.3 \
  --local-dir /opt/models/mistral-7b-instruct-v0.3
```

> ❗ **Important**
> The launcher does not download model files. Missing model directories block
> startup. Keep model files and other persistent data on mounted host storage, never
> only inside containers.

Use `--local-dir` to download into the configured model directory. Without it,
`hf download` normally stores files in the Hugging Face cache instead.


The deployment base is currently fixed as `/opt/litellm` in the launcher and
must exist even for help/list operations. Deployment settings are read from
`/opt/litellm/config/env`, model configuration from
`/opt/litellm/config/config.yaml`, and logs are stored in `/opt/litellm/logs`.
`LITELLM_CONFIG` is not used.

### config.yaml guide: models, groups, and runtime settings

Use `/opt/litellm/config/config.yaml` to define model entries, reusable groups,
and proxy settings. The following sections explain these parts together.

#### A realistic Mistral entry

This shortened example illustrates the supplied Mistral 7B profile and how its
`mistral` alias connects it to the `rag` group. It is for reference: **do not
replace or edit the existing `mistral_7b` entry for this walkthrough**, and do not
add a duplicate model or a second `model_list` key. The Guard, judge, and coder
entries also remain as supplied.

```yaml
model_list:
  - model_name: mistral_7b
    litellm_params:
      litellm_provider: openai
      model: openai/mistral_7b
      api_base: os.environ/MISTRAL_7B_API_BASE
      api_key: os.environ/LOCAL_VLLM_API_KEY
      request_timeout: 90
    rebuild: true
    model_info:
      alias: mistral
      model_dir: mistral-7b-instruct-v0.3
      generation_config: vllm
      override_generation_config:
        temperature: 0.7
        top_p: 0.9
      context_length: 32768
      max_input_tokens: 28000
      max_output_tokens: 2048
      max_tokens: 2048
      gpu_memory_utilization: 0.27
      ipc: host
      dtype: auto
      max_num_seqs: 1
      max_num_batched_tokens: 4096
      enforce_eager: false
      enable_prefix_caching: true
      description: "RAG answer generation with Mistral 7B"
      license: "See upstream model license"
      upstream: "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3"
```

These are hardware-dependent starting values, not universal sizing rules.
The example RAG fractions are Mistral 0.27 + Guard 0.16 + judge 0.28 = 0.71;
`coder` uses 0.90. Fractions are GPU memory budgets, not compute limits.
Weights, KV cache, context size, concurrency, and unified-memory behavior all
affect whether the models actually fit.

Keep proxy authentication configured at the YAML top level:

```yaml
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
```

**Names matter:** `model_info.alias` is a case-sensitive CLI shortcut and must
be unique. It does not rename an API model. With this entry, vLLM serves
`mistral_7b`, LiteLLM routes `openai/mistral_7b`, and API clients request
`"model": "mistral_7b"` regardless of which CLI name started it.

#### Define reusable groups

Replace the top-level `model_groups` mapping in your installed YAML with this
mapping, alongside `model_list`. The supplied example already has `rag` and
`quick`; **add `coding` before using the switching commands**:

```yaml
model_groups:
  rag:
    - mistral
    - guard
    - judge
    - litellm
  quick:
    - mistral
    - guard
  coding:
    - coder
    - litellm
```

Members are configured model names, aliases, or LiteLLM targets. Nested groups
and `all` are not allowed. Selection is ordered, with duplicates removed in
first-seen order. Positional models and `--group` cannot be combined.
Putting LiteLLM last starts it after backend readiness; stop operations also
follow the declared order.

#### Additional configuration keys

All model entries require `context_length`, `max_input_tokens`,
`max_output_tokens`, and `gpu_memory_utilization` under `model_info`, plus
`model_dir` for startup. `rebuild: true` belongs at the **model entry level**,
not inside `model_info`, and permits explicit recreation.

| Optional `model_info` fields | Purpose |
|---|---|
| `alias`, `description`, `detail`, `license`, `upstream` | CLI naming and metadata |
| `dtype`, `quantization`, `kv_cache_dtype` | Weight / cache precision; match the model and image |
| `max_num_seqs`, `max_num_batched_tokens` | Concurrency and batching |
| `ipc`, `enforce_eager`, `enable_prefix_caching`, `enable_chunked_prefill` | Docker IPC and vLLM execution settings |
| `generation_config` | Generation config source; defaults to `vllm`, avoiding inherited model defaults |
| `override_generation_config` | Backend generation defaults, forwarded as JSON to vLLM |
| `attention_backend`, `attention_config` | Attention implementation settings |
| `compilation_config`, `structured_outputs_config`, `speculative_config` | Version-dependent vLLM config objects or JSON strings |
| `enable_auto_tool_choice`, `tool_call_parser`, `reasoning_parser`, `reasoning_config` | Tool calling and reasoning parsing |

Config objects are serialized to JSON for vLLM. Check the selected image's
schema: the launcher does not validate every forwarded setting, and values
such as `compilation_config.mode: max-autotune` are not supported everywhere.
A parser nested in `reasoning_config` is moved to `--reasoning-parser`;
conflicting parsers and legacy `enable_reasoning` / `enable_thinking` keys
there are rejected.

Set backend generation defaults in `model_info.override_generation_config`,
a YAML mapping forwarded as JSON to vLLM's `--override-generation-config`:

```yaml
model_info:
  generation_config: vllm
  override_generation_config:
    temperature: 0.7
    top_p: 0.9
    repetition_penalty: 1.03
```

These settings are **active defaults used by vLLM**. Set `generation_config`
under the model's `model_info` in `/opt/litellm/config/config.yaml`, alongside
`override_generation_config`, as shown above. `auto` and `vllm` are literal
values defined by vLLM, not names of models or groups. The launcher passes
this setting as `--generation-config`; if omitted, the launcher explicitly
passes `vllm` (rather than using vLLM's own `auto` default).
It determines the starting point:

- `auto` (explicit opt-in): load the model's generation configuration, then replace
  the specified values with those in `override_generation_config`.
- `vllm` (the launcher default): skip the model's generation configuration and
  apply your overrides to vLLM's built-in defaults.

Use generation-config keys supported by the vLLM version in your container
image. The launcher forwards the mapping rather than filtering unsupported
keys. To apply changed defaults to an existing container, use
`--start --recreate` (requires `rebuild: true`).

Keep deployment sampling defaults in `model_info.override_generation_config`,
not duplicated in `litellm_params`. The launcher rejects a parameter defined in
both locations, even when the values are identical. Existing proxy-only defaults
in `litellm_params` remain supported and are handled by LiteLLM. The sampling
summary labels the two sources as `backend default` and `proxy default`;
these are configured defaults, not necessarily the values used by each request.

Explicit request sampling values take precedence over backend sampling defaults.
For example, a VS Code client sending `temperature: 0.2` overrides a backend
default of `0.7`, provided the client sends it and LiteLLM forwards it.
Server-wide limits still apply; for example,
`max_new_tokens` in the generation configuration can cap output length.

**Migrating an existing configuration:** move sampling defaults such as
`temperature`, `top_p`, and penalties from `litellm_params` into
`model_info.override_generation_config`, resolving conflicting values explicitly.
Keep routing, authentication, and proxy settings in `litellm_params`. Set
`generation_config: auto` explicitly if you want to retain inherited model
generation defaults; otherwise the launcher now uses `vllm`. Recreate backend
containers to apply the new startup settings, and restart LiteLLM after changing
its configuration.

## Run and switch workloads

After installing, downloading model files, and configuring the groups above:

```bash
harinezumigel-llm-stack --list                 # aliases and groups
harinezumigel-llm-stack                        # detailed model settings
# Preview container creation without launching model backends.
harinezumigel-llm-stack --group rag --start --dry-run
# Create and start missing backend containers; reuse any existing ones.
harinezumigel-llm-stack --group rag --start
harinezumigel-llm-stack --group rag --ps
```

No separate `docker create` command is needed: `--start` creates missing vLLM
containers from the configured image, mounts the model files, and starts them.
LiteLLM is launched separately as a host process. To explicitly replace existing
backend containers after changing their configuration, preview and then recreate:

```bash
harinezumigel-llm-stack --group rag --start --recreate --dry-run
harinezumigel-llm-stack --group rag --start --recreate
```

Recreation requires `rebuild: true` on each selected vLLM entry and removes
container-local data; mounted model files remain on the host.

Before switching, stop client traffic. Stop the old group **successfully**
before starting the new one:

```bash
# RAG -> coding (also stops/restarts the shared proxy)
harinezumigel-llm-stack --group rag --stop &&
  harinezumigel-llm-stack --group coding --start

# Coding -> RAG
harinezumigel-llm-stack --group coding --stop &&
  harinezumigel-llm-stack --group rag --start
```

This intentionally includes `litellm` in both groups, so switching restarts the
proxy as well as changing GPU workloads. If you want the proxy to stay up,
omit it from both groups and manage it separately with `litellm --start/--stop`.
LiteLLM can list configured models whose backends are currently stopped;
requests to those backends will fail.

If you have other models running, stopping `rag` alone will not free their
memory. Inspect `all --ps`, or deliberately stop the whole configured stack:

```bash
harinezumigel-llm-stack all --stop --dry-run
harinezumigel-llm-stack all --stop
```

`all --stop` stops LiteLLM first, then configured vLLM models.

### Startup, reuse, and failures

- Grouped `--start --recreate` validates the entire selection first, then stops
  all running vLLM containers belonging to the selected models (including old
  port variants) before checking the combined new memory budget and starting
  replacements sequentially. Containers outside the selection are untouched.
  If the memory check fails, selected backends remain stopped. LiteLLM is
  restarted only if included in the selection.
- Batch startup validates selected names, directories, runtime settings, ports,
  and recreation permissions before launching any selected vLLM container.
  It starts targets sequentially and waits for each backend's HTTP `/health`
  to return 200. Docker's `running` state alone is not readiness.
- Already healthy reused backends do not replay old logs. LiteLLM starts
  detached, without the vLLM readiness monitor.
- `--startup-timeout SECONDS` sets a per-backend deadline (default 900).
  `--quiet` suppresses routine startup logs, not progress or failure diagnostics.
- Failure or timeout stops the sequence with a nonzero exit and diagnostics.
  Earlier starts remain running; later targets are not attempted. Ctrl+C
  stops monitoring/further starts and leaves backends running (exit 130).
- Backend ports and vLLM runtime settings are configured in `config.yaml`;
  the CLI does not override them. Request-level sampling parameters remain
  available to API clients.
- Ordinary starts reuse running containers or start stopped containers with
  their **original Docker arguments**. To apply changed image, mount, context,
  or other startup settings, use `--start --recreate`. Every selected vLLM
  entry must permit this via top-level `rebuild: true`.
- Grouped starts reuse a LiteLLM process matching the config path without
  reloading it. Grouped `--start --recreate` restarts LiteLLM in group order;
  this does not require a `rebuild` setting for the proxy.
  Standalone `litellm --start` instead refuses a matching existing instance;
  stop then start it to reload configuration. An occupied port is not proof
  of a reusable proxy.

❗ **Important: Dry-run is not offline**

- It avoids model launches/recreation and readiness waiting, but still inspects
  Docker/GPU state.
- For backends needing launch, it checks available memory (dedicated GPU VRAM
  or GB10 shared system memory) and runs a short-lived GPU/image compatibility
  container.
- If missing locally, Docker pulls the vLLM server image specified by
  `VLLM_DOCKER_IMAGE` in `/opt/litellm/config/env`
  (`nvcr.io/nvidia/vllm:26.05.post1-py3` in the example configuration).
  This is the container image, not the model files; an existing local image is
  not updated by this check.
- Confirmed driver-version mismatches abort startup and dry runs; other
  compatibility-probe errors are warnings.
- Available memory must cover the configured GPU 0 memory budget; batch planning
  can also check the combined budget for new starts. Insufficient memory or
  unreadable memory data abort both startup and dry-run.
- On NVIDIA GB10, unavailable GPU memory counters use `/proc/meminfo`
  `MemAvailable` and `MemTotal` instead, matching DGX Dashboard's shared system
  memory counters. Output explicitly identifies shared system memory; CPU and
  GPU workloads compete for this pool. Other GPUs do not use this fallback.
- These checks do not guarantee successful loading.
- During grouped recreation dry-runs, selected containers remain running.
  If any selected backend is running, the memory preview checks the new budget
  against total capacity only and warns that post-stop available memory cannot
  be verified. The real start checks availability after stopping selected
  backends; individual checks still run before each replacement launch.

## CLI reference

Use `harinezumigel-llm-stack --help` for the full command help. A model's full
name and its alias are interchangeable in CLI selections.

| Task | Command |
|---|---|
| Select several backends without a group | `harinezumigel-llm-stack mistral guard judge --start` |
| Wait for one backend with bounded startup logs | `harinezumigel-llm-stack mistral --start --startup-logs` |
| Use a different readiness deadline | `harinezumigel-llm-stack --group rag --start --startup-timeout 600` |
| Preview selected recreation | `harinezumigel-llm-stack --group rag --start --recreate --dry-run` |
| Container status for configured models | `harinezumigel-llm-stack all --ps` |
| Recent grouped log snapshots | `harinezumigel-llm-stack --group rag --show-log --tail 50` |
| Follow one backend | `harinezumigel-llm-stack mistral --stream-log` |
| Follow proxy logs | `harinezumigel-llm-stack litellm --show-log --follow` |
| Locate Docker and proxy logs | `harinezumigel-llm-stack all --show-log-path` |
| Clean proxy log files | `harinezumigel-llm-stack litellm --clean-log` |

`--stream-log` is shorthand for `--show-log --follow`; do not combine them.
Single-model starts without log streaming print a reminder to add `--stream-log`,
plus a command to follow the running model's logs. After group/batch startup,
the launcher prints separate `<model> --stream-log` commands; streaming a whole
group indefinitely is not supported. Dry-runs do not print these reminders.
Batch selections reject indefinite following; follow one target separately
after startup. `all --show-log` shows log paths rather than merged streams.
JSON-escaped stack traces are expanded for readability.

Proxy logs are timestamped under `/opt/litellm/logs`; this directory is created
when needed, except during dry runs or cleanup. Proxy cleanup removes only
`.log` files inside that directory, without following symlinks. Backend/all
`--clean-log` can truncate Docker logs: inspect paths and preview with
`--dry-run` before deleting diagnostics.

### Clean logs

Use `--clean-log` with a model name or alias, a group, `litellm`, or `all`:

```bash
# Preview cleanup for the selected group without changing any files.
harinezumigel-llm-stack --group rag --clean-log --dry-run

# Clean only the selected group's logs.
harinezumigel-llm-stack --group rag --clean-log

# Other selections:
harinezumigel-llm-stack guard --clean-log
harinezumigel-llm-stack litellm --clean-log
harinezumigel-llm-stack all --clean-log
```

- **vLLM backends:** truncate the managed Docker log files; containers are not
  stopped, removed, or recreated.
- **LiteLLM:** remove managed `.log` files under `/opt/litellm/logs`, without
  following symlinks. If the directory exists but contains no managed log files,
  cleanup reports `Status: ✓ Cleaned`. A missing directory is left absent and
  reported as skipped.
- **Groups:** clean only the selected members' logs. Proxy logs are included
  only if `litellm` is a member of the group.
- **`all`:** clean configured vLLM backend logs and LiteLLM proxy logs.

Cleanup deletes diagnostic history, not model weights or configuration. Services
are not restarted; running backends can immediately write new log entries.
Preview with `--dry-run` and save any logs you need before cleaning.

### Call the proxy

With the RAG group running, use the target host's LAN IP from the example above
(replace it if yours differs) and replace the placeholder with your proxy master key:

```bash
curl http://192.168.100.50:4000/v1/chat/completions \
  -H 'Authorization: Bearer YOUR_LITELLM_MASTER_KEY' \
  -H 'Content-Type: application/json' \
  -d '{"model":"mistral_7b","messages":[{"role":"user","content":"Explain retrieval-augmented generation briefly."}],"max_tokens":128}'
```

This calls Mistral directly through the proxy, not the complete RAG workflow.

## Troubleshooting and safety

| Symptom | Check / action |
|---|---|
| CLI fails before showing help | Ensure `/opt/litellm`, `config/env`, and required env settings exist; use the installer. |
| Command or imports not found | Check `~/.local/bin` is on `PATH` and its symlink points to `/opt/litellm/src/harinezumigel-llm-stack.py`, with the `hls_*.py` helpers alongside it. |
| Model directory missing | Match `model_info.model_dir` exactly to a model directory under `MODEL_ROOT`; verify the `/models` mount. |
| Backend/proxy port occupied | Inspect `ss -ltn` and container/process status. Fix conflicting ports/URLs; do not assume another service is reusable. |
| Driver compatibility failure | Compare `nvidia-smi` with `VLLM_DOCKER_IMAGE` requirements. Choose a compatible image or upgrade the driver, then explicitly recreate affected containers. |
| Insufficient VRAM / OOM | Stop the other workload first. Reduce context, concurrency, or memory budget only as appropriate; too small a budget cannot hold the weights/cache. Recreate to apply changes. |
| Request authentication failure | Use `LITELLM_MASTER_KEY` for clients; verify `litellm_params.api_key` references `LOCAL_VLLM_API_KEY` for backends. Recreate/restart affected services after key changes. |
| API model not found | Use YAML `model_name`, not the CLI alias; keep `litellm_params.model` consistent with the backend's served name. |
| Configuration changes ignored | Recreate vLLM containers; stop/start LiteLLM or recreate its group to reload the proxy config. |
| Reused proxy venv has conflicts | Run `/opt/litellm/venv/bin/pip check`; rerun the installer and choose reinstallation if needed. |

For startup failures, inspect the relevant model's `--show-log --tail 100`
and preview `--start --recreate --dry-run` before changing the deployment.

- Container reuse is the default. There is no standalone container-removal
  option; removal happens implicitly during `--start --recreate`, which
  requires `rebuild: true` on each selected vLLM model. `--stop` leaves
  containers in place. Unselected models are not recreated.
  Container-local data is lost on recreation.
- Single-model, group, and `all` stop commands verify that selected backend
  containers are no longer running. LiteLLM stops wait for the matching proxy
  processes to exit after SIGTERM. Verification waits up to 10 seconds after
  the stop command/signal and exits with an error if anything selected remains
  running or its state cannot be queried. Dry-runs do not stop or wait.
- Model lifecycle operations target configured `vllm-` containers; LiteLLM
  process operations match its configuration path. Review selections before
  broad operations such as `all --stop` or log cleanup.
- Authentication is LiteLLM-controlled, not independently enforced by this
  launcher. Use strong keys, restrict config/backup permissions, firewall
  backend ports, and restrict the proxy and `/ui` appropriately.
- Keep debug logging off for sensitive workloads; do not publish raw logs,
  credentials, or request data. See [SECURITY.md](SECURITY.md) for scope and
  vulnerability reporting.

### Terminal output

Headings use bold, commands/aliases/groups use accent colors, and readiness,
warnings, and failures use status colors. No background colors are set; raw
backend logs are unchanged. Styling is disabled for redirected output,
unset/`dumb` `TERM`, or any `NO_COLOR` value (even empty).

Set `HLS_SUCCESS_COLOR`, `HLS_FAILURE_COLOR`, `HLS_WARNING_COLOR`,
`HLS_ACCENT_COLOR`, and `HLS_HEADING_COLOR` in the deployment env. Defaults are
`green`, `red`, `yellow`, `cyan`, and `default`. Accepted colors are the eight
standard terminal names, their `bright-` variants, and `default`. For
low-contrast palettes, use `NO_COLOR=1`.

## Development and project information

The launcher entry point and lifecycle logic live in
`src/harinezumigel-llm-stack.py`; `hls_selection.py`, `hls_startup.py`,
`hls_preflight.py`, and `hls_terminal.py` separate selection, monitoring,
GPU/image checks, and terminal presentation.

From your existing checkout root:

```bash
python3 -m venv .venv
.venv/bin/pip install pyyaml pytest
.venv/bin/python -m pytest tests/ -q
```

Tests mock Docker/GPU operations; startup tests also use local subprocesses to
verify readiness and log-follower cleanup. Their bootstrap uses
`config/env.example`, without requiring an installed deployment. Standard
library discovery also works: `.venv/bin/python -m unittest discover -s tests`.
The development `.venv` is separate from the proxy's `/opt/litellm/venv`.

For local development, `python3 src/fix_settings.py` applies its site-specific
env settings (with a private backup) and switches the user command symlink to
the launcher in that repository's `src/` directory. Review its settings before
running it; this helper is not part of the installed runtime. In this mode,
keep the checkout in place. Rerunning `./install.sh` restores the normal link
to `/opt/litellm/src/harinezumigel-llm-stack.py`.

See [CONTRIBUTING.md](CONTRIBUTING.md) for contributions,
[CHANGELOG.md](CHANGELOG.md) for changes, and
[GitHub issues](https://github.com/Harinezumigel/harinezumigel-llm-stack/issues)
for problems or questions.

### License and upstream projects

This repository is [MIT licensed](LICENSE). Referenced model weights have
their **own upstream licenses, access conditions, and usage restrictions**;
the repository license does not grant rights to those weights.

Built with [PyYAML](https://github.com/yaml/pyyaml),
[LiteLLM](https://github.com/BerriAI/litellm),
[vLLM](https://github.com/vllm-project/vllm), and
[Docker](https://www.docker.com/).

## Disclaimer

This software is provided **as is**, without warranty of any kind, as described
in the [MIT license](LICENSE). You are responsible for reviewing configuration,
testing your deployment, and deciding whether it is suitable for your use.

- **Operational risk:** Starting models consumes GPU memory, system memory,
  disk space, power, and network bandwidth. Resource checks do not guarantee
  successful loading, stability, or performance.
- **Data and availability:** Stopping services interrupts requests. Recreating
  containers removes container-local data, and log cleanup removes diagnostic
  history. Back up important data and review commands before running them.
- **Security and privacy:** You are responsible for authentication, network
  exposure, access controls, and protecting prompts, responses, credentials,
  and logs. Do not assume the proxy's API authentication protects its admin UI.
- **Model behavior:** Outputs may be inaccurate, unsafe, or unsuitable for your
  application. Guard and judge models do not guarantee safety or correctness;
  evaluate them independently and use appropriate human oversight.
- **Third-party terms:** Model licenses, provider terms, and applicable laws
  remain your responsibility. This project is not affiliated with LLMStack
  (llmstack.ai / Promptly), and naming upstream projects does not imply their
  endorsement.

To the extent permitted by applicable law, the authors and contributors are
not liable for damages arising from use of this software. See the
[license](LICENSE) for the governing warranty and liability terms.
