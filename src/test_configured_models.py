#!/usr/bin/env python3
"""Preview or run sequential model smoke tests through the local LiteLLM proxy."""

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, cast
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from hls_terminal import terminal_style


def _load_launcher() -> Any:
    path = Path(__file__).resolve().with_name("harinezumigel-llm-stack.py")
    spec = importlib.util.spec_from_file_location("model_check_launcher", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load launcher: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _missing_files(directory: Path) -> str | None:
    if not directory.is_dir():
        return "model directory missing"
    if not (directory / "config.json").is_file():
        return "config.json missing"
    if not any(directory.glob("tokenizer*")):
        return "tokenizer files missing"
    indexes = list(directory.glob("*.index.json"))
    for index in indexes:
        data = json.loads(index.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"Invalid weight index: {index}")
        mapping = cast(dict[str, Any], data).get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"Missing weight_map in {index}")
        names = cast(dict[str, Any], mapping).values()
        for name in names:
            if not isinstance(name, str):
                raise ValueError(f"Invalid shard name in {index}")
            shard = (directory / name).resolve()
            if not shard.is_relative_to(directory.resolve()):
                raise ValueError(f"Shard outside model directory in {index}")
            if not shard.is_file() or shard.stat().st_size == 0:
                return f"weight shard missing or empty: {name}"
    weights = list(directory.glob("*.safetensors")) + list(
        directory.glob("pytorch_model*.bin")
    )
    if not weights or any(path.stat().st_size == 0 for path in weights):
        return "weight files missing or empty"
    return None


def _running_backends() -> list[str]:
    result = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [name for name in result.stdout.splitlines() if name.startswith("vllm-")]


def _request(url: str, key: str, payload: dict[str, Any] | None = None) -> Any:
    data = json.dumps(payload).encode() if payload is not None else None
    request = Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )
    with build_opener(ProxyHandler({})).open(request, timeout=120) as response:
        return json.load(response)


def _wait_for_proxy(base: str, key: str, timeout: float) -> None:
    """Wait for LiteLLM's model-list endpoint to respond after startup."""
    deadline = time.monotonic() + timeout
    last_error: OSError | ValueError | None = None
    while True:
        try:
            _request(f"{base}/v1/models", key)
            return
        except (OSError, ValueError) as error:
            last_error = error
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(
                f"LiteLLM did not become ready within {timeout:g} seconds: "
                f"{last_error}"
            ) from last_error
        time.sleep(min(1, remaining))


def _run_launcher(model: str, *options: str) -> None:
    subprocess.run(["harinezumigel-llm-stack", model, *options], check=True)


def _normalize_model_selections(selections: list[str] | None) -> list[str] | None:
    if selections is None:
        return None
    names = [
        name.strip()
        for selection in selections
        for name in selection.split(",")
        if name.strip()
    ]
    return names or None


def _test_model(base: str, key: str, model: Any, max_tokens: int) -> str:
    """Call the endpoint appropriate for this model and validate its response."""
    mode = str(model.model_info.get("mode", "inference")).lower()
    if mode == "embedding":
        response = _request(
            f"{base}/v1/embeddings",
            key,
            {"model": model.name, "input": "A short model availability check."},
        )
        raw_embedding = response["data"][0]["embedding"]
        if not isinstance(raw_embedding, list) or not raw_embedding:
            raise ValueError("Model returned no embedding values")
        embedding = cast(list[Any], raw_embedding)
        return f"embedding dimensions={len(embedding)}"

    if mode == "rerank":
        response = _request(
            f"{base}/rerank",
            key,
            {
                "model": model.name,
                "query": "A short model availability check.",
                "documents": [
                    "A short model availability check.",
                    "An unrelated document about gardening.",
                ],
            },
        )
        raw_results = response["results"]
        if not isinstance(raw_results, list) or not raw_results:
            raise ValueError("Model returned no reranking results")
        results = cast(list[Any], raw_results)
        return f"reranked documents={len(results)}"

    response = _request(
        f"{base}/v1/chat/completions",
        key,
        {
            "model": model.name,
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with exactly CHECK_OK and no other text.",
                }
            ],
            "max_tokens": min(model.max_output_tokens, max_tokens),
            "stream": False,
        },
    )
    content = response["choices"][0]["message"].get("content")
    if not isinstance(content, str) or not content.strip():
        choice = response["choices"][0]
        message = choice["message"]
        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning.strip():
            return (
                "completion returned reasoning-only output "
                f"({len(reasoning.strip())} characters; final content was empty)"
            )
        raise ValueError(
            "Model returned no nonempty text content; "
            f"finish_reason={choice.get('finish_reason')}; usage={response.get('usage')}"
        )
    return content.strip()[:200]


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preview",
        action="store_true",
        help="List planned model checks without starting containers",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Prompt for confirmation, then recreate each model before testing",
    )
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument(
        "--models",
        "--model",
        nargs="+",
        dest="models",
        metavar="MODEL_OR_ALIAS",
        help="Test only these configured model names or aliases (default: all)",
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()
    args.models = _normalize_model_selections(args.models)
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if not math.isfinite(args.startup_timeout) or args.startup_timeout <= 0:
        parser.error("--startup-timeout must be finite and greater than zero")

    if args.recreate:
        print(
            "\n"
            + terminal_style(
                "WARNING: --recreate removes and rebuilds each selected model "
                "container. Container-local data will be lost.",
                code="1;31",
            )
        )
        try:
            confirmation = input("Type 'yes' to recreate the selected models: ")
        except EOFError:
            confirmation = ""
        if confirmation.strip().lower() != "yes":
            print("Recreation cancelled; no containers or requests changed.")
            return 1

    launcher = _load_launcher()
    stack = launcher.LLMStack(launcher.APP)
    stack.load_models()
    selected_models: set[str] | None = None
    if args.models:
        selected_models = set()
        unknown: list[str] = []
        for selection in args.models:
            resolved = stack._resolve_model_name(selection)
            if resolved is None:
                unknown.append(selection)
            else:
                selected_models.add(resolved)
        if unknown:
            parser.error("Unknown model names or aliases: " + ", ".join(unknown))
    models: list[Any] = []
    results: list[tuple[str, str]] = []
    for model in stack.models.values():
        if selected_models is not None and model.name not in selected_models:
            continue
        directory = Path(stack.config.model_root) / model.model_info["model_dir"]
        reason = _missing_files(directory)
        if reason:
            results.append((model.name, f"SKIP: {reason}"))
            print(f"SKIP {model.name}: {reason}")
        else:
            models.append(model)
            print(
                f"PLAN {model.name}: "
                f"{model.model_info.get('mode', 'inference')} -> start"
                f"{' with recreate' if args.recreate else ''} -> test -> stop"
            )

    if args.preview:
        print(
            "\nPreview only; no containers or requests changed. "
            "Run without --preview to start, test, and stop the listed models."
        )
        return 0

    key = os.environ.get("LITELLM_MASTER_KEY", "")
    if not key:
        raise RuntimeError("LITELLM_MASTER_KEY is required")
    host = stack.config.litellm_bind_host
    if host == "0.0.0.0":
        host = "127.0.0.1"
    elif host == "::":
        host = "::1"
    if ":" in host:
        host = f"[{host}]"
    base = f"http://{host}:{stack.config.litellm_port}"
    failed = bool(results)
    try:
        print("\nStopping LiteLLM and all configured model backends before testing.")
        _run_launcher("all", "--stop")
        running = _running_backends()
        if running:
            raise RuntimeError(
                "Backends are still running after 'harinezumigel-llm-stack all "
                "--stop': " + ", ".join(running)
            )

        print("\nStarting LiteLLM for API checks.")
        _run_launcher("litellm", "--start")
        proxy_status = "FAIL"
        try:
            _wait_for_proxy(base, key, args.startup_timeout)
            proxy_status = "PASS"
            print("PASS LiteLLM: model-list API responded.")
        finally:
            results.append(("LiteLLM", proxy_status))

        for model in models:
            if _running_backends():
                raise RuntimeError("Another vLLM backend started; refusing to continue")
            status = "FAIL"
            try:
                options = [
                    "--start",
                    "--startup-logs",
                    "--startup-timeout",
                    str(args.startup_timeout),
                ]
                if args.recreate:
                    options.append("--recreate")
                _run_launcher(model.name, *options)
                result = _test_model(base, key, model, args.max_tokens)
                status = "PASS"
                print(f"PASS {model.name}: {result}")
            except (
                OSError,
                ValueError,
                KeyError,
                IndexError,
                TypeError,
                subprocess.CalledProcessError,
            ) as error:
                detail = str(error)
                if isinstance(error, HTTPError):
                    detail += ": " + error.read(4096).decode(errors="replace")
                for secret in (key, model.api_key):
                    if secret:
                        detail = detail.replace(secret, "***REDACTED***")
                print(f"FAIL {model.name}: {detail}", file=sys.stderr)
                failed = True
            finally:
                results.append((model.name, status))
                # Stop even after a startup timeout; never overlap model allocations.
                _run_launcher(model.name, "--stop")
                if _running_backends():
                    raise RuntimeError("Backend still running after stop; aborting")
    finally:
        print("\nStopping LiteLLM and model backends.")
        _run_launcher("all", "--stop")
        print("\n=== Model check summary ===")
        for name, status in results:
            print(f"{name}: {status}")
        print("LiteLLM and configured model backends were stopped.")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(_main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
