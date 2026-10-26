"""Conservative terminal styling shared by launcher output and startup monitoring."""

import builtins
import os
import sys
from typing import TextIO

_COLORS = {
    "black": "30",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
    "white": "37",
    "default": "39",
}
_COLORS.update(
    {
        f"bright-{name}": str(int(code) + 60)
        for name, code in list(_COLORS.items())
        if name != "default"
    }
)
_PALETTE = {
    "31": ("HLS_FAILURE_COLOR", "red"),
    "32": ("HLS_SUCCESS_COLOR", "green"),
    "33": ("HLS_WARNING_COLOR", "yellow"),
    "36": ("HLS_ACCENT_COLOR", "cyan"),
    "1": ("HLS_HEADING_COLOR", "default"),
}
_STATUS_COLORS = {
    "ready": "32",
    "started": "32",
    "failed": "1;31",
    "interrupted": "1;33",
    "dry-run": "36",
}


def _configured_color(variable: str, default: str) -> str:
    value = os.environ.get(variable, default).strip().lower()
    if value not in _COLORS:
        raise ValueError(
            f"{variable} must be one of: {', '.join(_COLORS)}; got {value!r}"
        )
    return _COLORS[value]


def terminal_style(text: str, *, accent: bool = False, code: str | None = None) -> str:
    """Use standard palette colors only on capable interactive stdout."""
    if (
        not sys.stdout.isatty()
        or "NO_COLOR" in os.environ
        or os.environ.get("TERM", "").lower() in ("", "dumb")
    ):
        return text
    style = code or ("36" if accent else "1")
    base = style.removeprefix("1;")
    if base in _PALETTE:
        variable, default = _PALETTE[base]
        color = _configured_color(variable, default)
        if style == "1":
            style = "1" if color == "39" else f"1;{color}"
        else:
            style = f"1;{color}" if style.startswith("1;") else color
    return f"\033[{style}m{text}\033[0m"


def style_status_line(line: str) -> str:
    """Highlight launcher structure without recoloring arbitrary model logs."""
    if "\033[" in line:
        return line
    text = line.strip()
    if text.startswith(
        (
            "GPU VRAM preflight",
            "Shared system memory preflight",
            "NVIDIA pre-run compatibility check",
        )
    ):
        return terminal_style(line, code="32" if " passed" in text else "1;31")
    if text.startswith(("ERROR:", "Follow logs:")):
        return terminal_style(line, code="1;31" if text.startswith("ERROR:") else "36")
    if text.startswith(
        ("WARNING:", "Warning:", "Startup interrupted", "Startup stopped")
    ):
        return terminal_style(line, code="1;33")
    if text.startswith(("READY:", "SUCCESS:", "✓", "Stopped:", "Status:   ✓")):
        return terminal_style(line, code="32")
    if text.startswith(("✗", "Status:   ✗")):
        return terminal_style(line, code="1;31")
    if text.startswith("===") or text.endswith(":"):
        return terminal_style(line)
    if " - Unified LiteLLM" in text or " — Unified LiteLLM" in text:
        return terminal_style(line)
    if " port=" in line or text.endswith("no readiness check"):
        fields = text.split()
        if len(fields) >= 3 and (
            fields[2].startswith("port=") or fields[2:] == ["no", "readiness", "check"]
        ):
            status = fields[1]
            if status in _STATUS_COLORS:
                return terminal_style(line, code=_STATUS_COLORS[status])
    if ":" in line:
        label, rest = line.split(":", 1)
        if label and label[0].isupper() and "\n" not in label:
            return terminal_style(label + ":") + rest
        if label.startswith("  ") and len(label.strip().split()) <= 4:
            return terminal_style(label + ":") + rest
    if text.startswith(("Dry run", "Resolved alias", "Container already running")):
        return terminal_style(line, accent=True)
    return line


def terminal_print(
    *values: object,
    sep: str | None = " ",
    end: str | None = "\n",
    file: TextIO | None = None,
    flush: bool = False,
) -> None:
    """Print styled launcher text while leaving non-stdout destinations plain."""
    separator = " " if sep is None else sep
    text = separator.join(str(value) for value in values)
    if file is None or file is sys.stdout:
        text = "\n".join(style_status_line(line) for line in text.split("\n"))
    if end == "\n" and file is None and not flush:
        builtins.print(text)
    else:
        builtins.print(text, end=end, file=file, flush=flush)
