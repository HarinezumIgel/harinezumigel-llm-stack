"""Ordered model-group validation and selection, without lifecycle side effects."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any


def load_groups(
    raw: Any,
    names_and_aliases: Mapping[str, str | None],
    reserved: Sequence[str],
    allowed_reserved: Sequence[str] = (),
) -> dict[str, list[str]]:
    """Validate group structure and every reference against the model catalog."""
    if not isinstance(raw, dict):
        raise ValueError("model_groups must be a mapping of names to model lists")
    valid_names = set(names_and_aliases)
    valid_names.update(alias for alias in names_and_aliases.values() if alias)
    valid_names.update(allowed_reserved)
    allowed = {name.lower() for name in allowed_reserved}
    blocked = {name.lower() for name in reserved} - allowed
    groups: dict[str, list[str]] = {}
    for name, members in raw.items():
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(members, list)
            or not members
            or any(
                not isinstance(member, str) or not member.strip() for member in members
            )
        ):
            raise ValueError(
                "Each model group needs a name and a nonempty list of model names/aliases"
            )
        for member in members:
            if member.lower() in blocked or member not in valid_names:
                raise ValueError(
                    f"Unknown or reserved model '{member}' in group '{name}'"
                )
        groups[name] = list(members)
    return groups


def select_models(
    names: list[str],
    group: str | None,
    groups: Mapping[str, list[str]],
    resolve: Callable[[str], str | None],
    reserved: Sequence[str],
) -> list[str]:
    """Resolve and deduplicate an entire selection, preserving first-seen order."""
    if group is not None:
        if names:
            raise ValueError("--group cannot be combined with positional models")
        if group not in groups:
            raise ValueError(f"Unknown model group '{group}'")
        names = groups[group]
    selected: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name.lower() in reserved:
            raise ValueError("Batch selection does not accept the 'all' target")
        resolved = resolve(name)
        if resolved is None:
            raise ValueError(f"Unknown model '{name}'")
        if resolved not in seen:
            selected.append(resolved)
            seen.add(resolved)
    if not selected:
        raise ValueError("Select at least one model")
    return selected
