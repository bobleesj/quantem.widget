"""Shared tool visibility/locking registry and helpers."""

from __future__ import annotations

import json
import pathlib
from functools import lru_cache
from typing import Any

_REGISTRY_PATH = pathlib.Path(__file__).with_name("tool_parity.json")


@lru_cache(maxsize=1)
def _load_registry() -> dict[str, Any]:
    return json.loads(_REGISTRY_PATH.read_text())


def get_widget_tool_groups(widget_name: str) -> tuple[str, ...]:
    registry = _load_registry()
    widgets = registry.get("widgets", {})
    if widget_name not in widgets:
        supported = ", ".join(sorted(widgets))
        raise ValueError(f"Unknown widget {widget_name!r}. Supported widgets: {supported}.")
    return tuple(str(v).strip().lower() for v in widgets[widget_name].get("tool_groups", []))


def get_widget_tool_aliases(widget_name: str) -> dict[str, str]:
    registry = _load_registry()
    widgets = registry.get("widgets", {})
    if widget_name not in widgets:
        supported = ", ".join(sorted(widgets))
        raise ValueError(f"Unknown widget {widget_name!r}. Supported widgets: {supported}.")
    aliases = widgets[widget_name].get("aliases", {})
    return {str(k).strip().lower(): str(v).strip().lower() for k, v in aliases.items()}


def normalize_tool_groups(widget_name: str, tool_groups) -> list[str]:
    if tool_groups is None:
        return []
    if isinstance(tool_groups, str):
        values = [tool_groups]
    else:
        values = list(tool_groups)

    order = get_widget_tool_groups(widget_name)
    aliases = get_widget_tool_aliases(widget_name)
    supported = set(order)
    normalized: list[str] = []
    seen: set[str] = set()

    for raw in values:
        key = str(raw).strip().lower()
        if not key:
            continue
        key = aliases.get(key, key)
        if key not in supported:
            supported_values = ", ".join(f'"{k}"' for k in order)
            raise ValueError(
                f"Unknown tool group {raw!r}. Supported values: {supported_values}."
            )
        if key == "all":
            return ["all"]
        if key not in seen:
            seen.add(key)
            normalized.append(key)
    return normalized


def build_tool_groups(
    widget_name: str,
    *,
    tool_groups=None,
    all_flag: bool = False,
    flag_map: dict[str, bool] | None = None,
) -> list[str]:
    if all_flag:
        return ["all"]
    values: list[str] = []
    if tool_groups is not None:
        if isinstance(tool_groups, str):
            values.append(tool_groups)
        else:
            values.extend(tool_groups)
    for key, enabled in (flag_map or {}).items():
        if enabled:
            values.append(key)
    return normalize_tool_groups(widget_name, values)


def resolve_control_preset_hidden_tools(widget_name: str, preset_id: str) -> list[str]:
    preset_key = str(preset_id).strip().lower()
    presets = _load_registry().get("control_presets", {})
    if preset_key not in presets:
        supported = ", ".join(sorted(presets))
        raise ValueError(f"Unknown control preset {preset_id!r}. Supported presets: {supported}.")

    show_groups = [str(v).strip().lower() for v in presets[preset_key].get("show_groups", [])]
    supported_groups = [g for g in get_widget_tool_groups(widget_name) if g != "all"]
    if "*" in show_groups:
        return []
    show_set = set(show_groups)
    hidden = [group for group in supported_groups if group not in show_set]
    return normalize_tool_groups(widget_name, hidden)


def _flatten_groups(groups: tuple[Any, ...]) -> list[Any]:
    if len(groups) == 1 and isinstance(groups[0], (list, tuple, set)):
        return list(groups[0])
    return list(groups)


def _expanded_without_all(widget_name: str, values) -> list[str]:
    normalized = normalize_tool_groups(widget_name, values)
    if "all" not in normalized:
        return normalized
    return [group for group in get_widget_tool_groups(widget_name) if group != "all"]


def _ordered_groups(widget_name: str, values: set[str]) -> list[str]:
    return [group for group in get_widget_tool_groups(widget_name) if group != "all" and group in values]


def bind_tool_runtime_api(cls, widget_name: str) -> None:
    """Attach runtime lock/hide helpers to a widget class."""

    def set_disabled_tools(self, tool_groups) -> Any:
        self.disabled_tools = normalize_tool_groups(widget_name, tool_groups)
        return self

    def set_hidden_tools(self, tool_groups) -> Any:
        self.hidden_tools = normalize_tool_groups(widget_name, tool_groups)
        return self

    def lock_tool(self, *tool_groups) -> Any:
        new_groups = _flatten_groups(tool_groups)
        if not new_groups:
            return self
        current = _expanded_without_all(widget_name, self.disabled_tools)
        requested = _expanded_without_all(widget_name, new_groups)
        merged = set(current).union(requested)
        self.disabled_tools = _ordered_groups(widget_name, merged)
        return self

    def unlock_tool(self, *tool_groups) -> Any:
        remove_groups = _flatten_groups(tool_groups)
        if not remove_groups:
            return self
        current = set(_expanded_without_all(widget_name, self.disabled_tools))
        requested = set(_expanded_without_all(widget_name, remove_groups))
        current.difference_update(requested)
        self.disabled_tools = _ordered_groups(widget_name, current)
        return self

    def hide_tool(self, *tool_groups) -> Any:
        new_groups = _flatten_groups(tool_groups)
        if not new_groups:
            return self
        current = _expanded_without_all(widget_name, self.hidden_tools)
        requested = _expanded_without_all(widget_name, new_groups)
        merged = set(current).union(requested)
        self.hidden_tools = _ordered_groups(widget_name, merged)
        return self

    def show_tool(self, *tool_groups) -> Any:
        remove_groups = _flatten_groups(tool_groups)
        if not remove_groups:
            return self
        current = set(_expanded_without_all(widget_name, self.hidden_tools))
        requested = set(_expanded_without_all(widget_name, remove_groups))
        current.difference_update(requested)
        self.hidden_tools = _ordered_groups(widget_name, current)
        return self

    def apply_control_preset(self, preset: str) -> Any:
        self.hidden_tools = resolve_control_preset_hidden_tools(widget_name, preset)
        return self

    cls.set_disabled_tools = set_disabled_tools  # type: ignore[attr-defined]
    cls.set_hidden_tools = set_hidden_tools  # type: ignore[attr-defined]
    cls.lock_tool = lock_tool  # type: ignore[attr-defined]
    cls.unlock_tool = unlock_tool  # type: ignore[attr-defined]
    cls.hide_tool = hide_tool  # type: ignore[attr-defined]
    cls.show_tool = show_tool  # type: ignore[attr-defined]
    cls.apply_control_preset = apply_control_preset  # type: ignore[attr-defined]
