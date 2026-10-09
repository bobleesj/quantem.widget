"""Shared viewer UI chrome presets and constructor keyword checks."""

from collections.abc import Mapping
from typing import Literal

UiMode = Literal["interactive", "presentation", "report", "minimal"]

UI_MODE_DEFAULTS: dict[str, dict[str, bool]] = {
    "interactive": {},
    "presentation": {
        "controls_collapsed": True,
        "show_stats": False,
        "show_resize_handles": False,
        "show_zoom_indicator": False,
    },
    "report": {
        "show_controls": False,
        "controls_collapsed": False,
        "show_stats": False,
        "show_resize_handles": False,
        "show_zoom_indicator": False,
    },
    "minimal": {
        "show_title": False,
        "show_controls": False,
        "controls_collapsed": False,
        "show_stats": False,
        "show_panel_titles": False,
        "show_resize_handles": False,
        "show_zoom_indicator": False,
        "show_scale_bar": False,
        "show_legend": False,
        "show_grid": False,
        "show_crosshair": False,
    },
}


def resolve_ui_mode(
    ui_mode: UiMode | str,
    *,
    defaults: Mapping[str, object],
    overrides: Mapping[str, object],
) -> dict[str, object]:
    """Resolve a viewer UI preset with explicit keyword overrides.

    ``ui_mode`` applies broad defaults first. Values in ``overrides`` win when
    they are not ``None`` so callers can use a preset and still opt individual
    controls back in.
    """
    mode = str(ui_mode or "interactive").strip().lower()
    if mode not in UI_MODE_DEFAULTS:
        choices = ", ".join(repr(choice) for choice in UI_MODE_DEFAULTS)
        raise ValueError(f"ui_mode must be one of {choices}, got {ui_mode!r}")

    resolved = dict(defaults)
    resolved.update(
        {key: value for key, value in UI_MODE_DEFAULTS[mode].items() if key in defaults}
    )
    resolved.update({key: value for key, value in overrides.items() if value is not None})
    return resolved


def reject_unknown_kwargs(widget_class: type, kwargs: Mapping[str, object]) -> None:
    """Raise TypeError naming every keyword that is neither a parameter nor a trait of ``widget_class``.

    Widgets pass leftover keywords to ``anywidget.AnyWidget.__init__`` so any
    synced trait can be set at construction. traitlets only warns about a
    keyword it does not know, so without this check a removed parameter
    (``Show4DSTEM(..., compare_panel_gap_px=0)``) or a typo does nothing.
    """
    unknown = sorted(set(kwargs) - set(widget_class.class_trait_names()))
    if len(unknown) == 1:
        raise TypeError(f"{widget_class.__name__}() got an unexpected keyword argument {unknown[0]!r}")
    if unknown:
        names = ", ".join(repr(name) for name in unknown)
        raise TypeError(f"{widget_class.__name__}() got unexpected keyword arguments {names}")
