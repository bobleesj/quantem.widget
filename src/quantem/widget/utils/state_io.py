"""The versioned JSON envelope around a widget's ``state_dict``: ``save`` writes it, ``load_state_dict`` reads it."""

import json
import pathlib

import numpy as np

from quantem.widget import __version__

JSON_METADATA_VERSION = "1.0"


def read_state(state: dict | str | pathlib.Path, widget_name: str) -> dict:
    """The state dict to restore from a ``save`` file, its envelope, or a bare ``state_dict``.

    A file must be an envelope (``save`` always wraps). An envelope written by
    another widget class is refused: Show3D keys loaded into Show2D would set
    a wrong subset of traits without a word.
    """
    if isinstance(state, (str, pathlib.Path)):
        state = json.loads(pathlib.Path(state).read_text())
        if not isinstance(state, dict) or "state" not in state:
            raise ValueError("State JSON file must be a versioned envelope with top-level 'state'.")
    if not isinstance(state, dict):
        raise ValueError("State payload must be a dict.")
    if "state" not in state:
        return state
    written_by = state.get("widget_name")
    if written_by is not None and written_by != widget_name:
        raise ValueError(f"State envelope is for {written_by!r}, cannot load into {widget_name!r}")
    if not isinstance(state["state"], dict):
        raise ValueError("State envelope field 'state' must be a dict.")
    return state["state"]


def _numpy_safe(value):
    """JSON fallback for values ``json.dumps`` rejects: a NumPy scalar's Python value, anything else as text.

    NumPy scalars in ROI dicts (np.int64 from ``shape[0] // 2``) would
    otherwise raise TypeError in ``json.dumps``.
    """
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def save_state_file(path: str | pathlib.Path, widget_name: str, state: dict) -> None:
    """Write ``state`` in its versioned envelope as indented JSON, creating the parent folder.

    The envelope names the format version and the widget that wrote the file,
    so ``read_state`` can refuse another widget's file.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"metadata_version": JSON_METADATA_VERSION, "widget_name": widget_name,
                "widget_version": __version__, "state": state}
    path.write_text(json.dumps(envelope, indent=2, default=_numpy_safe))
