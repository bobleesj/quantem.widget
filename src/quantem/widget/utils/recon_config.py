"""
Shared QuantEM reconstruction ``config.json`` helpers.

Both Show3D and Show3DSlices accept a ``config=`` convenience argument so a
multislice ptychography z-stack can be calibrated (pixel size) and aligned
(in-plane rotation + post-crop) straight from the reconstruction metadata,
with no manual array math in the notebook. These helpers hold that logic in
one place so the two widgets stay consistent. The module has no widget
dependency: only numpy / json / math / pathlib.
"""
import json
import pathlib
from collections.abc import Mapping, Sequence

import numpy as np
from quantem.gpu.display.geometry import (
    normalize_rotation_degrees as _gpu_normalize_rotation_degrees,
)
from quantem.gpu.display.geometry import rotate_stack_inplane


def _load_quantem_config(config: Mapping | str | pathlib.Path | None) -> Mapping | None:
    """Accept a parsed QuantEM config dict or a path to its JSON file."""
    if config is None:
        return None
    if isinstance(config, (str, pathlib.Path)):
        return json.loads(pathlib.Path(config).read_text())
    if isinstance(config, Mapping):
        return config
    raise TypeError(
        "config must be a parsed mapping, a config.json path, or None; "
        f"got {type(config).__name__}"
    )


def _config_get(config: Mapping, *keys: str):
    current = config
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _config_float(config: Mapping, *keys: str) -> float | None:
    value = _config_get(config, *keys)
    if value is None or value == "":
        return None
    return float(value)


def _centered_crop_for_shape(
    source_shape: Sequence[int],
    target_shape: Sequence[int],
) -> tuple[int, int, int, int]:
    """Centered row/column crop needed to display target_shape from source_shape."""
    if len(source_shape) < 2 or len(target_shape) < 2:
        raise ValueError(
            "cropped_shape inference needs source and target shapes with row/col axes"
        )
    crop_y = max(0, int(source_shape[-2]) - int(target_shape[-2]))
    crop_x = max(0, int(source_shape[-1]) - int(target_shape[-1]))
    return (
        crop_y // 2,
        crop_y - crop_y // 2,
        crop_x // 2,
        crop_x - crop_x // 2,
    )


def _post_crop_from_quantem_config(
    source_shape: Sequence[int],
    config: Mapping,
) -> int | tuple[int, int, int, int]:
    """Infer the post-rotation crop from QuantEM reconstruction metadata."""
    cropped_shape = _config_get(config, "object", "cropped_shape")
    if cropped_shape:
        return _centered_crop_for_shape(source_shape, cropped_shape)
    post_crop_px = (
        _config_get(config, "reconstruction", "obj_padding_px")
        or _config_get(config, "input", "padding")
        or 0
    )
    return int(post_crop_px)


def _pixel_size_from_quantem_config(config: Mapping) -> list[float] | None:
    """Return [pz, py, px] sampling in Å from QuantEM reconstruction metadata."""
    z_sampling = _config_float(config, "reconstruction", "slice_thickness_A")
    xy_sampling = _config_float(config, "reconstruction", "obj_sampling_A_per_px")
    if z_sampling is None or xy_sampling is None:
        return None
    return [z_sampling, xy_sampling, xy_sampling]


def _is_default_pixel_size(pixel_size: float | Sequence[float] | None) -> bool:
    return pixel_size is None or (
        np.isscalar(pixel_size) and float(pixel_size) == 0.0
    )


def _normalize_rotation_deg(rotation_deg: float) -> float:
    """Compatibility shim for quantem.gpu rotation validation."""
    return _gpu_normalize_rotation_degrees(rotation_deg)


def _rotate_stack_inplane(data: np.ndarray, rotation_deg: float) -> np.ndarray:
    """Compatibility shim for :func:`quantem.gpu.display.geometry.rotate_stack_inplane`."""
    return rotate_stack_inplane(data, rotation_deg)
