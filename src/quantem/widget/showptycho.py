"""
ShowPtycho - interactive anywidget for ptychography aberration exploration.

Renders phase and optional FFT while tuning aberrations.  A full bright-field
disk gives the authoritative reconstruction; a smaller BF subset can be used
for fast exploratory drag previews.

Usage::

    from quantem.widget import ShowPtycho

    ssb.optimize()
    del data
    ShowPtycho(ssb)   # opens the widget
"""

import datetime
import json
import math
import numbers
import pathlib
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

import anywidget
import numpy as np
import traitlets


# Default path for auto-saved starred aberration snapshots.  Users can pass
# ``save_dir`` to override; otherwise we drop the JSON next to wherever the
# notebook is executing so the "next cell" can read it back.
_DEFAULT_STARS_FILENAME = "showptycho_stars.json"
_CALIBRATION_SCHEMA_VERSION = 1
_DEFAULT_DRAG_BF_FRACTION = 0.3


@dataclass
class PtychoCalibration:
    """Locked SSB calibration parameters saved by ``ShowPtycho``.

    Parameters
    ----------
    rotation_angle_deg : float
        Scan-detector rotation angle in degrees.
    aberrations : dict[str, float]
        Aberration coefficients. ``C10`` and ``C12`` are in nm, ``phi12`` is in
        radians. Higher-order magnitudes are stored in nm.
    flip_phase : bool
        Whether the displayed phase sign was flipped.
    """

    rotation_angle_deg: float
    aberrations: dict[str, float]
    higher_order: dict[str, float] = field(default_factory=dict)
    flip_phase: bool = False
    voltage_kV: float | None = None
    semiangle_mrad: float | None = None
    scan_sampling_A: float | None = None
    loss: float | None = None
    source_file: str | None = None
    source_stem: str | None = None
    label: str | None = None
    notes: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now().isoformat(timespec="seconds")
    )


def _atomic_write_json(path: pathlib.Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(path)


def _calibration_from_mapping(data: dict[str, Any]) -> PtychoCalibration:
    return PtychoCalibration(
        rotation_angle_deg=float(data["rotation_angle_deg"]),
        aberrations={
            str(k): float(v) for k, v in (data.get("aberrations") or {}).items()
        },
        higher_order={
            str(k): float(v) for k, v in (data.get("higher_order") or {}).items()
        },
        flip_phase=bool(data.get("flip_phase", False)),
        voltage_kV=data.get("voltage_kV"),
        semiangle_mrad=data.get("semiangle_mrad"),
        scan_sampling_A=data.get("scan_sampling_A"),
        loss=data.get("loss"),
        source_file=data.get("source_file"),
        source_stem=data.get("source_stem"),
        label=data.get("label"),
        notes=str(data.get("notes", "")),
        id=str(data.get("id", uuid.uuid4().hex[:12])),
        timestamp=str(
            data.get(
                "timestamp",
                datetime.datetime.now().isoformat(timespec="seconds"),
            )
        ),
    )


def load_ptycho_calibration(path: str | pathlib.Path) -> PtychoCalibration:
    """Load a single ``ShowPtycho`` calibration JSON file."""

    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    data = json.loads(path.read_text())
    if isinstance(data, list):
        raise ValueError(f"{path} is a calibration list, expected one object")
    return _calibration_from_mapping(data)


def save_ptycho_calibration(
    calibration: PtychoCalibration,
    path: str | pathlib.Path,
) -> pathlib.Path:
    """Save one ``ShowPtycho`` calibration JSON file."""

    path = pathlib.Path(path)
    payload = {
        "schema_version": _CALIBRATION_SCHEMA_VERSION,
        "version": "2.0",
        **asdict(calibration),
    }
    _atomic_write_json(path, payload)
    return path


def _coerce_calibration(calibration: object) -> PtychoCalibration:
    if isinstance(calibration, PtychoCalibration):
        return calibration
    if isinstance(calibration, (str, pathlib.Path)):
        return load_ptycho_calibration(calibration)
    if hasattr(calibration, "rotation_angle_deg") and hasattr(calibration, "aberrations"):
        return PtychoCalibration(
            rotation_angle_deg=float(getattr(calibration, "rotation_angle_deg")),
            aberrations={
                str(k): float(v)
                for k, v in dict(getattr(calibration, "aberrations")).items()
            },
            higher_order={
                str(k): float(v)
                for k, v in dict(getattr(calibration, "higher_order", {}) or {}).items()
            },
            flip_phase=bool(getattr(calibration, "flip_phase", False)),
            voltage_kV=getattr(calibration, "voltage_kV", None),
            semiangle_mrad=getattr(calibration, "semiangle_mrad", None),
            scan_sampling_A=getattr(calibration, "scan_sampling_A", None),
            loss=getattr(calibration, "loss", None),
            source_file=getattr(calibration, "source_file", None),
            source_stem=getattr(calibration, "source_stem", None),
            label=getattr(calibration, "label", None),
            notes=str(getattr(calibration, "notes", "")),
            id=str(getattr(calibration, "id", uuid.uuid4().hex[:12])),
            timestamp=str(
                getattr(
                    calibration,
                    "timestamp",
                    datetime.datetime.now().isoformat(timespec="seconds"),
                )
            ),
        )
    raise TypeError(
        "calibration must be a path or object with rotation_angle_deg and "
        f"aberrations, got {type(calibration).__name__}"
    )


def _higher_order_widget_payload(calibration: PtychoCalibration) -> dict[str, float]:
    """Translate saved calibration keys into the widget panel convention."""

    payload: dict[str, float] = {}
    source = {**calibration.higher_order, **calibration.aberrations}
    radial = {"C30", "C50"}
    for key, value in source.items():
        if key in {"C10", "C12", "phi12"}:
            continue
        value = float(value)
        if key.endswith("_angle"):
            payload[key] = value
        elif key.startswith("phi") and len(key) >= 5:
            payload[f"C{key[3:]}_angle"] = math.degrees(value)
        elif key in radial:
            payload[key] = value
        elif key.startswith("C"):
            payload[f"{key}_mag"] = value
    return payload


def _to_numpy(array: object) -> np.ndarray:
    """Convert a small reconstructed phase image to NumPy."""

    if hasattr(array, "get"):
        array = array.get()
    elif type(array).__module__.split(".", 1)[0] == "torch":
        array = array.detach().cpu().numpy()
    return np.asarray(array)


def _array_module_for_accel(accel: object):
    """Return the array module used for coefficient scratch buffers."""

    if getattr(accel, "backend", None) == "mps":
        return np
    import cupy as cp

    return cp


def _resolve_drag_bf_count(
    drag_bf: int | float | None,
    total_bf: int,
) -> int:
    """Resolve a user BF preview request to a positive detector-pixel count."""

    total = max(1, int(total_bf))
    if drag_bf is None:
        return max(1, min(total, int(round(total * _DEFAULT_DRAG_BF_FRACTION))))
    if isinstance(drag_bf, numbers.Integral) and not isinstance(drag_bf, bool):
        return max(1, min(total, int(drag_bf)))
    value = float(drag_bf)
    if not math.isfinite(value) or value <= 0:
        value = _DEFAULT_DRAG_BF_FRACTION
    if 0 < value <= 1:
        return max(1, min(total, int(round(total * value))))
    return max(1, min(total, int(round(value))))


def _bf_geometry_1d_numpy(
    kx: np.ndarray,
    ky: np.ndarray,
    *,
    wavelength: float,
    semiangle_rad: float,
    ang_y_rad: float,
    ang_x_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the BF-pixel one-dimensional geometry used by WebGPU export."""

    dx = np.asarray(kx, dtype=np.float32)
    dy = np.asarray(ky, dtype=np.float32)
    dx2 = dx * dx
    dy2 = dy * dy
    r2 = dx2 + dy2
    r = np.sqrt(r2).astype(np.float32, copy=False)
    alpha = r * np.float32(wavelength)
    alpha2 = alpha * alpha
    inv_r2 = np.zeros_like(r2, dtype=np.float32)
    np.divide(1.0, r2, out=inv_r2, where=r2 > np.float32(1e-30))
    cos2phi = (dx2 - dy2) * inv_r2
    sin2phi = np.float32(2.0) * dx * dy * inv_r2
    denom_num2 = (dx * np.float32(ang_y_rad)) ** 2 + (
        dy * np.float32(ang_x_rad)
    ) ** 2
    inv_r = np.zeros_like(r, dtype=np.float32)
    np.divide(1.0, r, out=inv_r, where=r > np.float32(1e-15))
    denom = np.sqrt(denom_num2).astype(np.float32, copy=False) * inv_r
    edge = np.ones_like(r, dtype=np.float32)
    valid = denom > np.float32(1e-15)
    edge[valid] = (
        (np.float32(semiangle_rad) - alpha[valid]) / denom[valid]
        + np.float32(0.5)
    )
    aperture = np.clip(edge, 0.0, 1.0).astype(np.float32, copy=False)
    return (
        alpha2.astype(np.float32, copy=False),
        cos2phi.astype(np.float32, copy=False),
        sin2phi.astype(np.float32, copy=False),
        aperture,
    )


class _ShowPtychoWidget(anywidget.AnyWidget):
    """ShowPtycho anywidget with GPU-accelerated reconstruction.

    During slider drag, uses a deterministic BF-pixel subset by default for
    responsive preview.  Slide the BF control to the full count when the
    microscopist wants the authoritative full-disk reconstruction.

    The widget exclusively owns the ``accel`` (SSBEngine) - no other code
    should call methods on it while the widget is active.

    Parameters
    ----------
    accel : SSBEngine
        The GPU accelerator (already holds G_qk in VRAM).
    rotation_rad : float
        Rotation angle in radians.
    auto_aberrations : dict
        Auto-optimized aberration values (C10, C12, phi12 in radians).
    auto_loss_val : float
        Loss from automatic optimization.
    c10_range, c12_range, phi12_range : tuple[float, float]
        Slider ranges.
    drag_bf : int
        Number of BF pixels for preview.  Float values in ``(0, 1)`` are
        interpreted as a fraction of the detected BF disk; the default is
        30 percent.
    save_dir : str or Path, optional
        Directory for saving results.
    ssb_ref : SSB, optional
        High-level SSB instance for applying aberrations.
    """

    _esm = pathlib.Path(__file__).with_name("static") / "showptycho.js"

    # -- Slider ranges (Python → JS, set once) --
    c10_min = traitlets.Float(-400.0).tag(sync=True)
    c10_max = traitlets.Float(400.0).tag(sync=True)
    c12_min = traitlets.Float(-100.0).tag(sync=True)
    c12_max = traitlets.Float(100.0).tag(sync=True)
    phi12_min = traitlets.Float(-90.0).tag(sync=True)
    phi12_max = traitlets.Float(90.0).tag(sync=True)
    rotation_min = traitlets.Float(-180.0).tag(sync=True)
    rotation_max = traitlets.Float(180.0).tag(sync=True)

    # -- Current rotation (JS ↔ Python).  Starts at the value passed to
    # __init__ and can be swept to find the best orientation of the BF mask. --
    rotation_deg = traitlets.Float(0.0).tag(sync=True)

    # -- Flip phase sign (JS → Python).  Phase is inherently ambiguous by sign
    # in SSB; this toggle lets the user pick the convention that matches
    # expected sample contrast without re-optimizing. --
    flip_phase = traitlets.Bool(False).tag(sync=True)

    # -- Auto reference (Python → JS, set once) --
    auto_c10 = traitlets.Float(0.0).tag(sync=True)
    auto_c12 = traitlets.Float(0.0).tag(sync=True)
    auto_phi12_deg = traitlets.Float(0.0).tag(sync=True)
    auto_loss = traitlets.Float(0.0).tag(sync=True)
    # Initial rotation angle at mount time - used by the Reset button so it
    # returns the rotation slider to where the user entered the widget.
    auto_rotation_deg = traitlets.Float(0.0).tag(sync=True)

    # -- Request from JS → Python --
    request_json = traitlets.Unicode("").tag(sync=True)

    # -- Response from Python → JS --
    phase_bytes = traitlets.Bytes(b"").tag(sync=True)
    phase_width = traitlets.Int(0).tag(sync=True)
    phase_height = traitlets.Int(0).tag(sync=True)
    result_json = traitlets.Unicode("").tag(sync=True)

    # -- Pixel size (Å) for scale bar --
    pixel_size = traitlets.Float(0.0).tag(sync=True)

    # -- Initial panel width and FFT visibility (Python → JS, set once at mount).
    # JS reads these as one-shot seeds for its local panel/showFFT state - the
    # user can still resize via the corner handle and toggle FFT via the switch. --
    initial_panel_size = traitlets.Int(800).tag(sync=True)
    initial_fft_on = traitlets.Bool(False).tag(sync=True)

    # -- Save/Apply trigger (JS → Python) --
    save_trigger = traitlets.Int(0).tag(sync=True)

    # -- User-editable notes persisted into calibration.json (JS ↔ Python) --
    notes = traitlets.Unicode("").tag(sync=True)

    # -- Pin event (JS → Python) --
    pin_json = traitlets.Unicode("").tag(sync=True)

    # -- Where starred snapshots are auto-saved (Python → JS, displayed in UI) --
    stars_path = traitlets.Unicode("").tag(sync=True)

    # -- Where calibration.json was last written (Python → JS, empty until save) --
    calibration_path = traitlets.Unicode("").tag(sync=True)
    calibration_saved_at = traitlets.Unicode("").tag(sync=True)

    # -- Optuna trial history (Python → JS, set once at init).
    # JSON list of {rank, C10, C12, phi12_deg, loss} sorted by ascending loss.
    # Empty string when the SSB instance hasn't been optimized yet. --
    trials_json = traitlets.Unicode("").tag(sync=True)

    # -- Preview BF count (JS → Python).
    # Positive count used while exploring aberrations. The full BF count is the
    # right edge of the slider rather than a separate "off" state. --
    drag_bf = traitlets.Int(0).tag(sync=True)
    total_bf = traitlets.Int(0).tag(sync=True)

    # -- Higher-order aberration panel (JS → Python).
    # JSON-encoded dict with keys a subset of
    #   {C21_mag, C21_angle, C23_mag, C23_angle, C30, C32_mag, C32_angle,
    #    C34_mag, C34_angle, C41_mag, C41_angle, C43_mag, C43_angle,
    #    C45_mag, C45_angle, C50, C52_mag, C52_angle, C54_mag, C54_angle,
    #    C56_mag, C56_angle}
    # All values default to 0 when absent.  Magnitudes in nm (displayed unit);
    # angles in degrees (displayed unit).  When any magnitude is non-zero the
    # reconstruct path switches from the fast 2-term kernel to the 14-coef
    # chi_full kernel via SSBEngine.reconstruct_full. --
    higher_order_json = traitlets.Unicode("{}").tag(sync=True)

    # -- Optional browser-side WebGPU SSB preview payload (Python → JS).
    # Default exports point the browser at the original compressed HDF5 source
    # and let WebGPU build transient BF reducers.  The g_bf fields are retained
    # only for explicit compatibility cache paths.
    webgpu_preview_enabled = traitlets.Bool(False).tag(sync=True)
    webgpu_cal_json = traitlets.Unicode("").tag(sync=True)
    webgpu_g_bf_bytes = traitlets.Bytes(b"").tag(sync=True)
    webgpu_g_bf_url = traitlets.Unicode("").tag(sync=True)
    webgpu_h5_source_json = traitlets.Unicode("").tag(sync=True)
    webgpu_preview_status = traitlets.Unicode("WebGPU preview not initialized.").tag(
        sync=True
    )
    webgpu_standalone = traitlets.Bool(False).tag(sync=True)

    def __init__(
        self,
        accel,
        rotation_rad: float,
        auto_aberrations: dict,
        auto_loss_val: float,
        c10_range: tuple[float, float],
        c12_range: tuple[float, float],
        phi12_range: tuple[float, float],
        rotation_range: "tuple[float, float] | None" = None,
        drag_bf: int | float | None = _DEFAULT_DRAG_BF_FRACTION,
        save_dir: "str | pathlib.Path | None" = None,
        ssb_ref=None,
        pixel_size: float = 0.0,
        source_file: "str | None" = None,
        size: int = 800,
        fft_on: bool = False,
        webgpu_preview: bool | str = "auto",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._accel = accel
        self._rotation_rad = rotation_rad
        self._ssb_ref = ssb_ref
        self._save_dir = pathlib.Path(save_dir) if save_dir else None
        # Identifies which 4D-STEM file this SSB was built from.  Persisted into
        # calibration.json and every starred entry so time-series `Live.watch(...,
        # calibrations=...)` can match a star to the right file by stem.
        self._source_file = str(source_file) if source_file else None
        self._pinned: list[dict] = []
        self._last_phase_np: np.ndarray | None = None
        self._inflight_id: int = -1
        self._drag_state: dict | None = None
        self._full_state: dict | None = None
        # Cached decode of result_json so `_current_c10/c12/phi12_deg()` skip
        # JSON parse on every higher-order observer tick (3 parses/frame
        # → 0 parses/frame).  Refreshed at the end of _do_reconstruct.
        self._last_result: dict = {}
        self._webgpu_asset_dir: pathlib.Path | None = None
        # Reusable coefficient scratch buffers for the higher-order kernel path.
        # CUDA uses CuPy arrays; MPS uses NumPy arrays because the current MPS
        # preview backend supports the 3-parameter C10/C12/phi12 path only.
        xp = _array_module_for_accel(accel)
        self._ho_mags_buf = xp.zeros(14, dtype=xp.float32)
        self._ho_angs_buf = xp.zeros(14, dtype=xp.float32)
        # Starred snapshots are the time-series payload: every star/unstar is
        # atomically written to ``self._stars_path`` so downstream cells can
        # re-open the file and ingest the full defocus/C12/phi/rotation history.
        stars_dir = self._save_dir if self._save_dir is not None else pathlib.Path.cwd()
        self._stars_path = stars_dir / _DEFAULT_STARS_FILENAME
        self.stars_path = str(self._stars_path.resolve())
        # Canonical location for the single-calibration file - same basename
        # ``3_live.ipynb`` reads via ``load_calibration()``.
        self._calibration_path = stars_dir / "calibration.json"

        # Cache rotation
        accel.cache_rotation(rotation_rad)

        # Set ranges
        self.c10_min, self.c10_max = c10_range
        self.c12_min, self.c12_max = c12_range
        self.phi12_min, self.phi12_max = phi12_range
        # Rotation range: default ±180° so the slider covers negative rotations
        # directly without the user mentally adding 180.  Technically scan/detector
        # rotation is mod-180, but the extra width costs nothing and matches
        # microscope conventions that sometimes report negative values.
        start_deg = math.degrees(rotation_rad)
        if rotation_range is None:
            rotation_range = (-180.0, 180.0)
        self.rotation_min, self.rotation_max = rotation_range
        # Set current rotation without firing the observer (guard against premature reconstruct
        # before _inflight_id and accel state are initialized).
        self._rotation_deg_init = True
        self.rotation_deg = start_deg
        self._rotation_deg_init = False

        # Set auto reference
        self.auto_c10 = auto_aberrations.get("C10", 0.0)
        self.auto_c12 = auto_aberrations.get("C12", 0.0)
        self.auto_phi12_deg = math.degrees(auto_aberrations.get("phi12", 0.0))
        self.auto_loss = auto_loss_val
        self.auto_rotation_deg = start_deg

        # Set pixel size for scale bar
        self.pixel_size = pixel_size

        # Seed the UI's initial panel size + FFT visibility.  JS reads these
        # once at mount; user interaction with the resize handle / FFT switch
        # takes over from there.
        self.initial_panel_size = int(size)
        self.initial_fft_on = bool(fft_on)

        # Listen for events
        self.observe(self._on_request, names=["request_json"])
        self.observe(self._on_save, names=["save_trigger"])
        self.observe(self._on_pin, names=["pin_json"])
        self.observe(self._on_drag_bf_change, names=["drag_bf"])
        self.observe(self._on_rotation_change, names=["rotation_deg"])
        self.observe(self._on_flip_change, names=["flip_phase"])
        self.observe(self._on_higher_order_change, names=["higher_order_json"])

        # Publish total BF count so the UI can clamp user input to valid range
        self.total_bf = int(accel._cache["num_bf"])

        # Publish Optuna trial history (sorted by ascending loss) so the widget
        # can render a browsable trials strip.  Capped at 50 for payload size.
        self.trials_json = self._build_trials_payload()

        # Initial reconstruction (full BF + loss) - also warms up chunk buffers
        self._inflight_id = 0
        self._do_reconstruct(
            0, self.auto_c10, self.auto_c12, self.auto_phi12_deg
        )

        # Start with an interactive BF fraction. Full BF is still available by
        # moving the slider to the total count.
        self.drag_bf = _resolve_drag_bf_count(drag_bf, self.total_bf)

        self._init_webgpu_preview(webgpu_preview)

    def _init_webgpu_preview(self, mode: bool | str) -> None:
        """Populate optional browser-side WebGPU payload when safe."""

        mode_text = str(mode).lower()
        if mode is False or mode_text in {"0", "false", "off", "none"}:
            self.webgpu_preview_enabled = False
            self.webgpu_preview_status = "WebGPU preview disabled by caller."
            return
        if mode_text not in {"cache", "legacy-cache", "g_bf"}:
            self.webgpu_preview_enabled = False
            self.webgpu_preview_status = (
                "Notebook WebGPU preview does not write a persistent BF-G cache by default. "
                "Use export_webgpu_folder() for compressed-source browser review, or pass "
                "webgpu_preview='cache' for the legacy local cache."
            )
            return
        try:
            from quantem.widget.showptycho_webgpu_export import (
                build_showptycho_webgpu_payload,
            )

            cal, g_bytes, status = build_showptycho_webgpu_payload(self)
        except Exception as exc:
            self.webgpu_preview_enabled = False
            self.webgpu_preview_status = f"WebGPU preview unavailable: {exc}"
            return
        self.webgpu_preview_status = status
        if cal is None or not g_bytes:
            self.webgpu_preview_enabled = False
            return
        # Jupyter refuses to serve hidden paths under /files/, so keep the
        # notebook-local WebGPU payload cache in a visible directory.
        asset_root = pathlib.Path.cwd() / "quantem_showptycho_webgpu" / uuid.uuid4().hex
        asset_root.mkdir(parents=True, exist_ok=True)
        g_path = asset_root / "g_bf.c64"
        g_path.write_bytes(g_bytes)
        self._webgpu_asset_dir = asset_root
        rel = g_path.relative_to(pathlib.Path.cwd()).as_posix()
        self.webgpu_cal_json = json.dumps(cal)
        # Keep the heavy BF-G payload out of widget comm state. Jupyter serves
        # notebook-relative files through /files/ on the same origin, matching
        # the folder export transport model.
        self.webgpu_g_bf_bytes = b""
        self.webgpu_g_bf_url = f"/files/{rel}"
        self.webgpu_preview_enabled = True
        self.webgpu_preview_status = (
            f"{status} Browser will fetch BF-G from a notebook-local asset."
        )

    # ------------------------------------------------------------------
    #  BF-subset drag preview state
    # ------------------------------------------------------------------

    def _build_drag_state(self, drag_bf: int) -> None:
        """Pre-compute deterministic BF subset for ~60 FPS drag preview."""
        accel = self._accel
        xp = _array_module_for_accel(accel)
        num_bf = int(accel._cache["num_bf"])
        ny = int(accel._cache["ny"])
        nx = int(accel._cache["nx"])

        # Deterministic uniform stride for even angular coverage
        step = max(1, num_bf // drag_bf)
        indices = xp.arange(0, num_bf, step, dtype=xp.int64)[:drag_bf]

        # Slice per-BF cache arrays
        drag_cache = dict(accel._cache)
        for k in ("kx_bf", "ky_bf", "alpha_k2_1d", "cos2phi_k_1d",
                  "sin2phi_k_1d", "aperture_k_1d"):
            if k in drag_cache:
                drag_cache[k] = xp.ascontiguousarray(accel._cache[k][indices])
        drag_cache["num_bf"] = int(indices.size)

        self._drag_state = {
            "G_qk": accel.G_qk[indices],
            "bf_inds_row": accel.bf_inds_row[indices],
            "bf_inds_col": accel.bf_inds_col[indices],
            "cache": drag_cache,
            "pk_buffer": xp.empty((int(indices.size),), dtype=xp.complex64),
            "result_buffer": xp.empty(
                (int(indices.size), ny, nx), dtype=xp.complex64,
            ),
            "mean_phase_buffer": xp.empty((ny, nx), dtype=xp.float32),
        }

        # Snapshot full state for restoration
        self._full_state = {
            "G_qk": accel.G_qk,
            "bf_inds_row": accel.bf_inds_row,
            "bf_inds_col": accel.bf_inds_col,
            "cache": accel._cache,
            "pk_buffer": accel._pk_buffer,
            "result_buffer": accel._result_buffer,
            "mean_phase_buffer": accel._mean_phase_buffer,
        }

    def _enter_drag(self) -> None:
        """Swap engine to BF-subset drag state (pointer swap, no copies)."""
        s = self._drag_state
        a = self._accel
        a.G_qk = s["G_qk"]
        a.bf_inds_row = s["bf_inds_row"]
        a.bf_inds_col = s["bf_inds_col"]
        a._cache = s["cache"]
        a._pk_buffer = s["pk_buffer"]
        a._result_buffer = s["result_buffer"]
        a._corrected_buffer = None
        a._mean_phase_buffer = s["mean_phase_buffer"]

    def _exit_drag(self) -> None:
        """Restore engine to full BF state."""
        a = self._accel
        # Persist any buffers the engine may have (re)allocated
        self._drag_state["result_buffer"] = a._result_buffer
        self._drag_state["mean_phase_buffer"] = a._mean_phase_buffer
        # Restore full state
        s = self._full_state
        a.G_qk = s["G_qk"]
        a.bf_inds_row = s["bf_inds_row"]
        a.bf_inds_col = s["bf_inds_col"]
        a._cache = s["cache"]
        a._pk_buffer = s["pk_buffer"]
        a._result_buffer = s["result_buffer"]
        a._corrected_buffer = None
        a._mean_phase_buffer = s["mean_phase_buffer"]

    def free_drag_state(self) -> None:
        """Explicitly release drag preview VRAM."""
        self._drag_state = None
        self._full_state = None

    def _on_drag_bf_change(self, change):
        """Rebuild or drop the BF-subset drag state when the user changes count."""
        new_count = int(change["new"])
        if new_count <= 0:
            self.free_drag_state()
            return
        # Clamp to valid range and rebuild. Existing state is dropped first to release VRAM.
        num_bf = int(self._accel._cache["num_bf"])
        count = max(1, min(new_count, num_bf))
        self._drag_state = None
        self._full_state = None
        if getattr(self._accel, "backend", None) == "mps":
            # MPS live reconstruction owns its BF selection inside the prepared
            # backend object. WebGPU folder mode uses this trait directly in the
            # browser, so keep the positive count without creating a fake Python
            # subset state.
            return
        self._build_drag_state(count)

    def _on_rotation_change(self, change):
        """Re-cache the rotation on the engine and re-reconstruct with current aberrations."""
        if getattr(self, "_rotation_deg_init", False):
            return  # initial trait assignment during __init__; skip
        new_deg = float(change["new"])
        new_rad = math.radians(new_deg)
        self._rotation_rad = new_rad
        # cache_rotation invalidates the drag BF subset (indices into kx/ky_bf change),
        # so drop drag state and rebuild on next drag if caller enabled it.
        drag_count = int(self.drag_bf)
        if drag_count > 0:
            self._drag_state = None
            self._full_state = None
        self._accel.cache_rotation(new_rad)
        if drag_count > 0:
            self._build_drag_state(drag_count)
        # Re-reconstruct with the last-committed aberrations.
        self._inflight_id += 1
        self._do_reconstruct(
            self._inflight_id,
            self.auto_c10 if self._last_phase_np is None else self._current_c10(),
            self.auto_c12 if self._last_phase_np is None else self._current_c12(),
            self.auto_phi12_deg if self._last_phase_np is None else self._current_phi12_deg(),
        )

    def _on_higher_order_change(self, change):
        """Re-reconstruct when the higher-order panel values change.

        Triggered by JS writing the ``higher_order_json`` trait.  We reuse
        the currently-displayed C10/C12/phi12 (from the last request) and
        just re-run the pipeline - the 14-coef kernel picks up whatever is
        now in the JSON.
        """
        if self._last_phase_np is None:
            return
        self._inflight_id += 1
        self._do_reconstruct(
            self._inflight_id,
            self._current_c10(),
            self._current_c12(),
            self._current_phi12_deg(),
            compute_loss=True,
        )

    def _on_flip_change(self, change):
        """Re-send the current phase with the new sign; no GPU recompute needed.

        Toggling flip is an involution: every toggle event means "negate the
        currently-displayed phase".  The _do_reconstruct path always starts
        from raw GPU output and re-applies flip_phase, so the cache state
        stays consistent across reconstruct + toggle events.
        """
        if self._last_phase_np is None:
            return
        self._last_phase_np = -self._last_phase_np
        self.phase_bytes = self._last_phase_np.astype(np.float32, copy=False).tobytes()

    def _current_c10(self) -> float:
        return float(self._last_result.get("C10", self.auto_c10))

    def _current_c12(self) -> float:
        return float(self._last_result.get("C12", self.auto_c12))

    def _current_phi12_deg(self) -> float:
        return float(self._last_result.get("phi12_deg", self.auto_phi12_deg))

    # ------------------------------------------------------------------
    #  Event handlers
    # ------------------------------------------------------------------

    @property
    def pinned(self) -> list[dict]:
        """All pinned snapshots (params + loss)."""
        return self._pinned

    @property
    def starred(self) -> list[dict]:
        """Starred pinned snapshots - the time-series payload.

        Each entry includes the aberration values, rotation, flip sign,
        loss, and an ISO-8601 timestamp.  Same content as
        ``self._stars_path`` on disk, minus the raw phase array.
        """
        return [
            {k: v for k, v in p.items() if k not in ("phase",)}
            for p in self._pinned
            if p.get("starred")
        ]

    def _build_trials_payload(self, max_trials: int = 50) -> str:
        """Serialize ``ssb._optuna_trials`` for the widget's trials panel.

        The SSB optimizer stores trials as
        ``list[{"loss": float, "params": {"C10_nm", "C12_nm", "phi12_deg"}}]``.
        We sort by ascending loss (best first), rename to the widget's
        ``C10 / C12 / phi12_deg`` convention, and cap at ``max_trials`` so a
        200-trial Optuna run doesn't bloat the initial trait payload.
        """
        ssb = self._ssb_ref
        raw = getattr(ssb, "_optuna_trials", None) if ssb is not None else None
        if not raw:
            return ""
        sorted_trials = sorted(raw, key=lambda t: t.get("loss", float("inf")))[:max_trials]
        payload = []
        for rank, trial in enumerate(sorted_trials):
            p = trial.get("params", {})
            payload.append({
                "rank": rank,
                "C10": float(p.get("C10_nm", 0.0)),
                "C12": float(p.get("C12_nm", 0.0)),
                "phi12_deg": float(p.get("phi12_deg", 0.0)),
                "loss": float(trial.get("loss", 0.0)),
            })
        return json.dumps(payload)

    def _write_stars_file(self) -> None:
        """Atomically dump starred entries to ``self._stars_path``.

        Each entry is written in the canonical Calibration shape (nested
        ``aberrations`` dict, phi12 in radians, ``rotation_angle_deg``,
        microscope metadata pulled off ``self._ssb_ref``) plus the three
        widget extras ``id`` / ``timestamp`` / ``starred``.  This means
        ``load_calibrations(stars_path)`` returns a ``list[Calibration]``
        directly - the bridge for time-series work in ``3_live.ipynb``.
        """
        self._stars_path.parent.mkdir(parents=True, exist_ok=True)

        ssb = self._ssb_ref
        voltage_kV = getattr(ssb, "voltage_kV", None) if ssb is not None else None
        semiangle_mrad = getattr(ssb, "semiangle_mrad", None) if ssb is not None else None
        scan_sampling = getattr(ssb, "scan_sampling", None) if ssb is not None else None
        if isinstance(scan_sampling, (tuple, list)):
            scan_sampling_A = float(scan_sampling[0])
        elif scan_sampling is not None:
            scan_sampling_A = float(scan_sampling)
        else:
            scan_sampling_A = None

        # Layout for unpacking the panel JSON into canonical names/angles.
        ho_layout = [
            ("C21",  True),  ("C23", True),
            ("C30",  False), ("C32", True), ("C34", True),
            ("C41",  True),  ("C43", True), ("C45", True),
            ("C50", False),  ("C52", True), ("C54", True), ("C56", True),
        ]

        payload = []
        for p in self._pinned:
            if not p.get("starred"):
                continue
            aberr = {
                "C10": float(p.get("C10", 0.0)),
                "C12": float(p.get("C12", 0.0)),
                "phi12": math.radians(float(p.get("phi12_deg", 0.0))),
            }
            # Merge starred higher-order coefs into the aberrations dict using
            # the canonical {name: value} / {"phi<n><m>": rad} shape so that
            # ``load_calibrations(stars_path)`` and ``SSB.reconstruct_full`` can
            # consume the file without translation.  Missing/zero values are
            # omitted to keep the JSON small.
            ho = p.get("higher_order", {}) or {}
            for name, has_angle in ho_layout:
                if has_angle:
                    mag = float(ho.get(f"{name}_mag", 0.0))
                    ang_deg = float(ho.get(f"{name}_angle", 0.0))
                else:
                    mag = float(ho.get(name, 0.0))
                    ang_deg = 0.0
                if mag == 0.0:
                    continue
                aberr[name] = mag
                if has_angle:
                    # Canonical name: phi21, phi23, phi32, ... (drop the 'C').
                    aberr[f"phi{name[1:]}"] = math.radians(ang_deg)
            payload.append({
                "id": p.get("id"),
                "timestamp": p.get("timestamp"),
                "starred": True,
                "rotation_angle_deg": float(p.get("rotation_deg", math.degrees(self._rotation_rad))),
                "aberrations": aberr,
                "flip_phase": bool(p.get("flip_phase", False)),
                "voltage_kV": voltage_kV,
                "semiangle_mrad": semiangle_mrad,
                "scan_sampling_A": scan_sampling_A,
                "loss": float(p["loss"]) if p.get("loss") is not None else None,
                "source_file": p.get("source_file"),
                "notes": None,
            })

        tmp = self._stars_path.with_suffix(self._stars_path.suffix + ".tmp")
        with open(tmp, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        tmp.replace(self._stars_path)

    def _on_request(self, change):
        """Handle reconstruction request from JS."""
        raw = change["new"]
        if not raw:
            return
        req = json.loads(raw)
        self._inflight_id = req["id"]
        self._do_reconstruct(
            req["id"], req["c10"], req["c12"], req["phi12_deg"],
            compute_loss=req.get("committed", False),
        )

    def _higher_order_arrays(
        self, c10: float, c12: float, phi12_deg: float,
    ) -> "tuple[object, object, bool]":
        """Read ``higher_order_json`` and pack all 14 Krivanek coefs into
        backend coefficient arrays ready for ``reconstruct_full``.

        The first two slots carry ``(C10, C12, phi12)`` from the existing
        3-slider UI so the legacy control path keeps working.  Returns
        ``(mags_m, angles_rad, any_higher_order_active)``.

        mags are in the engine's magnitude convention (nm-valued numeric -
        the engine's 2-term kernel treats the nm float as the coefficient,
        so the 14-coef kernel must do the same to match).
        """
        # Reuse scratch buffers allocated in __init__.  fill(0) is a single
        # memset, far cheaper than allocating two fresh (14,) arrays per frame.
        xp = _array_module_for_accel(self._accel)
        mags = self._ho_mags_buf
        angs = self._ho_angs_buf
        mags.fill(0)
        angs.fill(0)
        # C10, C12, phi12 come from the main sliders (nm, nm, rad).
        mags[0] = xp.float32(c10)
        mags[1] = xp.float32(c12)
        angs[1] = xp.float32(math.radians(phi12_deg))

        # Higher-order values come from the panel JSON.
        ho = json.loads(self.higher_order_json or "{}")
        any_active = False
        # (name, index, has_angle) for the 11 higher-order coefficients
        layout = [
            ("C21",  2, True), ("C23", 3, True),
            ("C30",  4, False), ("C32", 5, True), ("C34", 6, True),
            ("C41",  7, True),  ("C43", 8, True), ("C45", 9, True),
            ("C50", 10, False), ("C52", 11, True),
            ("C54", 12, True),  ("C56", 13, True),
        ]
        for name, idx, has_angle in layout:
            if has_angle:
                mag = float(ho.get(f"{name}_mag", 0.0))
                ang_deg = float(ho.get(f"{name}_angle", 0.0))
            else:
                mag = float(ho.get(name, 0.0))
                ang_deg = 0.0
            if mag != 0.0:
                any_active = True
            mags[idx] = xp.float32(mag)
            angs[idx] = xp.float32(math.radians(ang_deg))
        return mags, angs, any_active

    def _do_reconstruct(
        self,
        rid: int,
        c10: float,
        c12: float,
        phi12_deg: float,
        compute_loss: bool = True,
    ):
        """Run GPU reconstruction and push result to JS.

        Default (no higher-order active) uses the fast 2-term
        ``reconstruct_with_loss`` path (78 ms on 512×512).  When any
        higher-order slider is non-zero, routes through
        ``SSBEngine.reconstruct_full_with_loss`` with the full 14-coef
        Krivanek polynomial.  The variance loss is the same BF-pixel
        phase-variance metric the 3-param optimizer uses, so it is
        directly comparable across both paths and the user can watch it
        drop as they tune higher-order sliders.
        """
        phi12_rad = math.radians(phi12_deg)
        t0 = time.perf_counter()
        use_drag = not compute_loss and self._drag_state is not None
        mags_m, angles_rad, any_ho = self._higher_order_arrays(c10, c12, phi12_deg)

        # Higher-order path uses the full 14-coef kernel; the 3-param fast path
        # is the common case. During drag (compute_loss=False) we skip the
        # variance pass so slider frames stay under the drag budget.
        if use_drag:
            self._enter_drag()
        try:
            if any_ho and compute_loss:
                phase, loss_val = self._accel.reconstruct_full_with_loss(mags_m, angles_rad)
                loss = float(loss_val)
            elif any_ho:
                phase = self._accel.reconstruct_full(mags_m, angles_rad)
                loss = None
            elif compute_loss:
                phase, loss_val = self._accel.reconstruct_with_loss(c10, c12, phi12_rad)
                loss = float(loss_val)
            else:
                phase = self._accel.reconstruct(c10, c12, phi12_rad)
                loss = None
            t_gpu = time.perf_counter()
            phase_np = _to_numpy(phase)
            if phase_np.dtype != np.float32:
                phase_np = phase_np.astype(np.float32, copy=False)
            t_d2h = time.perf_counter()
        finally:
            if use_drag:
                self._exit_drag()

        if rid != self._inflight_id:
            return

        # Apply flip-phase sign convention BEFORE caching/sending.  SSB's phase
        # has an inherent ± ambiguity; we let the user pick the convention that
        # matches expected sample contrast.  Cached value is the displayed one.
        if bool(self.flip_phase):
            phase_np = -phase_np
        self._last_phase_np = phase_np

        # tobytes copies the ndarray into a Python bytes object. For 512×512 float32
        # = 1 MB; expected ~1-2 ms on a modern CPU.
        payload = phase_np.tobytes()
        t_bytes = time.perf_counter()

        h, w = phase_np.shape
        self.phase_height = h
        self.phase_width = w
        # This assignment triggers traitlets sync → Comm message to the frontend.
        # The work measured here is Python-side only (serialization + queue enqueue);
        # the actual wire time is Comm and lives in (UI − GPU − JS).
        self.phase_bytes = payload
        t_trait = time.perf_counter()

        entry = {
            "id": rid,
            "C10": round(c10, 2),
            "C12": round(c12, 2),
            "phi12_deg": round(phi12_deg, 2),
            "loss": loss,
            "time_ms": round((t_gpu - t0) * 1000, 1),        # GPU kernel only
            "d2h_ms":  round((t_d2h - t_gpu) * 1000, 1),     # cp.asnumpy + dtype
            "bytes_ms": round((t_bytes - t_d2h) * 1000, 1),   # ndarray.tobytes
            "trait_ms": round((t_trait - t_bytes) * 1000, 1), # Comm enqueue + sync broadcast
            "py_total_ms": round((t_trait - t0) * 1000, 1),   # everything on Python side
        }
        # Cache the dict so observers can read current slider values without
        # re-parsing result_json on every tick.
        self._last_result = entry
        self.result_json = json.dumps(entry)

    def _on_pin(self, change):
        """Handle pin/unpin event from JS."""
        raw = change["new"]
        if not raw:
            return
        evt = json.loads(raw)

        action = evt.get("action")
        if action == "pin":
            # source_file is captured AT PIN TIME so stars written later still
            # know which file this snapshot came from, even if the widget is
            # reused against a different dataset in the same kernel.
            # Capture the higher-order panel state at pin time.  A session
            # can pin multiple snapshots with different higher-order tunings,
            # and each one must remember the exact 14-coef configuration it
            # was taken with - otherwise re-viewing an old star would silently
            # fall back to whatever higher_order_json currently holds.
            higher_order_snapshot = json.loads(self.higher_order_json or "{}")
            pin_entry = {
                "id": evt.get("id"),
                "C10": evt.get("C10"),
                "C12": evt.get("C12"),
                "phi12_deg": evt.get("phi12_deg"),
                "rotation_deg": evt.get("rotation_deg", math.degrees(self._rotation_rad)),
                "flip_phase": bool(evt.get("flip_phase", self.flip_phase)),
                "loss": evt.get("loss"),
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "starred": False,
                "source_file": self._source_file,
                "higher_order": higher_order_snapshot,
            }
            if self._last_phase_np is not None:
                pin_entry["phase"] = self._last_phase_np.copy()
            self._pinned.append(pin_entry)
        elif action == "star" or action == "unstar":
            pin_id = evt.get("id")
            want_starred = (action == "star")
            for pin_entry in self._pinned:
                if pin_entry.get("id") != pin_id:
                    continue
                pin_entry["starred"] = want_starred
                # Refresh the timestamp on star so stars reflect when the user
                # actually decided to keep the snapshot (not when it was pinned).
                if want_starred:
                    pin_entry["timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")
                break
            self._write_stars_file()
        elif action == "view":
            pin_id = evt.get("id")
            for pin_entry in self._pinned:
                if pin_entry.get("id") != pin_id:
                    continue
                phase = pin_entry.get("phase")
                if phase is not None:
                    self._last_phase_np = np.array(phase, copy=True)
                    h, w = self._last_phase_np.shape
                    self.phase_height = h
                    self.phase_width = w
                    self.phase_bytes = self._last_phase_np.astype(np.float32).tobytes()
                self.result_json = json.dumps({
                    "id": pin_entry.get("id"),
                    "C10": round(float(pin_entry.get("C10", 0.0)), 2),
                    "C12": round(float(pin_entry.get("C12", 0.0)), 2),
                    "phi12_deg": round(float(pin_entry.get("phi12_deg", 0.0)), 2),
                    "loss": pin_entry.get("loss"),
                    "time_ms": None,
                })
                break
        elif action == "unpin":
            pin_id = evt.get("id")
            removed_starred = any(
                p.get("id") == pin_id and p.get("starred") for p in self._pinned
            )
            self._pinned = [p for p in self._pinned if p.get("id") != pin_id]
            if removed_starred:
                self._write_stars_file()

    def _on_save(self, change):
        """Write the currently-viewed aberrations as calibration.json.

        Mirrors the chosen aberrations onto ``self._ssb_ref.aberrations`` so
        any Python code still holding the SSB instance sees the update.
        """
        if self._last_phase_np is None or not self.result_json:
            return

        latest = json.loads(self.result_json)
        c10 = float(latest.get("C10", latest.get("c10", 0.0)))
        c12 = float(latest.get("C12", latest.get("c12", 0.0)))
        phi12_deg = float(latest.get("phi12_deg", 0.0))
        phi12_rad = math.radians(phi12_deg)
        loss = latest.get("loss")

        if self._ssb_ref is not None:
            self._ssb_ref.aberrations["C10"] = c10
            self._ssb_ref.aberrations["C12"] = c12
            self._ssb_ref.aberrations["phi12"] = phi12_rad
            if loss is not None:
                self._ssb_ref._best_loss = loss

        # Pull microscope metadata off the SSB instance - single source of
        # truth, avoids duplicating voltage/semiangle/scan_sampling in the
        # widget's constructor.
        ssb = self._ssb_ref
        voltage_kV = getattr(ssb, "voltage_kV", None) if ssb is not None else None
        semiangle_mrad = getattr(ssb, "semiangle_mrad", None) if ssb is not None else None
        scan_sampling = getattr(ssb, "scan_sampling", None) if ssb is not None else None
        if isinstance(scan_sampling, (tuple, list)):
            scan_sampling_A = float(scan_sampling[0])
        elif scan_sampling is not None:
            scan_sampling_A = float(scan_sampling)
        else:
            scan_sampling_A = None
        source_file = self._source_file

        # Merge in every non-zero higher-order coefficient from the HO panel.
        # The widget stores magnitudes in nm and angles in degrees; we keep
        # that convention in calibration.json so load-then-restore is a plain
        # copy of the dict.  phi12 remains in radians for historical reasons
        # (matches the existing Calibration contract downstream code expects).
        aberrations = {"C10": c10, "C12": c12, "phi12": phi12_rad}
        ho_dict = json.loads(self.higher_order_json or "{}")
        for k, v in ho_dict.items():
            fv = float(v)
            if k.endswith("_angle"):
                aberrations[k] = fv            # degrees
            elif abs(fv) > 0:
                aberrations[k] = fv            # nm (magnitude or single-coef like C30)

        cal_kwargs = dict(
            rotation_angle_deg=math.degrees(self._rotation_rad),
            aberrations=aberrations,
            flip_phase=bool(self.flip_phase),
            voltage_kV=voltage_kV,
            semiangle_mrad=semiangle_mrad,
            scan_sampling_A=scan_sampling_A,
            loss=float(loss) if loss is not None else None,
            source_file=source_file,
        )
        if (self.notes or None) is not None:
            cal_kwargs["notes"] = self.notes
        cal = PtychoCalibration(**cal_kwargs)
        saved = save_ptycho_calibration(cal, self._calibration_path)
        self.calibration_path = str(saved.resolve())
        self.calibration_saved_at = datetime.datetime.now().isoformat(timespec="seconds")

    def __repr__(self):
        n = len(self._pinned)
        drag = ""
        if self._drag_state is not None:
            bf = int(self._drag_state["cache"]["num_bf"])
            drag = f", drag_bf={bf}"
        return f"ShowPtycho({n} pinned{drag})"

    def export_webgpu_folder(
        self,
        out_dir: str | pathlib.Path,
        *,
        title: str | None = None,
        overwrite: bool = True,
        decode_dtype: str = "uint16",
    ) -> pathlib.Path:
        """Export a kernel-less WebGPU folder for the current ShowPtycho state."""

        from quantem.widget.showptycho_webgpu_export import (
            export_showptycho_webgpu_folder,
        )

        return export_showptycho_webgpu_folder(
            self, out_dir, title=title, overwrite=overwrite, decode_dtype=decode_dtype,
        )

    def export_webgpu_sidecar(
        self,
        out_dir: str | pathlib.Path,
        *,
        title: str | None = None,
        overwrite: bool = True,
        decode_dtype: str = "uint16",
    ) -> pathlib.Path:
        """Compatibility alias for :meth:`export_webgpu_folder`."""

        return self.export_webgpu_folder(
            out_dir, title=title, overwrite=overwrite, decode_dtype=decode_dtype,
        )

    def export_sidecar(
        self,
        out_dir: str | pathlib.Path,
        *,
        title: str | None = None,
        overwrite: bool = True,
        decode_dtype: str = "uint16",
    ) -> pathlib.Path:
        """Compatibility alias for :meth:`export_webgpu_folder`."""

        return self.export_webgpu_folder(
            out_dir, title=title, overwrite=overwrite, decode_dtype=decode_dtype,
        )


def _is_ssb_like(obj: object) -> bool:
    return (
        hasattr(obj, "_get_accelerator")
        and hasattr(obj, "aberrations")
        and hasattr(obj, "_rotation_angle_rad")
    )


def _scan_sampling_scalar(ssb: object) -> float:
    scan_sampling = getattr(ssb, "scan_sampling", 0.0)
    if isinstance(scan_sampling, (tuple, list)):
        return float(scan_sampling[0])
    return float(scan_sampling or 0.0)


def _voltage_kv_from_inputs(
    voltage_kV: float | None,
    energy: float | None,
) -> float:
    if voltage_kV is not None:
        return float(voltage_kV)
    if energy is None:
        raise ValueError(
            "ShowPtycho(data, ...) requires voltage_kV or energy when using "
            "the MPS data path."
        )
    value = float(energy)
    return value / 1000.0 if value > 1000.0 else value


def _looks_like_mps_data(data: object) -> bool:
    if hasattr(data, "_fields") and "data" in getattr(data, "_fields", ()):
        data = data.data
    module = type(data).__module__
    if "quantem.gpu" in module and "mps" in module:
        return True
    if hasattr(data, "chunks"):
        return True
    backend = getattr(data, "backend", None)
    return str(backend).lower() == "mps"


class _MpsPtychoAccelerator:
    """Adapter from the MLX MPS SSB preview backend to ``ShowPtycho``."""

    backend = "mps"

    def __init__(
        self,
        data: object,
        *,
        voltage_kV: float,
        semiangle_mrad: float,
        scan_sampling: float | tuple[float, float],
        det_sampling: float | tuple[float, float] | None,
        bf_intensity_threshold: float,
        bf_radius: int | None,
        rotation_angle_deg: float,
        chunk_bf: int = 16,
    ) -> None:
        from quantem.gpu.ssb.mps import (
            _as_chunked_frames,
            _as_sampling,
            _bf_pixels,
            _prepare_selection,
            _reconstruct_prepared,
            _scan_shape,
        )

        self._mps_reconstruct_prepared = _reconstruct_prepared
        self._mps_prepare_selection = _prepare_selection
        self._frames = _as_chunked_frames(data)
        self._scan_shape = _scan_shape(self._frames)
        self._det_shape = tuple(int(x) for x in self._frames.shape[-2:])
        self._voltage_kV = float(voltage_kV)
        self._semiangle_mrad = float(semiangle_mrad)
        self._scan_sampling = _as_sampling(scan_sampling)
        self._chunk_bf = max(1, int(chunk_bf))
        self._bf_row, self._bf_col, self._bf_center, self._bf_radius, detected_radius = (
            _bf_pixels(self._frames, bf_intensity_threshold, bf_radius)
        )
        if det_sampling is None:
            det_px = (2.0 * float(semiangle_mrad)) / float(detected_radius)
            self._det_sampling = (det_px, det_px)
        else:
            self._det_sampling = _as_sampling(det_sampling)
        self._rotation_angle_deg = float(rotation_angle_deg)
        self._cache = {
            "num_bf": int(self._bf_row.size),
            "ny": int(self._scan_shape[0]),
            "nx": int(self._scan_shape[1]),
        }
        self._prepared = None
        self._mean_phase_buffer = None
        self._sumsq_buffer = None
        self.cache_rotation(math.radians(self._rotation_angle_deg))

    @property
    def num_bf(self) -> int:
        return int(self._cache["num_bf"])

    def cache_rotation(self, rotation_rad: float) -> None:
        rotation_angle_deg = math.degrees(float(rotation_rad))
        if (
            self._prepared is not None
            and abs(rotation_angle_deg - self._rotation_angle_deg) < 1e-9
        ):
            if not hasattr(self, "G_qk"):
                self._sync_webgpu_export_state()
            return
        self._rotation_angle_deg = rotation_angle_deg
        self._prepared = self._mps_prepare_selection(
            self._frames,
            scan_shape=self._scan_shape,
            det_shape=self._det_shape,
            bf_row=self._bf_row,
            bf_col=self._bf_col,
            center=self._bf_center,
            voltage_kV=self._voltage_kV,
            semiangle_mrad=self._semiangle_mrad,
            scan_sampling=self._scan_sampling,
            det_sampling=self._det_sampling,
            rotation_angle_deg=self._rotation_angle_deg,
            chunk_bf=self._chunk_bf,
        )
        self._sync_webgpu_export_state()

    def _sync_webgpu_export_state(self) -> None:
        """Expose MPS-prepared BF-G data through the CUDA-style export contract."""

        prepared = self._prepared
        if prepared is None:
            return
        qx_1d = np.fft.fftfreq(
            int(self._scan_shape[0]), float(self._scan_sampling[0]),
        ).astype(np.float32)
        qy_1d = np.fft.fftfreq(
            int(self._scan_shape[1]), float(self._scan_sampling[1]),
        ).astype(np.float32)
        alpha_k2, cos2_k, sin2_k, aperture_k = _bf_geometry_1d_numpy(
            prepared.kx_np,
            prepared.ky_np,
            wavelength=float(prepared.wavelength),
            semiangle_rad=float(prepared.semiangle_rad),
            ang_y_rad=float(prepared.ang_y_rad),
            ang_x_rad=float(prepared.ang_x_rad),
        )
        self.G_qk = np.asarray(prepared.g_qk).astype(np.complex64, copy=False)
        self.bf_inds_row = self._bf_row.astype(np.int32, copy=False)
        self.bf_inds_col = self._bf_col.astype(np.int32, copy=False)
        self.bf_center = tuple(float(v) for v in self._bf_center)
        self.gpts = tuple(int(v) for v in self._det_shape)
        self.wavelength = np.float32(prepared.wavelength)
        self.sampling = (
            float(1.0 / ((float(prepared.ang_y_rad) / float(prepared.wavelength)) * self._det_shape[0])),
            float(1.0 / ((float(prepared.ang_x_rad) / float(prepared.wavelength)) * self._det_shape[1])),
        )
        self._dc_value_host = np.complex64(prepared.dc_value)
        self._cache = {
            "num_bf": int(prepared.num_bf),
            "ny": int(self._scan_shape[0]),
            "nx": int(self._scan_shape[1]),
            "kx_bf": prepared.kx_np.astype(np.float32, copy=False),
            "ky_bf": prepared.ky_np.astype(np.float32, copy=False),
            "qx_1d": qx_1d,
            "qy_1d": qy_1d,
            "aperture_k_1d": aperture_k,
            "alpha_k2_1d": alpha_k2,
            "cos2phi_k_1d": cos2_k,
            "sin2phi_k_1d": sin2_k,
            "semiangle_rad": np.float32(prepared.semiangle_rad),
            "ang_y_rad": np.float32(prepared.ang_y_rad),
            "ang_x_rad": np.float32(prepared.ang_x_rad),
        }

    def reconstruct_with_loss(self, c10: float, c12: float, phi12: float):
        _object_wave, loss, phase = self._mps_reconstruct_prepared(
            self._prepared,
            C10=float(c10),
            C12=float(c12),
            phi12=float(phi12),
            chunk_bf=self._chunk_bf,
            compute_loss=True,
            compute_object=True,
        )
        phase_np = np.asarray(phase, dtype=np.float32)
        self._mean_phase_buffer = phase_np
        self._sumsq_buffer = phase_np * phase_np * float(self._cache["num_bf"])
        return phase, float(loss)

    def reconstruct(self, c10: float, c12: float, phi12: float):
        _object_wave, _loss, phase = self._mps_reconstruct_prepared(
            self._prepared,
            C10=float(c10),
            C12=float(c12),
            phi12=float(phi12),
            chunk_bf=self._chunk_bf,
            compute_loss=False,
            compute_object=True,
        )
        return phase

    def _three_param_from_full(self, mags_m, angles_rad) -> tuple[float, float, float]:
        mags = np.asarray(mags_m, dtype=np.float32)
        angles = np.asarray(angles_rad, dtype=np.float32)
        if np.any(mags[2:] != 0):
            raise NotImplementedError(
                "ShowPtycho on MPS currently supports C10/C12/phi12 only. "
                "Higher-order controls require the CUDA SSB engine."
            )
        return float(mags[0]), float(mags[1]), float(angles[1])

    def reconstruct_full_with_loss(self, mags_m, angles_rad):
        c10, c12, phi12 = self._three_param_from_full(mags_m, angles_rad)
        return self.reconstruct_with_loss(c10, c12, phi12)

    def reconstruct_full(self, mags_m, angles_rad):
        c10, c12, phi12 = self._three_param_from_full(mags_m, angles_rad)
        return self.reconstruct(c10, c12, phi12)


class _MpsPtychoState:
    """Minimal SSB-like state object for ``ShowPtycho`` on MPS data."""

    def __init__(
        self,
        *,
        accel: _MpsPtychoAccelerator,
        aberrations: dict[str, float] | None,
        rotation_angle_deg: float,
        voltage_kV: float,
        semiangle_mrad: float,
        scan_sampling: float | tuple[float, float],
    ) -> None:
        defaults = {"C10": 0.0, "C12": 0.0, "phi12": 0.0}
        if aberrations:
            defaults.update(
                {k: float(v) for k, v in aberrations.items() if k in defaults}
            )
        self.aberrations = defaults
        self._rotation_angle_rad = math.radians(float(rotation_angle_deg))
        self._best_loss = float("inf")
        self._accel = accel
        self.voltage_kV = float(voltage_kV)
        self.semiangle_mrad = float(semiangle_mrad)
        self.scan_sampling = scan_sampling

    def _get_accelerator(self):
        return self._accel


def _apply_calibration(
    ssb: object,
    calibration: object,
    source_file: str | None,
) -> tuple[bool, dict[str, float], str | None]:
    cal = _coerce_calibration(calibration)
    primary = {
        "C10": float(cal.aberrations.get("C10", 0.0)),
        "C12": float(cal.aberrations.get("C12", 0.0)),
        "phi12": float(cal.aberrations.get("phi12", 0.0)),
    }
    for key, value in {**cal.higher_order, **cal.aberrations}.items():
        if key not in primary:
            primary[str(key)] = float(value)
    ssb.aberrations = primary
    ssb._rotation_angle_rad = math.radians(float(cal.rotation_angle_deg))
    if cal.loss is not None:
        ssb._best_loss = float(cal.loss)
    return (
        bool(cal.flip_phase),
        _higher_order_widget_payload(cal),
        source_file or cal.source_file,
    )


def _show_ptycho_from_ssb(
    ssb: object,
    *,
    c10_range: tuple[float, float] | None,
    c12_range: tuple[float, float] | None,
    phi12_range: tuple[float, float] | None,
    rotation_range: tuple[float, float] | None,
    drag_bf: int | float | None,
    save_dir: str | pathlib.Path | None,
    source_file: str | None,
    size: int,
    fft_on: bool,
    calibration: object | None,
    webgpu_preview: bool | str,
) -> _ShowPtychoWidget:
    flip_from_cal: bool | None = None
    ho_from_cal: dict[str, float] | None = None
    if calibration is not None:
        flip_from_cal, ho_from_cal, source_file = _apply_calibration(
            ssb, calibration, source_file,
        )

    aberrations = dict(getattr(ssb, "aberrations", {}) or {})
    auto_c10 = float(aberrations.get("C10", 0.0))
    auto_c12 = float(aberrations.get("C12", 0.0))
    auto_phi12 = float(aberrations.get("phi12", 0.0))

    if c10_range is None:
        c10_range = (auto_c10 - 200.0, auto_c10 + 200.0)
    if c12_range is None:
        c12_range = (-100.0, 100.0)
    if phi12_range is None:
        phi12_range = (-90.0, 90.0)

    accel = ssb._get_accelerator()
    rotation_rad = float(getattr(ssb, "_rotation_angle_rad", 0.0))
    accel.cache_rotation(rotation_rad)
    _, auto_loss_full = accel.reconstruct_with_loss(
        auto_c10, auto_c12, auto_phi12,
    )

    widget = _ShowPtychoWidget(
        accel=accel,
        rotation_rad=rotation_rad,
        auto_aberrations=aberrations,
        auto_loss_val=float(auto_loss_full),
        c10_range=c10_range,
        c12_range=c12_range,
        phi12_range=phi12_range,
        rotation_range=rotation_range,
        drag_bf=drag_bf,
        save_dir=save_dir,
        ssb_ref=ssb,
        pixel_size=_scan_sampling_scalar(ssb),
        source_file=source_file,
        size=size,
        fft_on=fft_on,
        webgpu_preview=webgpu_preview,
    )

    if flip_from_cal is not None:
        widget.flip_phase = flip_from_cal
    if ho_from_cal:
        widget.higher_order_json = json.dumps(ho_from_cal)
    ssb._showptycho_widget = widget
    return widget


def ShowPtycho(
    data_or_ssb: object,
    *,
    semiangle: float | None = None,
    scan_sampling: float | tuple[float, float] | None = None,
    det_sampling: float | tuple[float, float] | None = None,
    voltage_kV: float | None = None,
    energy: float | None = None,
    scan_shape: tuple[int, int] | None = None,
    bf_intensity_threshold: float = 0.5,
    bf_radius: int | None = None,
    aberrations: dict[str, float] | None = None,
    rotation_angle_deg: float = 0.0,
    c10_range: tuple[float, float] | None = None,
    c12_range: tuple[float, float] | None = None,
    phi12_range: tuple[float, float] | None = None,
    rotation_range: tuple[float, float] | None = None,
    drag_bf: int | float | None = _DEFAULT_DRAG_BF_FRACTION,
    save_dir: str | pathlib.Path | None = None,
    source_file: str | None = None,
    size: int = 800,
    fft_on: bool = False,
    calibration: object | None = None,
    webgpu_preview: bool | str = "auto",
) -> _ShowPtychoWidget:
    """Open an interactive ptychography aberration explorer.

    Parameters
    ----------
    data_or_ssb : object
        Either a prepared ``quantem.gpu.ssb.SSB`` instance or a 4D-STEM array.
        Passing an SSB instance is the preferred path because it reuses the
        existing GPU-resident preprocessing buffers.
    semiangle : float, optional
        Probe semi-convergence angle in mrad. Required when passing raw data.
    scan_sampling : float or tuple[float, float], optional
        Scan sampling in Angstroms. Required when passing raw data.
    det_sampling : float or tuple[float, float], optional
        Detector angular sampling in mrad per pixel.
    voltage_kV, energy : float, optional
        Accelerating voltage. ``voltage_kV`` is the public notebook-friendly
        form; ``energy`` is accepted for compatibility.
    calibration : path or object, optional
        Previously saved calibration used to seed aberrations, rotation, phase
        flip, and higher-order controls.
    webgpu_preview : bool or {"auto", "off"}, optional
        Enable browser-side WebGPU SSB preview when the prepared BF-indexed
        payload is small enough for notebook sync. ``"auto"`` currently enables
        only the validated 128x128 C10/C12/phi12 path.

    Returns
    -------
    anywidget.AnyWidget
        ShowPtycho widget instance backed by the ``quantem.gpu`` SSB engine.
    """

    if _is_ssb_like(data_or_ssb):
        ssb = data_or_ssb
    else:
        if semiangle is None or scan_sampling is None:
            raise ValueError(
                "ShowPtycho(data, ...) requires semiangle and scan_sampling. "
                "Pass a prepared quantem.gpu.ssb.SSB object to reuse an "
                "existing GPU-resident reconstruction."
            )
        if _looks_like_mps_data(data_or_ssb):
            voltage = _voltage_kv_from_inputs(voltage_kV, energy)
            accel = _MpsPtychoAccelerator(
                data_or_ssb,
                voltage_kV=voltage,
                semiangle_mrad=float(semiangle),
                scan_sampling=scan_sampling,
                det_sampling=det_sampling,
                bf_intensity_threshold=bf_intensity_threshold,
                bf_radius=bf_radius,
                rotation_angle_deg=rotation_angle_deg,
            )
            ssb = _MpsPtychoState(
                accel=accel,
                aberrations=aberrations,
                rotation_angle_deg=rotation_angle_deg,
                voltage_kV=voltage,
                semiangle_mrad=float(semiangle),
                scan_sampling=scan_sampling,
            )
        else:
            try:
                import cupy as cp
                from quantem.gpu.ssb import SSB
            except ImportError as exc:
                try:
                    voltage = _voltage_kv_from_inputs(voltage_kV, energy)
                    accel = _MpsPtychoAccelerator(
                        data_or_ssb,
                        voltage_kV=voltage,
                        semiangle_mrad=float(semiangle),
                        scan_sampling=scan_sampling,
                        det_sampling=det_sampling,
                        bf_intensity_threshold=bf_intensity_threshold,
                        bf_radius=bf_radius,
                        rotation_angle_deg=rotation_angle_deg,
                    )
                except Exception:
                    raise ImportError(
                        "ShowPtycho(data, ...) requires either CuPy for the "
                        "CUDA quantem.gpu SSB engine or MLX for the MPS path."
                    ) from exc
                ssb = _MpsPtychoState(
                    accel=accel,
                    aberrations=aberrations,
                    rotation_angle_deg=rotation_angle_deg,
                    voltage_kV=voltage,
                    semiangle_mrad=float(semiangle),
                    scan_sampling=scan_sampling,
                )
            else:
                data_gpu = (
                    data_or_ssb
                    if isinstance(data_or_ssb, cp.ndarray)
                    else cp.asarray(data_or_ssb)
                )
                ssb = SSB(
                    data_gpu,
                    semiangle=float(semiangle),
                    scan_sampling=scan_sampling,
                    det_sampling=det_sampling,
                    voltage_kV=voltage_kV,
                    energy=energy,
                    scan_shape=scan_shape,
                    bf_intensity_threshold=bf_intensity_threshold,
                    bf_radius=bf_radius,
                    aberrations=aberrations,
                    rotation_angle_deg=rotation_angle_deg,
                )

    return _show_ptycho_from_ssb(
        ssb,
        c10_range=c10_range,
        c12_range=c12_range,
        phi12_range=phi12_range,
        rotation_range=rotation_range,
        drag_bf=drag_bf,
        save_dir=save_dir,
        source_file=source_file,
        size=size,
        fft_on=fft_on,
        calibration=calibration,
        webgpu_preview=webgpu_preview,
    )
