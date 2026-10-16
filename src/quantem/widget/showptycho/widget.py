"""ShowPtycho: view a ptychographic reconstruction, and tune SSB aberrations on a quantem.gpu session.

Given an array (a phase image, or a complex object wave), the widget shows the phase, the
amplitude of a complex wave, their FFT and histograms with a calibrated scale bar, on any
computer. Given a fitted ``quantem.gpu.SSB`` session it owns the session: every slider move asks
it for a transient ``preview`` (C10, C12, phi12, scan rotation, optional higher-order terms,
optional thick-sample tilt) and ships the float32 phase to the browser, which draws it and its
FFT. Save writes the chosen aberrations to ``calibration.json``; starring a pinned snapshot
appends it to ``showptycho_stars.json``.

Usage::

    from quantem.widget import ShowPtycho

    ShowPtycho(object_wave, sampling=0.2, units="Å")   # NumPy or torch, any device

    from quantem.gpu import SSB

    ssb = SSB.open("scan_master.h5", voltage_kV=300, semiangle_mrad=30, scan_sampling_A=(0.264, 0.264))
    ssb.find_aberrations()
    ShowPtycho(ssb, show_fft=True)
"""

import contextlib
import datetime
import json
import math
import pathlib
import uuid

import anywidget
import numpy as np
import torch
import traitlets

from quantem.widget.adapters import core as core_adapter
from quantem.widget.adapters import gpu as gpu_adapter
from quantem.widget.device import gpu_path_hint, no_gpu_path
from quantem.widget.showptycho.export import write_showptycho_folder
from quantem.widget.units import axis_pair, image_axes, per_axis, stated, wrap_degrees
from quantem.widget.utils.array import device_array

STARS_FILENAME = "showptycho_stars.json"
CALIBRATION_FILENAME = "calibration.json"
# every stored record names its layout; a reader takes the current value only
CALIBRATION_SCHEMA = "quantem.ssb-calibration/4"
STARS_SCHEMA = "quantem.ssb-stars/1"
MIN_CROP_SPAN = 32


def session_note() -> str:
    """The line an array input shows in place of the aberration controls, true for this machine.

    With quantem.gpu installed the missing piece is a session, so the line says how to open one;
    without it, the install command for this machine's GPU, or why this machine has no GPU route.
    A fixed line naming both extras told users with quantem.gpu installed to install it again.
    """
    if gpu_adapter.available():
        return (
            "Aberration tuning (C10, C12, rotation, Save) needs an SSB session: "
            'ssb = quantem.gpu.SSB.open("scan_master.h5", voltage_kV=..., semiangle_mrad=..., scan_sampling_A=(row_A, col_A)), '
            "ssb.find_aberrations(), then ShowPtycho(ssb)."
        )
    hint = gpu_path_hint()
    if hint:
        return f"Aberration tuning (C10, C12, rotation, Save) needs an SSB session from quantem.gpu: {hint}."
    return f"Aberration tuning (C10, C12, rotation, Save) needs an SSB session from quantem.gpu. {no_gpu_path()}."


def finite_or_none(value: object) -> float | None:
    """A finite float for JSON, or None for missing, non-numeric or non-finite values."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def utc_now() -> str:
    """The present time as ISO 8601 UTC ending in Z: the ``generated_at_utc`` every file of this widget carries.

    A local time without a zone, written before, could not be compared across machines.
    """
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def propagator_frame_tilt(tilt_row_mrad: float, tilt_col_mrad: float, scan_detector_rotation_deg: float) -> list[float]:
    """SSB sample tilt (scan frame) in the ptychography object frame: ``[[cos, sin], [-sin, cos]]`` of the scan-detector rotation.

    quantem.thick places scan positions in the object frame by the same rotation with no sign change, so this value
    seeds its reconstruction directly (swap the components only if that reconstruction transposed the scan axes).
    Verified on a logic-device dataset (rotation -8.6 deg): SSB (-10.3, +4.7) -> (-10.9, +3.1) mrad against
    ptychography's learned (-11.6, +2.8), 2.4 deg apart.
    """
    rotation = math.radians(float(scan_detector_rotation_deg))
    return [
        tilt_row_mrad * math.cos(rotation) + tilt_col_mrad * math.sin(rotation),
        -tilt_row_mrad * math.sin(rotation) + tilt_col_mrad * math.cos(rotation),
    ]


def write_json(path: pathlib.Path, payload: object) -> None:
    """Write JSON through a sibling temp file so a reader never sees a half-written calibration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".tmp")
    partial.write_text(json.dumps(payload, indent=2, default=str))
    partial.replace(path)


def has_higher_order(panel_json: str) -> bool:
    """Whether the higher-order panel JSON holds a non-zero magnitude (``C21_nm``, ...; its azimuths do not count).

    Upsampled output carries C10/C12 only, so the two validators use it to refuse a
    higher-order term and upsampling together.
    """
    return any(value != 0 for key, value in json.loads(panel_json or "{}").items() if key.endswith("_nm"))


class ShowPtycho(anywidget.AnyWidget):
    """Ptychographic reconstruction viewer, and SSB aberration explorer over a ``quantem.gpu.SSB`` session.

    Parameters
    ----------
    data : array_like, quantem Dataset2d or quantem.gpu.SSB
        A 2D reconstruction (NumPy, torch, CuPy or anything ``np.asarray`` takes) or a prepared
        SSB session. A ``Dataset2d`` gives its array and its sampling (a keyword that
        contradicts it raises). A real array is the phase (rad); a complex array is the object wave, shown as
        its phase (``angle``) and amplitude (``abs``). Tensors and CuPy arrays stay on their
        device until the 2D result is drawn. Arrays need no quantem.gpu; the aberration controls then stay
        hidden and one line says what they need. An SSB session, normally after
        ``ssb.find_aberrations()``, is reused for its resident bright-field evidence and
        exclusively driven while the widget is open.
    sampling : float or (row, col), optional
        Pixel size of an array in ``units``; it draws the scale bar and calibrates the FFT. An SSB
        session carries its own scan sampling.
    units : str, optional
        Length unit of ``sampling``: any length of ``quantem.widget.units``
        (``"angstrom"``, ``"Å"``, ``"A"``, ``"nm"``, ``"pm"``, ...); angstrom when omitted.
    save_dir : str or Path, optional
        SSB session: where ``calibration.json`` and ``showptycho_stars.json`` are written; the
        working directory by default.
    show_fft : bool, default False
        Open with the FFT panel visible.

    A session opened from its master (``SSB.open("scan_master.h5", ...)``) records the file as its
    ``source_path``: the toolbar then offers Crop and Refit SSB, which reloads only the selected
    scan region and fits it afresh, and every saved calibration and star names the file.
    """

    _esm = pathlib.Path(__file__).parent.parent / "static" / "showptycho.js"

    # slider ranges: the physical span of the SSB fit's default search
    C10_min_nm = traitlets.Float(-40.0).tag(sync=True)
    C10_max_nm = traitlets.Float(40.0).tag(sync=True)
    scan_detector_rotation_min_deg = traitlets.Float(-180.0).tag(sync=True)
    scan_detector_rotation_max_deg = traitlets.Float(180.0).tag(sync=True)
    # the physical scan-to-detector rotation as the slider holds it; files store it in [0, 360)
    scan_detector_rotation_deg = traitlets.Float(0.0).tag(sync=True)
    # SSB phase is ambiguous by sign; the user picks the convention matching the expected contrast
    flip_phase = traitlets.Bool(False).tag(sync=True)
    # the fitted reference the Reset button returns to
    fitted_C10_nm = traitlets.Float(0.0).tag(sync=True)
    fitted_C12_nm = traitlets.Float(0.0).tag(sync=True)
    fitted_phi12_deg = traitlets.Float(0.0).tag(sync=True)
    fitted_loss = traitlets.Float(0.0).tag(sync=True)
    fitted_scan_detector_rotation_deg = traitlets.Float(0.0).tag(sync=True)
    # browser -> Python reconstruction request {"id", "C10_nm", "C12_nm", "phi12_deg", "committed"}
    request_json = traitlets.Unicode("").tag(sync=True)
    phase_bytes = traitlets.Bytes(b"").tag(sync=True)
    phase_width = traitlets.Int(0).tag(sync=True)
    phase_height = traitlets.Int(0).tag(sync=True)
    result_json = traitlets.Unicode("").tag(sync=True)
    # an array: the pixel size [row, col] of its image in angstrom. A session: the scan step [row, col] in angstrom,
    # which the browser shares out over the output columns (upsampling_factor). [] when unknown or not that input.
    object_sampling_A = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    scan_sampling_A = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    # one-shot seed the browser reads at mount; the FFT switch takes over
    initial_fft_on = traitlets.Bool(False).tag(sync=True)
    # SSB alias-order output sampling (CUDA sessions), not image interpolation
    upsampling_factor = traitlets.Int(1).tag(sync=True)
    upsampling_available = traitlets.Bool(False).tag(sync=True)
    save_trigger = traitlets.Int(0).tag(sync=True)
    notes = traitlets.Unicode("").tag(sync=True)
    pin_json = traitlets.Unicode("").tag(sync=True)
    stars_path = traitlets.Unicode("").tag(sync=True)
    calibration_path = traitlets.Unicode("").tag(sync=True)
    calibration_saved_at = traitlets.Unicode("").tag(sync=True)
    # fit history for the trials strip: JSON list of {rank, C10_nm, C12_nm, phi12_deg, loss}, best first
    trials_json = traitlets.Unicode("").tag(sync=True)
    # bright-field pixels used while dragging; the full count is the right edge of the slider
    num_drag_bright_field_pixels = traitlets.Int(0).tag(sync=True)
    num_bright_field_pixels = traitlets.Int(0).tag(sync=True)
    # higher-order panel: {C21_nm, phi21_deg, ..., C30_nm, ..., C50_nm, ...}
    higher_order_aberrations_json = traitlets.Unicode("{}").tag(sync=True)
    # browser-side WebGPU SSB source, set by the folder export
    webgpu_preview_enabled = traitlets.Bool(False).tag(sync=True)
    webgpu_cal_json = traitlets.Unicode("").tag(sync=True)
    webgpu_h5_source_json = traitlets.Unicode("").tag(sync=True)
    # browser SSB status; only an exported folder runs SSB in the browser (a live notebook asks Python)
    webgpu_preview_status = traitlets.Unicode("").tag(sync=True)
    webgpu_standalone = traitlets.Bool(False).tag(sync=True)
    # crop and refit: rebuilds SSB from the raw scan region, not a display crop
    scan_rows = traitlets.Int(0).tag(sync=True)
    scan_cols = traitlets.Int(0).tag(sync=True)
    crop_refit_available = traitlets.Bool(False).tag(sync=True)
    crop_refit_status = traitlets.Unicode("").tag(sync=True)
    crop_refit_request_json = traitlets.Unicode("").tag(sync=True)
    # thick-sample SSB: {"sample_tilt_mrad": [row, col], "depth_spread_nm"}; depth spread 0 is standard SSB
    sample_json = traitlets.Unicode("{}").tag(sync=True)
    sample_available = traitlets.Bool(False).tag(sync=True)
    sample_fit_request = traitlets.Int(0).tag(sync=True)
    sample_fit_status = traitlets.Unicode("").tag(sync=True)
    sample_fit_json = traitlets.Unicode("").tag(sync=True)
    # an array input has no session: the browser hides the aberration controls and shows session_note
    ssb_session = traitlets.Bool(True).tag(sync=True)
    session_note = traitlets.Unicode("").tag(sync=True)
    # float32 amplitude of a complex object wave, row-major like phase_bytes; empty for a phase
    amplitude_bytes = traitlets.Bytes(b"").tag(sync=True)

    def __init__(
        self,
        data,
        *,
        sampling: float | tuple[float, float] | None = None,
        units: str | None = None,
        save_dir: str | pathlib.Path | None = None,
        show_fft: bool = False,
    ) -> None:
        is_array = not gpu_adapter.is_ssb(data)
        # an array's pixel size [row, col] in angstrom, read through the one unit table (the widget draws in angstrom)
        if is_array and sampling is None:
            # a unit stated with no step, for an image that records none, raises as in every image widget
            image_axes(None, units, data if core_adapter.is_dataset(data, ndim=2) else None)
        given = None if sampling is None or not is_array else axis_pair(
            per_axis(sampling, 2), per_axis("Å" if units is None else units, 2), ("length",), "image")
        if core_adapter.is_dataset(data, ndim=2):
            # a calibrated reconstruction is the source of its pixel size, axis by axis: the keyword may
            # repeat a recorded step and gives the step of an axis the dataset leaves open
            recorded = axis_pair(data.sampling, data.units, ("length",), "image")
            given = stated("sampling", None if given is None else given[1], None if recorded is None else recorded[1])
        elif given is not None:
            given = given[1]
        if is_array:
            data = device_array(data)  # a tensor or CuPy array stays on its GPU until the 2D result is drawn
        if not is_array and sampling is not None:
            raise ValueError("sampling describes an array; an SSB session carries its own scan_sampling_A.")
        super().__init__()
        self._ssb = None if is_array else data
        self._pinned: list[dict] = []
        self._last_phase: np.ndarray | None = None
        self._last_result: dict = {}
        self._inflight_id = -1
        self._drag_context = None
        self.initial_fft_on = bool(show_fft)
        self.observe(self._on_flip_change, names=["flip_phase"])
        if is_array:
            # angle and abs run where the data lives (a CUDA or MPS tensor stays there until the 2D
            # result is copied for the browser), in the input's precision; the browser draws float32,
            # so wider input is rounded once and the largest change is printed, never silently
            if data.ndim != 2:
                needs = "" if gpu_adapter.available() else f" (a session needs quantem.gpu: {gpu_path_hint() or no_gpu_path()})"
                raise ValueError(f"ShowPtycho shows one 2D reconstruction or a quantem.gpu SSB session{needs}; "
                                 f"got an array of shape {tuple(data.shape)}. Pass one 2D slice.")
            is_tensor = isinstance(data, torch.Tensor)
            data = data.detach() if is_tensor else data
            if data.is_complex() if is_tensor else np.iscomplexobj(data):
                channels = {"phase": data.angle() if is_tensor else np.angle(data), "amplitude": data.abs() if is_tensor else np.abs(data)}
            else:
                channels = {"phase": data}
            drawn, changes = {}, []
            for name, values in channels.items():
                if is_tensor:
                    rounded = values.to(torch.float32)
                    change = float((rounded.to(values.dtype) - values).abs().max())
                    drawn[name] = np.ascontiguousarray(rounded.cpu().numpy())
                else:
                    rounded = np.ascontiguousarray(values, dtype=np.float32)
                    change = float(np.max(np.abs(rounded.astype(np.float64) - values.astype(np.float64))))
                    drawn[name] = rounded
                if change > 0.0:
                    changes.append(f"{name} {change:.3g}")
            if changes:
                print(f"ShowPtycho: {data.dtype} input is drawn in float32; largest change: {', '.join(changes)}.")
            self._last_phase = drawn["phase"]
            self.phase_height, self.phase_width = self._last_phase.shape
            self.phase_bytes = self._last_phase.tobytes()
            if "amplitude" in drawn:
                self.amplitude_bytes = drawn["amplitude"].tobytes()
            if given is not None and None not in given:  # both steps known: one open axis leaves the image uncalibrated
                self.object_sampling_A = [float(step) for step in given]
            self.ssb_session = False
            self.session_note = session_note()
            return
        self._ssb = gpu_adapter.SsbSession(data)
        state = self._ssb.state()
        self._source_path = state["source_path"]
        self._source_scan_shape = state["scan_shape"]
        self._scan_region = data.scan_region or (0, state["scan_shape"][0], 0, state["scan_shape"][1])
        save_root = pathlib.Path(save_dir) if save_dir else pathlib.Path.cwd()
        self._stars_path = save_root / STARS_FILENAME
        self._calibration_path = save_root / CALIBRATION_FILENAME
        self.stars_path = str(self._stars_path.resolve())
        self.upsampling_available = state["backend"] == "cuda"
        self.sample_available = True  # CUDA and MPS, the SSB backends, both fit the thick-sample model
        self.scan_sampling_A = state["scan_sampling_A"]
        self.crop_refit_available = bool(self._source_path and pathlib.Path(self._source_path).is_file())
        self.crop_refit_status = (
            "Ready to refit a native square scan crop." if self.crop_refit_available
            else "Crop refit needs a re-loadable master file."
        )
        self.observe(self._on_request, names=["request_json"])
        self.observe(self._on_upsampling_factor, names=["upsampling_factor"])
        self.observe(self._on_save, names=["save_trigger"])
        self.observe(self._on_pin, names=["pin_json"])
        self.observe(self._on_drag_pixels_change, names=["num_drag_bright_field_pixels"])
        self.observe(self._on_rotation_change, names=["scan_detector_rotation_deg"])
        self.observe(self._on_panel_change, names=["higher_order_aberrations_json", "sample_json"])
        self.observe(self._on_crop_refit_request, names=["crop_refit_request_json"])
        self.observe(self._on_sample_fit_request, names=["sample_fit_request"])
        self._adopt_session(initial=True)
        # the Sample panel opens on the session's latest sample-tilt fit
        if self.sample_available and state["sample_tilt_mrad"] is not None and state["depth_spread_nm"]:
            self.sample_json = json.dumps({"sample_tilt_mrad": state["sample_tilt_mrad"], "depth_spread_nm": float(state["depth_spread_nm"])})

    def _adopt_session(self, *, initial: bool) -> None:
        """Publish the session's geometry and fit as the widget's reference, then reconstruct at that fit.

        Runs once at construction and again after a crop refit replaces the session. The rotation slider
        spans at least +-180 deg so negative microscope conventions need no mental +180; the C10 range
        widens to include an outlier fit so the reference is always reachable.
        """
        state = self._ssb.state()
        self.scan_rows, self.scan_cols = state["scan_shape"]
        rotation = state["scan_detector_rotation_deg"]
        fitted = state["aberrations"]
        self._ssb.set_scan_detector_rotation(rotation)
        if initial:
            self.C10_min_nm, self.C10_max_nm = min(-30.0, fitted["C10_nm"]), max(30.0, fitted["C10_nm"])
            self.scan_detector_rotation_min_deg, self.scan_detector_rotation_max_deg = min(-180.0, rotation), max(180.0, rotation)
        # the session is already at this rotation: the observer must not rebuild and reconstruct
        self._adopting = True
        self.scan_detector_rotation_deg = rotation
        self._adopting = False
        self.fitted_scan_detector_rotation_deg = rotation
        self.fitted_C10_nm, self.fitted_C12_nm, self.fitted_phi12_deg = fitted["C10_nm"], fitted["C12_nm"], fitted["phi12_deg"]
        self.fitted_loss = state["loss"] or 0.0
        self.higher_order_aberrations_json = "{}"
        self.trials_json = self._trials_payload()
        self.num_bright_field_pixels = state["num_bright_field_pixels"]
        self._inflight_id += 1
        self._reconstruct(self._inflight_id, fitted, compute_loss=True)
        # drags start on the full disk; the slider's left side trades pixels for speed
        self._free_drag_context()
        self.num_drag_bright_field_pixels = self.num_bright_field_pixels
        self._drag_context = self._ssb.preview_context(self.num_bright_field_pixels)

    def _trials_payload(self, max_trials: int = 50) -> str:
        """The session's fit trials for the browser strip, best loss first, capped so a 200-trial run stays light."""
        finite = [trial for trial in self._ssb.trials() if finite_or_none(trial["loss"]) is not None]
        payload = [{"rank": rank, **trial, "loss": float(trial["loss"])}
                   for rank, trial in enumerate(sorted(finite, key=lambda trial: float(trial["loss"]))[:max_trials])]
        return json.dumps(payload) if payload else ""

    def _current(self) -> dict:
        """``{C10_nm, C12_nm, phi12_deg}`` of the displayed phase, the fitted reference before the first request."""
        fitted = {"C10_nm": self.fitted_C10_nm, "C12_nm": self.fitted_C12_nm, "phi12_deg": self.fitted_phi12_deg}
        return {name: float(self._last_result.get(name, value)) for name, value in fitted.items()}

    def _sample(self) -> dict | None:
        """The Sample panel as ``{sample_tilt_mrad, depth_spread_nm}``, or None for standard SSB (depth spread 0)."""
        values = json.loads(self.sample_json or "{}")
        depth_spread_nm = float(values.get("depth_spread_nm", 0.0))
        if depth_spread_nm <= 0.0:
            return None
        tilt_row, tilt_col = values.get("sample_tilt_mrad", (0.0, 0.0))
        return {"sample_tilt_mrad": (float(tilt_row), float(tilt_col)), "depth_spread_nm": depth_spread_nm}

    def _free_drag_context(self) -> None:
        """Release the reduced bright-field subset the session built for drags before a new one
        replaces it, so two subsets never hold device memory at once."""
        if self._drag_context is not None:
            self._drag_context.close()
        self._drag_context = None

    def _reconstruct(self, request_id: int, aberrations: dict, *, compute_loss: bool) -> None:
        """Ask the session for the phase at ``aberrations`` (C10_nm, C12_nm, phi12_deg) and push it to the browser.

        The higher-order panel joins them under its own keys (``C21_nm``, ``phi21_deg``, ...); a non-zero
        one switches the session to its full chi kernel. Drags (``compute_loss=False``) run inside the
        reduced bright-field context and skip the variance loss so slider frames stay under budget; a
        committed request computes the same phase-variance loss the optimizer minimised. A stale request
        (another one was issued meanwhile) is dropped after the compute.
        """
        higher_order = json.loads(self.higher_order_aberrations_json or "{}")
        sample = self._sample() if self.sample_available else None
        if sample is not None and has_higher_order(self.higher_order_aberrations_json):
            # the thick-sample kernel carries C10/C12/phi12 only; higher-order wins and the panel says so
            self.sample_fit_status = "Sample tilt is ignored while higher-order aberrations are non-zero."
            sample = None
        use_drag = not compute_loss and self._drag_context is not None
        with self._drag_context if use_drag else contextlib.nullcontext():
            phase, loss = self._ssb.preview({**higher_order, **aberrations}, compute_loss=compute_loss,
                                            upsampling_factor=self.upsampling_factor, **(sample or {}))
        if request_id != self._inflight_id:
            return
        if self.flip_phase:
            phase = -phase
        self._last_phase = phase
        self.phase_height, self.phase_width = phase.shape
        self.phase_bytes = phase.tobytes()
        # optimizer precision: the browser formats labels, rounding here would perturb saved calibrations
        self._last_result = {"id": request_id, "C10_nm": float(aberrations["C10_nm"]), "C12_nm": float(aberrations["C12_nm"]),
                             "phi12_deg": float(aberrations["phi12_deg"]), "upsampling_factor": self.upsampling_factor, "loss": loss}
        self.result_json = json.dumps(self._last_result)

    def _reconstruct_displayed(self, *, compute_loss: bool) -> None:
        """Reconstruct the displayed aberrations again, under a new request id, after a setting
        outside the three sliders (rotation, sampling, a panel) changed the phase."""
        self._inflight_id += 1
        self._reconstruct(self._inflight_id, self._current(), compute_loss=compute_loss)

    def _on_request(self, change) -> None:
        """A slider request from the browser: reconstruct at its aberrations; committed requests add the loss."""
        if not change["new"]:
            return
        request = json.loads(change["new"])
        self._inflight_id = request["id"]
        self._reconstruct(request["id"], {name: request[name] for name in ("C10_nm", "C12_nm", "phi12_deg")},
                          compute_loss=request.get("committed", False))

    def _on_upsampling_factor(self, change) -> None:
        """Output sampling changed: show the displayed aberrations at the new sampling."""
        self._reconstruct_displayed(compute_loss=False)

    @traitlets.validate("upsampling_factor")
    def _validate_upsampling_factor(self, proposal):
        """Refuse a factor the session cannot produce: 1, 2, 4 or 8, above 1 only on CUDA and
        only while every higher-order term is zero."""
        factor = proposal["value"]
        if factor not in (1, 2, 4, 8):
            raise ValueError("ShowPtycho upsampling_factor must be 1, 2, 4, or 8.")
        if factor > 1 and not self.upsampling_available:
            raise ValueError("This session supports native sampling only; use upsampling_factor=1.")
        if factor > 1 and has_higher_order(self.higher_order_aberrations_json):
            raise ValueError("Higher-order aberrations require upsampling_factor=1; reset them before upsampling.")
        return factor

    @traitlets.validate("higher_order_aberrations_json")
    def _validate_higher_order_sampling(self, proposal):
        """Refuse a non-zero higher-order term while the output is upsampled."""
        if self.upsampling_factor > 1 and has_higher_order(proposal["value"]):
            raise ValueError("Set upsampling_factor=1 before changing higher-order aberrations.")
        return proposal["value"]

    def _on_drag_pixels_change(self, change) -> None:
        """Rebuild the reduced bright-field context the session uses while the user drags."""
        self._free_drag_context()
        if int(change["new"]) > 0:
            self._drag_context = self._ssb.preview_context(min(int(change["new"]), self.num_bright_field_pixels))

    def _on_rotation_change(self, change) -> None:
        """A new scan-detector rotation re-indexes the bright-field geometry, so the drag context is rebuilt too."""
        if self._adopting:
            return
        self._free_drag_context()
        self._ssb.set_scan_detector_rotation(float(change["new"]))
        if self.num_drag_bright_field_pixels > 0:
            self._drag_context = self._ssb.preview_context(int(self.num_drag_bright_field_pixels))
        self._reconstruct_displayed(compute_loss=True)

    def _on_panel_change(self, change) -> None:
        """Higher-order or Sample panel changed: reconstruct the displayed aberrations with it.

        The first adoption resets the panel before any phase exists; there is nothing to redraw then.
        """
        if self._last_phase is None:
            return
        self._reconstruct_displayed(compute_loss=True)

    def _on_flip_change(self, change) -> None:
        """Negate the displayed phase; no reconstruction, the sign is a display convention."""
        if self._last_phase is None:
            return
        self._last_phase = -self._last_phase
        self.phase_bytes = self._last_phase.tobytes()

    def _on_sample_fit_request(self, change) -> None:
        """Fit C10, C12, phi12, sample tilt and depth spread together and hand the result to the sliders.

        The browser applies ``sample_fit_json`` to the aberration sliders and the Sample panel, which
        triggers the reconstruction. C10 comes back as the defocus at mid-depth, not the thin-sheet optimum.
        """
        if not change["new"]:
            return
        if not self.sample_available:
            self.sample_fit_status = "Tilt fit failed: needs a CUDA or MPS SSB session."
            return
        self.sample_fit_status = "Fitting defocus, astigmatism, sample tilt and depth spread..."
        try:
            result = self._ssb.find_aberrations(tilt=True, verbose=False)
        except (RuntimeError, ValueError, NotImplementedError, MemoryError) as exc:
            self.sample_fit_status = f"Tilt fit failed: {exc}"
            return
        tilt_row, tilt_col = result["sample_tilt_mrad"]
        propagator_tilt = propagator_frame_tilt(tilt_row, tilt_col, self.scan_detector_rotation_deg)
        self.sample_fit_json = json.dumps({
            **result["aberrations"], "sample_tilt_mrad": [tilt_row, tilt_col], "propagator_tilt_mrad": propagator_tilt,
            "depth_spread_nm": result["depth_spread_nm"], "sample_tilt_fit_gain": result["sample_tilt_fit_gain"],
        })
        self.sample_fit_status = (
            f"Tilt fit: ({tilt_row:+.1f}, {tilt_col:+.1f}) mrad scan frame = ({propagator_tilt[0]:+.1f}, {propagator_tilt[1]:+.1f}) mrad "
            f"ptychography frame, depth spread {result['depth_spread_nm']:.1f} nm, fit x{result['sample_tilt_fit_gain']:.2f} over standard SSB."
        )

    def _on_crop_refit_request(self, change) -> None:
        """Reload one raw scan crop from the master, fit SSB on it and make it the widget's session.

        The session keeps only its bright-field Fourier cache, so a scientifically valid fit on a region
        needs the raw frames again: the acquisition is opened encoded and only the crop is decoded. The
        browser's region is relative to the current crop; it is mapped back to the full scan. A crop is
        reconstruction input, not a display view: the SSB kernels take native square scans, so a
        rectangle must span at least 32 positions each way.
        """
        if not change["new"]:
            return
        try:
            request = json.loads(change["new"])
            region = request.get("scan_region")
            if not isinstance(region, (tuple, list)) or len(region) != 4:
                raise ValueError("scan_region must be (row_start, row_stop, col_start, col_stop)")
            num_trials = int(request.get("num_trials", 200))
            if num_trials < 1:
                raise ValueError("SSB refit needs at least one optimization trial.")
            if not self.crop_refit_available or not self._source_path:
                raise RuntimeError("Crop refit is unavailable: open the session from its master, SSB.open(<master.h5>, ...).")
            origin_row, _, origin_col, _ = self._scan_region
            row_start, row_stop, col_start, col_stop = (int(value) for value in region)
            region = (origin_row + row_start, origin_row + row_stop, origin_col + col_start, origin_col + col_stop)
            source_rows, source_cols = self._source_scan_shape
            if not (0 <= region[0] < region[1] <= source_rows and 0 <= region[2] < region[3] <= source_cols):
                raise ValueError(f"scan crop [{region[0]}:{region[1]}, {region[2]}:{region[3]}] is outside {source_rows}x{source_cols} data")
            if row_stop - row_start < MIN_CROP_SPAN or col_stop - col_start < MIN_CROP_SPAN:
                raise ValueError(
                    f"SSB refit crops must be at least {MIN_CROP_SPAN}x{MIN_CROP_SPAN}; got {row_stop - row_start}x{col_stop - col_start}."
                )
            self.crop_refit_status = f"Loading [{region[0]}:{region[1]}, {region[2]}:{region[3]}] and fitting SSB ({num_trials} trials)..."
            previous = self._ssb.state()
            with gpu_adapter.load_acquisition(self._source_path, backend=previous["backend"], scan_shape=self._source_scan_shape) as acquisition:
                crop = acquisition.read(scan_region=region)
            rebuilt = gpu_adapter.SsbSession.open(
                crop, backend=previous["backend"], voltage_kV=previous["voltage_kV"], semiangle_mrad=previous["semiangle_mrad"],
                scan_sampling_A=previous["scan_sampling_A"], detector_sampling_mrad=previous["detector_sampling_mrad"],
                aberrations=self._current(), scan_detector_rotation_deg=self.scan_detector_rotation_deg,
                bright_field_intensity_threshold=previous["bright_field_intensity_threshold"],
                bright_field_radius_px=previous["bright_field_radius_px"], source_path=previous["source_path"],
            )
            rebuilt.find_aberrations(num_trials=num_trials, verbose=False)
        except (RuntimeError, ValueError, OSError, MemoryError) as exc:
            # a refused request, an unreadable file or a solver failure; a code bug still raises
            self.crop_refit_status = f"Crop refit failed: {exc}"
            return
        self._free_drag_context()
        self._ssb = rebuilt
        self._scan_region = region
        self._adopt_session(initial=False)
        self.crop_refit_status = f"Refit complete: [{region[0]}:{region[1]}, {region[2]}:{region[3]}], {num_trials} trials."

    def _on_pin(self, change) -> None:
        """Pin, star, view or unpin a snapshot of the displayed aberrations.

        A pin captures the higher-order panel with it, so an old snapshot stays reproducible when the
        panel changes later; starring writes the stars file.
        """
        if not change["new"]:
            return
        event = json.loads(change["new"])
        action, pin_id = event.get("action"), event.get("id")
        if action == "pin":
            entry = {
                "id": pin_id,
                "aberrations": {**json.loads(self.higher_order_aberrations_json or "{}"),
                                **{name: float(event[name]) for name in ("C10_nm", "C12_nm", "phi12_deg")}},
                "scan_detector_rotation_deg": float(event.get("scan_detector_rotation_deg", self.scan_detector_rotation_deg)),
                "flip_phase": bool(event.get("flip_phase", self.flip_phase)), "loss": event.get("loss"),
                "generated_at_utc": utc_now(), "starred": False,
            }
            if self._last_phase is not None:
                entry["phase"] = self._last_phase.copy()
            self._pinned.append(entry)
        elif action in ("star", "unstar"):
            for entry in self._pinned:
                if entry["id"] == pin_id:
                    entry["starred"] = action == "star"
                    if action == "star":
                        # the star time, not the pin time, is when the user decided to keep it
                        entry["generated_at_utc"] = utc_now()
            self._write_stars()
        elif action == "view":
            for entry in self._pinned:
                if entry["id"] == pin_id:
                    if "phase" in entry:
                        self._last_phase = entry["phase"].copy()
                        self.phase_height, self.phase_width = self._last_phase.shape
                        self.phase_bytes = self._last_phase.tobytes()
                    # the viewed snapshot becomes the displayed state, so Save and later slider moves start from it
                    self._last_result = {"id": pin_id, **{name: entry["aberrations"][name] for name in ("C10_nm", "C12_nm", "phi12_deg")},
                                         "loss": entry.get("loss")}
                    self.result_json = json.dumps(self._last_result)
        elif action == "unpin":
            starred = any(entry["id"] == pin_id and entry["starred"] for entry in self._pinned)
            self._pinned = [entry for entry in self._pinned if entry["id"] != pin_id]
            if starred:
                self._write_stars()

    def _calibration(self, aberrations: dict, scan_detector_rotation_deg: float, flip_phase: bool, loss) -> dict:
        """One SSB calibration under the vocabulary's names: what ``calibration.json`` and every star hold.

        Aberrations are nm and degrees (``C10_nm``, ``C12_nm``, ``phi12_deg`` and the non-zero higher
        orders ``C21_nm`` / ``phi21_deg`` ...), the rotation is the physical angle in [0, 360) with
        ``detector_transposed`` beside it, ``scan_sampling_A`` and ``detector_sampling_mrad`` are
        ``[row, col]``, and the optics and ``source_path`` are the session's. Both files are built
        here, so a reader of one reads the other.
        """
        state = self._ssb.state()
        return {
            "aberrations": {name: float(value) for name, value in aberrations.items()
                            # a higher-order term is kept with its azimuth only while its magnitude is non-zero
                            if name in ("C10_nm", "C12_nm", "phi12_deg")
                            or float(aberrations.get(f"C{name[3:-4]}_nm" if name.startswith("phi") else name, 0.0)) != 0.0},
            "scan_detector_rotation_deg": wrap_degrees(scan_detector_rotation_deg), "detector_transposed": False,
            "flip_phase": bool(flip_phase), "voltage_kV": state["voltage_kV"], "semiangle_mrad": state["semiangle_mrad"],
            "scan_sampling_A": state["scan_sampling_A"], "detector_sampling_mrad": state["detector_sampling_mrad"],
            "loss": None if loss is None else float(loss), "source_path": self._source_path, "scan_region": list(self._scan_region),
        }

    def _write_stars(self) -> None:
        """Write the starred snapshots: ``{"schema", "stars": [calibration, ...]}``, each as ``_calibration`` builds it."""
        stars = [{"id": entry["id"], "generated_at_utc": entry["generated_at_utc"],
                  **self._calibration(entry["aberrations"], entry["scan_detector_rotation_deg"], entry["flip_phase"], entry.get("loss"))}
                 for entry in self._pinned if entry["starred"]]
        write_json(self._stars_path, {"schema": STARS_SCHEMA, "stars": stars})

    def _on_save(self, change) -> None:
        """Write the displayed aberrations as ``calibration.json`` and mirror them onto the session.

        The file is one ``_calibration`` record with the Sample panel beside it (``sample_tilt_mrad``
        in the scan frame, ``propagator_tilt_mrad`` in the ptychography frame, ``depth_spread_nm``).
        The session's aberrations are updated so Python code still holding it sees the user's choice.
        """
        if self._last_phase is None or not self.result_json:
            return
        aberrations = self._current()
        loss = self._last_result.get("loss")
        self._ssb.store(aberrations, loss)
        sample = self._sample() if self.sample_available else None
        payload = {
            "schema": CALIBRATION_SCHEMA, "id": uuid.uuid4().hex[:12], "generated_at_utc": utc_now(), "notes": self.notes,
            **self._calibration({**json.loads(self.higher_order_aberrations_json or "{}"), **aberrations},
                                self.scan_detector_rotation_deg, self.flip_phase, loss),
            "sample_tilt_mrad": list(sample["sample_tilt_mrad"]) if sample else None,
            "propagator_tilt_mrad": propagator_frame_tilt(*sample["sample_tilt_mrad"], self.scan_detector_rotation_deg) if sample else None,
            "depth_spread_nm": sample["depth_spread_nm"] if sample else None,
            "upsampling_factor": self.upsampling_factor,
        }
        write_json(self._calibration_path, payload)
        self.calibration_path = str(self._calibration_path.resolve())
        self.calibration_saved_at = payload["generated_at_utc"]

    def freeze(self):
        """Keep the displayed phase visible while detaching its live SSB controls.

        The session's owner can then close it and release GPU data. This does
        not close a shared SSB session or change the displayed scientific array.
        Contrast, zoom and FFT inspection remain available.

        Returns
        -------
        ShowPtycho
            This viewer, now showing a fixed phase array.

        Examples
        --------
        >>> viewer.freeze()
        >>> ssb.close()
        """
        if self._ssb is None:
            return self
        self._free_drag_context()
        for callback, names in (
            (self._on_request, ["request_json"]),
            (self._on_upsampling_factor, ["upsampling_factor"]),
            (self._on_save, ["save_trigger"]),
            (self._on_pin, ["pin_json"]),
            (self._on_drag_pixels_change, ["num_drag_bright_field_pixels"]),
            (self._on_rotation_change, ["scan_detector_rotation_deg"]),
            (self._on_panel_change, ["higher_order_aberrations_json", "sample_json"]),
            (self._on_crop_refit_request, ["crop_refit_request_json"]),
            (self._on_sample_fit_request, ["sample_fit_request"]),
        ):
            self.unobserve(callback, names=names)
        self.object_sampling_A = [step / self.upsampling_factor for step in self.scan_sampling_A]
        self._ssb = None
        self.ssb_session = self.sample_available = self.upsampling_available = False
        self.crop_refit_available = False
        self.session_note = "Saved SSB phase. Open a new SSB review to edit aberrations."
        return self

    def export(self, path: str | pathlib.Path | None = None, *, title: str | None = None) -> pathlib.Path:
        """Write a browser folder that reviews this SSB with no kernel: ``index.html``, exact bright-field
        counts under ``source/``, the calibration under ``snapshots/`` and a double-click ``ShowPtycho.command``.

        ``path`` defaults to ``<stem>_showptycho`` beside the source master (or ``showptycho_export`` in the
        working directory). The browser rebuilds the SSB reducers on its GPU from the stored counts, so
        C10, C12, phi12, rotation and the BF count all move live without Python.
        """
        if self._ssb is None:
            raise ValueError("export() writes an SSB review folder from a session's bright-field counts; this widget shows an array.")
        return write_showptycho_folder(self, path, title)

    def __repr__(self) -> str:
        """What the widget shows (array kind, size, sampling) or, for a session, the pins and drag subset."""
        if self._ssb is None:
            kind = "object wave" if self.amplitude_bytes else "phase"
            sampling = f"{self.object_sampling_A[-1]:g} Å/px" if self.object_sampling_A else "no sampling"
            return f"ShowPtycho({kind} {self.phase_height}x{self.phase_width}, {sampling})"
        drag = f", {self.num_drag_bright_field_pixels} bright-field pixels while dragging" if self._drag_context is not None else ""
        return f"ShowPtycho({len(self._pinned)} pinned{drag})"
