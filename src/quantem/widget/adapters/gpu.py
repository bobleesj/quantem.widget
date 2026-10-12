"""quantem.gpu acquisitions, detector sessions and SSB, whether or not quantem.gpu is installed.

quantem.gpu is optional (``quantem.widget[cuda]`` or ``quantem.widget[mps]``). This is the only
module that imports it: encoded acquisitions from ``quantem.gpu.io.load``, their
detector sessions, master discovery and inspection, SSB, centre of mass and the
release of cached GPU memory.
Without quantem.gpu every name below is None, ``available()`` is False and a
function that needs it raises ``ImportError`` with the install command; the
widgets then read files densely with ``show4dstem.reader`` and reduce with
their own torch code.
"""

import math

import numpy as np
import torch

from quantem.widget.device import device_name, gpu_path_hint, no_gpu_path, print_once
from quantem.widget.units import acquisition_calibration

try:
    from quantem.gpu import SSB
    from quantem.gpu import io as gpu_io
    from quantem.gpu.detector import prepare
    from quantem.gpu.device import release_cached_memory as gpu_release_cached_memory
    from quantem.gpu.device import runtime_notice as gpu_runtime_notice
    from quantem.gpu.dpc import center_of_mass as gpu_center_of_mass
    from quantem.gpu.io import Dataset4dstemGPU
except ImportError:
    SSB = gpu_io = prepare = gpu_release_cached_memory = gpu_runtime_notice = gpu_center_of_mass = Dataset4dstemGPU = None


def available() -> bool:
    """Whether quantem.gpu imported."""
    return gpu_io is not None


def require(feature: str) -> None:
    """Raise ``ImportError`` naming ``feature`` and the install command when quantem.gpu is missing."""
    if not available():
        raise ImportError(f"{feature} needs quantem.gpu, which is not installed. {gpu_path_hint() or no_gpu_path()}.")


def accelerator_ready() -> bool:
    """Whether quantem.gpu can load files on this machine: installed, and a CUDA or Metal backend works."""
    if not available():
        return False
    from quantem.gpu.device import detect

    try:
        detect()
    except RuntimeError:
        return False
    return True


def is_acquisition(obj) -> bool:
    """Whether ``obj`` is an acquisition from ``quantem.gpu.io.load``."""
    return available() and isinstance(obj, Dataset4dstemGPU)


def is_ssb(obj) -> bool:
    """Whether ``obj`` is a ``quantem.gpu.SSB`` session."""
    return available() and isinstance(obj, SSB)


def load_acquisition(path, device: str | None = None, *, backend: str = "auto", scan_shape=None):
    """Load one master (or a list) into encoded GPU storage with ``quantem.gpu.io.load``.

    ``device`` is a ``resolve_device`` string passed through: ``"cuda:N"``
    selects that GPU, ``"mps"`` the Apple GPU, None (``device="auto"``) lets
    quantem.gpu choose, and the choice is printed once, as ``resolve_device``
    prints it for arrays.
    """
    require("Loading an acquisition on the GPU")
    loaded = gpu_io.load(path, backend=backend, device=device, scan_shape=scan_shape, verbose=False)
    if device is None:
        first = loaded[0] if isinstance(loaded, list) else loaded
        print_once(f'quantem.widget: device="auto" selected {device_name(torch.device(first.device))}.')
    return loaded


def acquisition_from_tensor(values_t: torch.Tensor, metadata: dict):
    """A ``Dataset4dstemGPU`` over a 4D tensor already on the GPU, with ``metadata`` as its calibration and name.

    quantem.gpu holds such a tensor as a dense acquisition: indexing, ``read``
    and its detector sessions work as on an encoded one, so a tensor computed
    on the GPU (a strided preview of a loaded file) keeps the GPU dataset type.
    """
    require("A GPU acquisition")
    return Dataset4dstemGPU(values_t, dict(metadata))


def discover_masters(folder, *, pattern: str = "*_master.h5", recursive: bool = True, scan_shape=None) -> list[str]:
    """Readable candidate masters under ``folder`` (``quantem.gpu.io.discover``)."""
    require("Master discovery")
    return list(gpu_io.discover(str(folder), pattern=pattern, recursive=recursive, scan_shape=scan_shape, verbose=False))


def inspect_master(path, *, scan_shape=None):
    """Readiness report of one master (``quantem.gpu.io.inspect``): shapes, dtype, reason, action."""
    require("Master inspection")
    return gpu_io.inspect(str(path), scan_shape=scan_shape)


def header_calibration(metadata: dict) -> dict:
    """The calibration in the header metadata ``inspect_master`` returns, under the vocabulary's flat names.

    quantem.gpu's flat pair ``detector_sampling`` with ``detector_sampling_unit``
    becomes ``detector_sampling_mrad`` or ``detector_sampling_inv_A`` here, as
    the axes of a loaded dataset do in ``units.acquisition_calibration``. A
    ``.qem`` viewer and the command-line export read the header this way, so
    their scale bars carry the file's calibration and not ``1.0 px``.
    """
    # quantem.gpu name -> vocabulary name: detector_sampling + detector_sampling_unit ->
    # detector_sampling_mrad | detector_sampling_inv_A; scan_sampling_A, voltage_kV, semiangle_mrad unchanged
    steps = [metadata.get("scan_sampling_A"), metadata.get("detector_sampling")]
    sampling = [step for pair in steps for step in ([pair] * 2 if pair is None or np.isscalar(pair) else pair)]
    return acquisition_calibration(sampling, ["angstrom"] * 2 + [metadata.get("detector_sampling_unit")] * 2, metadata)


def prepare_session(source):
    """A ``quantem.gpu`` detector session over an acquisition view."""
    require("Encoded detector reductions")
    return prepare(source)


def center_of_mass(source, scan_shape) -> tuple[np.ndarray, np.ndarray]:
    """Mean-subtracted detector centre of mass ``(com_row, com_col)`` of every scan position."""
    require("Centre of mass")
    return gpu_center_of_mass(source, scan_shape=scan_shape)


class SsbSession:
    """A ``quantem.gpu.SSB`` session as the SSB widgets read and drive it.

    quantem.gpu and the widgets use one name per concept
    (``scan_detector_rotation_deg``, ``detector_sampling_mrad``, ``C10_nm`` /
    ``C12_nm`` / ``phi12_deg``, ``bright_field_radius_px``,
    ``num_bright_field_pixels``, ``sample_tilt_mrad``), so arguments reach
    quantem.gpu as given. This class collects what the widgets read from a
    session into plain dicts (``state``, ``trials``, ``browser_state``,
    ``result``); every file and message of the SSB widgets is built from them.
    """

    def __init__(self, ssb):
        self.ssb = ssb

    @classmethod
    def open(cls, source, **parameters):
        """A new session ``quantem.gpu.SSB(source, **parameters)``; use it as a context manager.

        quantem.gpu raises ``NotImplementedError`` for
        ``detector_transposed=True``: its SSB engines have no transposition
        input, and reconstructing such a scan untransposed would give a mirrored
        phase without a word.
        """
        require("Compute SSB")
        return cls(SSB(source, **parameters))

    def __enter__(self):
        self.ssb.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.ssb.__exit__(exc_type, exc_value, traceback)

    @property
    def num_bright_field_pixels(self) -> int:
        """Pixels of the detected bright-field disk; the first read prepares the session."""
        return int(self.ssb.num_bright_field_pixels)

    def state(self) -> dict:
        """Geometry, optics and current fit of the session.

        ``scan_sampling_A`` and ``detector_sampling_mrad`` are ``[row, col]``
        lists and the rotation is the physical angle in [0, 360). Every file and
        message of the SSB widgets is built from this dict, so they cannot
        disagree on a name or a unit.
        """
        ssb = self.ssb
        return {
            "backend": ssb.backend, "scan_shape": tuple(int(size) for size in ssb.scan_shape), "source_path": ssb.source_path,
            "voltage_kV": ssb.voltage_kV, "semiangle_mrad": ssb.semiangle_mrad, "scan_sampling_A": list(ssb.scan_sampling_A),
            "detector_sampling_mrad": None if ssb.detector_sampling_mrad is None else list(ssb.detector_sampling_mrad),
            "scan_detector_rotation_deg": ssb.scan_detector_rotation_deg, "detector_transposed": ssb.detector_transposed,
            "aberrations": dict(ssb.aberrations),
            "loss": float(ssb.best_loss) if math.isfinite(ssb.best_loss) else None,
            "bright_field_radius_px": ssb.bright_field_radius_px,
            "bright_field_intensity_threshold": ssb.bright_field_intensity_threshold,
            "num_bright_field_pixels": self.num_bright_field_pixels,
            "sample_tilt_mrad": None if ssb.sample_tilt_mrad is None else list(ssb.sample_tilt_mrad),
            "depth_spread_nm": ssb.depth_spread_nm,
        }

    def trials(self) -> list[dict]:
        """The fit's trials as ``{C10_nm, C12_nm, phi12_deg, loss}``, in the order they ran."""
        return [{"C10_nm": float(trial["params"].get("C10_nm", 0.0)), "C12_nm": float(trial["params"].get("C12_nm", 0.0)),
                 "phi12_deg": float(trial["params"].get("phi12_deg", 0.0)), "loss": trial.get("loss")}
                for trial in self.ssb.trial_history]

    def set_scan_detector_rotation(self, scan_detector_rotation_deg: float) -> None:
        """Set the physical scan-to-detector rotation and refresh the session's geometry."""
        self.ssb.set_rotation(float(scan_detector_rotation_deg))

    def store(self, aberrations: dict, loss: float | None) -> None:
        """Mirror the aberrations a user chose onto the session, so code still holding it sees them."""
        self.ssb.aberrations.update(aberrations)
        if loss is not None:
            self.ssb.best_loss = float(loss)

    def preview_context(self, num_bright_field_pixels: int):
        """A reduced bright-field context for slider drags (``quantem.gpu.SSB.preview_context``)."""
        return self.ssb.preview_context(int(num_bright_field_pixels))

    def preview(self, aberrations: dict, **options):
        """Transient ``(phase, loss)`` at ``aberrations`` (``quantem.gpu.SSB.preview(aberrations, **options)``).

        ``aberrations`` holds ``C10_nm``, ``C12_nm``, ``phi12_deg`` and the
        higher-order panel's terms (``C21_nm`` with ``phi21_deg``; ``C30_nm`` and
        ``C50_nm`` have no azimuth). The higher orders reach quantem.gpu only
        while one of their magnitudes is non-zero: a named term, even at zero,
        selects quantem.gpu's 14-slot path, which holds C10, C12 and phi12 in
        float32, so a panel left at zero changed the last bits of most pixels of
        the two-term phase.
        """
        higher_order = any(value != 0.0 for name, value in aberrations.items() if name.endswith("_nm") and name not in ("C10_nm", "C12_nm"))
        primary = aberrations if higher_order else {name: aberrations[name] for name in ("C10_nm", "C12_nm", "phi12_deg")}
        return self.ssb.preview(primary, **options)

    def browser_state(self) -> dict:
        """The compact state a browser SSB engine rebuilds its reducers from.

        quantem.gpu's ``SSBExportState``: the bright-field pixel list
        (``bright_field_rows`` / ``_cols``), their reciprocal coordinates in the
        scan frame (``bright_field_k_row`` / ``_k_col``, quantem.gpu's ``kx_bf``
        / ``ky_bf``: its x is the scan row axis), the scan frequencies
        (``scan_q_row`` / ``scan_q_col``, its ``qx_1d`` / ``qy_1d``) and the
        aperture terms, with the detector step in mrad. ``ShowPtycho.export``
        writes ``snapshots/cal.json`` from this dict alone.
        """
        state = self.ssb.browser_state()
        return {
            "scan_shape": tuple(int(size) for size in state.scan_shape),
            "num_bright_field_pixels": int(state.num_bright_field_pixels),
            "bright_field_radius_px": state.brightfield.bright_field_radius_px,
            "bright_field_center_px": list(state.brightfield.bright_field_center_px),
            "detector_shape": list(state.brightfield.detector_shape),
            "bright_field_rows": state.brightfield.rows, "bright_field_cols": state.brightfield.cols,
            "bright_field_k_row": state.kx_bf, "bright_field_k_col": state.ky_bf,
            "scan_q_row": state.qx_1d, "scan_q_col": state.qy_1d,
            "aperture_k": state.aperture_k, "alpha_k2": state.alpha_k2,
            "cos2phi_k": state.cos2phi_k, "sin2phi_k": state.sin2phi_k,
            "wavelength_A": state.wavelength_A,
            "detector_sampling_mrad": [float(step) * 1e3 for step in state.angular_sampling_rad],
            "dc_value": state.dc_value,
            "bright_field_source_path": state.bf_source_path, "bright_field_source_dtype": state.bf_source_dtype,
            "bright_field_source_max_value": state.bf_source_max_value,
        }

    def export_bright_field(self, path) -> None:
        """Write the exact bright-field counts beside an exported page (``quantem.gpu.SSB.export_brightfield``)."""
        self.ssb.export_brightfield(path)

    def reconstruct(self) -> dict:
        """Reconstruct at the session's aberrations (``quantem.gpu.SSB.reconstruct``); see ``result`` for the keys."""
        return self.result(self.ssb.reconstruct())

    def find_aberrations(self, **options) -> dict:
        """Fit the aberrations (``quantem.gpu.SSB.find_aberrations(**options)``); see ``result`` for the keys."""
        return self.result(self.ssb.find_aberrations(**options))

    @staticmethod
    def result(result) -> dict:
        """The values of a quantem.gpu ``SSBResult`` the widgets show and store.

        ``phase``, ``aberrations`` (``C10_nm``, ``C12_nm``, ``phi12_deg``),
        ``scan_detector_rotation_deg`` (the physical angle in [0, 360)),
        ``loss``, ``elapsed_s``, and after a sample fit ``sample_tilt_mrad``,
        ``depth_spread_nm`` and ``sample_tilt_fit_gain`` (quantem.gpu's
        ``tilt_fit_gain``).
        """
        fitted = {
            "phase": result.phase, "aberrations": dict(result.aberrations),
            "scan_detector_rotation_deg": result.scan_detector_rotation_deg,
            "loss": None if result.loss is None else float(result.loss),
            "elapsed_s": None if result.elapsed_s is None else float(result.elapsed_s),
        }
        if result.sample_tilt_mrad is not None:
            fitted.update(sample_tilt_mrad=[float(value) for value in result.sample_tilt_mrad],
                          depth_spread_nm=float(result.depth_spread_nm), sample_tilt_fit_gain=float(result.tilt_fit_gain))
        return fitted


class AcquisitionView:
    """One acquisition, or a scan region of it, read in bounded windows.

    ``source`` is a ``Dataset4dstemGPU`` from ``io.load`` or a 4D tensor on the
    GPU. ``quantem.gpu.detector.prepare`` recognises the marker attributes and
    sums virtual detectors on the encoded counts; reads return float32 with the
    detector pixels the source flagged set to zero.
    """

    _bounded_detector_source = True

    def __init__(self, source, region=None):
        self.source = source
        self.region = region or (0, source.shape[0], 0, source.shape[1])
        row_start, row_stop, col_start, col_stop = self.region
        if not (0 <= row_start < row_stop <= source.shape[0] and 0 <= col_start < col_stop <= source.shape[1]):
            raise ValueError(f"Scan region {self.region} is outside {tuple(source.shape[:2])}.")
        self.shape = (row_stop - row_start, col_stop - col_start, *source.shape[2:])
        device = torch.device(source.device)
        self.device = torch.device(device.type, device.index or 0)
        self._detector_source = source
        self._detector_region = self.region
        self.valid = None
        if is_acquisition(source):
            session = prepare(source)
            validity = session.detector_validity
            session.close()
            if validity is not None:
                self.valid = torch.as_tensor(validity, device=self.device).reshape(self.shape[-2:])

    def read(self, *, scan_region):
        """``scan_region`` relative to this view, as a float32 tensor on the source GPU."""
        row_start, row_stop, col_start, col_stop = scan_region
        row_offset, _, col_offset, _ = self.region
        region = (row_start + row_offset, row_stop + row_offset, col_start + col_offset, col_stop + col_offset)
        if torch.is_tensor(self.source):
            return self.source[region[0] : region[1], region[2] : region[3]]
        values_t = self.source.read(scan_region=region).float()
        return values_t if self.valid is None else values_t.masked_fill(~self.valid, 0)


class AcquisitionSeries:
    """Several acquisitions of one scan and detector geometry as a growing series."""

    def __init__(self, sources):
        self.frames = [AcquisitionView(source) for source in sources]
        first = self.frames[0]
        if any(frame.shape != first.shape or frame.device != first.device for frame in self.frames):
            raise ValueError("Comparison sources must share scan/detector shape and device.")
        self.device = first.device

    @property
    def shape(self) -> tuple[int, ...]:
        """``(n_acquisitions, scan_rows, scan_cols, det_rows, det_cols)``."""
        return (len(self.frames), *self.frames[0].shape)

    def append(self, source) -> None:
        """Add one acquisition that arrived while a folder is watched."""
        frame = AcquisitionView(source)
        if frame.shape != self.frames[0].shape or frame.device != self.device:
            raise ValueError(
                f"Acquisition shape {frame.shape} on {frame.device} does not match "
                f"{self.frames[0].shape} on {self.device}."
            )
        self.frames.append(frame)
