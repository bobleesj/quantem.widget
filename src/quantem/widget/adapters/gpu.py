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


# quantem.gpu's 14-coefficient Krivanek vector: (name, slot, carries an azimuth); slots 0 and 1 are C10 and C12 / phi12
HIGHER_ORDER_LAYOUT = (
    ("C21", 2, True), ("C23", 3, True),
    ("C30", 4, False), ("C32", 5, True), ("C34", 6, True),
    ("C41", 7, True), ("C43", 8, True), ("C45", 9, True),
    ("C50", 10, False), ("C52", 11, True), ("C54", 12, True), ("C56", 13, True),
)


class SsbSession:
    """A ``quantem.gpu.SSB`` session spoken to in the names of the vocabulary.

    The widgets, their messages and the files they write use one name per
    concept (``scan_detector_rotation_deg``, ``detector_sampling_mrad``,
    ``C10_nm`` / ``C12_nm`` / ``phi12_deg``, ``bright_field_radius_px``,
    ``num_bright_field_pixels``, ``sample_tilt_mrad``). quantem.gpu's own
    argument and attribute names appear in this class only, each beside the
    name it stands for, so this is the one place to change when quantem.gpu
    takes the same vocabulary.

    quantem.gpu name -> vocabulary name:

    ====================================================  ==========================================
    ``det_sampling``                                      ``detector_sampling_mrad``
    ``rotation_angle_deg`` + ``com_reversed``,            ``scan_detector_rotation_deg`` (physical,
    ``physical_rotation_deg``, ``set_rotation``           [0, 360)); ``detector_transposed`` raises
    ``aberrations`` ``C10``, ``C12`` (nm), ``phi12``      ``aberrations`` ``C10_nm``, ``C12_nm``,
    (rad)                                                 ``phi12_deg``
    ``higher_order_magnitudes`` / ``higher_order_angles`` ``aberrations`` ``C21_nm``, ``phi21_deg``, ...
    (nm and rad in 14 slots)
    ``bf_radius``, ``bf_intensity_threshold``,            ``bright_field_radius_px``,
    ``bf_center``, ``num_bf``                             ``bright_field_intensity_threshold``,
                                                          ``bright_field_center_px``,
                                                          ``num_bright_field_pixels``
    ``tilt_mrad``, ``tilt_fit_gain``                      ``sample_tilt_mrad``, ``sample_tilt_fit_gain``
    ``trials``, ``refinement``, ``refine_lock``           ``num_trials``, ``refine``, ``locked``
    ``best_loss``, ``elapsed``                            ``loss``, ``elapsed_s``
    ``upsample`` (result)                                 ``upsampling_factor``
    ====================================================  ==========================================

    ``voltage_kV``, ``semiangle_mrad``, ``scan_sampling_A``, ``depth_spread_nm``,
    ``source_path``, ``scan_shape`` and ``upsampling_factor`` (``preview``) carry
    the same name on both sides.
    """

    def __init__(self, ssb):
        self.ssb = ssb

    @classmethod
    def open(cls, source, *, voltage_kV: float, semiangle_mrad: float, scan_sampling_A, detector_sampling_mrad=None,
             scan_shape=None, aberrations: dict | None = None, scan_detector_rotation_deg: float = 0.0,
             detector_transposed: bool = False, bright_field_intensity_threshold: float = 0.0,
             bright_field_radius_px: float | None = None, backend: str = "auto", source_path: str | None = None):
        """A new session over ``source``; use it as a context manager.

        ``detector_transposed=True`` raises: quantem.gpu's SSB engines have no
        transposition input, and reconstructing such a scan untransposed would
        give a mirrored phase without a word.
        """
        require("Compute SSB")
        if detector_transposed:
            raise NotImplementedError("SSB cannot reconstruct a scan whose detector axes are transposed "
                                      "(detector_transposed=True); the engines take the rotation only.")
        return cls(SSB(
            source, backend=backend, voltage_kV=float(voltage_kV), semiangle_mrad=float(semiangle_mrad),
            scan_sampling_A=scan_sampling_A, scan_shape=scan_shape, det_sampling=detector_sampling_mrad,
            aberrations=None if aberrations is None else cls.engine_aberrations(aberrations),
            # quantem.gpu folds any angle into [0, 180) plus com_reversed itself, so the physical angle goes in whole
            rotation_angle_deg=float(scan_detector_rotation_deg),
            bf_intensity_threshold=float(bright_field_intensity_threshold),
            bf_radius=None if bright_field_radius_px is None else float(bright_field_radius_px),
            source_path=source_path,
        ))

    def __enter__(self):
        self.ssb.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.ssb.__exit__(exc_type, exc_value, traceback)

    @staticmethod
    def engine_aberrations(aberrations: dict) -> dict:
        """``{C10_nm, C12_nm, phi12_deg}`` as quantem.gpu's ``{C10, C12 (nm), phi12 (rad)}``; a missing key is 0."""
        return {"C10": float(aberrations.get("C10_nm", 0.0)), "C12": float(aberrations.get("C12_nm", 0.0)),
                "phi12": math.radians(float(aberrations.get("phi12_deg", 0.0)))}

    @staticmethod
    def vocabulary_aberrations(aberrations: dict) -> dict:
        """quantem.gpu's ``{C10, C12 (nm), phi12 (rad)}`` as ``{C10_nm, C12_nm, phi12_deg}``."""
        return {"C10_nm": float(aberrations.get("C10", 0.0)), "C12_nm": float(aberrations.get("C12", 0.0)),
                "phi12_deg": math.degrees(float(aberrations.get("phi12", 0.0)))}

    @property
    def num_bright_field_pixels(self) -> int:
        """Pixels of the detected bright-field disk (quantem.gpu ``num_bf``); the first read prepares the session."""
        return int(self.ssb.num_bf)

    def state(self) -> dict:
        """Geometry, optics and current fit of the session under the vocabulary's names.

        ``scan_sampling_A`` and ``detector_sampling_mrad`` are ``[row, col]``
        (quantem.gpu holds one number when both axes share it), and the
        rotation is the physical angle, with the reversal quantem.gpu keeps in
        ``com_reversed`` folded in. Every file and message of the SSB widgets is
        built from this dict, so they cannot disagree on a name or a unit.
        """
        ssb = self.ssb
        steps = {"scan_sampling_A": ssb.scan_sampling_A, "detector_sampling_mrad": ssb.det_sampling}
        pairs = {name: None if step is None else [float(step)] * 2 if np.isscalar(step) else [float(value) for value in step]
                 for name, step in steps.items()}
        return {
            "backend": ssb.backend, "scan_shape": tuple(int(size) for size in ssb.scan_shape), "source_path": ssb.source_path,
            "voltage_kV": float(ssb.voltage_kV), "semiangle_mrad": float(ssb.semiangle_mrad), **pairs,
            "scan_detector_rotation_deg": float(ssb.physical_rotation_deg), "detector_transposed": False,
            "aberrations": self.vocabulary_aberrations(ssb.aberrations),
            "loss": float(ssb.best_loss) if math.isfinite(ssb.best_loss) else None,
            "bright_field_radius_px": ssb.bf_radius, "bright_field_intensity_threshold": ssb.bf_intensity_threshold,
            "num_bright_field_pixels": self.num_bright_field_pixels,
            "sample_tilt_mrad": None if ssb.tilt_mrad is None else [float(value) for value in ssb.tilt_mrad],
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
        self.ssb.aberrations.update(self.engine_aberrations(aberrations))
        if loss is not None:
            self.ssb.best_loss = float(loss)

    def preview_context(self, num_bright_field_pixels: int):
        """A reduced bright-field context for slider drags (``quantem.gpu.SSB.preview_context``)."""
        return self.ssb.preview_context(int(num_bright_field_pixels))

    def preview(self, aberrations: dict, *, compute_loss: bool, upsampling_factor: int = 1,
                sample_tilt_mrad=None, depth_spread_nm: float = 0.0):
        """Transient ``(phase, loss)`` at ``aberrations``: ``C10_nm``, ``C12_nm``, ``phi12_deg`` and any higher order.

        A higher-order term is ``C21_nm`` with ``phi21_deg`` (``C30_nm`` and
        ``C50_nm`` have no azimuth). They fill quantem.gpu's 14-slot Krivanek
        vector in nm and radians; with every one of them zero the session keeps
        its C10 / C12 kernel. ``depth_spread_nm > 0`` turns on the thick-sample
        model with ``sample_tilt_mrad`` ``(row, col)``.
        """
        magnitudes = np.zeros(14, dtype=np.float32)
        angles = np.zeros(14, dtype=np.float32)
        magnitudes[0], magnitudes[1] = aberrations.get("C10_nm", 0.0), aberrations.get("C12_nm", 0.0)
        angles[1] = math.radians(aberrations.get("phi12_deg", 0.0))
        for name, slot, has_angle in HIGHER_ORDER_LAYOUT:
            magnitudes[slot] = float(aberrations.get(f"{name}_nm", 0.0))
            angles[slot] = math.radians(float(aberrations.get(f"phi{name[1:]}_deg", 0.0))) if has_angle else 0.0
        higher_order = bool(np.any(magnitudes[2:]))
        sample = {} if depth_spread_nm <= 0.0 else {"tilt_mrad": tuple(sample_tilt_mrad), "depth_spread_nm": float(depth_spread_nm)}
        return self.ssb.preview(
            self.engine_aberrations(aberrations), compute_loss=compute_loss, upsampling_factor=int(upsampling_factor),
            higher_order_magnitudes=magnitudes if higher_order else None,
            higher_order_angles=angles if higher_order else None, **sample,
        )

    def browser_state(self) -> dict:
        """The compact state a browser SSB engine rebuilds its reducers from, under the vocabulary's names.

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
            "num_bright_field_pixels": int(state.num_bf),
            "bright_field_radius_px": state.brightfield.radius_px,
            "bright_field_center_px": list(state.brightfield.center_row_col),
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

    def find_aberrations(self, *, num_trials: int = 200, refine: bool = True, search_ranges: dict | None = None,
                         locked=(), seed: int = 42, sample_tilt: bool = False, verbose: bool = False) -> dict:
        """Fit the aberrations (``quantem.gpu.SSB.find_aberrations``); see ``result`` for the keys.

        ``search_ranges`` maps ``C10_nm``, ``C12_nm`` and ``phi12_deg`` to a
        ``(low, high)`` range or to one pinned value, and ``locked`` names the
        ones Nelder-Mead holds. ``sample_tilt=True`` fits the sample tilt and
        the depth spread together with them.
        """
        refinement = "nelder-mead" if refine else None
        # quantem.gpu locks by its bare names: C10_nm -> C10, C12_nm -> C12, phi12_deg -> phi12
        refine_lock = [name.rsplit("_", 1)[0] for name in locked] if refinement and locked else None
        if sample_tilt:
            return self.result(self.ssb.find_aberrations(tilt=True, verbose=verbose))
        return self.result(self.ssb.find_aberrations(trials=int(num_trials), refinement=refinement, search_ranges=search_ranges,
                                                     refine_lock=refine_lock, seed=int(seed), verbose=verbose))

    @classmethod
    def result(cls, result) -> dict:
        """A quantem.gpu ``SSBResult`` under the vocabulary's names.

        ``phase``, ``aberrations`` (``C10_nm``, ``C12_nm``, ``phi12_deg``),
        ``scan_detector_rotation_deg`` (the angle quantem.gpu reports below 180
        plus 180 when it reversed the centre of mass: storing the folded angle
        alone loses the reversal), ``loss``, ``elapsed_s``, and after a sample
        fit ``sample_tilt_mrad``, ``depth_spread_nm`` and ``sample_tilt_fit_gain``.
        """
        fitted = {
            "phase": result.phase, "aberrations": cls.vocabulary_aberrations(result.aberrations),
            "scan_detector_rotation_deg": (float(result.rotation_angle_deg) + (180.0 if result.com_reversed else 0.0)) % 360.0,
            "loss": None if result.loss is None else float(result.loss),
            "elapsed_s": None if result.elapsed is None else float(result.elapsed),
        }
        if result.tilt_mrad is not None:
            fitted.update(sample_tilt_mrad=[float(value) for value in result.tilt_mrad],
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
