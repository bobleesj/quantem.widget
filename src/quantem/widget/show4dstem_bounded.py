"""Live Show4DSTEM views over acquisitions that stay in their encoded GPU storage.

``quantem.gpu.io.load`` keeps an acquisition ANS encoded on CUDA or MPS; a
512 x 512 x 192 x 192 Arina scan (18 GiB dense) occupies about 0.1 GiB. The
viewer never expands it: diffraction patterns come from bounded ``read`` calls,
and quantem.gpu sums virtual detectors on the encoded storage when a view names
its owner and scan region (``_detector_source``, ``_detector_region``).
"""

import math
import pathlib

import torch

from quantem.gpu.io import Dataset4dstemGPU
from quantem.widget.show4dstem import Show4DSTEM


class _View:
    """One acquisition, or a scan region of it, as the base viewer's 4D frame.

    ``source`` is a ``Dataset4dstemGPU`` returned by ``io.load`` or a 4D torch
    tensor on the GPU. Reads return float32 with detector pixels the source
    flagged invalid set to zero, as the base viewer expects of every frame.
    """

    _is_gpu_frames = True
    _bounded_detector_source = True
    ndim = 4
    dtype = torch.float32
    nbytes = 0  # Borrowed: a view allocates no measurement storage.

    def __init__(self, source, region=None):
        self.source = source
        self.region = region or (0, source.shape[0], 0, source.shape[1])
        row_start, row_stop, col_start, col_stop = self.region
        if not (
            0 <= row_start < row_stop <= source.shape[0]
            and 0 <= col_start < col_stop <= source.shape[1]
        ):
            raise ValueError(f"Scan region {self.region} is outside {tuple(source.shape[:2])}.")
        self.shape = (row_stop - row_start, col_stop - col_start, *source.shape[2:])
        device = torch.device(source.device)
        self.device = torch.device(device.type, device.index or 0)
        self._detector_source = source
        self._detector_region = self.region
        # Encoded counts record the detector pixels their source flagged; dense
        # tensors carry no such mask.
        validity = (
            getattr(source.data, "valid_pixels", None)
            if isinstance(source, Dataset4dstemGPU)
            else None
        )
        self.valid = None if validity is None else torch.as_tensor(
            validity, device=self.device, dtype=torch.bool
        ).reshape(self.shape[-2:])

    def read(self, *, scan_region):
        """Return ``scan_region`` (relative to this view) as a tensor on the source GPU."""
        row_start, row_stop, col_start, col_stop = scan_region
        row_offset, _, col_offset, _ = self.region
        region = (
            row_start + row_offset,
            row_stop + row_offset,
            col_start + col_offset,
            col_stop + col_offset,
        )
        if torch.is_tensor(self.source):
            return self.source[region[0] : region[1], region[2] : region[3]]
        values_t = self.source.read(scan_region=region)
        return values_t if self.valid is None else values_t.float().masked_fill(~self.valid, 0)

    def numel(self):
        return math.prod(self.shape)

    def __getitem__(self, row):
        if isinstance(row, tuple):
            row, col = row
            return self.read(scan_region=(row, row + 1, col, col + 1))[0, 0]
        # The base viewer uses one initial row only for display range estimation.
        return self.read(scan_region=(row, row + 1, 0, min(32, self.shape[1])))[0]


class _Views:
    """Several acquisitions of one scan and detector geometry as a 5D series."""

    _is_gpu_frames = True
    ndim = 5
    dtype = torch.float32
    nbytes = 0

    def __init__(self, sources):
        self.frames = [_View(source) for source in sources]
        first = self.frames[0]
        if any(frame.shape != first.shape or frame.device != first.device for frame in self.frames):
            raise ValueError("Comparison sources must share scan/detector shape and device.")
        self.shape = (len(self.frames), *first.shape)
        self.device = first.device

    def append(self, source) -> None:
        """Add one acquisition that arrived while a folder is watched."""
        frame = _View(source)
        if frame.shape != self.frames[0].shape or frame.device != self.device:
            raise ValueError(
                f"Acquisition shape {frame.shape} on {frame.device} does not match "
                f"{self.frames[0].shape} on {self.device}."
            )
        self.frames.append(frame)
        self.shape = (len(self.frames), *self.frames[0].shape)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        return self.frames[index]

    def numel(self):
        return math.prod(self.shape)


def acquisition_name(path) -> str:
    """Name an acquisition by its file without the ``_master.h5`` suffix."""
    return pathlib.Path(str(path)).name.removesuffix("_master.h5")


def acquisition_label(source) -> str | None:
    """Name a loaded acquisition by its source file, when it records one."""
    if not isinstance(source, Dataset4dstemGPU) or "source_path" not in source.metadata:
        return None
    return acquisition_name(source.metadata["source_path"])


def _live_options(options: dict) -> dict:
    """Defaults shared by every bounded viewer; encoded storage cannot go offline."""
    if options.get("offline") or options.get("data_url"):
        raise ValueError(
            "Encoded acquisitions need a live kernel; offline export is not supported."
        )
    options.setdefault("precompute_virtual_images", False)
    options.setdefault("verbose", False)
    options.setdefault("offline", False)
    return options


def show_bounded(sources, *, scan_region=None, **options):
    """Open the live viewer over borrowed acquisitions without stacking them.

    One source opens as a single 4D viewer, optionally restricted to
    ``scan_region=(row_start, row_stop, col_start, col_stop)``. Several sources
    of one geometry open as a dataset comparison labelled by source file.
    """
    if len(sources) == 1:
        return Show4DSTEM(_View(sources[0], scan_region), **_live_options(options))
    if scan_region is not None:
        raise ValueError("Select the same source regions before multi-source comparison.")
    return show_series(sources, **options)


def show_series(sources, **options):
    """Open acquisitions as a dataset series that can grow while a folder is watched."""
    labels = [acquisition_label(source) for source in sources]
    options.setdefault("view_mode", "multiple")
    options.setdefault("compare_dp_mode", "selected")
    options.setdefault("frame_dim_label", "Dataset")
    if all(label is not None for label in labels):
        options.setdefault("frame_labels", labels)
    return Show4DSTEM(_Views(sources), **_live_options(options))
