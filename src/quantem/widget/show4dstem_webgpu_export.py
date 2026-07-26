"""No-server Show4DSTEM WebGPU bundle export.

Mirrors the ShowPtycho handoff protocol: the recipient double-clicks one
``Show4DSTEM.command``, a local range-capable HTTP server starts over the data
folder, and Chrome opens a fully vendored viewer page that decodes the raw
bslz4 HDF5 family in WebGPU. No Python, no network, no folder-grant click at
view time. Everything the page needs (require.js, the Jupyter widget manager,
anywidget, the server script) ships from this package's ``static/vendor``.
"""

from __future__ import annotations

import gzip
import pathlib
import re
from typing import Any, Sequence

from quantem.widget.command_launcher import write_command_launcher

_VENDOR = pathlib.Path(__file__).parent / "vendor"
# CDN references embed_minimal_html emits; each is replaced by a vendored copy so
# the bundle works with no network at all (conference wifi is not a dependency).
# Patterns, not exact URLs: the emitted version specifiers drift across
# ipywidgets/anywidget releases (e.g. anywidget@0.11.0 vs anywidget@~0.11.*).
_CDN_REWRITES = (
    (re.compile(r"https://cdnjs\.cloudflare\.com/[^\"']*/require(\.min)?\.js"), "./require.min.js"),
    (re.compile(r"https://cdn\.jsdelivr\.net/[^\"']*html-manager[^\"']*/embed-amd\.js"), "./embed-amd.js"),
    (re.compile(r"\"https://cdn\.jsdelivr\.net/npm/anywidget@[^\"]*\""), '"./anywidget.min"'),
)


def _write_vendor_asset(name: str, viewer: pathlib.Path) -> None:
    """Expand a compressed browser-manager asset into the export viewer."""
    source = _VENDOR / f"{name}.gz"
    if not source.is_file():
        raise FileNotFoundError(
            f"missing vendored Show4DSTEM browser asset: {source}; "
            "rebuild the package with src/quantem/widget/vendor included"
        )
    with gzip.open(source, "rb") as src, (viewer / name).open("wb") as dst:
        dst.write(src.read())


# Promoted WebGPU decode configuration. Native uint16 is the conservative
# default; audited uint8 browse sources can use the low8-only kernel to skip the
# high bitplanes that only hold masked detector sentinels.
def _tuning(*, h5_uint8_lossless: bool) -> str:
    dtype = "uint8" if h5_uint8_lossless else "u2"
    low8 = "true" if h5_uint8_lossless else "false"
    return (
        "<script>\n"
        f'globalThis.__QT_H5_DECODE_DTYPE ??= "{dtype}";\n'
        f"globalThis.__QT_H5_FORCE_LOW8 ??= {low8};\n"
        f"globalThis.__BSLZ4_LOW8_ONLY ??= {low8};\n"
        "globalThis.__BSLZ4_FRAME_WG ??= 64;\n"
        "globalThis.__QT_H5_FETCH_WINDOW ??= 8;\n"
        "globalThis.__QT_H5_DECODE_QUEUE ??= 8;\n"
        "globalThis.__QT_H5_PRELOAD_WINDOW ??= 1;\n"
        "globalThis.__QT_H5_LOCAL_GROUP ??= 8;\n"
        "globalThis.__QT_H5_LOCAL_WORKERS ??= 8;\n"
        "</script>\n"
    )


def export_show4dstem_webgpu_bundle(
    widget: Any,
    out_dir: str | pathlib.Path,
    *,
    port: int = 8794,
    title: str | None = None,
) -> pathlib.Path:
    """Write a double-clickable Show4DSTEM WebGPU bundle into ``out_dir``.

    ``out_dir`` must be the folder holding the ``*_master.h5`` family the widget
    was built from (``h5_url``/``h5_urls`` given as bare basenames or ``../``
    relative names). Produces ``Show4DSTEM.command`` at the root and a hidden
    ``.viewer/`` with the vendored page and the range-capable server. Returns
    the path to the launcher. Without this bundle the recipient needs Python,
    the CDNs, and a folder-grant click; with it the demo is one double-click.
    """
    root = pathlib.Path(out_dir)
    if not root.is_dir():
        raise ValueError(f"bundle out_dir must be an existing data folder: {root}")
    masters = sorted(root.glob("*_master.h5"))
    if not masters:
        raise ValueError(f"no *_master.h5 files in {root}; the bundle serves the data folder itself")
    viewer = root / ".viewer"
    viewer.mkdir(exist_ok=True)
    html = viewer / "Show4DSTEM.html"
    widget._write_html_export(html, dtype="uint16", det_bin=1, scan_bin=1, title=title)
    text = html.read_text(encoding="utf-8")
    text = text.replace(
        "<head>",
        "<head>\n"
        + _tuning(h5_uint8_lossless=bool(getattr(widget, "_h5_uint8_lossless", False))),
        1,
    )
    for pattern, local in _CDN_REWRITES:
        text = pattern.sub(local, text)
    html.write_text(text, encoding="utf-8")
    for name in ("require.min.js", "embed-amd.js", "anywidget.min.js"):
        _write_vendor_asset(name, viewer)
    return write_command_launcher(
        root,
        "Show4DSTEM",
        viewer_html=".viewer/Show4DSTEM.html",
        port=int(port),
    )


def bundle_master_urls(folder: str | pathlib.Path, names: Sequence[str] | None = None) -> list[str]:
    """Viewer-relative URLs (``../<basename>``) for masters in a bundle folder.

    The viewer page lives one level down in ``.viewer/``, so data references
    must climb back to the served root; a bare basename would resolve inside
    ``.viewer/`` and 404. ``names`` filters by substring, preserving its order.
    """
    folder = pathlib.Path(folder)
    masters = sorted(p.name for p in folder.glob("*_master.h5"))
    if names:
        picked = []
        for token in names:
            hits = [m for m in masters if token in m]
            if not hits:
                raise ValueError(f"no master matches {token!r} in {folder}")
            picked.append(hits[0])
        masters = picked
    return [f"../{name}" for name in masters]
