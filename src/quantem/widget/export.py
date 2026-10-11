"""Standalone HTML export shared by every widget with an export button.

``HtmlExportMixin`` owns the toolbar handshake (the ``export_*`` traits and
the request observer), the export options, the in-memory bytes path and the
standalone write. A widget lists the packings its page can hold
(``_HTML_ENCODINGS``, ``_HTML_DOWNSAMPLES``) and supplies
``_clone_for_html_export`` (an export-only copy of itself with that packing)
and ``_default_html_export_path``. ``write_widget_html`` embeds any widget as a
one-file page that opens with no network and on phones.
"""

import base64
import functools
import gzip
import html
import json
import pathlib
import tempfile

import traitlets

VENDOR = pathlib.Path(__file__).parent / "vendor"
# The toolbar menus send mode "exact" or "quantized"; Python calls name the encoding.
_TOOLBAR_ENCODINGS = {"exact": "full", "quantized": "uint8"}
_RUNTIME_LOADER = """<script>
(async () => {
  // require.js, anywidget and the jupyter-widgets manager travel gzipped in this page, so it opens offline
  for (const packed of document.querySelectorAll('script[type="application/gzip"][id^="quantem-widget-runtime-"]')) {
    const gzipped = await fetch(`data:application/gzip;base64,${packed.textContent}`);
    const script = document.createElement("script");
    script.textContent = await new Response(gzipped.body.pipeThrough(new DecompressionStream("gzip"))).text();
    document.head.appendChild(script);
  }
})();
</script>"""
MOBILE_VIEWPORT_META = '<meta name="viewport" content="width=device-width, initial-scale=1">'
STANDALONE_EXPORT_STYLE = """<style id="quantem-widget-export-layout">
html, body {
  margin: 0;
  padding: 0;
  width: 100%;
  max-width: 100%;
  overflow-x: hidden;
}
body {
  box-sizing: border-box;
}
*, *::before, *::after {
  box-sizing: inherit;
}
</style>"""


@functools.cache
def widget_runtime_html() -> str:
    """The script tags that start the widget manager from the copies vendored in the wheel, with no network.

    The ipywidgets embed page loads require.js, the jupyter-widgets
    html-manager and anywidget from CDNs, so it is blank offline (an
    air-gapped microscope PC, a plane) without a word. The vendored files ship gzipped
    (about 0.9 MB); they travel base64 in the page, the browser gunzips them
    (``DecompressionStream``) and runs them in order. anywidget is an anonymous
    AMD module, so it is named ``"anywidget"`` here for html-manager to find it
    without a fetch. The manager renders on load, or at once when the page has
    already loaded. Built once per process: every export carries the same tags.
    """
    anywidget = gzip.decompress((VENDOR / "anywidget.min.js.gz").read_bytes()).decode("utf-8")
    anywidget = anywidget.replace('define(["@jupyter-widgets/base"]', 'define("anywidget",["@jupyter-widgets/base"]', 1)
    packed = [
        (VENDOR / "require.min.js.gz").read_bytes(),
        gzip.compress(anywidget.encode("utf-8"), mtime=0),
        (VENDOR / "embed-amd.js.gz").read_bytes(),
    ]
    tags = [
        f'<script type="application/gzip" id="quantem-widget-runtime-{index}">{base64.b64encode(blob).decode("ascii")}</script>'
        for index, blob in enumerate(packed)
    ]
    return "\n".join([*tags, _RUNTIME_LOADER]) + "\n"


def export_slug(title: str, fallback: str) -> str:
    """File-name stem from a widget title: lowercase, one underscore between words."""
    label = (title or "").strip() or fallback
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or fallback


def write_widget_html(path: str | pathlib.Path, widget, title: str) -> pathlib.Path:
    """Embed one widget with its full state as a standalone HTML page that opens with no network.

    The page is the ``ipywidgets.embed`` one (widget state plus view script)
    with the vendored runtime in place of its CDN script tags, a mobile
    viewport and the full-width layout, written in one pass.
    """
    from ipywidgets.embed import dependency_state, embed_data, escape_script, widget_view_template

    export_path = pathlib.Path(path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    data = embed_data([widget], drop_defaults=False, state=dependency_state([widget], drop_defaults=False))
    views = "\n".join(widget_view_template.format(view_spec=escape_script(json.dumps(spec))) for spec in data["view_specs"])
    export_path.write_text(
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
        f"    {MOBILE_VIEWPORT_META}\n    <meta charset=\"UTF-8\">\n    <title>{html.escape(title)}</title>\n"
        f"    {STANDALONE_EXPORT_STYLE}\n</head>\n<body>\n"
        f"{widget_runtime_html()}"
        f"<script type=\"application/vnd.jupyter.widget-state+json\">\n{escape_script(json.dumps(data['manager_state'], indent=2))}\n</script>\n"
        f"{views}\n</body>\n</html>\n",
        encoding="utf-8",
    )
    return export_path


class HtmlExportMixin(traitlets.HasTraits):
    """Toolbar export handshake and standalone HTML writing.

    The frontend writes a JSON request into ``export_request``; Python answers
    through ``export_status`` and, for browser downloads, ``export_payload``
    with the matching ``export_payload_id`` / ``export_filename``. Widgets
    register ``_on_export_request_change`` on ``export_request`` in their
    ``__init__`` so a subclass override still receives the request.
    """

    export_request = traitlets.Unicode("").tag(sync=True)
    export_status = traitlets.Unicode("").tag(sync=True)
    export_enabled = traitlets.Bool(True).tag(sync=True)
    export_payload = traitlets.Bytes(b"").tag(sync=True)
    export_payload_id = traitlets.Unicode("").tag(sync=True)
    export_filename = traitlets.Unicode("").tag(sync=True)
    # The packings this widget's page can hold: exact float32 ("full"), uint8 per
    # panel, and the mean-bin factors of its embedded images.
    _HTML_ENCODINGS: tuple[str, ...] = ("full",)
    _HTML_DOWNSAMPLES: tuple[int, ...] = (1,)

    def _on_export_request_change(self, change: dict) -> None:
        """React to a toolbar request on ``export_request``: clear the download payload or serve the export."""
        raw = str(change.get("new") or "")
        if not raw:
            return
        try:
            payload = json.loads(raw)
            mode = str(payload.get("mode", ""))
            if mode == "clear":
                self.export_payload = b""
                self.export_payload_id = ""
                self.export_filename = ""
                return
            self._export_request(payload, mode)
        except (ValueError, TypeError, KeyError, OSError, RuntimeError, NotImplementedError, MemoryError) as exc:
            # the toolbar shows the failure; raising would only reach the kernel log
            self.export_status = f"Export failed: {exc}"

    def _export_request(self, payload: dict, mode: str) -> None:
        """Serve one toolbar request: the page as bytes for the browser to save."""
        options = self._html_export_options(payload, mode)
        filename = str(payload.get("filename") or self._default_html_export_path(**options).name)
        self.export_status = f"Preparing {filename}..."
        with tempfile.TemporaryDirectory(prefix=f"{type(self).__name__.lower()}-export-") as tmp:
            path = pathlib.Path(tmp) / filename
            self._write_html_export(path, **options)
            html = path.read_bytes()
        self.export_filename = filename
        self.export_payload = html
        self.export_payload_id = str(payload.get("id") or "")
        self.export_status = f"Ready {filename} ({len(html) / (1024 * 1024):.1f} MB, {self._export_mode_label(**options)})"

    def _html_export_options(self, payload: dict, mode: str) -> dict:
        """The packing options of a toolbar request, checked as an ``export_html`` call is."""
        return self._html_export_choice(_TOOLBAR_ENCODINGS.get(mode, payload.get("encoding", "full")), payload.get("downsample"))

    def _html_export_choice(self, encoding: str, downsample) -> dict:
        """The ``quantized`` / ``downsample`` options of an export, refusing a packing this widget's page cannot hold.

        Only the options a widget offers are returned (Show3DSlices takes no
        downsample, Show1D no uint8), so each widget's export hooks take
        exactly its own. A downsample of an exact float32 page is refused: that
        page promises every native pixel.
        """
        name = type(self).__name__
        if encoding not in self._HTML_ENCODINGS:
            raise ValueError(f"{name} export encoding must be one of {list(self._HTML_ENCODINGS)}, got {encoding!r}")
        factor = 1 if downsample is None else downsample
        if isinstance(factor, bool) or factor not in self._HTML_DOWNSAMPLES:
            raise ValueError(f"{name} export downsample must be one of {list(self._HTML_DOWNSAMPLES)}, got {downsample!r}")
        if factor > 1 and encoding == "full" and "uint8" in self._HTML_ENCODINGS:
            raise ValueError(f"{name} exact float32 export keeps every native pixel; downsample needs encoding='uint8'")
        options = {}
        if len(self._HTML_ENCODINGS) > 1:
            options["quantized"] = encoding == "uint8"
        if len(self._HTML_DOWNSAMPLES) > 1:
            options["downsample"] = int(factor)
        return options

    def _write_export_and_report(self, path, *, title: str | None, max_mb: float | None = None, **options) -> pathlib.Path:
        """Write the page of an ``export_html`` call and name its file, size and packing in ``export_status``.

        With ``max_mb`` a page estimated above it is refused before anything is
        written, naming the uint8 size: Chrome often fails to open a much larger
        single file from disk.
        """
        if max_mb is not None:
            estimate_mb = self._estimate_html_export_mb(**options)
            if estimate_mb > float(max_mb):
                uint8_mb = self._estimate_html_export_mb(quantized=True, downsample=options["downsample"])
                raise ValueError(
                    f"This export would embed about {estimate_mb:.0f} MB into one HTML file, "
                    f"above the {float(max_mb):.0f} MB safe limit (large single-file exports often "
                    f"fail to open under Chrome file://). Options: encoding='uint8' "
                    f"(about {uint8_mb:.0f} MB), downsample=2 or 4 to shrink spatially, or pass "
                    f"max_mb={estimate_mb:.0f} to force this size."
                )
        export_path = pathlib.Path(path) if path is not None else self._default_html_export_path(**options)
        self._write_html_export(export_path, title=title, **options)
        size_mb = export_path.stat().st_size / (1024 * 1024)
        self.export_status = f"Exported {export_path.name} ({size_mb:.1f} MB, {self._export_mode_label(**options)})"
        return export_path

    def _export_mode_label(self, quantized: bool = False, downsample: int = 1) -> str:
        """Packing named in the toolbar status, so the user sees whether the page holds float32 or uint8 pixels."""
        if not quantized:
            return "full float32"
        if int(downsample) > 1:
            return f"uint8, {int(downsample)}x downsample"
        return "uint8"

    def _write_html_export(self, path: str | pathlib.Path, *, title: str | None = None, **options) -> pathlib.Path:
        """Write the standalone page from an export clone, then release the clone."""
        export_widget = self._clone_for_html_export(**options)
        try:
            return write_widget_html(path, export_widget, title or self.title or type(self).__name__)
        finally:
            self._release_export_clone(export_widget)
            # Widget.close leaves the child Layout model open, one more per export
            export_widget.layout.close()

    def _release_export_clone(self, clone) -> None:
        """Close the export clone so its comm and pixel buffers do not outlive the export; widgets override it."""
        clone.close()
