"""``quantem`` command-line interface: a folder (or file) of images or 4D-STEM
masters becomes a rendered, standalone HTML viewer in one command, no notebook.

    quantem show ./frames/                # PNG/TIFF folder -> Show3D scrub HTML
    quantem show scan.png                 # single image    -> Show2D HTML
    quantem show ./masters/ --bin 8       # *_master.h5     -> offline WebGPU Show4DSTEM
    quantem ptycho scan_master.h5         # raw 4D-STEM    -> ShowPtycho WebGPU folder
    quantem html tutorial.ipynb           # run a notebook  -> standalone shareable HTML

The CLI only orchestrates existing pieces: ``io.read_image`` / ``read_image_stack``
for images, ``io.discover_masters`` + ``io.load(det_bin=...)`` for 4D-STEM and
ptychography review, the ``Show2D`` / ``Show3D`` / ``Show4DSTEM`` / ``ShowPtycho``
widgets, and each widget's export helpers. 4D-STEM is packed offline so the browser
does all compute on WebGPU, which is what lets a laptop browse data that never fit
full resolution (bin the detector first).
"""
import argparse
import email.utils
import http.server
import json
import mimetypes
import os
import pathlib
import posixpath
import re
import shutil
import socketserver
import sys
import threading
import urllib.parse
import webbrowser

# Single image -> Show2D, a folder of frames -> Show3D, a folder of differently
# sized images -> a Show2D gallery. These are the formats read_image understands.
IMAGE_EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".dm3", ".dm4", ".emd", ".npy"}
MASTER_PATTERN = "*_master.h5"
SHOWPTYCHO_MASTER_PATTERNS = ("*_master.h5", "*_master_wrapper.h5")
SHOWPTYCHO_FOLDER_FORMAT = "quantem.showptycho.webgpu.folder"
DEFAULT_PTYCHO_SEMIANGLE_MRAD = 30.0
DEFAULT_PTYCHO_SCAN_SAMPLING_A = 0.5
DEFAULT_PTYCHO_VOLTAGE_KV = 300.0
_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)$")
_RANGE_FALLBACK_CHUNK_BYTES = 16 * 1024 * 1024


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``quantem`` console script. Parse args, dispatch to the
    ``show`` subcommand, return a process exit code."""
    parser = argparse.ArgumentParser(
        prog="quantem",
        description="Render images or 4D-STEM masters as a viewer (HTML, or a live notebook for 4D).",
    )
    sub = parser.add_subparsers(dest="command")
    # `show` auto-detects; show2d/show3d/show4dstem force the widget so the command
    # reads exactly like the widget it opens. All share the same options + engine.
    forced = {
        "show": "auto",
        "show2d": "2d",
        "show3d": "3d",
        "show4dstem": "4dstem",
        "ptycho": "ptycho",
        "showptycho": "ptycho",
    }
    helps = {
        "show": "Auto-detect PATH(s) and render the matching viewer.",
        "show2d": "Render an image (or a folder of images) as Show2D.",
        "show3d": "Render a folder of frames as a Show3D scrub.",
        "show4dstem": "Render 4D-STEM master(s) as Show4DSTEM (live notebook, or --html).",
        "ptycho": "Open or build a ptychography WebGPU review folder.",
        "showptycho": "Compatibility alias for `quantem ptycho`.",
    }
    for name in forced:
        _add_show_args(sub.add_parser(name, help=helps[name]))
    # `html` is a different shape (one .ipynb in, one HTML out), so it gets its own
    # parser rather than the shared show* options.
    _add_html_args(sub.add_parser(
        "html", help="Execute a notebook and export it to a standalone, offline shareable HTML."))
    # `jupyter` starts JupyterLab here on the GPU box and prints a URL to paste into the
    # laptop browser - kernel + GPU here, UI in your browser. You bring your own tunnel
    # (SSH -L / VS Code) the same way quantem.live does.
    _add_jupyter_args(sub.add_parser(
        "jupyter", help="Start JupyterLab on this GPU box and print a URL to open from your laptop."))
    # `github` shrinks a widget notebook to a form GitHub can display: drop the heavy offline
    # widget-state, keep the auto-snapshot widget render (re-encoded JPEG) + print outputs.
    _add_github_args(sub.add_parser(
        "github", help="Make a widget notebook GitHub-displayable (strip offline state, snapshots to JPEG)."))
    _add_showfolder_args(sub.add_parser(
        "showfolder", help="Browse a microscopy folder with ShowFolder: inventory, thumbnails, and selection state."))
    _add_data_transfer_args(sub.add_parser(
        "data-transfer", help="Plan, inspect, and copy microscopy data layouts across folders/disks."))
    args = parser.parse_args(argv)
    try:
        if args.command == "html":
            return _render_html(args)
        if args.command == "jupyter":
            return _launch_jupyter(args)
        if args.command == "github":
            return _prepare_github(args)
        if args.command == "showfolder":
            return _showfolder(args)
        if args.command == "data-transfer":
            return _data_transfer(args)
        if args.command not in forced:
            parser.print_help()
            return 0
        args.widget = forced[args.command]
        return _show(args)
    except (FileNotFoundError, ValueError) as err:
        print(f"quantem: {err}", file=sys.stderr)
        return 1


def _add_html_args(parser: argparse.ArgumentParser) -> None:
    """Attach options for the ``html`` subcommand."""
    parser.add_argument("path", help="The .ipynb to render.")
    parser.add_argument("--out", default=None,
                        help="Output path or directory for the HTML. Default: ~/Downloads.")
    parser.add_argument("--no-execute", action="store_true",
                        help="Export the notebook's already-saved outputs without re-running it.")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Per-cell execution timeout in seconds (default 600).")
    parser.add_argument("--no-open", action="store_true", help="Write the HTML but do not open it.")


def _add_showfolder_args(parser: argparse.ArgumentParser) -> None:
    """Attach options for the ``showfolder`` subcommand."""
    parser.add_argument("folder", help="Folder of microscopy files to browse.")
    parser.add_argument("--html", default=None, help="Execute the ShowFolder notebook and write this HTML file.")
    parser.add_argument("--notebook", default=None, help="Write this ShowFolder notebook path.")
    parser.add_argument("--thumb", type=int, default=512, help="Thumbnail size for the HAADF/STEM gallery.")
    parser.add_argument("--glob", default="*.emd", help="Glob within the folder (default '*.emd').")
    parser.add_argument("--title", default=None, help="ShowFolder title.")
    parser.add_argument("--group-by", default="session", choices=("session", "fov", "none"),
                        help="ShowFolder layout grouping mode (default 'session').")
    parser.add_argument("--group-view", default="stack", choices=("stack", "gallery"),
                        help="Grouped image display mode (default 'stack').")
    parser.add_argument("--timeout", type=int, default=900, help="Notebook execution timeout in seconds.")
    parser.add_argument("--no-open", action="store_true", help="Write outputs but do not launch/open them.")


def _add_data_transfer_args(parser: argparse.ArgumentParser) -> None:
    """Attach options for the ``data-transfer`` subcommand."""
    parser.add_argument("action", choices=("plan", "inspect", "copy", "update", "masters", "show4dstem"),
                        help=(
                            "Plan a transfer, inspect/update a manifest, copy from a manifest, "
                            "print ready target masters, or write a Show4DSTEM handoff notebook."
                        ))
    parser.add_argument("source", nargs="?", help="Source folder or master path(s) for the plan action.")
    parser.add_argument("targets", nargs="*", help="Target folder(s) for the plan action.")
    parser.add_argument("--manifest", default=None,
                        help="Manifest path. Plan writes it; inspect/copy read it.")
    parser.add_argument("--pattern", default=MASTER_PATTERN, help="Master glob for planning.")
    parser.add_argument("--strategy", default="balance-by-size",
                        choices=("balance-by-size", "round-robin"),
                        help="Target assignment strategy.")
    parser.add_argument("--require-ready", action="store_true",
                        help="Skip masters that do not pass readiness checks.")
    parser.add_argument("--hash", dest="hash_algorithm", default=None,
                        choices=("none", "sha256"),
                        help="Include optional source hashes in the manifest.")
    parser.add_argument("--verify", default="size",
                        choices=("size", "none", "hash", "sha256"),
                        help="Verification mode for inspect/copy.")
    parser.add_argument("--execute", action="store_true",
                        help="Actually copy files. Without this, copy is a dry-run.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing targets after verification failure.")
    parser.add_argument("--show-masters", action="store_true",
                        help="Also print ready target master paths during inspect/update/copy.")
    parser.add_argument("--all-masters", action="store_true",
                        help="masters: print planned target masters even when targets are incomplete.")
    parser.add_argument("--gpus", default=None,
                        help="show4dstem: comma-separated CUDA GPU ids, e.g. 0 or 0,1.")
    parser.add_argument("--page-budget", default="auto",
                        help="show4dstem: resident dataset cache, e.g. auto, 1, 2, or none (default auto).")
    parser.add_argument("--dtype", default="u8", choices=("auto", "u8", "uint8", "u16", "uint16", "float32"),
                        help="show4dstem browse dtype (default u8 for fast visual screening).")
    parser.add_argument("--bin", type=int, default=1, dest="det_bin",
                        help="show4dstem detector binning factor (default 1: no detector binning).")
    parser.add_argument("--out", default=None,
                        help="show4dstem notebook output directory (default: ~/Downloads).")
    parser.add_argument("--no-open", action="store_true",
                        help="show4dstem: write the notebook but do not launch Jupyter.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")


def _data_transfer(args: argparse.Namespace) -> int:
    """Run data-transfer plan/inspect/copy commands."""
    from dataclasses import asdict
    from quantem.widget.io import (
        copy_data_transfer,
        data_transfer_load_warnings,
        inspect_data_transfer,
        plan_data_transfer,
        read_data_transfer_manifest,
        summarize_data_transfer,
        target_masters,
        update_data_transfer_plan,
        write_data_transfer_manifest,
    )

    if args.action == "plan":
        if not args.source or not args.targets:
            raise ValueError("data-transfer plan requires SOURCE and at least one TARGET.")
        plan = plan_data_transfer(
            pathlib.Path(args.source).expanduser(),
            [pathlib.Path(target).expanduser() for target in args.targets],
            pattern=args.pattern,
            strategy=args.strategy,
            require_ready=args.require_ready,
            hash_algorithm=args.hash_algorithm,
        )
        manifest = pathlib.Path(args.manifest).expanduser() if args.manifest else (
            pathlib.Path.cwd() / f"{plan.logical_name}_data_transfer.json"
        )
        write_data_transfer_manifest(plan, manifest)
        states = inspect_data_transfer(plan, verify=args.verify)
        summary = summarize_data_transfer(states)
        if args.json:
            print(json.dumps({"manifest": str(manifest), "plan": plan.to_dict(), "summary": asdict(summary)}, indent=2))
        else:
            print(f"manifest: {manifest}")
            _print_data_transfer_plan(plan)
            _print_data_transfer_summary(summary)
            _print_data_transfer_warnings(data_transfer_load_warnings(plan, devices=args.gpus, verify=args.verify))
        return 0

    if not args.manifest:
        raise ValueError(f"data-transfer {args.action} requires --manifest.")
    plan = read_data_transfer_manifest(args.manifest)
    manifest = pathlib.Path(args.manifest).expanduser()
    if args.action == "inspect":
        states = inspect_data_transfer(plan, verify=args.verify)
        summary = summarize_data_transfer(states)
        masters = target_masters(plan, verify=args.verify)
        warnings = data_transfer_load_warnings(plan, devices=args.gpus, verify=args.verify)
        if args.json:
            print(json.dumps({
                "summary": asdict(summary),
                "states": [asdict(state) for state in states],
                "masters": [str(master) for master in masters],
                "warnings": warnings,
            }, indent=2))
        else:
            _print_data_transfer_plan(plan)
            _print_data_transfer_summary(summary)
            _print_data_transfer_states(states)
            if args.show_masters:
                _print_data_transfer_masters(masters)
            _print_data_transfer_warnings(warnings)
        return 0
    if args.action == "copy":
        results = copy_data_transfer(
            plan,
            dry_run=not args.execute,
            verify=args.verify,
            overwrite=args.overwrite,
        )
        states = inspect_data_transfer(plan, verify=args.verify)
        summary = summarize_data_transfer(states)
        masters = target_masters(plan, verify=args.verify)
        warnings = data_transfer_load_warnings(plan, devices=args.gpus, verify=args.verify)
        if args.json:
            print(json.dumps({
                "executed": bool(args.execute),
                "results": [asdict(result) for result in results],
                "summary": asdict(summary),
                "masters": [str(master) for master in masters],
                "warnings": warnings,
            }, indent=2))
        else:
            print("copy executed" if args.execute else "copy dry-run")
            _print_data_transfer_results(results)
            _print_data_transfer_summary(summary)
            if args.show_masters:
                _print_data_transfer_masters(masters)
            _print_data_transfer_warnings(warnings)
        return 0
    if args.action == "update":
        updated = update_data_transfer_plan(
            plan,
            source=pathlib.Path(args.source).expanduser() if args.source else None,
            pattern=args.pattern,
            require_ready=args.require_ready,
            hash_algorithm=args.hash_algorithm,
        )
        write_data_transfer_manifest(updated, manifest)
        states = inspect_data_transfer(updated, verify=args.verify)
        summary = summarize_data_transfer(states)
        masters = target_masters(updated, verify=args.verify)
        warnings = data_transfer_load_warnings(updated, devices=args.gpus, verify=args.verify)
        if args.json:
            print(json.dumps({
                "manifest": str(manifest),
                "plan": updated.to_dict(),
                "summary": asdict(summary),
                "masters": [str(master) for master in masters],
                "warnings": warnings,
            }, indent=2))
        else:
            print(f"updated manifest: {manifest}")
            _print_data_transfer_plan(updated)
            _print_data_transfer_summary(summary)
            if args.show_masters:
                _print_data_transfer_masters(masters)
            _print_data_transfer_warnings(warnings)
        return 0
    if args.action == "masters":
        masters = target_masters(
            plan,
            existing_only=not args.all_masters,
            require_complete=not args.all_masters,
            verify=args.verify,
        )
        warnings = data_transfer_load_warnings(plan, devices=args.gpus, verify=args.verify)
        if args.json:
            print(json.dumps({
                "masters": [str(master) for master in masters],
                "ready_only": not args.all_masters,
                "warnings": warnings,
            }, indent=2))
        else:
            _print_data_transfer_masters(masters, ready_only=not args.all_masters)
            _print_data_transfer_warnings(warnings)
        return 0
    if args.action == "show4dstem":
        masters = target_masters(plan, verify=args.verify)
        if not masters:
            raise ValueError("no complete target masters are ready; run data-transfer copy --execute first.")
        warnings = data_transfer_load_warnings(plan, devices=args.gpus, verify=args.verify)
        _print_data_transfer_warnings(warnings)
        notebook = _render_data_transfer_show4dstem_notebook(manifest, plan.logical_name, masters, args)
        _launch_notebook(notebook, no_open=args.no_open)
        return 0
    raise ValueError(f"unknown data-transfer action: {args.action}")


def _print_data_transfer_plan(plan) -> None:
    print(
        f"plan: {plan.logical_name}  groups={len(plan.entries)}  "
        f"files={sum(len(entry.files) for entry in plan.entries)}  "
        f"bytes={_fmt_bytes(plan.total_bytes)}"
    )
    for target, size in plan.totals_by_target.items():
        print(f"  target {target}: {_fmt_bytes(size)}")
    if plan.skipped:
        print(f"  skipped not-ready groups: {len(plan.skipped)}")


def _print_data_transfer_summary(summary) -> None:
    counts = ", ".join(f"{key}={value}" for key, value in sorted(summary.status_counts.items()))
    print(
        f"state: {counts or 'no files'}  "
        f"complete={_fmt_bytes(summary.complete_bytes)} / {_fmt_bytes(summary.total_bytes)}  "
        f"problems={summary.problem_files}"
    )


def _print_data_transfer_states(states) -> None:
    for state in states[:80]:
        print(f"  {state.status:14s} {_fmt_bytes(state.size_bytes):>9s} {state.target}")
    if len(states) > 80:
        print(f"  ... {len(states) - 80} more files")


def _print_data_transfer_masters(masters, *, ready_only: bool = True) -> None:
    label = "ready masters" if ready_only else "planned masters"
    print(f"{label}: {len(masters)}")
    for master in masters:
        print(f"  {master}")


def _print_data_transfer_warnings(warnings) -> None:
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)


def _print_data_transfer_results(results) -> None:
    for result in results[:80]:
        print(f"  {result.status:8s} {_fmt_bytes(result.size_bytes):>9s} {result.target}")
    if len(results) > 80:
        print(f"  ... {len(results) - 80} more files")


def _fmt_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1000 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1000.0
    return f"{size:.1f} TB"


def _showfolder(args: argparse.Namespace) -> int:
    """Generate a microscopy folder browser notebook, optionally render it to HTML."""
    import shutil
    import subprocess
    from quantem.widget.showfolder_core import write_showfolder_notebook

    folder = pathlib.Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"not a folder: {folder}")
    if shutil.which("jupyter") is None and args.html:
        raise ValueError("jupyter not found; install jupyter to render survey HTML")

    html_out = pathlib.Path(args.html).expanduser().resolve() if args.html else None
    if args.notebook:
        notebook = pathlib.Path(args.notebook).expanduser().resolve()
    elif html_out is not None:
        notebook = html_out.with_suffix(".ipynb")
    else:
        notebook = _default_out_dir() / f"{folder.name}_showfolder.ipynb"

    write_showfolder_notebook(
        folder,
        notebook,
        glob=args.glob,
        thumb=args.thumb,
        title=args.title,
        group_by=args.group_by,
        group_view=args.group_view,
    )
    print(f"notebook: {notebook}")

    if html_out is None:
        _launch_notebook(notebook, no_open=args.no_open)
        return 0

    html_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "jupyter",
        "nbconvert",
        "--to",
        "html",
        "--execute",
        str(notebook),
        "--output-dir",
        str(html_out.parent),
        "--output",
        html_out.stem,
        f"--ExecutePreprocessor.timeout={args.timeout}",
        # explicit store_widget_state: ambient nbconvert config must not strip
        # the ShowFolder hydration state out of the share artifact
        "--ExecutePreprocessor.store_widget_state=True",
    ]
    print(f"executing + rendering ShowFolder -> {html_out}")
    if subprocess.run(cmd).returncode != 0:
        raise ValueError("ShowFolder nbconvert failed (see output above)")
    size_mb = html_out.stat().st_size / 1e6
    print(f"HTML: {size_mb:.1f} MB")
    _open_html(html_out, serve=False, no_open=args.no_open)
    return 0


def _render_html(args: argparse.Namespace) -> int:
    """Execute a notebook and export it to a standalone, shareable HTML.

    Wraps ``jupyter nbconvert --to html [--execute]``: a finished notebook becomes a
    kernel-less HTML page whose saved widget state is hydrated by the ipywidgets HTML
    manager. Show2D / Show3D / Show3DSlices / ShowEDS controls remain interactive in the browser,
    but changes are browser-local and do not write back to the notebook or HTML file.
    The live ``.ipynb`` stays the editable surface; this is the share artifact.
    ``--no-execute`` exports the saved outputs as-is, which is what a notebook's own
    in-cell ``!jupyter nbconvert`` does after a run."""
    import shutil
    import subprocess
    notebook = pathlib.Path(args.path).expanduser().resolve()
    if not notebook.exists():
        raise FileNotFoundError(f"notebook not found: {notebook}")
    if notebook.suffix.lower() != ".ipynb":
        raise ValueError(f"expected a .ipynb, got {notebook.suffix!r}")
    if shutil.which("jupyter") is None:
        raise ValueError("jupyter not found; install jupyter to render a notebook")
    out_dir = _out_dir(args.out)
    cmd = ["jupyter", "nbconvert", "--to", "html", str(notebook),
           "--output-dir", str(out_dir), "--output", notebook.stem]
    if not args.no_execute:
        # explicit store_widget_state: a jupyter_nbconvert_config.py that disables it
        # for heavy notebook sweeps would otherwise silently strip the hydration state
        # this share artifact exists to carry
        cmd += ["--execute", f"--ExecutePreprocessor.timeout={args.timeout}",
                "--ExecutePreprocessor.store_widget_state=True"]
    print(f"{'rendering' if args.no_execute else 'executing + rendering'} {notebook.name} -> HTML")
    if subprocess.run(cmd).returncode != 0:
        raise ValueError("nbconvert failed (see output above)")
    out = out_dir / f"{notebook.stem}.html"
    # Report the file size so the audience knows how heavy the share artifact is: baked
    # widget images make these big (a Show2D gallery can be >100 MB), which matters for
    # email limits and browser open time.
    size_mb = out.stat().st_size / 1e6
    note = "large - widget images baked in; trim panels if emailing" if size_mb > 50 else "self-contained, offline"
    print(f"HTML: {size_mb:.1f} MB ({note})")
    print(f"  {out}")
    _open_html(out, serve=False, no_open=args.no_open)
    return 0


# ---------------------------------------------------------------------------
_WIDGET_CELL = ("Show2D(", "Show3D(", "Show4DSTEM(", "Show3DSlices(", "ShowEDS(")


def _add_github_args(parser: argparse.ArgumentParser) -> None:
    """Attach options for the ``github`` subcommand."""
    parser.add_argument("path", help="The .ipynb to make GitHub-displayable (edited in place).")
    parser.add_argument("--no-execute", action="store_true",
                        help="Use the notebook's existing outputs instead of re-running it.")
    parser.add_argument("--quality", type=int, default=92,
                        help="JPEG quality for the embedded renders (default 92).")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Per-cell execution timeout in seconds (default 600).")


def _strip_state(nb: dict) -> None:
    """Drop the heavy offline live-widget manager-state + the dead widget-view output refs."""
    nb.get("metadata", {}).pop("widgets", None)
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []):
            (out.get("data") or {}).pop("application/vnd.jupyter.widget-view+json", None)


def _embed_jpeg(cell: dict, png_or_jpeg: bytes, quality: int) -> bool:
    """Replace a cell's visual output with one JPEG.

    Widget outputs usually have only ``application/vnd.jupyter.widget-view+json``,
    not an existing ``image/*`` slot.  For GitHub display we must add a normal
    image output before stripping the widget MIME bundle.
    """
    import base64
    from io import BytesIO
    from PIL import Image
    img = Image.open(BytesIO(png_or_jpeg)).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    done = False
    for out in cell.get("outputs", []):
        data = out.get("data")
        if data and (
            any(k.startswith("image/") for k in data)
            or "application/vnd.jupyter.widget-view+json" in data
        ):
            for k in [k for k in data if k.startswith("image/")]:
                del data[k]
            data["image/jpeg"] = b64
            out.setdefault("metadata", {})
            done = True
            break
    if not done:
        cell.setdefault("outputs", []).append({
            "output_type": "display_data",
            "metadata": {},
            "data": {"image/jpeg": b64},
        })
        done = True
    return done


def _cell_has_image_output(cell: dict) -> bool:
    """Return true when a notebook cell already has a GitHub-renderable image."""
    for out in cell.get("outputs", []):
        data = out.get("data") or {}
        if any(key.startswith("image/") for key in data):
            return True
    return False


def _cell_has_widget_view_output(cell: dict) -> bool:
    """Return true when a notebook cell still depends on live widget MIME output."""
    for out in cell.get("outputs", []):
        data = out.get("data") or {}
        if "application/vnd.jupyter.widget-view+json" in data:
            return True
    return False


def _capture_full_ui(html: pathlib.Path, n_expected: int) -> list[bytes]:
    """Screenshot each widget's FULL UI (toolbar + toggles + panels + histograms) from the
    rendered live-widget HTML, deterministically, via Playwright on the real GPU. The widget
    UI is React+MUI+WebGPU, so a browser engine is required; Playwright manages the lifecycle
    (waits for mount + paint) and ``locator.screenshot`` grabs each widget element exactly."""
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("DISPLAY", ":1")
    from playwright.sync_api import sync_playwright
    shots: list[bytes] = []
    launch_kwargs = {}
    for candidate in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        executable = shutil.which(candidate)
        if executable:
            launch_kwargs["executable_path"] = executable
            print(f"  browser executable: {executable}")
            break
    with sync_playwright() as play:
        browser = play.chromium.launch(headless=False, args=[
            "--enable-unsafe-webgpu", "--use-angle=vulkan", "--enable-features=Vulkan",
            "--ignore-gpu-blocklist", "--disable-gpu-sandbox", "--no-sandbox"], **launch_kwargs)
        page = browser.new_page(viewport={"width": 1300, "height": 2400}, device_scale_factor=2)
        page.goto(html.as_uri(), wait_until="load", timeout=90000)
        page.wait_for_timeout(13000)  # anywidget mount + WebGPU paint
        arch = page.evaluate("async()=>{const a=await navigator.gpu?.requestAdapter();"
                             "return a?(a.info?.architecture||'?'):'none';}")
        print(f"  GPU adapter: {arch}")
        if arch == "swiftshader":
            print("  warning: WebGPU reported SwiftShader; continuing because GitHub snapshots only need pixels")
        outs = page.locator(".jp-OutputArea-output")
        for i in range(outs.count()):
            el = outs.nth(i)
            if el.locator("canvas").count() > 0:
                el.scroll_into_view_if_needed()
                page.wait_for_timeout(700)
                shots.append(el.screenshot())
        browser.close()
    if len(shots) != n_expected:
        print(f"  warning: captured {len(shots)} widget UIs for {n_expected} widget cells")
    return shots


def _prepare_github(args: argparse.Namespace) -> int:
    """Make a widget notebook GitHub/VS-Code-displayable: embed a screenshot of each widget's
    FULL live UI (toolbar+toggles+panels) - because the whole reason to use the widget over
    ``show_2d`` is the UI, so the static render shows it (captured deterministically via
    Playwright on the real GPU). Drops the offline ``metadata.widgets`` state (tens of MB;
    GitHub won't render it and can't run widgets anyway) and JPEG-encodes each render (noisy
    science images compress ~10x). Keeps every other output (matplotlib PNGs, prints).

    The interactive widget still comes from re-running the notebook or ``quantem html``. Needs
    Playwright + a real GPU (NVIDIA Vulkan ICD + a display); errors clearly if unavailable."""
    import json
    import shutil
    import subprocess
    notebook = pathlib.Path(args.path).expanduser().resolve()
    if not notebook.exists():
        raise FileNotFoundError(f"notebook not found: {notebook}")
    if notebook.suffix.lower() != ".ipynb":
        raise ValueError(f"expected a .ipynb, got {notebook.suffix!r}")
    if shutil.which("jupyter") is None:
        raise ValueError("jupyter not found; install jupyter")
    before = notebook.stat().st_size
    if not args.no_execute:
        print(f"executing {notebook.name} ...")
        if subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                           str(notebook), f"--ExecutePreprocessor.timeout={args.timeout}"]).returncode != 0:
            raise ValueError("nbconvert --execute failed (see output above)")
    nb = json.loads(notebook.read_text())
    widget_cells = [c for c in nb["cells"]
                    if c["cell_type"] == "code" and any(w in "".join(c["source"]) for w in _WIDGET_CELL)]
    capture_cells = [
        c for c in widget_cells
        if not _cell_has_image_output(c)
    ]
    if capture_cells:
        try:
            html = notebook.with_suffix(".fullui.html")
            subprocess.run(["jupyter", "nbconvert", "--to", "html", str(notebook),
                            "--output-dir", str(notebook.parent), "--output", notebook.stem + ".fullui"],
                           check=True)
            print(f"capturing {len(capture_cells)} widget UI(s) on the GPU ...")
            shots = _capture_full_ui(html, len(capture_cells))
            html.unlink(missing_ok=True)
            if len(shots) != len(capture_cells):
                raise ValueError(
                    f"captured {len(shots)} widget UI screenshot(s) for {len(capture_cells)} widget cell(s)"
                )
            for cell, png in zip(capture_cells, shots):
                _embed_jpeg(cell, png, args.quality)
            mode = f"{len(shots)} full-UI screenshots"
        except (ImportError, RuntimeError, OSError) as err:
            raise ValueError(
                "full-UI capture needs Playwright + a real GPU (NVIDIA Vulkan ICD + a display): "
                f"{err}") from err
    elif widget_cells:
        mode = f"{len(widget_cells)} existing image output(s)"
    else:
        mode = "no widget cells - state stripped only"
    _strip_state(nb)
    notebook.write_text(json.dumps(nb, indent=1))
    after = notebook.stat().st_size
    print(f"github-ready: {notebook.name}  {before / 1e6:.1f} MB -> {after / 1e6:.1f} MB"
          f"  ({mode}, JPEG q{args.quality}, offline state stripped)")
    if after > 5e6:
        print("  warning: still > 5 MB - GitHub may not render. Lower --quality or the widget's size=.")
    return 0


# ---------------------------------------------------------------------------
def _add_jupyter_args(parser: argparse.ArgumentParser) -> None:
    """Attach options for the ``jupyter`` subcommand."""
    parser.add_argument("path", nargs="?", default=None,
                        help="Notebook or directory to open (relative to where you run this, or "
                             "absolute). A .ipynb opens directly.")
    parser.add_argument("--env", default="live-env",
                        help="Conda/mamba env to activate before launching JupyterLab "
                             "(default: live-env). Pass --env '' to skip activation.")
    parser.add_argument("--port", type=int, default=None,
                        help="Port to serve on (default: auto-pick a free port).")
    parser.add_argument("--no-open", action="store_true",
                        help="Start JupyterLab but do not open a browser (the box is usually "
                             "headless - copy the printed URL into your laptop browser).")


def _free_port() -> int:
    """Pick a free local TCP port by binding to port 0 and reading back what the OS chose.
    Runs on the same box JupyterLab will, so the port is guaranteed free for the server."""
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _config_path() -> pathlib.Path:
    """Where the per-user jupyter config lives. Honors XDG_CONFIG_HOME, else ~/.config."""
    base = os.environ.get("XDG_CONFIG_HOME")
    root = pathlib.Path(base) if base else pathlib.Path.home() / ".config"
    return root / "quantem" / "jupyter.json"


def _ssh_target() -> str:
    """The `user@host` a laptop uses to SSH into this box, for the copy-paste tunnel line.

    Saved once in the per-user config so the printed `ssh -L` command is ready to paste
    with no placeholder to edit. On first interactive launch we auto-detect `whoami@fqdn`,
    show it, and let the user accept (Enter) or correct it, then persist it. Later launches
    read it silently. Non-interactive launches (no TTY) fall back to the auto-detected guess
    without prompting or saving, so a headless run still prints a usable command."""
    import getpass
    import json
    import socket
    cfg = _config_path()
    if cfg.exists():
        try:
            saved = json.loads(cfg.read_text()).get("ssh_target")
            if saved:
                return saved
        except (OSError, ValueError):
            pass
    default = f"{getpass.getuser()}@{socket.getfqdn()}"
    if not sys.stdin.isatty():
        return default
    answer = input(f"SSH target your laptop uses to reach this box [{default}]: ").strip()
    target = answer or default
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(json.dumps({"ssh_target": target}, indent=2))
    print(f"  saved to {cfg} (edit it anytime to change)")
    return target


def _strip_json_line_comments(text: str) -> str:
    """Remove the line comments JupyterLab may put in .jupyterlab-settings files."""
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("//"))


def _jupyterlab_user_settings_dir() -> pathlib.Path:
    """Return the JupyterLab user-settings dir for this process.

    JupyterLab stores per-plugin settings under
    ``<jupyter-config>/lab/user-settings`` unless ``JUPYTERLAB_SETTINGS_DIR`` is
    set.  We write the widgets-manager setting there before launching Lab so a
    normal notebook save also saves widget state.
    """
    override = os.environ.get("JUPYTERLAB_SETTINGS_DIR")
    if override:
        return pathlib.Path(override)
    config = os.environ.get("JUPYTER_CONFIG_DIR")
    root = pathlib.Path(config) if config else pathlib.Path.home() / ".jupyter"
    return root / "lab" / "user-settings"


def _enable_jupyterlab_widget_state_save() -> pathlib.Path:
    """Enable JupyterLab's automatic widget-state save setting.

    Without this setting, Cmd+S saves code/output but may omit ``metadata.widgets``.
    Then reopening a notebook later has no frontend model state to hydrate.  The
    fixed Lab manager can restore saved widgets, but only if the state is saved.
    """
    path = (
        _jupyterlab_user_settings_dir()
        / "@jupyter-widgets"
        / "jupyterlab-manager"
        / "plugin.jupyterlab-settings"
    )
    settings: dict[str, object] = {}
    if path.exists():
        raw = path.read_text()
        try:
            settings = json.loads(_strip_json_line_comments(raw) or "{}")
        except ValueError:
            backup = path.with_suffix(path.suffix + ".bak")
            backup.write_text(raw)
            settings = {}
    settings["saveState"] = True
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n")
    return path


def _launch_jupyter(args: argparse.Namespace) -> int:
    """Start JupyterLab here on the GPU box and print a URL to paste into the laptop
    browser - kernel + GPU run here, UI in your browser. anywidget rides the Jupyter Comm
    channel, so one server serves every quantem widget (Show2D/3D/4DSTEM, SSB, any
    tutorial) with no per-widget setup. You bring your own laptop->box hop (SSH ``-L`` or
    VS Code Remote-SSH), same as quantem.live; this command does no SSH or tunnel work.

    Runs under a login shell so conda init is sourced, in the foreground so JupyterLab
    lives as long as the command does (Ctrl-C stops it)."""
    import secrets
    import shlex
    import subprocess
    import threading
    import webbrowser
    settings_path = _enable_jupyterlab_widget_state_save()
    port = args.port or _free_port()
    token = secrets.token_hex(16)
    # --log-level=WARN silences the ~20 INFO log lines that otherwise scroll the
    # paste-line banner off-screen. Errors + warnings still surface.
    launch = (f"jupyter lab --no-browser --ip=127.0.0.1 --port={port} --log-level=WARN "
              f"--IdentityProvider.token={token} --ServerApp.token={token}")
    if args.env:
        # `conda` is usually NOT on a non-interactive login shell's PATH, so a bare
        # `conda activate` fails. Source conda.sh from the common install locations
        # first, then activate. Covers miniforge/miniconda/anaconda/mambaforge.
        src = ("for c in ~/miniforge3 ~/miniconda3 ~/anaconda3 ~/mambaforge; do "
               "[ -f \"$c/etc/profile.d/conda.sh\" ] && . \"$c/etc/profile.d/conda.sh\" && break; done")
        launch = f"{src} && conda activate {shlex.quote(args.env)} && {launch}"
    if args.path and not args.path.endswith(".ipynb"):
        launch = f"cd {shlex.quote(args.path)} && {launch}"
    sub = ""
    if args.path and args.path.endswith(".ipynb"):
        sub = "/tree/" + args.path.lstrip("/")
    url = f"http://localhost:{port}/lab{sub}?token={token}"
    # Resolve (and on first run, save) the SSH target BEFORE printing, so its one-time
    # prompt doesn't interrupt the URL block.
    target = _ssh_target()
    # Three short one-liners — one launcher per OS, no URL repetition. macOS
    # uses `open`, Linux uses `xdg-open`, Windows (PowerShell / Git Bash / cmd)
    # uses `start ""` (PowerShell alias for Start-Process; Git Bash forwards to
    # Windows start; the empty-title arg works in all three Windows shells).
    tunnel = f"ssh -fN -L {port}:127.0.0.1:{port} {target}"
    line_mac = f'{tunnel} && open "{url}"'
    line_lnx = f'{tunnel} && xdg-open "{url}"'
    line_win = f'{tunnel} ; start "" "{url}"'
    YL = "\033[1;33m"      # bold yellow — frame
    CY = "\033[1;96m"      # bold bright cyan — the paste command
    GN = "\033[1;32m"      # bold green — status
    DM = "\033[2;37m"      # dim white — alternates
    AR = "\033[1;31m"      # bold red — arrows
    LB = "\033[1;35m"      # bold magenta — OS labels
    RST = "\033[0m"
    bar = "═" * 72
    arrow = "↓ ↓ ↓"
    print()
    print(f"{YL}╔{bar}╗{RST}")
    print(f"{YL}║{RST}    {AR}{arrow}{RST}  {YL}COPY-PASTE ONE LINE INTO YOUR LAPTOP TERMINAL{RST}  {AR}{arrow}{RST}        {YL}║{RST}")
    print(f"{YL}╚{bar}╝{RST}")
    print()
    print(f"  {LB}macOS:{RST}    {CY}{line_mac}{RST}")
    print(f"  {LB}Linux:{RST}    {CY}{line_lnx}{RST}")
    print(f"  {LB}Windows:{RST}  {CY}{line_win}{RST}")
    print()
    print(f"  {GN}✓ JupyterLab running on this box, port {port}. Ctrl-C here to stop.{RST}")
    print(f"  {GN}✓ Widget state auto-save enabled for Cmd+S.{RST} {DM}({settings_path}){RST}")
    print(f"  {DM}(URL alone): {url}{RST}")
    print(f"  {DM}(tunnel alone): ssh -L {port}:127.0.0.1:{port} {target}{RST}")
    print(f"  {DM}(no SSH key yet? https://github.com/ophusgroup/dev#appendix-c-ssh-for-github-and-gpu-servers){RST}")
    print()
    # Flush now: the foreground server below never returns, so block-buffered stdout
    # (when redirected to a file/pipe, not a TTY) would otherwise hide the URL forever.
    sys.stdout.flush()
    if not args.no_open:
        # If a browser is reachable (rare on a headless box), open it after the server
        # has had a moment to come up. The foreground process below keeps it alive.
        threading.Timer(3.0, lambda: webbrowser.open(url)).start()
    return subprocess.run(["bash", "-lc", launch]).returncode


def _add_show_args(parser: argparse.ArgumentParser) -> None:
    """Attach the shared options to a show* subparser (one source of truth)."""
    parser.add_argument("path", nargs="+",
                        help="An image, a folder of images, a 4D-STEM master, a folder of masters, "
                             "or several master files (-> one 5D multi-tilt viewer).")
    parser.add_argument("--bin", type=int, default=None, dest="det_bin",
                        help=(
                            "Detector binning factor. Show4DSTEM defaults to 8 "
                            "for laptop browsing; ShowPtycho defaults to 1."
                        ))
    parser.add_argument("--combined", action="store_true",
                        help="Many 4D masters -> one 5D HTML viewer (with --html; needs a local serve).")
    parser.add_argument("--out", default=None,
                        help="Output path (single) or directory (batch). Default: ~/Downloads.")
    parser.add_argument("--quantized", action="store_true",
                        help="Image widgets: uint8 pack (smaller file).")
    parser.add_argument("--html", action="store_true",
                        help="4D-STEM: export a standalone offline-WebGPU HTML instead of a live notebook.")
    parser.add_argument("--watch", action="store_true",
                        help="Folder: write a live ShowFolder-watched notebook that appends new files.")
    parser.add_argument("--watch-interval", type=float, default=2.0,
                        help="Polling interval in seconds for --watch live folders (default 2).")
    parser.add_argument("--gpus", default=None,
                        help="4D-STEM --watch: comma-separated CUDA GPU ids, e.g. 0 or 0,1. Default preserves loader device.")
    parser.add_argument("--page-budget", default="auto",
                        help="4D-STEM --watch: resident dataset cache, e.g. auto, 1, 2, or none (default auto).")
    parser.add_argument("--dtype", default="u8", choices=("u8", "uint8", "u16", "uint16", "float32"),
                        help="4D-STEM/ShowPtycho browse dtype (default u8).")
    parser.add_argument("--scan-size", type=int, default=None,
                        help="4D-STEM --watch: only include masters with this square scan size.")
    parser.add_argument("--no-open", action="store_true", help="Write the file(s) but do not launch anything.")
    parser.add_argument("--serve", action="store_true",
                        help="Open via a local HTTP server even for self-contained files (tunnelable URL).")
    parser.add_argument("--port", type=int, default=None,
                        help="Folder exports: port for the local HTTP server (default: auto-pick).")
    parser.add_argument("--bind", default="127.0.0.1",
                        help="Folder exports: bind address for the local HTTP server (default: 127.0.0.1).")
    parser.add_argument("--title", default=None, help="Viewer page title.")
    parser.add_argument("--backend", default="auto", choices=("auto", "cuda", "mps", "cpu"),
                        help="ShowPtycho master generation: HDF5 load backend (default auto).")
    parser.add_argument("--webgpu-source", default="compressed-hdf5",
                        choices=("compressed-hdf5", "hdf5", "bf-columns", "auto"),
                        help=(
                            "ShowPtycho folder browser source. Default compressed-hdf5 uses "
                            "the original HDF5 files and browser WebGPU decompression; "
                            "bf-columns is an explicit fallback/comparison companion."
                        ))
    parser.add_argument("--calibration", default="auto",
                        help=(
                            "ShowPtycho master generation: calibration JSON, "
                            "'auto' to search nearby QuantEM results, or 'none'."
                        ))
    parser.add_argument("--semiangle", "--semiangle-mrad", dest="semiangle_mrad",
                        type=float, default=None,
                        help="ShowPtycho master generation: probe semi-angle in mrad.")
    parser.add_argument("--scan-sampling", "--scan-sampling-A", dest="scan_sampling_A",
                        type=float, default=None,
                        help="ShowPtycho master generation: scan pixel size in Angstrom.")
    parser.add_argument("--det-sampling", "--det-sampling-mrad-px", dest="det_sampling_mrad_px",
                        type=float, default=None,
                        help="ShowPtycho master generation: detector angular sampling in mrad/pixel.")
    parser.add_argument("--voltage-kv", dest="voltage_kv", type=float, default=None,
                        help="ShowPtycho master generation: accelerating voltage in kV.")
    parser.add_argument("--bf-radius", type=float, default=None,
                        help="ShowPtycho master generation: bright-field radius in detector pixels.")
    parser.add_argument("--bf-threshold", type=float, default=0.5,
                        help="ShowPtycho master generation: BF intensity threshold fraction (default 0.5).")
    parser.add_argument("--drag-bf", type=float, default=1.0,
                        help="ShowPtycho browser BF fraction/count: 1.0 is full BF, 0.3 is 30%%, values greater than 1 are explicit BF-pixel counts (default 1.0).")
    parser.add_argument("--size", type=int, default=800,
                        help="ShowPtycho initial panel size in pixels (default 800).")
    parser.add_argument("--fft", action="store_true",
                        help="ShowPtycho opens with the FFT panel visible.")
    parser.add_argument("--force", action="store_true",
                        help="ShowPtycho master generation: rebuild an existing output folder.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose progress.")


# ---------------------------------------------------------------------------
def _show(args: argparse.Namespace) -> int:
    """Resolve the content, render the matching widget(s), open the result.

    Images render to a standalone HTML (light, shareable, opens with a double-click).
    4D-STEM renders to a live Jupyter notebook by default (full real-time WebGPU, no
    large file); ``--html`` instead exports the self-contained offline-WebGPU HTML.
    One path can be a file or a folder; several paths are taken as a list of 4D-STEM
    masters and become one 5D multi-tilt viewer."""
    paths = [pathlib.Path(p).expanduser().resolve() for p in args.path]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("path does not exist: " + ", ".join(missing))
    # Several explicit paths: a list of masters -> one 5D viewer (multi-tilt), or a
    # set of image files -> a gallery. A single path falls through to _detect.
    if len(paths) > 1:
        if args.watch:
            raise ValueError("--watch requires one folder path, not multiple explicit paths.")
        if args.widget != "4dstem" and all(p.suffix.lower() in IMAGE_EXTS for p in paths):
            out = _render_gallery(paths, "gallery", args)
            _open_html(out, serve=args.serve, no_open=args.no_open)
            return 0
        masters = [str(p) for p in paths]
        return _do_4dstem(masters, f"{len(masters)}_datasets", args)
    path = paths[0]
    kind = _detect(path, args.widget)
    if kind == "showptycho-master":
        master = _showptycho_master_source(path)
        folder = _render_showptycho_master(master, args)
        _serve_showptycho_folder(folder, bind=args.bind, port=args.port, no_open=args.no_open)
        return 0
    if kind == "showptycho":
        folder = _showptycho_folder(path)
        _serve_showptycho_folder(folder, bind=args.bind, port=args.port, no_open=args.no_open)
        return 0
    if kind == "4dstem":
        if args.watch:
            if args.html:
                raise ValueError("--watch writes a live notebook; omit --html.")
            if not path.is_dir():
                raise ValueError("--watch requires a folder path containing *_master.h5 files.")
        from quantem.widget.io import discover_masters
        masters = [str(path)] if path.is_file() else discover_masters(str(path), verbose=args.verbose)
        if not masters:
            raise ValueError(f"no *_master.h5 found in {path}")
        label = pathlib.Path(masters[0]).stem.replace("_master", "") if path.is_file() else path.name
        if args.watch:
            notebook = _render_4dstem_watch_notebook(path, label, args)
            _launch_notebook(notebook, no_open=args.no_open)
            return 0
        return _do_4dstem(masters, label, args)
    if args.watch:
        if args.html:
            raise ValueError("--watch writes a live notebook; omit --html.")
        if kind != "images" or not path.is_dir():
            raise ValueError("--watch requires one folder path.")
        widget = "show3d" if args.widget == "3d" else "show2d"
        notebook = _render_image_watch_notebook(path, path.name, args, widget=widget)
        _launch_notebook(notebook, no_open=args.no_open)
        return 0
    out = _render_images(path, kind, args)
    _open_html(out, serve=args.serve, no_open=args.no_open)
    return 0


def _do_4dstem(masters: list[str], label: str, args: argparse.Namespace) -> int:
    """Dispatch 4D-STEM master(s) to either a live notebook (default) or an offline
    HTML (``--html``), then launch/open it. One master loads alone; many load stacked
    into a 5D viewer with a dataset slider (the multi-tilt case)."""
    args.det_bin = _effective_det_bin(args, default=8)
    if args.html:
        outputs = _render_4dstem(masters, label, args)
        _open_html(outputs[0], serve=args.serve or args.combined, no_open=args.no_open)
        if len(outputs) > 1:
            print(f"wrote {len(outputs)} HTML files to {outputs[0].parent}")
        return 0
    notebook = _render_4dstem_notebook(masters, label, args)
    _launch_notebook(notebook, no_open=args.no_open)
    return 0


def _detect(path: pathlib.Path, forced: str) -> str:
    """Return the content kind: image, images, 4dstem, showptycho, or ptycho master.

    A single file is always 'image' unless it is a master or 4D is forced (a lone
    file can't be a 3D scrub). For a folder: the command's forced widget wins, else a
    ``*_master.h5`` makes it 4D and image files make it 'images'. The stack-vs-gallery
    split for 'images' is decided later from the forced widget."""
    if forced == "ptycho":
        if _is_showptycho_folder_export(path):
            return "showptycho"
        if path.is_file() and _is_showptycho_master_name(path.name):
            return "showptycho-master"
        if path.is_dir() and _showptycho_master_candidates(path):
            return "showptycho-master"
        _showptycho_folder(path)
        return "showptycho"
    if _is_showptycho_folder_export(path):
        return "showptycho"
    if path.is_file():
        if forced == "4dstem" or path.name.endswith("_master.h5"):
            return "4dstem"
        if forced in ("2d", "3d", "auto") and path.suffix.lower() in IMAGE_EXTS:
            return "image"
        raise ValueError(f"unsupported file type {path.suffix!r}; expected an image or *_master.h5")
    if forced == "4dstem":
        return "4dstem"
    if forced in ("2d", "3d"):
        return "images"
    masters = sorted(path.glob(MASTER_PATTERN))
    if masters:
        return "4dstem"
    if any(p.suffix.lower() in IMAGE_EXTS for p in path.iterdir()):
        return "images"
    raise ValueError(f"no images or *_master.h5 found in {path}")


def _effective_det_bin(args: argparse.Namespace, *, default: int) -> int:
    """Return a positive detector bin, using the command-specific default."""

    raw = getattr(args, "det_bin", None)
    value = default if raw is None else raw
    try:
        det_bin = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"--bin must be a positive integer, got {value!r}") from exc
    if det_bin < 1:
        raise ValueError(f"--bin must be a positive integer, got {det_bin}")
    return det_bin


def _showptycho_decode_dtype(args: argparse.Namespace) -> str:
    """Return the explicit browser decode dtype for ShowPtycho source HDF5."""

    raw = str(getattr(args, "dtype", "u8") or "u8").lower()
    if raw in {"u8", "uint8"}:
        return "uint8"
    if raw in {"u16", "uint16"}:
        return "uint16"
    if raw == "float32":
        return "float32"
    raise ValueError(f"ShowPtycho --dtype must be u8, u16, or float32; got {raw!r}")


def _showptycho_webgpu_source(args: argparse.Namespace) -> str:
    """Return the browser source policy for generated ShowPtycho folders."""

    raw = str(getattr(args, "webgpu_source", "compressed-hdf5") or "compressed-hdf5").lower()
    return {
        "compressed-hdf5": "compressed_hdf5",
        "hdf5": "compressed_hdf5",
        "bf-columns": "bf_columns",
        "auto": "auto",
    }[raw]


def _is_showptycho_master_name(name: str) -> bool:
    """Return whether ``name`` is a supported ShowPtycho source master."""

    return name.endswith("_master.h5") or name.endswith("_master_wrapper.h5")


def _showptycho_master_candidates(path: pathlib.Path) -> list[pathlib.Path]:
    """Find supported ShowPtycho source masters in a folder."""

    masters: set[pathlib.Path] = set()
    for pattern in SHOWPTYCHO_MASTER_PATTERNS:
        masters.update(path.glob(pattern))
    return sorted(masters)


def _showptycho_folder(path: pathlib.Path) -> pathlib.Path:
    """Return the folder for a ShowPtycho WebGPU export, or raise with next steps."""
    folder = path.parent if path.is_file() and path.name == "index.html" else path
    if not folder.is_dir():
        raise FileNotFoundError(f"not a ShowPtycho folder export: {path}")
    index = folder / "index.html"
    manifest = folder / "manifest.json"
    if not index.is_file():
        raise ValueError(f"ShowPtycho folder export is missing index.html: {folder}")
    if not manifest.is_file():
        raise ValueError(f"ShowPtycho folder export is missing manifest.json: {folder}")
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"ShowPtycho manifest is not valid JSON: {manifest}") from exc
    format_name = str(payload.get("format", ""))
    if not format_name.startswith(SHOWPTYCHO_FOLDER_FORMAT):
        raise ValueError(
            "not a ShowPtycho WebGPU folder export; expected manifest format "
            f"{SHOWPTYCHO_FOLDER_FORMAT!r}, got {format_name!r}"
        )
    return folder


def _is_showptycho_folder_export(path: pathlib.Path) -> bool:
    """Best-effort detector used by ``quantem show`` before normal image/4D routing."""
    folder = path.parent if path.is_file() and path.name == "index.html" else path
    manifest = folder / "manifest.json"
    if not manifest.is_file():
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return str(payload.get("format", "")).startswith(SHOWPTYCHO_FOLDER_FORMAT)


def _showptycho_master_source(path: pathlib.Path) -> pathlib.Path:
    """Resolve a ShowPtycho source master from a file or one-master folder."""

    if path.is_file():
        if not _is_showptycho_master_name(path.name):
            raise ValueError(
                "ShowPtycho generation expects *_master.h5 or "
                f"*_master_wrapper.h5, got {path.name}"
            )
        return path
    masters = _showptycho_master_candidates(path)
    if not masters:
        raise ValueError(f"no ShowPtycho master HDF5 file found in {path}")
    if len(masters) > 1:
        raise ValueError(
            "ShowPtycho generation expects one master at a time; pass the "
            f"specific master file (found {len(masters)} in {path})."
        )
    return masters[0]


def _showptycho_source_stem(master: pathlib.Path) -> str:
    """Return the microscope source stem without the Arina ``_master`` suffix."""

    stem = master.stem
    if stem.endswith("_master_wrapper"):
        return stem[:-len("_master_wrapper")]
    return stem[:-len("_master")] if stem.endswith("_master") else stem


def _showptycho_output_dir(master: pathlib.Path, args: argparse.Namespace) -> pathlib.Path:
    """Resolve the ShowPtycho folder output path."""

    if args.out:
        target = pathlib.Path(args.out).expanduser()
        if target.suffix.lower() in {".html", ".htm", ".ipynb"}:
            raise ValueError("ShowPtycho writes a folder export; pass --out as a directory.")
        target.mkdir(parents=True, exist_ok=True)
        return target
    target = _default_out_dir() / f"{_showptycho_source_stem(master)}_showptycho"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _showptycho_calibration_search_paths(master: pathlib.Path) -> list[pathlib.Path]:
    """Nearby calibration files created by the QuantEM ptychography workflow."""

    stem = _showptycho_source_stem(master)
    root = master.parent
    return [
        root / "quantem" / "screen" / "_calibrations.json",
        root / "quantem" / "screen" / stem / "calibration.json",
        root / "quantem" / "ptycho" / stem / "calibration.json",
    ]


def _showptycho_master_calib_paths(master: pathlib.Path) -> list[pathlib.Path]:
    """Nearby run metadata files that can fill microscope geometry."""

    stem = _showptycho_source_stem(master)
    root = master.parent
    return [
        root / "quantem" / "ptycho" / stem / "master_calib.json",
        root / "quantem" / "ptycho" / stem / "_runspec.json",
    ]


def _mapping_matches_showptycho_source(
    payload: dict,
    *,
    master: pathlib.Path,
) -> bool:
    """Return whether a calibration object belongs to ``master``."""

    stem = _showptycho_source_stem(master)
    candidates = {
        str(payload.get("source_stem", "")),
        str(payload.get("label", "")),
    }
    source_file = payload.get("source_file") or payload.get("master_path")
    if source_file:
        candidates.add(_showptycho_source_stem(pathlib.Path(str(source_file))))
    return stem in candidates or master.stem in candidates


def _calibration_from_showptycho_mapping(payload: dict):
    """Convert a calibration mapping to ``PtychoCalibration``."""

    from quantem.widget.showptycho import PtychoCalibration

    aberrations = {
        str(k): float(v) for k, v in (payload.get("aberrations") or {}).items()
    }
    if "C10" not in aberrations and "C10_nm" in payload:
        aberrations["C10"] = float(payload["C10_nm"])
    if "C12" not in aberrations and "C12_nm" in payload:
        aberrations["C12"] = float(payload["C12_nm"])
    if "phi12" not in aberrations and "phi12_rad" in payload:
        aberrations["phi12"] = float(payload["phi12_rad"])
    if "phi12" not in aberrations and "phi12_deg" in payload:
        import math

        aberrations["phi12"] = math.radians(float(payload["phi12_deg"]))
    if "rotation_angle_deg" not in payload:
        raise ValueError("calibration is missing rotation_angle_deg")
    return PtychoCalibration(
        rotation_angle_deg=float(payload["rotation_angle_deg"]),
        aberrations=aberrations,
        higher_order={
            str(k): float(v) for k, v in (payload.get("higher_order") or {}).items()
        },
        flip_phase=bool(payload.get("flip_phase", False)),
        voltage_kV=payload.get("voltage_kV") or payload.get("voltage_kv"),
        semiangle_mrad=payload.get("semiangle_mrad") or payload.get("semiangle"),
        scan_sampling_A=(
            payload.get("scan_sampling_A")
            or payload.get("scan_sampling")
            or payload.get("scan_sampling_A_per_px")
        ),
        loss=payload.get("loss"),
        source_file=payload.get("source_file"),
        source_stem=payload.get("source_stem"),
        label=payload.get("label"),
        notes=str(payload.get("notes", "")),
    )


def _load_showptycho_calibration(path: pathlib.Path, *, master: pathlib.Path):
    """Load a single calibration, choosing the matching entry from a list file."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("calibrations"), list):
        payload = payload["calibrations"]
    if isinstance(payload, list):
        matches = [
            item for item in payload
            if isinstance(item, dict)
            and "rotation_angle_deg" in item
            and _mapping_matches_showptycho_source(item, master=master)
        ]
        if not matches:
            raise ValueError(f"no calibration in {path} matches {_showptycho_source_stem(master)}")

        def score(item: dict) -> float:
            loss = item.get("loss")
            try:
                return float(loss)
            except (TypeError, ValueError):
                return float("inf")

        payload = min(matches, key=score)
    if not isinstance(payload, dict):
        raise ValueError(f"calibration must be a JSON object or list: {path}")
    return _calibration_from_showptycho_mapping(payload)


def _resolve_showptycho_calibration(master: pathlib.Path, args: argparse.Namespace):
    """Resolve an explicit, automatic, or disabled ShowPtycho calibration."""

    raw = str(getattr(args, "calibration", "auto") or "auto").strip()
    if raw.lower() in {"none", "off", "false", "0"}:
        return None, None
    if raw.lower() != "auto":
        path = pathlib.Path(raw).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"ShowPtycho calibration file not found: {path}")
        return _load_showptycho_calibration(path, master=master), path
    for path in _showptycho_calibration_search_paths(master):
        if not path.is_file():
            continue
        try:
            return _load_showptycho_calibration(path, master=master), path
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
    return None, None


def _read_showptycho_master_calib(master: pathlib.Path) -> dict:
    """Read optional ptychography run metadata next to a master."""

    for path in _showptycho_master_calib_paths(master):
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _first_number(mapping: dict, *keys: str) -> float | None:
    """Return the first finite numeric value found under ``keys``."""

    for key in keys:
        value = mapping.get(key)
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number == number:
            return number
    return None


def _positive_cli_number(value: object, option: str) -> float | None:
    """Return a positive finite CLI number, or raise for an invalid explicit value."""

    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{option} must be a positive finite number, got {value!r}") from exc
    if not (number == number and number > 0):
        raise ValueError(f"{option} must be a positive finite number, got {value!r}")
    return number


def _positive_optional_number(value: object) -> float | None:
    """Return ``value`` when it is positive and finite, otherwise ``None``."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and number > 0 else None


def _first_positive_number(mapping: dict, *keys: str) -> float | None:
    """Return the first positive finite numeric value found under ``keys``."""

    number = _first_number(mapping, *keys)
    return number if number is not None and number > 0 else None


def _showptycho_default_warning(label: str, value: float, unit: str, option: str) -> str:
    """Format one visible quick-start geometry default warning."""

    return (
        f"using default ptychography {label} {value:g} {unit}; pass {option} "
        "or a calibration JSON for microscope-specific geometry"
    )


def _resolve_showptycho_geometry(
    args: argparse.Namespace,
    calibration,
    meta: dict,
) -> tuple[float, float, float, float | None, list[str]]:
    """Resolve ShowPtycho geometry from CLI args, calibration, metadata, or defaults."""

    semiangle = (
        _positive_cli_number(getattr(args, "semiangle_mrad", None), "--semiangle")
        or _positive_optional_number(
            calibration.semiangle_mrad if calibration is not None else None
        )
        or _first_positive_number(meta, "semiangle_mrad", "semiangle")
    )
    scan_sampling = (
        _positive_cli_number(getattr(args, "scan_sampling_A", None), "--scan-sampling")
        or _positive_optional_number(
            calibration.scan_sampling_A if calibration is not None else None
        )
        or _first_positive_number(
            meta, "scan_sampling_A", "scan_sampling", "scan_sampling_A_per_px"
        )
    )
    voltage = (
        _positive_cli_number(getattr(args, "voltage_kv", None), "--voltage-kv")
        or _positive_optional_number(
            calibration.voltage_kV if calibration is not None else None
        )
        or _first_positive_number(meta, "voltage_kV", "voltage_kv", "voltage")
    )
    det_sampling = (
        _positive_cli_number(getattr(args, "det_sampling_mrad_px", None), "--det-sampling")
        or _first_positive_number(
            meta, "det_sampling_mrad_per_px", "det_sampling_mrad_px"
        )
    )

    warnings: list[str] = []
    if semiangle is None:
        semiangle = DEFAULT_PTYCHO_SEMIANGLE_MRAD
        warnings.append(_showptycho_default_warning(
            "semiangle", semiangle, "mrad", "--semiangle",
        ))
    if scan_sampling is None:
        scan_sampling = DEFAULT_PTYCHO_SCAN_SAMPLING_A
        warnings.append(_showptycho_default_warning(
            "scan sampling", scan_sampling, "A", "--scan-sampling",
        ))
    if voltage is None:
        voltage = DEFAULT_PTYCHO_VOLTAGE_KV
        warnings.append(_showptycho_default_warning(
            "voltage", voltage, "kV", "--voltage-kv",
        ))
    return semiangle, scan_sampling, voltage, det_sampling, warnings


def _render_showptycho_master(
    master: pathlib.Path,
    args: argparse.Namespace,
) -> pathlib.Path:
    """Build a ShowPtycho WebGPU folder from one raw ``*_master.h5`` file."""

    master = master.expanduser().resolve()
    out_dir = _showptycho_output_dir(master, args)
    if _is_showptycho_folder_export(out_dir) and not getattr(args, "force", False):
        print(f"ShowPtycho folder already exists: {out_dir}")
        print("  using existing folder; pass --force to rebuild")
        return out_dir

    det_bin = _effective_det_bin(args, default=1)
    calibration, calibration_path = _resolve_showptycho_calibration(master, args)
    meta = _read_showptycho_master_calib(master)
    semiangle, scan_sampling, voltage, det_sampling, geometry_warnings = (
        _resolve_showptycho_geometry(args, calibration, meta)
    )

    from quantem.widget import ShowPtycho
    from quantem.widget.io import load

    print(f"{master.name}: *_master.h5 -> ShowPtycho WebGPU folder")
    print(f"  output: {out_dir}")
    print(f"  detector bin: {det_bin} ({'native' if det_bin == 1 else 'downsampled'})")
    if calibration_path is not None:
        print(f"  calibration: {calibration_path}")
    else:
        print("  calibration: none; using resolved geometry and zero aberration start")
    print(
        f"  geometry: semiangle={semiangle:g} mrad, "
        f"scan_sampling={scan_sampling:g} A, voltage={voltage:g} kV"
    )
    if det_sampling is not None:
        print(f"  detector sampling: {det_sampling:g} mrad/pixel")
    for warning in geometry_warnings:
        print(f"warning: {warning}", file=sys.stderr)
    load_kwargs = {
        "det_bin": det_bin,
        "dtype": _showptycho_decode_dtype(args),
        "verbose": bool(getattr(args, "verbose", False)),
        "backend": getattr(args, "backend", "auto"),
    }
    result = load(str(master), **load_kwargs)
    data = result
    payload = getattr(result, "data", None)
    if payload is not None and not hasattr(payload, "chunks"):
        data = payload
    aberrations = dict(getattr(calibration, "aberrations", {}) or {}) if calibration else None
    rotation = float(calibration.rotation_angle_deg) if calibration else 0.0
    widget = ShowPtycho(
        data,
        semiangle=float(semiangle),
        scan_sampling=float(scan_sampling),
        det_sampling=float(det_sampling) if det_sampling is not None else None,
        voltage_kV=float(voltage),
        bf_intensity_threshold=float(args.bf_threshold),
        bf_radius=int(round(float(args.bf_radius))) if args.bf_radius is not None else None,
        aberrations=aberrations,
        rotation_angle_deg=rotation,
        calibration=calibration,
        source_file=str(master),
        drag_bf=float(args.drag_bf),
        size=int(args.size),
        fft_on=bool(args.fft),
        webgpu_preview="off",
    )
    return widget.export(
        out_dir,
        title=args.title or f"{_showptycho_source_stem(master)} ShowPtycho",
        overwrite=True,
        decode_dtype=_showptycho_decode_dtype(args),
        webgpu_source=_showptycho_webgpu_source(args),
    )


# ---------------------------------------------------------------------------
def _render_images(path: pathlib.Path, kind: str, args: argparse.Namespace) -> pathlib.Path:
    """Render one image (Show2D), a same-size folder (Show3D scrub), or a mixed
    folder (Show2D gallery), and write the HTML. Returns the written path."""
    from quantem.widget import Show2D, Show3D
    from quantem.widget.io import read_image_stack
    title = args.title
    if kind == "image":
        print(f"{path.name}: 1 image -> Show2D")
        widget = Show2D(_load_2d(path), title=title or path.stem)
        out = _out_path(args.out, path, suffix="show2d")
        widget.export_html(out)
        return out
    # Folder of images: try to stack into a Show3D scrub; differently-sized frames
    # cannot stack (np.stack raises) so fall back to a Show2D gallery.
    if args.widget != "2d":
        try:
            stack = read_image_stack(path, progress=args.verbose)
            widget = Show3D(stack, title=title or path.name)
            out = _out_path(args.out, path, suffix="show3d", from_dir=True)
            widget.export_html(out, quantized=args.quantized)
            return out
        except ValueError:
            if args.verbose:
                print("frames differ in size; rendering a Show2D gallery instead")
    files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    arrays = [_load_2d(p) for p in files]
    widget = Show2D(arrays, title=title or path.name)
    out = _out_path(args.out, path, suffix="gallery", from_dir=True)
    widget.export_html(out)
    return out


def _load_2d(path: pathlib.Path):
    """Decode one image file to a 2D float32 array. ``.npy`` / ``.emd`` / Gatan go
    through ``read_image`` (calibration-aware); raster formats use the same frame
    decoder ``read_image_stack`` uses, since this repo's ``read_image`` only knows
    ``.emd`` / ``.npy``."""
    from quantem.widget.io import read_image
    from quantem.widget.io.image import _read_frame
    if path.suffix.lower() in (".npy", ".emd", ".dm3", ".dm4"):
        return read_image(path).array
    return _read_frame(path)


def _render_4dstem_notebook(masters: list[str], label: str, args: argparse.Namespace) -> pathlib.Path:
    """Write a live Jupyter notebook that loads the 4D-STEM master(s) and opens a
    kernel-backed ``Show4DSTEM`` (full real-time WebGPU, no baked HTML). One master
    loads on its own; many load stacked into a 5D viewer with a dataset slider (the
    multi-tilt case). The notebook is the editable, real-use surface; ``--html`` is
    the share artifact."""
    import json
    print(f"{len(masters)} master(s), bin {args.det_bin} -> Show4DSTEM (live notebook)")
    arg = repr(masters[0]) if len(masters) == 1 else repr(masters)
    source = (
        "from quantem.widget import load, Show4DSTEM\n"
        f"Show4DSTEM(load({arg}, det_bin={args.det_bin}))"
    )
    nb = {
        "cells": [
            {"cell_type": "markdown", "id": "title", "metadata": {}, "source": [f"# {label}\n", f"\n{len(masters)} master(s), detector bin {args.det_bin}."]},
            {"cell_type": "code", "id": "viewer", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = _out_dir(args.out) / f"{label}.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    return out


def _python_page_budget(value: str | int | None) -> str:
    """Return a source literal for a Show4DSTEM page budget CLI value."""
    if value is None:
        return "None"
    if isinstance(value, int):
        return str(value)
    text = str(value).strip()
    if text.lower() in {"none", "off", "false", "no"}:
        return "None"
    if text.isdigit():
        return str(int(text))
    return repr(text)


def _python_gpus(value: str | None) -> str:
    """Return a source literal for comma-separated CUDA GPU ids."""
    if value is None or not str(value).strip():
        return "None"
    try:
        ids = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError("--gpus must be a comma-separated list of integer ids, e.g. 0 or 0,1") from exc
    if not ids:
        return "None"
    return repr(ids)


def _render_data_transfer_show4dstem_notebook(
    manifest: pathlib.Path,
    label: str,
    masters,
    args: argparse.Namespace,
) -> pathlib.Path:
    """Write a live Show4DSTEM notebook from ready target masters in a manifest."""
    import json

    devices = _python_gpus(args.gpus)
    page_budget = _python_page_budget(args.page_budget)
    print(
        f"{len(masters)} transferred target master(s), bin {args.det_bin}, dtype {args.dtype}, "
        f"devices {devices} -> Show4DSTEM (live notebook)"
    )
    source = (
        "from pathlib import Path\n"
        "from quantem.widget import Show4DSTEM, load\n"
        "from quantem.widget.io import data_transfer_load_warnings, read_data_transfer_manifest, target_masters\n"
        "\n"
        f"manifest = Path({str(manifest)!r})\n"
        "plan = read_data_transfer_manifest(manifest)\n"
        "masters = [str(path) for path in target_masters(plan)]\n"
        f"devices = {devices}\n"
        "warnings = data_transfer_load_warnings(plan, devices=devices)\n"
        "for warning in warnings:\n"
        "    print(f\"warning: {warning}\")\n"
        "print(f\"ready masters: {len(masters)}\")\n"
        "print(\"devices:\", devices)\n"
        "print(\"dtype:\", " + repr(args.dtype) + ", \"det_bin:\", " + str(int(args.det_bin)) + ")\n"
        "data = load(\n"
        "    masters,\n"
        f"    det_bin={int(args.det_bin)},\n"
        f"    dtype={args.dtype!r},\n"
        "    devices=devices,\n"
        "    verbose=True,\n"
        ")\n"
        "Show4DSTEM(\n"
        "    data,\n"
        f"    page_budget={page_budget},\n"
        "    page_device=devices,\n"
        "    verbose=True,\n"
        ")\n"
    )
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "title",
                "metadata": {},
                "source": [
                    f"# {label} transferred Show4DSTEM\n",
                    f"\nManifest: `{manifest}`\n",
                    f"\nReady target masters: {len(masters)}. Detector bin {args.det_bin}; "
                    f"dtype `{args.dtype}`; devices `{devices}`; page budget `{args.page_budget}`.",
                ],
            },
            {
                "cell_type": "code",
                "id": "transferred-viewer",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source.splitlines(keepends=True),
            },
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = _out_dir(args.out) / f"{label}_transferred_show4dstem.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    return out


def _render_4dstem_watch_notebook(folder: pathlib.Path, label: str, args: argparse.Namespace) -> pathlib.Path:
    """Write a live ShowFolder-watched notebook for a 4D-STEM acquisition folder."""
    import json

    print(
        f"{folder.name}: watched folder, bin {args.det_bin}, page_budget {args.page_budget} "
        "-> ShowFolder + lazy Show4DSTEM"
    )
    gpus = _python_gpus(args.gpus)
    page_budget = _python_page_budget(args.page_budget)
    scan_size = "None" if args.scan_size is None else str(int(args.scan_size))
    source = (
        "from quantem.widget import ShowFolder\n"
        "\n"
        f"folder = ShowFolder({str(folder)!r}, thumb=256, group_by='none')\n"
        "folder.browser.attach_selection_panel()\n"
        "folder.browser.open_show4dstem(\n"
        f"    gpus={gpus},\n"
        f"    page_budget={page_budget},\n"
        f"    det_bin={int(args.det_bin)},\n"
        f"    dtype={args.dtype!r},\n"
        f"    scan_size={scan_size},\n"
        ")\n"
        f"folder.watch(interval={float(args.watch_interval)!r})\n"
        "folder\n"
    )
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "title",
                "metadata": {},
                "source": [
                    f"# {label} live Show4DSTEM\n",
                    f"\nWatched folder: `{folder}`\n",
                    f"\nDetector bin {args.det_bin}; page budget `{args.page_budget}`; "
                    f"watch interval {args.watch_interval:g}s.",
                ],
            },
            {
                "cell_type": "code",
                "id": "live-viewer",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source.splitlines(keepends=True),
            },
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = _out_dir(args.out) / f"{label}_live.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    return out


def _render_image_watch_notebook(
    folder: pathlib.Path,
    label: str,
    args: argparse.Namespace,
    *,
    widget: str,
) -> pathlib.Path:
    """Write a live ShowFolder-watched notebook for image folder previews."""
    import json

    method = "open_show3d" if widget == "show3d" else "open_show2d"
    title = "Show3D" if widget == "show3d" else "Show2D"
    print(f"{folder.name}: watched folder -> ShowFolder + live all-image {title}")
    source = (
        "from quantem.widget import ShowFolder\n"
        "\n"
        f"folder = ShowFolder({str(folder)!r}, thumb=256, group_by='none')\n"
        "folder.browser.attach_selection_panel()\n"
        f"folder.browser.{method}(all_images=True)\n"
        f"folder.watch(interval={float(args.watch_interval)!r})\n"
        "folder\n"
    )
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "title",
                "metadata": {},
                "source": [
                    f"# {label} live {title}\n",
                    f"\nWatched folder: `{folder}`\n",
                    f"\nNew readable image files append on the next poll; "
                    f"watch interval {args.watch_interval:g}s.",
                ],
            },
            {
                "cell_type": "code",
                "id": "live-viewer",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source.splitlines(keepends=True),
            },
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = _out_dir(args.out) / f"{label}_{widget}_live.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    return out


def _render_gallery(files: list[pathlib.Path], label: str, args: argparse.Namespace) -> pathlib.Path:
    """Render several explicit image files as one Show2D gallery HTML."""
    from quantem.widget import Show2D
    print(f"{len(files)} images -> Show2D gallery")
    widget = Show2D([_load_2d(p) for p in files], title=args.title or label)
    out = _out_dir(args.out) / f"{label}.html"
    widget.export_html(out)
    return out


def _launch_notebook(notebook: pathlib.Path, *, no_open: bool) -> None:
    """Open the notebook for the user. Locally (a Mac or any box with a display) start
    ``jupyter lab`` on it, which opens the browser. On a headless/remote box a browser
    cannot be reached, so print the path plus the ``mj jupyter`` hint instead (never
    start a server the user cannot see)."""
    import shutil
    import subprocess
    headless = sys.platform != "darwin" and not os.environ.get("DISPLAY")
    if no_open or headless or shutil.which("jupyter") is None:
        print(f"wrote {notebook}")
        if headless:
            print(f"  open it from your Mac:  mj jupyter cuda-env quantem   (then open {notebook.name})")
        elif shutil.which("jupyter") is None:
            print("  jupyter not found; install it or open the notebook in your editor")
        return
    print(f"launching jupyter lab on {notebook}")
    subprocess.Popen(["jupyter", "lab", str(notebook)],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _master_to_binned_numpy(master: str, det_bin: int):
    """Load one master with detector binning and return a mean-binned 4D numpy array
    ``(scan_row, scan_col, det_row, det_col)``. Binning happens at LOAD time (so the
    full 19 GB stack never materializes - fits a laptop), and since the loader
    integer-SUMS over det_bin^2 we divide by that to get the MEAN, which keeps values
    in the raw range so the uint8 pack never clips. Works on CUDA / MPS (zero-copy
    ChunkedFrames, materialized via its chunks) / CPU."""
    import numpy as np
    import torch
    from quantem.widget import load
    result = load(master, det_bin=det_bin)
    data = result.data if hasattr(result, "data") else result
    meta = getattr(result, "metadata", {}) or {}
    if hasattr(data, "chunks"):
        arr = np.concatenate([np.asarray(chunk) for chunk in data.chunks], axis=0)
    elif hasattr(data, "get"):
        arr = data.get()
    elif isinstance(data, torch.Tensor):
        arr = data.detach().to("cpu").numpy()
    else:
        arr = np.asarray(data)
    if arr.ndim == 3:
        scan = meta.get("scan_shape")
        rows, cols = scan if scan else (int(round(arr.shape[0] ** 0.5)),) * 2
        arr = arr.reshape(rows, cols, arr.shape[-2], arr.shape[-1])
    if det_bin > 1:
        arr = np.round(arr.astype(np.float32) / (det_bin * det_bin))  # loader summed -> mean
    return np.ascontiguousarray(arr.astype(np.float32))


def _render_4dstem(masters: list[str], label: str, args: argparse.Namespace) -> list[pathlib.Path]:
    """Render 4D-STEM master(s) as offline WebGPU Show4DSTEM HTML.

    Each master is loaded with detector binning (``--bin``, default 8) so the stack
    fits a laptop and packs inline as a self-contained file. ``--combined`` instead
    stacks every master into one 5D viewer (a bslz4 companion folder + a local
    serve, since file:// cannot fetch the companion)."""
    import numpy as np
    from quantem.widget import Show4DSTEM
    out_dir = _out_dir(args.out)
    if args.combined and len(masters) > 1:
        # Stack the masters into one 5D numpy array and pass THAT to the viewer. A
        # 5D array routes to the universal Show4DSTEM (which has the offline
        # multi-volume WebGPU frame-flip), not the MacBook live-Metal viewer (whose
        # offline export can't switch volumes kernel-lessly).
        volumes = [_master_to_binned_numpy(m, args.det_bin) for m in masters]
        stack = np.stack(volumes, axis=0)
        data_url = out_dir / "widget-data"
        widget = Show4DSTEM(
            stack, backend="web", offline_codec="bslz4", data_url=str(data_url),
            frame_dim_label="Dataset",
            frame_labels=[pathlib.Path(m).stem.replace("_master", "") for m in masters],
        )
        out = out_dir / f"{label}_combined.html"
        widget.export_html(str(out), title=args.title)
        return [out]
    outputs = []
    iterator = masters
    if args.verbose:
        try:
            from tqdm import tqdm
            iterator = tqdm(masters, desc="export")
        except ImportError:
            pass
    for master in iterator:
        stem = pathlib.Path(master).stem.replace("_master", "")
        try:
            # Mean-bin at load (memory-safe: the full 19 GB stack never materializes)
            # so uint8 never clips the bright field. Data is already binned, so the
            # export does no further binning.
            arr = _master_to_binned_numpy(master, args.det_bin)
            widget = Show4DSTEM(arr, backend="web")
            out = out_dir / f"{stem}.html"
            widget.export_html(str(out), title=args.title or stem)
            outputs.append(out)
        except (RuntimeError, ValueError, OSError, MemoryError) as err:
            print(f"quantem: skipped {stem}: {err}", file=sys.stderr)
    if not outputs:
        raise ValueError("every master failed to export (see messages above)")
    return outputs


# ---------------------------------------------------------------------------
def _out_path(out: str | None, src: pathlib.Path, *, suffix: str, from_dir: bool = False) -> pathlib.Path:
    """Resolve a single output HTML path from ``--out`` (file or dir) or default to
    ``<source-stem>_<suffix>.html`` beside the input."""
    base = (src.name if from_dir else src.stem)
    default_name = f"{base}_{suffix}.html"
    if out is None:
        return _default_out_dir() / default_name
    target = pathlib.Path(out).expanduser()
    if target.is_dir() or out.endswith("/"):
        target.mkdir(parents=True, exist_ok=True)
        return target / default_name
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _out_dir(out: str | None) -> pathlib.Path:
    """Resolve the output directory (``--out`` or the default ``~/Downloads``)."""
    target = pathlib.Path(out).expanduser() if out else _default_out_dir()
    target.mkdir(parents=True, exist_ok=True)
    return target


def _default_out_dir() -> pathlib.Path:
    """Default save location: the user's ``~/Downloads`` (where a shareable artifact
    is expected and always writable), falling back to the current directory when no
    Downloads folder exists (servers, CI)."""
    downloads = pathlib.Path.home() / "Downloads"
    return downloads if downloads.is_dir() else pathlib.Path.cwd()


class _RangeRequestHandler(http.server.BaseHTTPRequestHandler):
    """Static file handler with single byte-range support for folder exports."""

    root: pathlib.Path

    def log_message(self, format: str, *args) -> None:  # noqa: A003 - stdlib API name.
        if os.environ.get("QUANTEM_CLI_HTTP_LOG"):
            super().log_message(format, *args)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_common_headers()
        self.end_headers()

    def do_HEAD(self) -> None:
        self._serve(send_body=False)

    def do_GET(self) -> None:
        self._serve(send_body=True)

    def _serve(self, *, send_body: bool) -> None:
        path = self._resolve_path()
        if path is None:
            self.send_error(404, "file not found")
            return
        if path.is_dir():
            path = path / "index.html"
        if not path.is_file():
            self.send_error(404, "file not found")
            return

        size = path.stat().st_size
        start, end, partial = 0, size - 1, False
        range_header = self.headers.get("Range")
        if range_header:
            parsed = _parse_http_range(range_header, size)
            if parsed is None:
                self.send_response(416)
                self._send_common_headers()
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            start, end = parsed
            partial = True

        content_length = max(0, end - start + 1)
        self.send_response(206 if partial else 200)
        self._send_common_headers()
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(content_length))
        self.send_header("Last-Modified", email.utils.formatdate(path.stat().st_mtime, usegmt=True))
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()

        if not send_body:
            return
        with path.open("rb") as handle:
            self._send_file_body(handle, start=start, length=content_length)

    def _send_common_headers(self) -> None:
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Range, Content-Type")

    def _send_file_body(self, handle, *, start: int, length: int) -> None:
        """Send a file range with zero-copy when the local socket supports it."""
        if length <= 0:
            return
        if not os.environ.get("QUANTEM_DISABLE_HTTP_SENDFILE") and hasattr(self.connection, "sendfile"):
            try:
                self.connection.sendfile(handle, offset=start, count=length)
                return
            except BrokenPipeError:
                return
            except (OSError, ValueError):
                pass
        handle.seek(start)
        remaining = length
        chunk_bytes = _range_fallback_chunk_bytes()
        while remaining > 0:
            chunk = handle.read(min(chunk_bytes, remaining))
            if not chunk:
                break
            try:
                self.wfile.write(chunk)
            except BrokenPipeError:
                break
            remaining -= len(chunk)

    def _resolve_path(self) -> pathlib.Path | None:
        parsed = urllib.parse.urlsplit(self.path)
        raw_path = urllib.parse.unquote(parsed.path)
        norm = posixpath.normpath(raw_path)
        rel = pathlib.Path(norm.lstrip("/"))
        root = self.root.resolve()
        candidate = root / rel
        try:
            candidate.relative_to(root)
        except ValueError:
            return None
        return candidate


def _range_fallback_chunk_bytes() -> int:
    """Return the fallback HTTP chunk size for large local folder exports."""
    raw = os.environ.get("QUANTEM_HTTP_RANGE_CHUNK_MB", "")
    if raw:
        try:
            mb = int(raw)
        except ValueError:
            mb = 16
        return max(1, mb) * 1024 * 1024
    return _RANGE_FALLBACK_CHUNK_BYTES


def _parse_http_range(value: str, size: int) -> tuple[int, int] | None:
    """Parse a single HTTP byte range header."""
    match = _RANGE_RE.fullmatch(value.strip())
    if not match or size < 0:
        return None
    start_text, end_text = match.groups()
    if start_text == "" and end_text == "":
        return None
    if start_text == "":
        suffix = int(end_text)
        if suffix <= 0:
            return None
        start = max(0, size - suffix)
        end = size - 1
    else:
        start = int(start_text)
        end = int(end_text) if end_text else size - 1
    if start >= size or end < start:
        return None
    return start, min(end, size - 1)


def _showptycho_manifest(folder: pathlib.Path) -> dict:
    """Read a ShowPtycho folder manifest after validation."""
    return json.loads((folder / "manifest.json").read_text(encoding="utf-8"))


def _host_for_url(bind: str) -> str:
    return "127.0.0.1" if bind in {"", "0.0.0.0", "::"} else bind


def _serve_showptycho_folder(
    folder: pathlib.Path,
    *,
    bind: str,
    port: int | None,
    no_open: bool,
) -> None:
    """Serve a ShowPtycho WebGPU folder export and open it for the user."""
    folder = _showptycho_folder(folder)
    manifest = _showptycho_manifest(folder)
    arrays = manifest.get("arrays", {})
    g_bf = arrays.get("g_bf", {})
    g_path = folder / str(g_bf.get("path", "g_bf.c64"))
    source = manifest.get("source", {})
    shape = g_bf.get("shape", [])
    print(f"ShowPtycho folder: {folder}")
    if source.get("kind") == "hdf5":
        data_files = source.get("data_files") or []
        link_mode = ", ".join(source.get("link_mode") or []) or "linked"
        preferred = source.get("preferred_browser_source") or "compressed_hdf5"
        print(
            "  source: compressed HDF5 "
            f"{source.get('master', 'source master')} + {len(data_files)} data file(s) "
            f"({link_mode}); no persistent BF-G cache"
        )
        print(f"  browser source: {preferred}")
    elif len(shape) == 3 and g_path.exists():
        print(
            "  BF payload: "
            f"{shape[0]} BF terms x {shape[1]} x {shape[2]} phase grid, "
            f"{_fmt_bytes(g_path.stat().st_size)}"
        )
    elif g_path.exists():
        print(f"  BF payload: {_fmt_bytes(g_path.stat().st_size)}")
    else:
        print(f"  warning: BF payload missing: {g_path}", file=sys.stderr)
    if no_open:
        print("  ready: run without --no-open to serve and open this folder.")
        return

    handler = type("QuantemShowPtychoRangeHandler", (_RangeRequestHandler,), {"root": folder})
    server = http.server.ThreadingHTTPServer((bind, port or 0), handler)
    actual_port = server.server_address[1]
    url = f"http://{_host_for_url(bind)}:{actual_port}/index.html"
    print(f"  open: {url}")
    print("  serving folder export; press Ctrl-C to stop")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    headless = sys.platform != "darwin" and not os.environ.get("DISPLAY")
    if not headless:
        webbrowser.open(url)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.shutdown()
    finally:
        server.server_close()


def _open_html(path: pathlib.Path, *, serve: bool, no_open: bool) -> None:
    """Open the HTML for the user: a self-contained file via ``file://``, or behind
    a local HTTP server when serving (required for bslz4 companions, and the only
    way a remote/SSH user can tunnel in). On a headless box, just print the path."""
    headless = sys.platform != "darwin" and not os.environ.get("DISPLAY")
    if no_open:
        print(f"wrote {path}")
        return
    if serve:
        directory = str(path.parent)
        handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=directory, **k)
        httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{port}/{path.name}"
        print(f"serving {url}  (Ctrl-C to stop)")
        if not headless:
            webbrowser.open(url)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            httpd.shutdown()
        return
    if headless:
        print(f"wrote {path}  (open it in a browser)")
        return
    webbrowser.open(path.as_uri())
    print(f"opened {path}")


if __name__ == "__main__":
    raise SystemExit(main())
