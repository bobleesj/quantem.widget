"""
Batch and CLI runner for Show4DSTEM programmatic exports.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Any

import numpy as np

from quantem.widget.show4dstem import Show4DSTEM


def _parse_pair(text: str | None, name: str) -> tuple[float, float] | None:
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 2:
        raise ValueError(f"{name} must be in 'row,col' format")
    return float(parts[0]), float(parts[1])


def _parse_int_pair(text: str | None, name: str) -> tuple[int, int] | None:
    pair = _parse_pair(text, name)
    if pair is None:
        return None
    return int(pair[0]), int(pair[1])


def _parse_frame_indices(text: str | None) -> list[int] | None:
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    return [int(p.strip()) for p in raw.split(",") if p.strip()]


def _parse_frame_range(text: str | None) -> tuple[int, int] | None:
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(":")]
    if len(parts) != 2:
        raise ValueError("frame-range must be in 'start:end' format")
    return int(parts[0]), int(parts[1])


def _parse_adaptive_weights(text: str | None) -> dict[str, float] | None:
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 3:
        raise ValueError(
            "adaptive-weights must be 'vi_gradient,vi_local_std,dp_variance'"
        )
    return {
        "vi_gradient": float(parts[0]),
        "vi_local_std": float(parts[1]),
        "dp_variance": float(parts[2]),
    }


def _load_path_points(path: str | None) -> list[tuple[int, int]] | None:
    if path is None:
        return None
    payload = json.loads(pathlib.Path(path).read_text())
    points_raw = payload.get("points") if isinstance(payload, dict) and "points" in payload else payload
    if not isinstance(points_raw, list):
        raise ValueError("path JSON must be a list of [row, col] points or {'points': [...]} ")
    points: list[tuple[int, int]] = []
    for point in points_raw:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("Each path point must be a [row, col] pair")
        points.append((int(point[0]), int(point[1])))
    return points


def _load_array(path: pathlib.Path, npz_key: str | None) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
        return np.asarray(arr)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            key = npz_key
            if key is None:
                keys = list(data.keys())
                if not keys:
                    raise ValueError(f"NPZ file has no arrays: {path}")
                key = keys[0]
            if key not in data:
                raise ValueError(f"NPZ key '{key}' not found in {path}")
            return np.asarray(data[key])
    raise ValueError(f"Unsupported input suffix '{suffix}' for {path}")


def _iter_input_files(input_path: pathlib.Path, pattern: str, recursive: bool) -> list[pathlib.Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise ValueError(f"Input path does not exist: {input_path}")
    if recursive:
        files = sorted([p for p in input_path.rglob(pattern) if p.is_file()])
    else:
        files = sorted([p for p in input_path.glob(pattern) if p.is_file()])
    return files


def _prompt_with_default(label: str, default: str) -> str:
    value = input(f"{label} [{default}]: ").strip()
    return value if value else default


def _maybe_interactive_config(args: argparse.Namespace) -> None:
    if not bool(args.interactive):
        return
    args.mode = _prompt_with_default("mode (single/path/raster/frames)", str(args.mode))
    args.view = _prompt_with_default("view (diffraction/virtual/fft/all)", str(args.view))
    args.format = _prompt_with_default("format (png/pdf)", str(args.format))
    args.include_overlays = _prompt_with_default(
        "include_overlays (true/false)",
        "true" if bool(args.include_overlays) else "false",
    ).lower() in {"true", "1", "yes", "y"}
    args.include_scalebar = _prompt_with_default(
        "include_scalebar (true/false)",
        "true" if bool(args.include_scalebar) else "false",
    ).lower() in {"true", "1", "yes", "y"}

    if str(args.mode).strip().lower() == "path" and not args.path_json:
        args.path_json = _prompt_with_default("path_json file", "")
        if not str(args.path_json).strip():
            args.path_json = None

    if str(args.mode).strip().lower() == "frames" and not args.frame_range and not args.frame_indices:
        args.frame_range = _prompt_with_default("frame_range start:end (blank=all)", "")
        if not str(args.frame_range).strip():
            args.frame_range = None


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch runner for Show4DSTEM programmatic exports."
    )
    parser.add_argument("--input", required=True, help="Input file or folder")
    parser.add_argument("--output-dir", required=True, help="Output folder")
    parser.add_argument("--pattern", default="*.npy", help="Glob pattern for folder mode")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    parser.add_argument("--npz-key", default=None, help="NPZ array key (for .npz input)")

    parser.add_argument(
        "--mode",
        default="single",
        choices=["single", "path", "raster", "frames", "adaptive"],
        help="Export mode",
    )
    parser.add_argument(
        "--view",
        default="all",
        choices=["diffraction", "virtual", "fft", "all"],
        help="Export view",
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=["png", "pdf"],
        help="Export image format",
    )
    parser.add_argument("--position", default=None, help="Fixed scan position as 'row,col'")
    parser.add_argument("--frame-idx", type=int, default=None, help="Fixed frame index for path/raster")
    parser.add_argument("--frame-indices", default=None, help="Frame list for frames mode, e.g. 0,5,10")
    parser.add_argument("--frame-range", default=None, help="Frame range for frames mode, e.g. 0:20")
    parser.add_argument("--path-json", default=None, help="Path points JSON for path mode")
    parser.add_argument("--raster-step", type=int, default=1, help="Raster step size")
    parser.add_argument("--raster-bidirectional", action="store_true", help="Use snake raster")
    parser.add_argument("--adaptive-coarse-step", type=int, default=4, help="Adaptive planner coarse step")
    parser.add_argument("--adaptive-target-fraction", type=float, default=0.25, help="Adaptive planner target scan fraction")
    parser.add_argument("--adaptive-min-spacing", type=int, default=2, help="Adaptive planner min spacing")
    parser.add_argument("--adaptive-dose-lambda", type=float, default=0.25, help="Adaptive planner coarse dose penalty")
    parser.add_argument("--adaptive-local-window", type=int, default=5, help="Adaptive planner local std window")
    parser.add_argument(
        "--adaptive-weights",
        default=None,
        help="Adaptive utility weights as vi_gradient,vi_local_std,dp_variance (comma-separated)",
    )
    parser.add_argument("--filename-prefix", default=None, help="Output filename prefix")
    parser.add_argument("--manifest-name", default="save_sequence_manifest.json", help="Sequence manifest filename")

    parser.add_argument("--include-overlays", action="store_true", default=True, help="Include ROI/crosshair overlays")
    parser.add_argument("--no-include-overlays", action="store_false", dest="include_overlays")
    parser.add_argument("--include-scalebar", action="store_true", default=True, help="Include scale bars")
    parser.add_argument("--no-include-scalebar", action="store_false", dest="include_scalebar")
    parser.add_argument("--include-metadata", action="store_true", default=True, help="Write sidecar metadata")
    parser.add_argument("--no-include-metadata", action="store_false", dest="include_metadata")
    parser.add_argument("--dpi", type=int, default=300, help="Export DPI")

    parser.add_argument("--pixel-size", type=float, default=None, help="Real-space calibration (Å/px)")
    parser.add_argument("--k-pixel-size", type=float, default=None, help="Reciprocal calibration (mrad/px)")
    parser.add_argument("--center", default=None, help="Detector center as 'row,col'")
    parser.add_argument("--bf-radius", type=float, default=None, help="BF radius in detector pixels")

    parser.add_argument("--report-name", default="reproducibility_report.json", help="Per-run reproducibility report name")
    parser.add_argument("--batch-manifest", default="show4dstem_batch_manifest.jsonl", help="Batch manifest filename")
    parser.add_argument("--interactive", action="store_true", help="Interactive prompt for key options")
    parser.add_argument("--quiet-progress", action="store_true", help="Suppress per-file status output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _maybe_interactive_config(args)

    input_path = pathlib.Path(args.input)
    output_root = pathlib.Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    position = _parse_int_pair(args.position, "position")
    center_pair = _parse_pair(args.center, "center")
    center = (float(center_pair[0]), float(center_pair[1])) if center_pair is not None else None
    frame_indices = _parse_frame_indices(args.frame_indices)
    frame_range = _parse_frame_range(args.frame_range)
    path_points = _load_path_points(args.path_json)
    adaptive_weights = _parse_adaptive_weights(args.adaptive_weights)

    files = _iter_input_files(input_path, pattern=str(args.pattern), recursive=bool(args.recursive))
    if not files:
        raise ValueError(f"No files found for input={input_path} pattern={args.pattern}")

    batch_manifest_path = output_root / str(args.batch_manifest)
    with batch_manifest_path.open("w"):
        pass

    n_ok = 0
    n_err = 0
    start_time = time.perf_counter()

    for idx, file_path in enumerate(files, start=1):
        t0 = time.perf_counter()
        status = "ok"
        error_text = ""
        per_file_output = output_root / file_path.stem if len(files) > 1 else output_root
        per_file_output.mkdir(parents=True, exist_ok=True)
        report_path: pathlib.Path | None = None
        image_path: pathlib.Path | None = None
        sequence_manifest_path: pathlib.Path | None = None
        adaptive_summary: dict[str, Any] | None = None

        try:
            array = _load_array(file_path, npz_key=args.npz_key)
            widget = Show4DSTEM(
                array,
                pixel_size=args.pixel_size,
                k_pixel_size=args.k_pixel_size,
                center=center,
                bf_radius=args.bf_radius,
            )
            mode_key = str(args.mode).strip().lower()

            if mode_key == "single":
                prefix = str(args.filename_prefix).strip() if args.filename_prefix else file_path.stem
                out_name = f"{prefix}.{args.format}"
                image_path = per_file_output / out_name
                widget.save_image(
                    image_path,
                    view=str(args.view),
                    position=position,
                    frame_idx=args.frame_idx,
                    format=str(args.format),
                    include_metadata=bool(args.include_metadata),
                    include_overlays=bool(args.include_overlays),
                    include_scalebar=bool(args.include_scalebar),
                    dpi=int(args.dpi),
                )
            elif mode_key == "adaptive":
                plan = widget.suggest_adaptive_path(
                    coarse_step=int(args.adaptive_coarse_step),
                    target_fraction=float(args.adaptive_target_fraction),
                    min_spacing=int(args.adaptive_min_spacing),
                    local_window=int(args.adaptive_local_window),
                    dose_lambda=float(args.adaptive_dose_lambda),
                    weights=adaptive_weights,
                    update_widget_path=False,
                )
                adaptive_summary = {
                    "target_count": int(plan["target_count"]),
                    "coarse_count": int(plan["coarse_count"]),
                    "dense_count": int(plan["dense_count"]),
                    "path_count": int(plan["path_count"]),
                    "selected_fraction": float(plan["selected_fraction"]),
                }
                sequence_manifest_path = widget.save_sequence(
                    output_dir=per_file_output,
                    mode="path",
                    view=str(args.view),
                    format=str(args.format),
                    include_metadata=bool(args.include_metadata),
                    include_overlays=bool(args.include_overlays),
                    include_scalebar=bool(args.include_scalebar),
                    frame_idx=args.frame_idx,
                    path_points=plan["path_points"],
                    filename_prefix=args.filename_prefix,
                    manifest_name=str(args.manifest_name),
                    dpi=int(args.dpi),
                )
            else:
                if mode_key == "path" and path_points is None:
                    raise ValueError("Path mode requires --path-json or --interactive path entry.")
                sequence_manifest_path = widget.save_sequence(
                    output_dir=per_file_output,
                    mode=mode_key,
                    view=str(args.view),
                    format=str(args.format),
                    include_metadata=bool(args.include_metadata),
                    include_overlays=bool(args.include_overlays),
                    include_scalebar=bool(args.include_scalebar),
                    frame_idx=args.frame_idx,
                    position=position,
                    path_points=path_points,
                    raster_step=int(args.raster_step),
                    raster_bidirectional=bool(args.raster_bidirectional),
                    frame_indices=frame_indices,
                    frame_range=frame_range,
                    filename_prefix=args.filename_prefix,
                    manifest_name=str(args.manifest_name),
                    dpi=int(args.dpi),
                )

            report_path = widget.save_reproducibility_report(
                per_file_output / str(args.report_name)
            )
            n_ok += 1
        except Exception as exc:
            status = "error"
            error_text = str(exc)
            n_err += 1

        elapsed = float(time.perf_counter() - t0)
        row: dict[str, Any] = {
            "index": int(idx),
            "total": int(len(files)),
            "status": status,
            "input": str(file_path),
            "output_dir": str(per_file_output),
            "mode": str(args.mode),
            "view": str(args.view),
            "format": str(args.format),
            "elapsed_sec": elapsed,
        }
        if image_path is not None:
            row["image_path"] = str(image_path)
        if sequence_manifest_path is not None:
            row["sequence_manifest_path"] = str(sequence_manifest_path)
        if report_path is not None:
            row["report_path"] = str(report_path)
        if adaptive_summary is not None:
            row["adaptive"] = adaptive_summary
        if error_text:
            row["error"] = error_text

        with batch_manifest_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

        if not bool(args.quiet_progress):
            name = file_path.name
            print(
                f"[{idx}/{len(files)}] {status:<5} {name} "
                f"t={elapsed:.2f}s mode={args.mode} view={args.view}"
            )
            if error_text:
                print(f"  error: {error_text}")

    total_elapsed = float(time.perf_counter() - start_time)
    print(
        "Processed: "
        f"ok={n_ok}, error={n_err}, total={len(files)}, "
        f"elapsed={total_elapsed:.2f}s, manifest={batch_manifest_path}"
    )
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
