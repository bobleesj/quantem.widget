/**
 * Shared scale bar utilities for all canvas-based widgets.
 * Provides HiDPI-aware scale bar rendering with automatic unit conversion.
 */

/** Round a physical value to a "nice" number (1, 2, 5, 10, 20, 50, ...) */
export function roundToNiceValue(value: number): number {
  if (value <= 0) return 1;
  const magnitude = Math.pow(10, Math.floor(Math.log10(value)));
  const normalized = value / magnitude;
  if (normalized < 1.5) return magnitude;
  if (normalized < 3.5) return 2 * magnitude;
  if (normalized < 7.5) return 5 * magnitude;
  return 10 * magnitude;
}

/** Format scale bar label with appropriate unit and auto-conversion (Å→nm, mrad→rad) */
export function formatScaleLabel(value: number, unit: "Å" | "mrad" | "px"): string {
  const nice = roundToNiceValue(value);
  if (unit === "Å") {
    if (nice >= 10) return `${Math.round(nice / 10)} nm`;
    return nice >= 1 ? `${Math.round(nice)} Å` : `${nice.toFixed(2)} Å`;
  }
  if (unit === "px") {
    return nice >= 1 ? `${Math.round(nice)} px` : `${nice.toFixed(1)} px`;
  }
  if (nice >= 1000) return `${Math.round(nice / 1000)} rad`;
  return nice >= 1 ? `${Math.round(nice)} mrad` : `${nice.toFixed(2)} mrad`;
}

/**
 * Draw scale bar and zoom indicator on a high-DPI UI canvas.
 * Renders crisp text/lines independent of the image resolution.
 */
export function drawScaleBarHiDPI(
  canvas: HTMLCanvasElement,
  dpr: number,
  zoom: number,
  pixelSize: number,
  unit: "Å" | "mrad" | "px",
  imageWidth: number,
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.save();
  ctx.scale(dpr, dpr);

  const cssWidth = canvas.width / dpr;
  const cssHeight = canvas.height / dpr;
  const scaleX = cssWidth / imageWidth;
  const effectiveZoom = zoom * scaleX;

  const targetBarPx = 60;
  const barThickness = 5;
  const fontSize = 16;
  const margin = 12;

  const targetPhysical = (targetBarPx / effectiveZoom) * pixelSize;
  const nicePhysical = roundToNiceValue(targetPhysical);
  const barPx = (nicePhysical / pixelSize) * effectiveZoom;

  const barY = cssHeight - margin;
  const barX = cssWidth - barPx - margin;

  ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
  ctx.shadowBlur = 2;
  ctx.shadowOffsetX = 1;
  ctx.shadowOffsetY = 1;

  ctx.fillStyle = "white";
  ctx.fillRect(barX, barY, barPx, barThickness);

  const label = formatScaleLabel(nicePhysical, unit);
  ctx.font = `${fontSize}px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif`;
  ctx.fillStyle = "white";
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  ctx.fillText(label, barX + barPx / 2, barY - 4);

  ctx.textAlign = "left";
  ctx.textBaseline = "bottom";
  ctx.fillText(`${zoom.toFixed(1)}×`, margin, cssHeight - margin + barThickness);

  ctx.restore();
}
