// A view's drawn size, "small", "medium" or "large" (the model's size trait), and the S M L control that sets it; and
// the complex-wave canvas: QuantEM's convention for a probe in real space, hue the phase and brightness the amplitude.
import * as React from "react";

export type ViewSize = "small" | "medium" | "large";
export const VIEW_SIZES: ViewSize[] = ["small", "medium", "large"];

/** One value a size: what each view scales by its preset. */
export function bySize<T>(size: string | undefined, small: T, medium: T, large: T): T {
  return size === "small" ? small : size === "large" ? large : medium;
}

/** Three compact buttons, the chosen one filled: a click redraws this view and writes the size back to the model. */
export function SizeControl({ size, onSize, colors }: { size: string; onSize: (size: ViewSize) => void; colors: any }) {
  return (
    <span style={{ display: "inline-flex", border: `1px solid ${colors.border}`, borderRadius: 3, overflow: "hidden", flex: "none" }} role="group" aria-label="View size">
      {VIEW_SIZES.map((option) => {
        const on = (size || "medium") === option;
        return (
          <button key={option} type="button" aria-pressed={on} title={`${option}: ${option === "large" ? "images dominant" : option === "small" ? "a compact strip" : "the default"}`}
            onClick={() => onSize(option)}
            style={{ border: "none", padding: "0 6px", font: "inherit", fontSize: 10, lineHeight: "16px", cursor: "pointer",
              background: on ? colors.accent : "transparent", color: on ? "#fff" : colors.textMuted }}>
            {option[0].toUpperCase()}
          </button>
        );
      })}
    </span>
  );
}

function hsvToRgb(h: number, s: number, v: number): [number, number, number] {
  const i = Math.floor(h * 6), f = h * 6 - i, p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
  switch (i % 6) {
    case 0: return [v, t, p];
    case 1: return [q, v, p];
    case 2: return [p, v, t];
    case 3: return [p, q, v];
    case 4: return [t, p, v];
    default: return [v, p, q];
  }
}

/** A complex wave at its native pixels: hue the phase, brightness the amplitude over its own peak (ShowPtycho's
 *  convention), drawn into width × height with true proportions kept by the caller. */
export function ComplexCanvas({ re, im, rows, cols, width, height, smooth }: {
  re: Float32Array; im: Float32Array; rows: number; cols: number; width: number; height: number; smooth?: boolean;
}) {
  const ref = React.useRef<HTMLCanvasElement>(null);
  React.useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !rows || !cols || re.length < rows * cols) return;
    canvas.width = cols; canvas.height = rows;
    const context = canvas.getContext("2d");
    if (!context) return;
    const image = context.createImageData(cols, rows);
    let peak = 0;
    for (let index = 0; index < rows * cols; index++) peak = Math.max(peak, Math.hypot(re[index], im[index]));
    peak = peak || 1;
    for (let index = 0; index < rows * cols; index++) {
      const hue = ((Math.atan2(im[index], re[index]) + Math.PI) / (2 * Math.PI)) % 1;
      const [r, g, b] = hsvToRgb(hue, 0.92, Math.min(1, Math.hypot(re[index], im[index]) / peak));
      image.data[4 * index] = Math.round(255 * r); image.data[4 * index + 1] = Math.round(255 * g); image.data[4 * index + 2] = Math.round(255 * b); image.data[4 * index + 3] = 255;
    }
    context.putImageData(image, 0, 0);
  }, [re, im, rows, cols]);
  return <canvas ref={ref} style={{ width, height, display: "block", imageRendering: smooth ? "auto" : "pixelated" }} />;
}
