export type DetectorRoiMode = "point" | "circle" | "square" | "rect" | "annular" | "off";

export type DetectorResizeGeometry = {
  radius?: number;
  radiusInner?: number;
  width?: number;
  height?: number;
};

export function clampDetectorCenter(
  row: number,
  col: number,
  detectorRows: number,
  detectorCols: number,
): { row: number; col: number } {
  return {
    row: Math.max(0, Math.min(detectorRows - 1, row)),
    col: Math.max(0, Math.min(detectorCols - 1, col)),
  };
}

export function resizeDetectorFromPointer({
  mode,
  centerRow,
  centerCol,
  pointerRow,
  pointerCol,
  radius,
  radiusInner,
  resizeInner = false,
  aspectRatio = null,
  preserveAspect = false,
}: {
  mode: DetectorRoiMode;
  centerRow: number;
  centerCol: number;
  pointerRow: number;
  pointerCol: number;
  radius: number;
  radiusInner: number;
  resizeInner?: boolean;
  aspectRatio?: number | null;
  preserveAspect?: boolean;
}): DetectorResizeGeometry | null {
  const rowDistance = Math.abs(pointerRow - centerRow);
  const colDistance = Math.abs(pointerCol - centerCol);

  if (resizeInner && mode === "annular") {
    return {
      radiusInner: Math.max(1, Math.min(radius - 1, Math.hypot(rowDistance, colDistance))),
    };
  }

  if (mode === "rect") {
    let width = Math.max(2, colDistance * 2);
    let height = Math.max(2, rowDistance * 2);
    if (preserveAspect && aspectRatio != null) {
      if (width / height > aspectRatio) height = Math.max(2, width / aspectRatio);
      else width = Math.max(2, height * aspectRatio);
    }
    return { width, height };
  }

  if (mode === "circle" || mode === "square" || mode === "annular") {
    const nextRadius = mode === "square"
      ? Math.max(rowDistance, colDistance)
      : Math.hypot(rowDistance, colDistance);
    return {
      radius: Math.max(mode === "annular" ? radiusInner + 1 : 1, nextRadius),
    };
  }

  return null;
}
