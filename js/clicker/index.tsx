import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import Slider from "@mui/material/Slider";
import Button from "@mui/material/Button";
import Box from "@mui/material/Box";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import Select, { type SelectChangeEvent } from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import "./clicker.css";
import { useTheme } from "../theme";

type MarkerShape = "circle" | "triangle" | "square" | "diamond" | "star";
type Point = { x: number; y: number; shape: MarkerShape; color: string };
type ZoomState = { zoom: number; panX: number; panY: number };

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 10;
const DRAG_THRESHOLD = 3;
const DEFAULT_ZOOM: ZoomState = { zoom: 1, panX: 0, panY: 0 };
const CANVAS_TARGET_SIZE = 400;
const GALLERY_TARGET_SIZE = 300;

const SPACING = {
  XS: 4,
  SM: 8,
  MD: 12,
  LG: 16,
};

const typography = {
  label: { fontSize: 11 },
  labelSmall: { fontSize: 10 },
  value: { fontSize: 10, fontFamily: "monospace" as const },
};

const controlRow = {
  display: "flex",
  alignItems: "center",
  gap: `${SPACING.SM}px`,
  px: 1,
  py: 0.5,
};

const compactButton = {
  fontSize: 10,
  minWidth: 0,
  px: 1,
  py: 0.25,
  "&.Mui-disabled": {
    color: "#666",
    borderColor: "#444",
  },
};

const sliderStyles = {
  small: {
    "& .MuiSlider-thumb": { width: 12, height: 12 },
    "& .MuiSlider-rail": { height: 3 },
    "& .MuiSlider-track": { height: 3 },
  },
};

const containerStyles = {
  root: { p: 2, bgcolor: "transparent", color: "inherit", fontFamily: "monospace", overflow: "visible" },
  imageBox: { bgcolor: "#000", border: "1px solid #444", overflow: "hidden", position: "relative" as const },
};

const MARKER_COLORS = [
  "#f44336", // red
  "#4caf50", // green
  "#2196f3", // blue
  "#ff9800", // orange
  "#9c27b0", // purple
  "#00bcd4", // cyan
  "#ffeb3b", // yellow
  "#e91e63", // pink
  "#8bc34a", // lime
  "#ff5722", // deep orange
];

const MARKER_SHAPES: MarkerShape[] = ["circle", "triangle", "square", "diamond", "star"];

function drawMarker(ctx: CanvasRenderingContext2D, x: number, y: number, r: number, shape: MarkerShape, fillColor: string, strokeColor: string) {
  ctx.beginPath();
  switch (shape) {
    case "circle":
      ctx.arc(x, y, r, 0, Math.PI * 2);
      break;
    case "triangle":
      ctx.moveTo(x, y - r);
      ctx.lineTo(x + r * 0.87, y + r * 0.5);
      ctx.lineTo(x - r * 0.87, y + r * 0.5);
      ctx.closePath();
      break;
    case "square":
      ctx.rect(x - r * 0.75, y - r * 0.75, r * 1.5, r * 1.5);
      break;
    case "diamond":
      ctx.moveTo(x, y - r);
      ctx.lineTo(x + r * 0.7, y);
      ctx.lineTo(x, y + r);
      ctx.lineTo(x - r * 0.7, y);
      ctx.closePath();
      break;
    case "star": {
      const spikes = 5;
      const outerR = r;
      const innerR = r * 0.4;
      for (let s = 0; s < spikes * 2; s++) {
        const rad = (s * Math.PI) / spikes - Math.PI / 2;
        const sr = s % 2 === 0 ? outerR : innerR;
        if (s === 0) ctx.moveTo(x + sr * Math.cos(rad), y + sr * Math.sin(rad));
        else ctx.lineTo(x + sr * Math.cos(rad), y + sr * Math.sin(rad));
      }
      ctx.closePath();
      break;
    }
  }
  ctx.fillStyle = fillColor;
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = strokeColor;
  ctx.stroke();
}

function extractBytes(dataView: DataView | ArrayBuffer | Uint8Array): Uint8Array {
  if (dataView instanceof Uint8Array) return dataView;
  if (dataView instanceof ArrayBuffer) return new Uint8Array(dataView);
  if (dataView && "buffer" in dataView) {
    return new Uint8Array(dataView.buffer, dataView.byteOffset, dataView.byteLength);
  }
  return new Uint8Array(0);
}

const render = createRender(() => {
  const { colors: tc } = useTheme();

  // Model state
  const [nImages] = useModelState<number>("n_images");
  const [width] = useModelState<number>("width");
  const [height] = useModelState<number>("height");
  const [frameBytes] = useModelState<DataView>("frame_bytes");
  const [imgMin] = useModelState<number[]>("img_min");
  const [imgMax] = useModelState<number[]>("img_max");
  const [selectedIdx, setSelectedIdx] = useModelState<number>("selected_idx");
  const [ncols] = useModelState<number>("ncols");
  const [labels] = useModelState<string[]>("labels");
  const [scale] = useModelState<number>("scale");
  const [selectedPoints, setSelectedPoints] = useModelState<Point[] | Point[][]>("selected_points");
  const [dotSize, setDotSize] = useModelState<number>("dot_size");
  const [maxPoints, setMaxPoints] = useModelState<number>("max_points");

  const isGallery = nImages > 1;

  // Current marker style (user selects before placing)
  const [currentShape, setCurrentShape] = React.useState<MarkerShape>("circle");
  const [currentColor, setCurrentColor] = React.useState<string>(MARKER_COLORS[0]);

  // Refs
  const canvasRefs = React.useRef<(HTMLCanvasElement | null)[]>([]);
  const offscreenRefs = React.useRef<(HTMLCanvasElement | null)[]>([]);
  const canvasContainerRefs = React.useRef<(HTMLDivElement | null)[]>([]);

  const [hover, setHover] = React.useState<{
    x: number;
    y: number;
    raw?: number;
    norm?: number;
  } | null>(null);

  // Per-image zoom state
  const [zoomStates, setZoomStates] = React.useState<Map<number, ZoomState>>(new Map());
  const getZoom = React.useCallback((idx: number): ZoomState => zoomStates.get(idx) || DEFAULT_ZOOM, [zoomStates]);
  const setZoom = React.useCallback((idx: number, zs: ZoomState) => {
    setZoomStates(prev => new Map(prev).set(idx, zs));
  }, []);

  const dragRef = React.useRef<{
    startX: number;
    startY: number;
    startPanX: number;
    startPanY: number;
    dragging: boolean;
    wasDrag: boolean;
    imageIdx: number;
  } | null>(null);

  // Resize state
  const [mainCanvasSize, setMainCanvasSize] = React.useState(CANVAS_TARGET_SIZE);
  const [galleryCanvasSize, setGalleryCanvasSize] = React.useState(GALLERY_TARGET_SIZE);
  const [isResizing, setIsResizing] = React.useState(false);
  const [resizeStart, setResizeStart] = React.useState<{ x: number; y: number; size: number } | null>(null);
  const initialCanvasSizeRef = React.useRef<number>(CANVAS_TARGET_SIZE);

  // Sync initial size when image loads — never shrink below target
  React.useEffect(() => {
    if (width > 0 && height > 0) {
      const sz = Math.max(CANVAS_TARGET_SIZE, Math.round(Math.max(width, height) * scale));
      if (!isGallery) setMainCanvasSize(sz);
      initialCanvasSizeRef.current = CANVAS_TARGET_SIZE;
    }
  }, [width, height, scale, isGallery]);

  // Compute display dimensions
  const targetSize = isGallery ? galleryCanvasSize : mainCanvasSize;
  const displayScale = width > 0 && height > 0 ? targetSize / Math.max(width, height) : 1;
  const canvasW = width > 0 ? Math.round(width * displayScale) : targetSize;
  const canvasH = height > 0 ? Math.round(height * displayScale) : targetSize;

  // Parse frame_bytes into per-image Float32Arrays
  const floatsPerImage = width * height;
  const perImageData = React.useMemo(() => {
    if (!frameBytes || !width || !height) return [];
    const bytes = extractBytes(frameBytes);
    if (bytes.length === 0) return [];
    const allFloats = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
    const result: Float32Array[] = [];
    for (let i = 0; i < nImages; i++) {
      const start = i * floatsPerImage;
      result.push(allFloats.subarray(start, start + floatsPerImage));
    }
    return result;
  }, [frameBytes, nImages, floatsPerImage, width, height]);

  // Build offscreen canvases from float32 data (JS normalizes)
  React.useEffect(() => {
    if (perImageData.length === 0 || !width || !height) return;
    for (let i = 0; i < nImages; i++) {
      const f32 = perImageData[i];
      if (!f32) continue;
      if (!offscreenRefs.current[i]) {
        offscreenRefs.current[i] = document.createElement("canvas");
      }
      const offscreen = offscreenRefs.current[i]!;
      offscreen.width = width;
      offscreen.height = height;
      const ctx = offscreen.getContext("2d");
      if (!ctx) continue;

      const min = imgMin?.[i] ?? 0;
      const max = imgMax?.[i] ?? 1;
      const range = max - min || 1;

      const imageData = ctx.createImageData(width, height);
      const dst = imageData.data;
      for (let j = 0; j < f32.length; j++) {
        const normalized = Math.max(0, Math.min(255, Math.round(((f32[j] - min) / range) * 255)));
        const k = j * 4;
        dst[k] = normalized;
        dst[k + 1] = normalized;
        dst[k + 2] = normalized;
        dst[k + 3] = 255;
      }
      ctx.putImageData(imageData, 0, 0);
    }
    offscreenRefs.current.length = nImages;
  }, [perImageData, nImages, width, height, imgMin, imgMax]);

  // Per-image points helpers
  const getPointsForImage = React.useCallback((idx: number): Point[] => {
    if (!isGallery) return (selectedPoints as Point[]) || [];
    const nested = (selectedPoints as Point[][]) || [];
    return nested[idx] || [];
  }, [isGallery, selectedPoints]);

  const setPointsForImage = React.useCallback((idx: number, points: Point[]) => {
    if (!isGallery) {
      setSelectedPoints(points);
    } else {
      setSelectedPoints((prev) => {
        const nested = [...((prev as Point[][]) || [])];
        while (nested.length < nImages) nested.push([]);
        nested[idx] = points;
        return nested;
      });
    }
  }, [isGallery, nImages, setSelectedPoints]);

  // Dot size
  const size = Number.isFinite(dotSize) && dotSize > 0 ? dotSize : 12;

  // Render all canvases
  React.useEffect(() => {
    if (!width || !height || perImageData.length === 0) return;
    for (let i = 0; i < nImages; i++) {
      const canvas = canvasRefs.current[i];
      const offscreen = offscreenRefs.current[i];
      if (!canvas || !offscreen) continue;
      const ctx = canvas.getContext("2d");
      if (!ctx) continue;

      canvas.width = canvasW;
      canvas.height = canvasH;

      const { zoom, panX, panY } = getZoom(i);
      const cx = canvasW / 2;
      const cy = canvasH / 2;

      ctx.clearRect(0, 0, canvasW, canvasH);
      ctx.save();
      ctx.imageSmoothingEnabled = false;

      ctx.translate(cx + panX, cy + panY);
      ctx.scale(zoom, zoom);
      ctx.translate(-cx, -cy);

      ctx.drawImage(offscreen, 0, 0, canvasW, canvasH);

      // Draw points for this image
      const pts = getPointsForImage(i);
      const dotRadius = (size / 2) * displayScale;
      for (let j = 0; j < pts.length; j++) {
        const p = pts[j];
        const px = (p.x / width) * canvasW;
        const py = (p.y / height) * canvasH;
        const color = p.color || MARKER_COLORS[j % MARKER_COLORS.length];
        const shape = p.shape || MARKER_SHAPES[j % MARKER_SHAPES.length];

        drawMarker(ctx, px, py, dotRadius, shape, color, tc.bg);

        const fontSize = Math.max(10, size * 0.9);
        ctx.font = `bold ${fontSize}px sans-serif`;
        ctx.fillStyle = tc.text;
        ctx.textAlign = "center";
        ctx.textBaseline = "bottom";
        ctx.shadowColor = "rgba(0,0,0,0.7)";
        ctx.shadowBlur = 3;
        ctx.fillText(`${j + 1}`, px, py - dotRadius - 2);
        ctx.shadowBlur = 0;
      }

      ctx.restore();

    }
  }, [perImageData, width, height, canvasW, canvasH, displayScale, zoomStates, selectedPoints, size, tc.accent, tc.bg, tc.text, nImages, isGallery, selectedIdx, getZoom, getPointsForImage]);

  // Map screen coordinates to image pixel coordinates
  const clientToImage = React.useCallback(
    (clientX: number, clientY: number, idx: number): { x: number; y: number } | null => {
      const canvas = canvasRefs.current[idx];
      if (!canvas || !width || !height) return null;
      const rect = canvas.getBoundingClientRect();
      const cx = canvasW / 2;
      const cy = canvasH / 2;
      const { zoom, panX, panY } = getZoom(idx);

      const canvasX = ((clientX - rect.left) / rect.width) * canvasW;
      const canvasY = ((clientY - rect.top) / rect.height) * canvasH;

      const imgDisplayX = (canvasX - cx - panX) / zoom + cx;
      const imgDisplayY = (canvasY - cy - panY) / zoom + cy;

      const x = Math.floor((imgDisplayX / canvasW) * width);
      const y = Math.floor((imgDisplayY / canvasH) * height);
      if (x < 0 || y < 0 || x >= width || y >= height) return null;
      return { x, y };
    },
    [width, height, canvasW, canvasH, getZoom],
  );

  // Prevent page scroll on canvas containers
  React.useEffect(() => {
    const preventDefault = (e: WheelEvent) => e.preventDefault();
    const containers = canvasContainerRefs.current.filter(Boolean);
    containers.forEach(el => el?.addEventListener("wheel", preventDefault, { passive: false }));
    return () => {
      containers.forEach(el => el?.removeEventListener("wheel", preventDefault));
    };
  }, [nImages]);

  // Scroll to zoom
  const handleWheel = React.useCallback(
    (e: React.WheelEvent, idx: number) => {
      e.preventDefault();
      if (isGallery && idx !== selectedIdx) return;
      const canvas = canvasRefs.current[idx];
      if (!canvas || !width || !height) return;
      const rect = canvas.getBoundingClientRect();

      const mouseX = ((e.clientX - rect.left) / rect.width) * canvasW;
      const mouseY = ((e.clientY - rect.top) / rect.height) * canvasH;

      const factor = e.deltaY < 0 ? 1.1 : 0.9;
      const prev = getZoom(idx);
      const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, prev.zoom * factor));
      const cx = canvasW / 2;
      const cy = canvasH / 2;
      const wx = (mouseX - cx - prev.panX) / prev.zoom + cx;
      const wy = (mouseY - cy - prev.panY) / prev.zoom + cy;
      const newPanX = mouseX - cx - (wx - cx) * newZoom;
      const newPanY = mouseY - cy - (wy - cy) * newZoom;
      setZoom(idx, { zoom: newZoom, panX: newPanX, panY: newPanY });
    },
    [width, height, canvasW, canvasH, isGallery, selectedIdx, getZoom, setZoom],
  );

  // Track when we just switched focus (don't place a point on the same click)
  const justSwitchedRef = React.useRef(false);

  // Mouse down
  const handleMouseDown = React.useCallback(
    (e: React.MouseEvent, idx: number) => {
      if (e.button !== 0) return;
      justSwitchedRef.current = false;
      if (isGallery && idx !== selectedIdx) {
        setSelectedIdx(idx);
        justSwitchedRef.current = true;
        return;
      }
      const zs = getZoom(idx);
      dragRef.current = {
        startX: e.clientX,
        startY: e.clientY,
        startPanX: zs.panX,
        startPanY: zs.panY,
        dragging: false,
        wasDrag: false,
        imageIdx: idx,
      };
    },
    [isGallery, selectedIdx, setSelectedIdx, getZoom],
  );

  // Mouse move
  const handleMouseMove = React.useCallback(
    (e: React.MouseEvent, idx: number) => {
      const drag = dragRef.current;
      if (drag && drag.imageIdx === idx) {
        const dx = e.clientX - drag.startX;
        const dy = e.clientY - drag.startY;
        if (!drag.dragging && Math.abs(dx) + Math.abs(dy) > DRAG_THRESHOLD) {
          drag.dragging = true;
        }
        if (drag.dragging) {
          drag.wasDrag = true;
          const canvas = canvasRefs.current[idx];
          if (!canvas) return;
          const rect = canvas.getBoundingClientRect();
          const scaleX = canvasW / rect.width;
          const scaleY = canvasH / rect.height;
          setZoom(idx, {
            zoom: getZoom(idx).zoom,
            panX: drag.startPanX + dx * scaleX,
            panY: drag.startPanY + dy * scaleY,
          });
          return;
        }
      }

      // Hover readout (only for selected image in gallery)
      if (isGallery && idx !== selectedIdx) return;
      const p = clientToImage(e.clientX, e.clientY, idx);
      if (!p) { setHover(null); return; }
      let raw: number | undefined;
      let norm: number | undefined;
      const f32 = perImageData[idx];
      if (f32) {
        raw = f32[p.y * width + p.x];
        const min = imgMin?.[idx] ?? 0;
        const max = imgMax?.[idx] ?? 1;
        const denom = max > min ? max - min : 1;
        norm = (raw - min) / denom;
      }
      setHover({ x: p.x, y: p.y, raw, norm });
    },
    [clientToImage, width, canvasW, canvasH, perImageData, imgMin, imgMax, isGallery, selectedIdx, getZoom, setZoom],
  );

  // Mouse up — place point
  const handleMouseUp = React.useCallback(
    (e: React.MouseEvent, idx: number) => {
      const drag = dragRef.current;
      dragRef.current = null;
      if (drag?.wasDrag) return;
      if (justSwitchedRef.current) { justSwitchedRef.current = false; return; }
      if (isGallery && idx !== selectedIdx) return;

      const coords = clientToImage(e.clientX, e.clientY, idx);
      if (!coords) return;
      redoStackRef.current.set(idx, []); // clear redo on new point
      const p: Point = { x: coords.x, y: coords.y, shape: currentShape, color: currentColor };
      const currentPts = getPointsForImage(idx);
      const limit = Number.isFinite(maxPoints) && maxPoints > 0 ? maxPoints : 3;
      const next = [...currentPts, p];
      setPointsForImage(idx, next.length <= limit ? next : next.slice(next.length - limit));
    },
    [clientToImage, maxPoints, isGallery, selectedIdx, getPointsForImage, setPointsForImage],
  );

  // Double-click — reset zoom
  const handleDoubleClick = React.useCallback((idx: number) => {
    if (isGallery && idx !== selectedIdx) return;
    setZoom(idx, DEFAULT_ZOOM);
  }, [isGallery, selectedIdx, setZoom]);

  const activeIdx = isGallery ? selectedIdx : 0;

  // Redo stack: per-image undone points
  const redoStackRef = React.useRef<Map<number, Point[]>>(new Map());

  const resetPoints = React.useCallback(() => {
    if (!isGallery) {
      setSelectedPoints([]);
    } else {
      setSelectedPoints([...Array(nImages)].map(() => []));
    }
    setHover(null);
    setDotSize(12);
    setMaxPoints(10);
    redoStackRef.current = new Map();
  }, [isGallery, nImages, setSelectedPoints, setDotSize, setMaxPoints]);

  const undoPoint = React.useCallback(() => {
    const pts = getPointsForImage(activeIdx);
    if (pts.length === 0) return;
    const removed = pts[pts.length - 1];
    const stack = redoStackRef.current.get(activeIdx) || [];
    redoStackRef.current.set(activeIdx, [...stack, removed]);
    setPointsForImage(activeIdx, pts.slice(0, -1));
  }, [activeIdx, getPointsForImage, setPointsForImage]);

  const redoPoint = React.useCallback(() => {
    const stack = redoStackRef.current.get(activeIdx) || [];
    if (stack.length === 0) return;
    const point = stack[stack.length - 1];
    redoStackRef.current.set(activeIdx, stack.slice(0, -1));
    const pts = getPointsForImage(activeIdx);
    const limit = Number.isFinite(maxPoints) && maxPoints > 0 ? maxPoints : 3;
    if (pts.length < limit) {
      setPointsForImage(activeIdx, [...pts, point]);
    }
  }, [activeIdx, getPointsForImage, setPointsForImage, maxPoints]);

  const canRedo = (redoStackRef.current.get(activeIdx) || []).length > 0;

  // Resize handlers
  const handleResizeStart = (e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    setIsResizing(true);
    setResizeStart({ x: e.clientX, y: e.clientY, size: isGallery ? galleryCanvasSize : mainCanvasSize });
  };

  React.useEffect(() => {
    if (!isResizing) return;
    const handleMouseMove = (e: MouseEvent) => {
      if (!resizeStart) return;
      const delta = Math.max(e.clientX - resizeStart.x, e.clientY - resizeStart.y);
      const minSize = isGallery ? 100 : initialCanvasSizeRef.current;
      const maxSize = isGallery ? 600 : 800;
      const newSize = Math.max(minSize, Math.min(maxSize, resizeStart.size + delta));
      if (isGallery) {
        setGalleryCanvasSize(newSize);
      } else {
        setMainCanvasSize(newSize);
      }
    };
    const handleMouseUp = () => {
      setIsResizing(false);
      setResizeStart(null);
    };
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizing, resizeStart]);

  const activeZoom = getZoom(activeIdx);
  const needsReset = activeZoom.zoom !== 1 || activeZoom.panX !== 0 || activeZoom.panY !== 0;
  const maxPtsVal = Number.isFinite(maxPoints) && maxPoints > 0 ? maxPoints : 3;
  const activePts = getPointsForImage(activeIdx);
  const hasAnyPoints = isGallery
    ? (selectedPoints as Point[][])?.some((pts) => pts?.length > 0)
    : (selectedPoints as Point[])?.length > 0;

  // Render a single canvas box (shared between single and gallery mode)
  const renderCanvasBox = (idx: number, showResizeHandle: boolean) => (
    <Box
      ref={(el: HTMLDivElement | null) => { canvasContainerRefs.current[idx] = el; }}
      sx={{
        ...containerStyles.imageBox,
        width: canvasW,
        height: canvasH,
        cursor: isGallery && idx !== selectedIdx
          ? "pointer"
          : dragRef.current?.dragging ? "grabbing" : "crosshair",
        border: isGallery && idx === selectedIdx
          ? `1px solid ${tc.accent}`
          : containerStyles.imageBox.border,
      }}
      onMouseDown={(e) => handleMouseDown(e, idx)}
      onMouseMove={(e) => handleMouseMove(e, idx)}
      onMouseUp={(e) => handleMouseUp(e, idx)}
      onMouseLeave={() => { dragRef.current = null; setHover(null); }}
      onWheel={(e) => handleWheel(e, idx)}
      onDoubleClick={() => handleDoubleClick(idx)}
    >
      <canvas
        ref={(el) => { canvasRefs.current[idx] = el; }}
        width={canvasW}
        height={canvasH}
        style={{ width: canvasW, height: canvasH, imageRendering: "pixelated" }}
      />
      {showResizeHandle && (
        <Box
          onMouseDown={handleResizeStart}
          sx={{
            position: "absolute",
            bottom: 0,
            right: 0,
            width: 16,
            height: 16,
            cursor: "nwse-resize",
            opacity: 0.6,
            background: `linear-gradient(135deg, transparent 50%, ${tc.accent} 50%)`,
            borderRadius: "0 0 4px 0",
            "&:hover": { opacity: 1 },
          }}
        />
      )}
    </Box>
  );

  return (
    <Box className="clicker-root" sx={{ ...containerStyles.root, bgcolor: tc.bg, color: tc.text }}>
      {/* Header row */}
      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 28, width: isGallery ? undefined : canvasW, maxWidth: isGallery ? ncols * canvasW + (ncols - 1) * 8 : canvasW }}>
        <Typography variant="caption" sx={{ ...typography.label, color: tc.text }}>
          Clicker
          {isGallery && labels?.[activeIdx] && (
            <Box component="span" sx={{ color: tc.textMuted, ml: 1 }}>
              {labels[activeIdx]}
            </Box>
          )}
          {activePts.length > 0 && (
            <Box component="span" sx={{ color: tc.textMuted, ml: 1 }}>
              {activePts.length}/{maxPtsVal} pts
            </Box>
          )}
        </Typography>
        <Stack direction="row" spacing={`${SPACING.SM}px`} alignItems="center">
          <Button size="small" sx={compactButton} onClick={undoPoint} disabled={!activePts.length}>UNDO</Button>
          <Button size="small" sx={compactButton} onClick={redoPoint} disabled={!canRedo}>REDO</Button>
          <Button size="small" sx={compactButton} disabled={!needsReset} onClick={() => handleDoubleClick(activeIdx)}>RESET VIEW</Button>
          <Button size="small" sx={compactButton} onClick={resetPoints} disabled={!hasAnyPoints}>RESET ALL</Button>
        </Stack>
      </Stack>

      {/* Canvas area */}
      {isGallery ? (
        <Box sx={{ display: "inline-grid", gridTemplateColumns: `repeat(${ncols}, ${canvasW}px)`, gap: 1 }}>
          {Array.from({ length: nImages }).map((_, i) => (
            <Box key={i}>
              {renderCanvasBox(i, i === selectedIdx)}
              <Typography sx={{ fontSize: 10, color: i === selectedIdx ? tc.accent : tc.textMuted, textAlign: "center", mt: 0.25 }}>
                {labels?.[i] || `Image ${i + 1}`}
              </Typography>
            </Box>
          ))}
        </Box>
      ) : (
        renderCanvasBox(0, true)
      )}

      {/* Readout bar */}
      <Box sx={{ mt: 0.5, px: 1, py: 0.5, bgcolor: tc.bgAlt, display: "flex", gap: 2, minHeight: 20, width: isGallery ? undefined : canvasW, maxWidth: isGallery ? ncols * canvasW + (ncols - 1) * 8 : canvasW, boxSizing: "border-box" }}>
        <Typography sx={{ fontSize: 11, color: tc.textMuted }}>
          x: <Box component="span" sx={{ color: tc.accent }}>{hover ? hover.x : "\u2013"}</Box>
        </Typography>
        <Typography sx={{ fontSize: 11, color: tc.textMuted }}>
          y: <Box component="span" sx={{ color: tc.accent }}>{hover ? hover.y : "\u2013"}</Box>
        </Typography>
        {hover?.raw !== undefined && (
          <Typography sx={{ fontSize: 11, color: tc.textMuted }}>
            Raw <Box component="span" sx={{ color: tc.accent }}>{hover.raw.toFixed(4)}</Box>
          </Typography>
        )}
        {hover?.norm !== undefined && (
          <Typography sx={{ fontSize: 11, color: tc.textMuted }}>
            Norm <Box component="span" sx={{ color: tc.accent }}>{hover.norm.toFixed(3)}</Box>
          </Typography>
        )}
        {activeZoom.zoom !== 1 && (
          <Typography sx={{ fontSize: 11, color: tc.accent, fontWeight: "bold", ml: "auto" }}>
            {activeZoom.zoom.toFixed(1)}x
          </Typography>
        )}
      </Box>

      {/* Controls — single row with dot size + max points */}
      {/* Controls row 1: Marker size + Shape + Max */}
      <Box sx={{ ...controlRow, border: `1px solid ${tc.border}`, bgcolor: tc.controlBg, mt: 0.5, maxWidth: canvasW, width: "fit-content", boxSizing: "border-box" }}>
        <Typography sx={{ ...typography.labelSmall, color: tc.textMuted }}>Marker:</Typography>
        <Slider
          value={size}
          min={4}
          max={40}
          step={1}
          onChange={(_, v) => { if (typeof v === "number") setDotSize(v); }}
          size="small"
          sx={{ ...sliderStyles.small, width: 60 }}
        />
        <Typography sx={{ ...typography.value, color: tc.textMuted, minWidth: 20 }}>{size}px</Typography>
        <Typography sx={{ ...typography.labelSmall, color: tc.textMuted }}>Max:</Typography>
        <Select
          value={maxPtsVal}
          onChange={(e: SelectChangeEvent<number>) => {
            const v = Number(e.target.value);
            setMaxPoints(v);
            if (!isGallery) {
              setSelectedPoints((prev) => {
                const flat = (prev as Point[]) || [];
                return flat.length <= v ? flat : flat.slice(flat.length - v);
              });
            } else {
              setSelectedPoints((prev) => {
                const nested = ((prev as Point[][]) || []).map(pts =>
                  pts.length <= v ? pts : pts.slice(pts.length - v)
                );
                return nested;
              });
            }
          }}
          size="small"
          variant="outlined"
          MenuProps={{ anchorOrigin: { vertical: "top", horizontal: "left" }, transformOrigin: { vertical: "bottom", horizontal: "left" } }}
          sx={{
            minWidth: 50,
            fontSize: 10,
            bgcolor: tc.controlBg,
            color: tc.text,
            "& .MuiSelect-select": { py: 0.25, px: 1 },
            "& .MuiOutlinedInput-notchedOutline": { borderColor: tc.border },
            "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: tc.accent },
          }}
        >
          {Array.from({ length: 20 }, (_, i) => i + 1).map(n => (
            <MenuItem key={n} value={n} sx={{ fontSize: 10 }}>{n}</MenuItem>
          ))}
        </Select>
      </Box>

      {/* Shape + Color picker row */}
      <Box sx={{ mt: 0.5, display: "flex", alignItems: "center", gap: `${SPACING.SM}px`, maxWidth: canvasW }}>
        <Box sx={{ display: "flex", gap: "3px" }}>
          {MARKER_SHAPES.map(s => {
            const sz = 16;
            const half = sz / 2;
            const r = half * 0.7;
            const selected = s === currentShape;
            let path: React.ReactNode;
            switch (s) {
              case "circle": path = <circle cx={half} cy={half} r={r} />; break;
              case "triangle": path = <polygon points={`${half},${half - r} ${half + r * 0.87},${half + r * 0.5} ${half - r * 0.87},${half + r * 0.5}`} />; break;
              case "square": path = <rect x={half - r * 0.75} y={half - r * 0.75} width={r * 1.5} height={r * 1.5} />; break;
              case "diamond": path = <polygon points={`${half},${half - r} ${half + r * 0.7},${half} ${half},${half + r} ${half - r * 0.7},${half}`} />; break;
              case "star": {
                const pts: string[] = [];
                for (let i = 0; i < 10; i++) {
                  const angle = (i * Math.PI) / 5 - Math.PI / 2;
                  const sr = i % 2 === 0 ? r : r * 0.4;
                  pts.push(`${half + sr * Math.cos(angle)},${half + sr * Math.sin(angle)}`);
                }
                path = <polygon points={pts.join(" ")} />;
                break;
              }
            }
            return (
              <Box
                key={s}
                onClick={() => setCurrentShape(s)}
                sx={{
                  width: sz, height: sz, cursor: "pointer", borderRadius: "2px",
                  border: selected ? `2px solid ${tc.text}` : "2px solid transparent",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  "&:hover": { opacity: 0.8 },
                }}
              >
                <svg width={sz} height={sz} style={{ display: "block" }}>
                  <g fill={currentColor} stroke={tc.bg} strokeWidth={1}>{path}</g>
                </svg>
              </Box>
            );
          })}
        </Box>
        <Box sx={{ display: "flex", gap: "3px", flexWrap: "wrap" }}>
          {MARKER_COLORS.map(c => (
            <Box
              key={c}
              onClick={() => setCurrentColor(c)}
              sx={{
                width: 16, height: 16, bgcolor: c, borderRadius: "2px", cursor: "pointer",
                border: c === currentColor ? `2px solid ${tc.text}` : "2px solid transparent",
                "&:hover": { opacity: 0.8 },
              }}
            />
          ))}
        </Box>
      </Box>

      {/* Selected points list */}
      {isGallery ? (
        hasAnyPoints && (
          <Box sx={{ mt: 0.5 }}>
            {Array.from({ length: nImages }).map((_, imgIdx) => {
              const pts = getPointsForImage(imgIdx);
              if (pts.length === 0) return null;
              return (
                <Box key={imgIdx} sx={{ mb: 0.5 }}>
                  <Typography sx={{ fontSize: 10, fontFamily: "monospace", color: imgIdx === selectedIdx ? tc.accent : tc.textMuted, fontWeight: "bold", lineHeight: 1.6 }}>
                    {labels?.[imgIdx] || `Image ${imgIdx + 1}`}
                  </Typography>
                  <Box sx={{ display: "grid", gridTemplateColumns: "repeat(5, auto)", gap: `0 ${SPACING.MD}px`, width: "fit-content", pl: 1 }}>
                    {pts.map((p, i) => (
                      <Typography key={`pt-${imgIdx}-${i}`} sx={{ fontSize: 10, fontFamily: "monospace", color: tc.textMuted, lineHeight: 1.6 }}>
                        <Box component="span" sx={{ color: p.color || MARKER_COLORS[i % MARKER_COLORS.length] }}>{i + 1}</Box> ({p.x}, {p.y})
                      </Typography>
                    ))}
                  </Box>
                </Box>
              );
            })}
          </Box>
        )
      ) : (
        activePts.length > 0 && (
          <Box sx={{ mt: 0.5, display: "grid", gridTemplateColumns: "repeat(5, auto)", gap: `0 ${SPACING.MD}px`, width: "fit-content" }}>
            {activePts.map((p, i) => (
              <Typography key={`pt-${p.x}-${p.y}-${i}`} sx={{ fontSize: 10, fontFamily: "monospace", color: tc.textMuted, lineHeight: 1.6 }}>
                <Box component="span" sx={{ color: MARKER_COLORS[i % MARKER_COLORS.length] }}>{i + 1}</Box> ({p.x}, {p.y})
              </Typography>
            ))}
          </Box>
        )
      )}
    </Box>
  );
});

export default { render };
