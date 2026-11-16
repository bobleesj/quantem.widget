import React from "react";
import {usePanelTouch, transformPanelView} from "../shared/panelTouch";
import type {standalonePhase} from "./standaloneDirect";

type Result = Awaited<ReturnType<typeof standalonePhase>>;

export function DirectPhaseDialog({result, onClose}: {result: Result; onClose: () => void}) {
  const dialog = React.useRef<HTMLDialogElement>(null);
  const canvas = React.useRef<HTMLCanvasElement>(null);
  const viewport = React.useRef<HTMLDivElement>(null);
  const view = React.useRef({zoom: 1, panX: 0, panY: 0});
  const touch = usePanelTouch();
  const paint = React.useRef(0);
  const redraw = React.useCallback(() => {
    if (paint.current) return;
    paint.current = requestAnimationFrame(() => {
      paint.current = 0;
      const {zoom, panX, panY} = view.current;
      if (canvas.current) canvas.current.style.transform = `translate(${panX}px,${panY}px) scale(${zoom})`;
    });
  }, []);
  const fit = () => { view.current = {zoom: 1, panX: 0, panY: 0}; redraw(); };
  const move = React.useCallback((el: Element, from: {x:number;y:number}, to: {x:number;y:number}, scale: number) => {
    const rect = el.getBoundingClientRect();
    const local = (p: typeof from) => ({x:p.x-rect.left, y:p.y-rect.top});
    const next = transformPanelView(view.current, local(from), local(to), scale, 1, 32);
    next.panX = Math.min(0,Math.max(rect.width*(1-next.zoom),next.panX));
    next.panY = Math.min(0,Math.max(rect.height*(1-next.zoom),next.panY));
    view.current = next; redraw();
  }, [redraw]);
  React.useEffect(() => {
    const el = viewport.current!;
    const wheel = (e: WheelEvent) => {
      e.preventDefault();
      const anchor = {x:e.clientX,y:e.clientY};
      move(el,anchor,anchor,Math.exp(-0.002*e.deltaY*(e.deltaMode === 1 ? 16 : 1)));
    };
    el.addEventListener("wheel",wheel,{passive:false});
    return () => { el.removeEventListener("wheel",wheel); cancelAnimationFrame(paint.current); paint.current=0; };
  }, [move]);
  React.useEffect(() => {
    const modal = dialog.current!;
    modal.showModal();
    return () => modal.close();
  }, []);
  React.useEffect(() => {
    const {width, height, phase} = result.phase;
    const ctx = canvas.current?.getContext("2d");
    if (!ctx) return;
    const pixels = ctx.createImageData(width, height);
    const sorted = phase.slice().sort();
    const lo = sorted[Math.floor(sorted.length * .01)], hi = sorted[Math.floor(sorted.length * .99)];
    for (let i = 0; i < phase.length; i++) {
      const j = i * 4;
      pixels.data[j] = pixels.data[j + 1] = pixels.data[j + 2] = 255 * (phase[i] - lo) / (hi - lo || 1);
      pixels.data[j + 3] = 255;
    }
    ctx.putImageData(pixels, 0, 0);
  }, [result]);
  return <dialog ref={dialog} aria-label="Direct Ptycho phase" onCancel={onClose}
    style={{maxWidth:"96vw",maxHeight:"96dvh",padding:12,boxSizing:"border-box",border:0}}>
    <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",gap:16}}>
      <strong>Direct Ptycho phase</strong><span><button onClick={fit}>Fit</button> <button onClick={onClose}>Close</button></span>
    </div>
    <div ref={viewport} {...touch(move)} onDoubleClick={fit}
      onPointerDown={e => { if (e.pointerType === "mouse" && e.button === 0) e.currentTarget.setPointerCapture(e.pointerId); }}
      onPointerMove={e => { if (e.pointerType === "mouse" && e.buttons === 1) move(e.currentTarget,
        {x:e.clientX-e.movementX,y:e.clientY-e.movementY},{x:e.clientX,y:e.clientY},1); }}
      style={{width:"min(87vw, max(96px,calc(92dvh - 7rem)))", aspectRatio:"1", overflow:"hidden", touchAction:"none", margin:"8px auto"}}>
    <canvas ref={canvas} width={result.phase.width} height={result.phase.height}
      aria-label="Full-field Direct Ptycho phase"
      style={{display:"block",width:"100%",height:"100%",transformOrigin:"0 0",imageRendering:"pixelated"}} />
    </div>
    <div>{result.phase.width} × {result.phase.height} · {result.phase.upsample}× · damping {result.phase.phaseDamp}</div>
    <div>Fit {((result.trialsMs + result.refinementMs)/1000).toFixed(1)} s · phase {(result.phase.elapsedMs/1000).toFixed(1)} s</div>
  </dialog>;
}
