import { useCallback, useRef, type TouchEvent } from "react";
import { touchDistance, touchMidpoint } from "./geometry";

type Point = { x: number; y: number };
type Contact = Point & { distance: number; count: number };

function contact(touches: TouchEvent["targetTouches"]): Contact | null {
  if (touches.length === 2) {
    return { ...touchMidpoint(touches[0], touches[1]), distance: touchDistance(touches[0], touches[1]), count: 2 };
  }
  if (touches.length === 1) {
    return { x: touches[0].clientX, y: touches[0].clientY, distance: 0, count: 1 };
  }
  return null;
}

/** Restart the anchor whenever a finger lands or lifts, including pinch-to-pan. */
export function usePanelTouch() {
  const previous = useRef<{ target: Element; point: Contact } | null>(null);
  const track = useCallback((event: TouchEvent) => {
    const point = contact(event.targetTouches);
    previous.current = point ? { target: event.currentTarget, point } : null;
  }, []);
  return useCallback((move: (target: Element, from: Point, to: Point, scale: number, count: number) => void, finish?: () => void) => ({
    onTouchStart: track,
    onTouchMove: (event: TouchEvent) => {
      const old = previous.current;
      const point = contact(event.targetTouches);
      if (old?.target === event.currentTarget && point && point.count === old.point.count) {
        move(old.target, old.point, point, old.point.distance > 0 ? point.distance / old.point.distance : 1, point.count);
      }
      track(event);
    },
    onTouchEnd: (event: TouchEvent) => { track(event); if (!event.targetTouches.length) finish?.(); },
    onTouchCancel: (event: TouchEvent) => { track(event); finish?.(); },
  }), [track]);
}

export function transformPanelView(
  view: { zoom: number; panX: number; panY: number }, from: Point, to: Point,
  scale: number, min: number, max: number,
) {
  const zoom = Math.max(min, Math.min(max, view.zoom * scale));
  const ratio = zoom / view.zoom;
  return { zoom, panX: to.x - (from.x - view.panX) * ratio, panY: to.y - (from.y - view.panY) * ratio };
}
