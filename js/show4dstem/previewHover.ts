const SOURCE_GAP_MS = 75;

/** Keep a preview through the space between source buttons, then clear a parked pointer. */
export function createPreviewHover(setSource: (update: (previous: string | null) => string | null) => void) {
  let gap: { source: string; timer: number } | null = null;
  const cancel = () => {
    if (gap) window.clearTimeout(gap.timer);
    gap = null;
  };
  const clear = (source?: string) => {
    if (source === undefined || gap?.source === source) cancel();
    setSource(previous => source === undefined || previous === source ? null : previous);
  };
  return {
    show(source: string) {
      cancel();
      setSource(() => source);
    },
    leave(source: string, group: Element | null, relatedTarget: EventTarget | null) {
      if (!(relatedTarget instanceof Element) || !group?.contains(relatedTarget)) {
        clear(source);
        return;
      }
      if (relatedTarget.closest("[data-preview-source]")?.parentElement === group) return;
      cancel();
      gap = { source, timer: window.setTimeout(() => clear(source), SOURCE_GAP_MS) };
    },
    leaveGroup() {
      if (gap) clear(gap.source);
    },
    clear,
    cancel,
  };
}
