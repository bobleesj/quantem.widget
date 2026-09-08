/** Serialize point-pattern queries while retaining the latest requested position. */
export function createLatestFrameQueue(
  run: (isCurrent: () => boolean) => Promise<void>,
  onError: (error: unknown) => void,
) {
  let version = 0, busy = false, scheduled = false, closed = false;
  async function drain() {
    if (closed) return;
    busy = true;
    try {
      while (!closed) {
        const active = version;
        const current = () => !closed && active === version;
        try { await run(current); }
        catch (error) { if (current()) onError(error); }
        if (active === version) break;
      }
    } finally { busy = false; }
  }
  return {
    request() {
      if (closed) return;
      version++;
      if (busy || scheduled) return;
      scheduled = true;
      // Row, column, and selection notifications in one event share one query.
      queueMicrotask(() => { scheduled = false; void drain(); });
    },
    close() { closed = true; version++; },
  };
}
