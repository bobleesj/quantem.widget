/** The scan the operator chose while its images are still on their way from the kernel.
 *
 *  One choice is with the kernel at a time. A choice made meanwhile replaces the one waiting, so the
 *  kernel answers in the order asked and an answer for an older choice cannot move the view back. */
export function createScanChoice(
  send: (source: string) => void,
  later: (run: () => void, ms: number) => unknown = (run, ms) => setTimeout(run, ms),
  answerWithinMs = 2000,
) {
  let sent: string | null = null;
  let waiting: string | null = null;
  let ticket = 0;
  function dispatch(source: string) {
    sent = source;
    const mine = ++ticket;
    send(source);
    // A scan released meanwhile is never shown: stop waiting for it.
    later(() => { if (mine === ticket && sent === source) settle(); }, answerWithinMs);
  }
  function settle() {
    sent = null;
    if (waiting === null) return;
    const next = waiting;
    waiting = null;
    dispatch(next);
  }
  return {
    /** Ask for `source`; `shown` is the scan on screen now. */
    choose(source: string, shown: string | undefined) {
      if (sent !== null) { waiting = source; return; }
      if (source === shown) return;
      dispatch(source);
    },
    /** The kernel now shows `source`. */
    shown(source: string | undefined) {
      if (sent !== null && source === sent) settle();
    },
    /** The scan the operator is waiting for, or null when the view shows it. */
    pending(): string | null { return waiting ?? sent; },
  };
}

/** The index a frame step starts from: the scan chosen last, else the one shown. */
export function chosenFrameIndex(pending: string | null, identities: string[], shownIndex: number): number {
  const index = pending === null ? -1 : identities.indexOf(pending);
  return index >= 0 ? index : shownIndex;
}
