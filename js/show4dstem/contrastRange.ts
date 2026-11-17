/** The contrast ranges an image offers beside its histogram: the full range of the data, or the data
 *  between two percentiles. The percentile choices are those of quantem.live's Browse. */
export type Percentiles = [number, number];

export const CONTRAST_PERCENTILES: Percentiles[] = [[0.5, 99.5], [2, 98], [5, 95], [10, 90]];

/** The percentiles in force: a valid `percentiles` pair, else 1 to 99 when auto contrast is on, else
 *  null (the full range, or the two handles on the histogram). */
export function chosenPercentiles(percentiles: unknown, autoContrast: boolean): Percentiles | null {
  if (Array.isArray(percentiles) && percentiles.length === 2) {
    const [low, high] = percentiles.map(Number);
    if (Number.isFinite(low) && Number.isFinite(high) && low >= 0 && high <= 100 && low < high) return [low, high];
  }
  return autoContrast ? [1, 99] : null;
}

/** "2–98%", as the menu shows it. */
export function percentilesLabel([low, high]: Percentiles): string {
  return `${low}–${high}%`;
}

/** The menu entry that is selected: "full", a percentile pair as "low,high", or "handles" once a
 *  handle on the histogram has been moved off the full range. */
export function contrastRangeChoice(percentiles: Percentiles | null, vminPct: number, vmaxPct: number): string {
  if (percentiles) return percentiles.join(",");
  return vminPct <= 0 && vmaxPct >= 100 ? "full" : "handles";
}

/** A menu entry back to percentiles; null for "full" and "handles". */
export function percentilesOfChoice(choice: string): Percentiles | null {
  return chosenPercentiles(choice.split(","), false);
}

/** Where the two histogram handles sit (0 to 100 across the data's range) for a window of values. */
export function handlePositions(vmin: number, vmax: number, dataMin: number, dataMax: number): Percentiles | null {
  const span = dataMax - dataMin;
  if (!(span > 0)) return null;
  const position = (value: number) => Math.max(0, Math.min(100, ((value - dataMin) / span) * 100));
  return [position(vmin), position(vmax)];
}
