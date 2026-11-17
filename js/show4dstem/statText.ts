/** A statistic as the stats line shows it: two or three decimals, scientific notation for very small or large values. */
export function formatStat(value: number | null | undefined): string {
  // No statistic yet (the scans in view were just replaced by those of another size): a dash, never a throw
  // that takes the whole view off the page.
  if (value === null || value === undefined) return "–";
  if (value === 0) return "0";
  const abs = Math.abs(value);
  if (abs < 0.001 || abs >= 10000) {
    return value.toExponential(2);
  }
  if (abs < 0.01) return value.toFixed(4);
  if (abs < 1) return value.toFixed(3);
  return value.toFixed(2);
}

/** The mean of an image whose spread is `std`. A mean a million times smaller than the spread is zero
 *  at every digit shown: it reads "0", so a phase image's mean (2.4e-11 one scan, -3e-12 the next) does
 *  not change the line's length, and with it the panel's height, from scan to scan. */
export function formatMean(mean: number | null | undefined, std: number | null | undefined): string {
  return mean !== null && mean !== undefined && std !== null && std !== undefined && std > 0 && Math.abs(mean) < std * 1e-6 ? "0" : formatStat(mean);
}
