/** A statistic as the stats line shows it: two or three decimals, scientific notation for very small or large values. */
export function formatStat(value: number | null | undefined): string {
  // No statistic yet (the scans in view were just replaced by those of another size): a dash, never a throw
  // that takes the whole view off the page.
  if (value === null || value === undefined) return "–";
  if (value === 0) return "0";
  const abs = Math.abs(value);
  if (abs < 0.001 || abs >= 10000) return powerOfTen(value, 2);
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

const SUPERSCRIPT = "⁰¹²³⁴⁵⁶⁷⁸⁹";

/** A number as digits and a power of ten, the way it is written by hand: 13600 with 2 decimals is "1.36×10⁴",
 *  2.73e-5 is "2.73×10⁻⁵" (11-18: "1.36e+4" and a six-digit dose were numbers to decode). */
export function powerOfTen(value: number, decimals: number): string {
  const [mantissa, exponent] = value.toExponential(decimals).split("e");
  const power = Number(exponent);
  return `${mantissa}×10${power < 0 ? "⁻" : ""}${[...String(Math.abs(power))].map((digit) => SUPERSCRIPT[Number(digit)]).join("")}`;
}

/** The electron dose: two digits and a power of ten from 1000 up ("3.4×10⁵"), so the size is read at a glance
 *  (344612 was a number to count the digits of); smaller doses stay plain. */
export function formatDose(dose: number): string {
  if (!Number.isFinite(dose)) return "–";
  if (dose < 1) return dose.toPrecision(2);
  if (dose < 100) return dose.toFixed(1);
  if (dose < 1000) return dose.toFixed(0);
  return powerOfTen(dose, 1);
}
