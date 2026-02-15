/** Compute normalized histogram bins from Float32Array. Returns array of 0-1 values. */
export function computeHistogramFromBytes(data: Float32Array | null, numBins = 256): number[] {
  if (!data || data.length === 0) return new Array(numBins).fill(0);
  const bins = new Array(numBins).fill(0);
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (isFinite(v)) { if (v < min) min = v; if (v > max) max = v; }
  }
  if (!isFinite(min) || !isFinite(max) || min === max) return bins;
  const range = max - min;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (isFinite(v)) bins[Math.min(numBins - 1, Math.floor(((v - min) / range) * numBins))]++;
  }
  const maxCount = Math.max(...bins);
  if (maxCount > 0) for (let i = 0; i < numBins; i++) bins[i] /= maxCount;
  return bins;
}
