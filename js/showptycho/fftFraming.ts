/** Preserve the native scan frequency range without cropping the underlying FFT. */
export function fftAutoFrameZoom(
  width: number, height: number, sampling: readonly number[],
  nativeSampling: readonly number[] = [],
): number {
  const valid = (steps: readonly number[]) => steps.length === 2
    && steps.every(step => Number.isFinite(step) && step > 0);
  if (!(width > 0 && height > 0) || !valid(sampling) || !valid(nativeSampling)) return 1;
  // Each axis retains its native Nyquist extent as output sampling increases.
  // The FFT may be padded; use its actual shape and the phase's physical sampling.
  const nativeWidth = width * sampling[1] / nativeSampling[1];
  const nativeHeight = height * sampling[0] / nativeSampling[0];
  return Math.max(1, Math.max(width, height) / Math.max(nativeWidth, nativeHeight));
}
