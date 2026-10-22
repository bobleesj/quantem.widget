import { expect, test } from 'vitest';
import { fftAutoFrameZoom } from './fftFraming';

test('Native frequency range is identical at 1x, 2x, padded 3x and 4x', () => {
  const step = 0.373, q = 0.8;
  for (const factor of [1, 2, 3, 4]) {
    const fft = 2 ** Math.ceil(Math.log2(512 * factor));
    const sampling = step / factor;
    const zoom = fftAutoFrameZoom(fft, fft, [sampling, sampling], [step, step]);
    expect(zoom).toBeCloseTo(factor, 12);
    expect(0.5 / sampling / zoom).toBeCloseTo(0.5 / step, 12);
    expect(q * sampling * zoom).toBeCloseTo(q * step, 12);
  }
});

test('rectangular anisotropic scans retain their original frequency extents', () => {
  expect(fftAutoFrameZoom(512, 256, [0.06, 0.04], [0.06, 0.04])).toBe(1);
  expect(fftAutoFrameZoom(2048, 1024, [0.02, 0.04 / 3], [0.06, 0.04])).toBeCloseTo(3);
});

test('uncalibrated arrays and invalid metadata retain the full FFT', () => {
  for (const sampling of [[], [0, 0], [NaN, 1]]) {
    expect(fftAutoFrameZoom(512, 512, sampling, [0.4, 0.4])).toBe(1);
    expect(fftAutoFrameZoom(512, 512, [0.1, 0.1], sampling)).toBe(1);
  }
  expect(fftAutoFrameZoom(512, 512, [0.1, 0.1])).toBe(1);
  expect(fftAutoFrameZoom(0, 0, [0.1, 0.1], [0.4, 0.4])).toBe(1);
});
