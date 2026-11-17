import { expect, test, vi } from 'vitest';
import { DefocusPreview } from './defocusPreview';

test('the fitted interval can draw while the wider grid fills without duplicate planes', () => {
  vi.stubGlobal('GPUBufferUsage', { STORAGE: 1, COPY_DST: 2, UNIFORM: 4 });
  try {
    const device = {createComputePipeline: vi.fn(), createShaderModule: vi.fn(),
      createBuffer: () => ({ destroy: vi.fn() }), queue: {writeBuffer: vi.fn()}};
    const cache = new DefocusPreview(device as unknown as GPUDevice, -31, 63, 2, 1, 'optics');
    const plane = new DataView(new ArrayBuffer(4));
    cache.add(18, plane);
    expect(cache.covers(-12.5)).toBe(false);
    cache.add(19, plane);
    expect(cache.covers(-12.5)).toBe(true);
    expect(cache.covers(20)).toBe(false);
    while (cache.loaded < cache.count) {
      const {index, length} = cache.nextBatch(20);
      expect(length).toBeGreaterThan(0);
      expect(length).toBeLessThanOrEqual(3);
      for (let offset = 0; offset < length; offset++) cache.add(index + offset, plane);
    }
    expect(cache.covers(-30)).toBe(true);
    expect(cache.covers(30)).toBe(true);
    expect(cache.covers(31)).toBe(false);
    cache.destroy();
  } finally {
    vi.unstubAllGlobals();
  }
});
