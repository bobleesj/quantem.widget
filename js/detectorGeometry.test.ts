import { describe, expect, it } from "vitest";

import { annulusMask, diskMask } from "./detectorGeometry";
import { buildDetectorMask } from "./.generated/engine/detector/compute/webgpu/backend";

describe("shared detector mask reference", () => {
  it("rasterizes an inclusive disk in row/column order", () => {
    expect(Array.from(diskMask(3, 5, 1, 2, 1))).toEqual([
      0, 0, 1, 0, 0,
      0, 1, 1, 1, 0,
      0, 0, 1, 0, 0,
    ]);
  });

  it("includes both annulus boundaries", () => {
    expect(Array.from(annulusMask(3, 5, 1, 2, 1, 1))).toEqual([
      0, 0, 1, 0, 0,
      0, 1, 0, 1, 0,
      0, 0, 1, 0, 0,
    ]);
    const values: Record<string, unknown> = {
      roi_center_row: 1,
      roi_center_col: 2,
      roi_mode: "annular",
      roi_radius_inner: 1,
      roi_radius: 1,
    };
    expect(Array.from(buildDetectorMask({ get: name => values[name] }, 3, 5))).toEqual(
      Array.from(annulusMask(3, 5, 1, 2, 1, 1)),
    );
  });

  it("supports fractional centers and nonsquare shapes", () => {
    const mask = diskMask(3, 5, 0.5, 1.5, Math.sqrt(0.5));
    expect(Array.from(mask)).toEqual([
      0, 1, 1, 0, 0,
      0, 1, 1, 0, 0,
      0, 0, 0, 0, 0,
    ]);
  });
});
