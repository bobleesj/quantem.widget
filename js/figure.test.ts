import { describe, expect, it } from "vitest";

import { formatScaleLabel, formatZoomLabel, unitSymbol } from "./figure";

describe("formatZoomLabel", () => {
  it("uses one decimal and the multiplication sign", () => {
    expect(formatZoomLabel(1)).toBe("1.0×");
    expect(formatZoomLabel(2)).toBe("2.0×");
    expect(formatZoomLabel(5.06)).toBe("5.1×");
  });
});

describe("formatScaleLabel", () => {
  it("prints a length as a whole number in the unit that reads cleanest", () => {
    expect(formatScaleLabel(0.5, "nm")).toBe("5 Å");
    expect(formatScaleLabel(0.005, "nm")).toBe("5 pm");
    expect(formatScaleLabel(20, "Å")).toBe("2 nm");
    expect(formatScaleLabel(5, "angstroms")).toBe("5 Å");
    expect(formatScaleLabel(1500, "um")).toBe("2 mm");
    expect(formatScaleLabel(0.2, "μm")).toBe("200 nm");
  });

  it("rescales every spelling of a length unit, meters included", () => {
    // these spellings printed "0.00 m" or kept a decimal in the given unit before
    expect(formatScaleLabel(1e-9, "m")).toBe("1 nm");
    expect(formatScaleLabel(0.002, "millimeter")).toBe("2 µm");
    expect(formatScaleLabel(0.05, "micrometers")).toBe("50 nm");
  });

  it("keeps a unit that is not a length", () => {
    expect(formatScaleLabel(5, "mrad")).toBe("5 mrad");
    expect(formatScaleLabel(0.2, "ps")).toBe("0.20 ps");
    expect(formatScaleLabel(50, "px")).toBe("50 px");
  });
});

describe("unitSymbol", () => {
  it("labels uncalibrated axes px, as every widget does", () => {
    expect(unitSymbol("pixels")).toBe("px");
    expect(unitSymbol("Pixel")).toBe("px");
    expect(formatScaleLabel(20, "pixels")).toBe("20 px");
  });
});
