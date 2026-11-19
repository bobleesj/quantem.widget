import { describe, expect, it } from "vitest";
import { formatNumber } from "../format";
import { formatDose, formatMean, formatStat } from "./statText";

describe("stats line", () => {
  it("shows a dash for a statistic that is not there yet, and never throws", () => {
    expect(formatStat(undefined)).toBe("–");
    expect(formatStat(null)).toBe("–");
    expect(formatMean(undefined, 0.04)).toBe("–");
    expect(formatNumber(undefined)).toBe("–");
  });

  it("formats by magnitude", () => {
    expect(formatStat(0)).toBe("0");
    expect(formatStat(0.187)).toBe("0.187");
    expect(formatStat(19)).toBe("19.00");
    expect(formatStat(2.73e-5)).toBe("2.73×10⁻⁵");
    expect(formatStat(13600)).toBe("1.36×10⁴");
    expect(formatStat(-27700)).toBe("-2.77×10⁴");
  });

  it("reads a mean far below the spread as zero", () => {
    expect(formatMean(2.44e-11, 0.04)).toBe("0");
    expect(formatMean(-3e-12, 0.04)).toBe("0");
  });

  it("keeps a mean that is small but real", () => {
    expect(formatMean(2.73e-5, 0.041)).toBe("2.73×10⁻⁵");
    expect(formatMean(1.31, 2.53)).toBe("1.31");
    expect(formatMean(2.44e-11, 0)).toBe("2.44×10⁻¹¹");
  });

  it("writes a large dose as two digits and a power of ten", () => {
    expect(formatDose(344612)).toBe("3.4×10⁵");
    expect(formatDose(2400)).toBe("2.4×10³");
    expect(formatDose(9960)).toBe("1.0×10⁴");
    expect(formatDose(3.2e12)).toBe("3.2×10¹²");
    expect(formatDose(243.4)).toBe("243");
    expect(formatDose(12.34)).toBe("12.3");
    expect(formatDose(0.123)).toBe("0.12");
  });
});
