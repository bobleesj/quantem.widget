import { describe, expect, it } from "vitest";
import { formatMean, formatStat } from "./statText";

describe("stats line", () => {
  it("formats by magnitude", () => {
    expect(formatStat(0)).toBe("0");
    expect(formatStat(0.187)).toBe("0.187");
    expect(formatStat(19)).toBe("19.00");
    expect(formatStat(2.73e-5)).toBe("2.73e-5");
  });

  it("reads a mean far below the spread as zero", () => {
    expect(formatMean(2.44e-11, 0.04)).toBe("0");
    expect(formatMean(-3e-12, 0.04)).toBe("0");
  });

  it("keeps a mean that is small but real", () => {
    expect(formatMean(2.73e-5, 0.041)).toBe("2.73e-5");
    expect(formatMean(1.31, 2.53)).toBe("1.31");
    expect(formatMean(2.44e-11, 0)).toBe("2.44e-11");
  });
});
