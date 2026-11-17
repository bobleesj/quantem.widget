import { describe, expect, it } from "vitest";
import {
  CONTRAST_PERCENTILES, chosenPercentiles, contrastRangeChoice, handlePositions, percentilesLabel,
  percentilesOfChoice,
} from "./contrastRange";

describe("contrast range", () => {
  it("shows the full range unless a range is chosen", () => {
    expect(chosenPercentiles(null, false)).toBeNull();
    expect(chosenPercentiles(undefined, false)).toBeNull();
    expect(contrastRangeChoice(null, 0, 100)).toBe("full");
  });

  it("uses the chosen percentiles, and 1 to 99 for auto contrast", () => {
    expect(chosenPercentiles([2, 98], false)).toEqual([2, 98]);
    expect(chosenPercentiles([2, 98], true)).toEqual([2, 98]);
    expect(chosenPercentiles(null, true)).toEqual([1, 99]);
  });

  it("ignores a pair that is not a range", () => {
    for (const bad of [[98, 2], [5, 5], [-1, 50], [0, 101], [1], [1, 2, 3], ["a", "b"], "2,98"]) {
      expect(chosenPercentiles(bad, false)).toBeNull();
    }
  });

  it("offers the ranges of Browse and names them", () => {
    expect(CONTRAST_PERCENTILES.map(percentilesLabel)).toEqual(["0.5–99.5%", "2–98%", "5–95%", "10–90%"]);
  });

  it("round-trips every menu entry", () => {
    for (const pair of CONTRAST_PERCENTILES) {
      expect(percentilesOfChoice(contrastRangeChoice(pair, 0, 100))).toEqual(pair);
    }
    expect(percentilesOfChoice("full")).toBeNull();
    expect(percentilesOfChoice("handles")).toBeNull();
  });

  it("says when the handles were moved", () => {
    expect(contrastRangeChoice(null, 10, 100)).toBe("handles");
    expect(contrastRangeChoice(null, 0, 80)).toBe("handles");
  });

  it("places the handles on the window within the data's range", () => {
    expect(handlePositions(2, 6, 0, 10)).toEqual([20, 60]);
    expect(handlePositions(-5, 20, 0, 10)).toEqual([0, 100]);
    expect(handlePositions(1, 1, 3, 3)).toBeNull();
  });
});
