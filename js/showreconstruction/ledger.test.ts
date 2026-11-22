import { describe, expect, it } from "vitest";
import { ledger, shown } from "./ledger";

describe("status ledger", () => {
  const stages = [
    { name: "search", refused: false, changed: [] },
    { name: "10 modes", refused: false, changed: [{ key: "probe.modes", before: "8", after: "10" }] },
    { name: "slab 450 Å", refused: false, changed: [{ key: "object.thickness_A", before: "400.0", after: "450.0" }] },
    { name: "slab 500 Å", refused: true, changed: [{ key: "object.thickness_A", before: "450.0", after: "500.0" }] },
    { name: "slab 425 Å", refused: false, changed: [{ key: "object.thickness_A", before: "450.0", after: "425" }] },
  ];

  it("carries a value forward from the stage that set it, and a refused stage leaves it as it was", () => {
    const groups = ledger(stages);
    expect(groups.map((group) => group.part)).toEqual(["probe", "object"]);
    const thickness = groups[1].rows[0];
    expect(thickness.cells.map((cell) => cell.value)).toEqual(["400.0", "400.0", "450.0", "500.0", "425"]);
    expect(thickness.cells.map((cell) => cell.changed)).toEqual([false, false, true, true, true]);
    expect(thickness.cells[3].refused).toBe(true);
    expect(thickness.cells[4].before).toBe("450.0");  // the refused 500 never became the value
    expect(thickness.now).toBe("425");
    expect(groups[0].rows[0].cells.map((cell) => cell.value)).toEqual(["8", "10", "10", "10", "10"]);
  });

  it("lists a setting no stage changed after the ones that moved, with its one value", () => {
    const groups = ledger(stages, { "object.thickness_A": "425", "object.slices": "27", "scan.lr": "None" });
    const slices = groups.find((group) => group.part === "object")!.rows[1];
    expect(slices.moved).toBe(false);
    expect(slices.cells.every((cell) => cell.value === "27" && !cell.changed)).toBe(true);
    expect(groups.map((group) => group.part)).toEqual(["probe", "object", "scan"]);
  });

  it("writes values as a person reads them: five decimals at most, no exponent", () => {
    expect(shown("0.0019754888426523187")).toBe("0.00198");  // five decimals at most, as every number here
    expect(shown("0.07905694150420949")).toBe("0.07906");
    expect(shown("450.0")).toBe("450");
    expect(shown("425")).toBe("425");
    expect(shown("None")).toBe("–");
    expect(shown("True")).toBe("on");
    expect(shown("tensor")).toBe("learned");
    expect(shown("1e-06")).toBe("0");
  });
});
