import { describe, expect, it } from "vitest";
import { contrastOf, groupsOf, plain, signed, type Candidate } from "./search";

const row = (name: string, extra: Partial<Candidate> = {}): Candidate => ({
  name, stage: "geometry", thickness_nm: 40, focus_nm: 0, slices: 12, modes: 6, epochs: 4, state: "done", held_out: null, parent: null, mark: "", added: false, ...extra,
});

describe("ShowSearch numbers", () => {
  it("prints at most five decimals and never scientific notation", () => {
    expect(plain(0.459072)).toBe("0.45907");
    expect(plain(0.00000012)).toBe("0.00000");
    expect(plain(0.17951, 8)).toBe("0.17951");
    expect(plain(null)).toBe("–");
    expect(signed(5)).toBe("+5.0");
    expect(signed(-7.6e-7)).toBe("0.0");
    expect(signed(-5)).toBe("-5.0");
  });

  it("takes contrast from the 1st and 99.5th percentiles", () => {
    const values = Float32Array.from({ length: 1000 }, (_, index) => index);
    values[999] = 1e9;
    const [low, high] = contrastOf(values);
    expect(low).toBe(9);
    expect(high).toBe(994);
  });
});

describe("ShowSearch tiles", () => {
  it("folds a continuation of its own stage into its parent's tile, and keeps another stage's child apart", () => {
    const groups = groupsOf([
      row("pass1-007", { held_out: 0.19499 }),
      row("pass1-1003", { parent: "pass1-007", epochs: 2, held_out: 0.19480, mark: "best" }),
      row("pass2-000", { stage: "slices", parent: "pass1-1003", slices: 27, epochs: 12, state: "running" }),
    ], { "pass1-007": [[0, 0.4], [4, 0.19499]], "pass1-1003": [[4, 0.19499], [6, 0.1948]], "pass2-000": [[0, 0.22]] });
    expect(groups.map((group) => group.key)).toEqual(["pass1-007", "pass2-000"]);
    const [geometry, slices] = groups;
    expect(geometry.members).toEqual(["pass1-007", "pass1-1003"]);
    expect(geometry.landed).toBe("pass1-1003");
    expect(geometry.row.held_out).toBe(0.1948);
    expect(geometry.row.mark).toBe("best");
    expect(geometry.epochs).toBe(6);
    expect(geometry.points).toEqual([[0, 0.4], [4, 0.19499], [6, 0.1948]]);
    expect(slices.row.state).toBe("running");
    expect(slices.landed).toBe("pass2-000");
  });
});
