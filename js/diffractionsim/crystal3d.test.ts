import { describe, expect, it } from "vitest";

import { cellGeometry } from "./crystal3d";
import { parseCrystal } from "./physics";
import { PRESETS } from "../diffractionsim-web/presets";

function geometry(name: string) {
  return cellGeometry(parseCrystal(PRESETS[name])!, [1, 1, 1], true);
}

function edges(verts: number[][]): number[] {
  return verts.map((v, i) => {
    const w = verts[(i + 1) % verts.length];
    return Math.hypot(v[0] - w[0], v[1] - w[1], v[2] - w[2]);
  });
}

describe("diffsim coordination polyhedra", () => {
  it("draws Si tetrahedra on one fcc sublattice, inside the cell", () => {
    const g = geometry("Si (diamond cubic)");
    // 4 tetrahedra around the interior sublattice, with vertices on the drawn corner and face atoms
    expect(g.faces.length).toBe(16);
    for (const face of g.faces) {
      for (const e of edges(face.verts)) expect(e).toBeCloseTo(5.431 / Math.SQRT2, 2);
      for (const v of face.verts) {
        expect(g.atoms.some((a) => Math.hypot(a.pos[0] - v[0], a.pos[1] - v[1], a.pos[2] - v[2]) < 1e-6)).toBe(true);
      }
    }
  });

  it("draws TiO6 octahedra in SrTiO3, without Sr", () => {
    const g = geometry("SrTiO3 (perovskite)");
    expect(g.faces.length).toBe(8);
    for (const face of g.faces) {
      expect(face.verts.length).toBe(3);
      for (const e of edges(face.verts)) expect(e).toBeCloseTo(3.905 / Math.SQRT2, 2);
    }
  });

  it("draws no polyhedra for an fcc metal", () => {
    expect(geometry("Al (fcc)").faces.length).toBe(0);
  });
});
