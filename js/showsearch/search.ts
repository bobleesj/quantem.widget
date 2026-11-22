// ShowSearch's arithmetic, apart from React so it is tested alone: plain numbers, contrast, and candidates grouped
// into tiles (a candidate that continues another of its own stage extends that tile instead of adding one).

export type Candidate = {
  name: string; stage: string; thickness_nm: number; focus_nm: number; slices: number; modes: number; epochs: number;
  state: string; held_out: number | null; parent: string | null; mark: string; added: boolean;
};

export type Group = { key: string; members: string[]; landed: string; row: Candidate; points: [number, number][]; epochs: number };

// At most five decimals, never scientific notation.
export function plain(value: number | null | undefined, digits = 5): string {
  if (value == null || !Number.isFinite(value)) return "–";
  return value.toFixed(Math.min(5, digits));
}

export function signed(value: number | null | undefined, digits = 1): string {
  if (value == null || !Number.isFinite(value)) return "–";
  const text = Math.abs(value) < 0.5 * 10 ** -digits ? (0).toFixed(digits) : value.toFixed(digits);
  return value > 0 && !/^0\.?0*$/.test(text) ? `+${text}` : text.replace(/^-(0\.?0*)$/, "$1");
}

// The 1st and 99.5th percentiles: a few hot pixels must not flatten the picture.
export function contrastOf(data: Float32Array): [number, number] {
  const finite = Float32Array.from(data.filter((value) => Number.isFinite(value))).sort();
  if (finite.length === 0) return [0, 1];
  const low = finite[Math.floor(0.01 * (finite.length - 1))], high = finite[Math.floor(0.995 * (finite.length - 1))];
  return high > low ? [low, high] : [low, low + 1];
}

// One tile a candidate, a continuation of the same stage folded into its parent's tile: its epochs extend the curve,
// its held-out and pictures replace the parent's, its mark (★, ▲) is the tile's.
export function groupsOf(candidates: Candidate[], curves: { [name: string]: [number, number][] }): Group[] {
  const byName = new Map(candidates.map((row) => [row.name, row]));
  const root = (row: Candidate): string => {
    let current = row;
    while (current.parent && byName.has(current.parent) && byName.get(current.parent)!.stage === current.stage) current = byName.get(current.parent)!;
    return current.name;
  };
  const groups = new Map<string, Group>();
  for (const row of candidates) {
    const key = root(row);
    const group = groups.get(key) || { key, members: [], landed: key, row: { ...row }, points: [], epochs: 0 };
    group.members.push(row.name);
    group.epochs += row.epochs;
    const merged = new Map<number, number>(group.points);
    for (const [epoch, value] of curves[row.name] || []) merged.set(epoch, value);
    group.points = [...merged.entries()].sort((a, b) => a[0] - b[0]);
    if (row.held_out != null) { group.landed = row.name; group.row = { ...group.row, held_out: row.held_out }; }
    if (row.mark) group.row = { ...group.row, mark: row.mark };
    if (row.state === "running" || (row.state !== "queued" && group.row.state === "queued")) group.row = { ...group.row, state: row.state };
    groups.set(key, group);
  }
  return [...groups.values()];
}
