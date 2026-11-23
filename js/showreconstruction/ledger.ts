// The status view's arithmetic, apart from React so it is tested alone: every setting a stage changed, as one row
// across the stages in the order they ran, with the value each setting had at each stage. A stage that was tried and
// refused shows what it tried and leaves the value as it was.

export type Change = { key: string; before: string; after: string };
export type LedgerStage = { name: string; refused: boolean; changed: Change[] };
export type Cell = { value: string; before: string | null; changed: boolean; refused: boolean };
export type Row = { key: string; part: string; name: string; cells: Cell[]; now: string; moved: boolean };
export type Group = { part: string; rows: Row[] };

// A stored value as a person reads it: at most five decimals and four significant figures, never an exponent;
// None is "–", a boolean "on" or "off", an array training holds "learned".
export function shown(value: string | null | undefined): string {
  if (value == null || value === "None" || value === "") return "–";
  if (value === "True") return "on";
  if (value === "False") return "off";
  if (value === "tensor") return "learned";
  const number = Number(value);
  if (!Number.isFinite(number) || !/^[-+0-9.eE]+$/.test(value.trim())) return value;
  if (Number.isInteger(number)) return String(number);
  const digits = Math.max(0, Math.min(5, 3 - Math.floor(Math.log10(Math.abs(number)))));
  return number.toFixed(digits).replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, "");
}

/** The rows of the ledger, grouped by part (object, probe, scan, ...) in the order a setting first changed. settings is
 *  the newest run's whole settings, when the view has them: settings no stage changed become rows with moved false. */
export function ledger(stages: LedgerStage[], settings: { [key: string]: string } = {}): Group[] {
  const keys: string[] = [];
  for (const stage of stages) for (const change of stage.changed) if (!keys.includes(change.key)) keys.push(change.key);
  const still = Object.keys(settings).filter((key) => !keys.includes(key));
  const rows: Row[] = [...keys, ...still].map((key) => {
    const first = stages.flatMap((stage) => stage.changed).find((change) => change.key === key);
    let current = first ? first.before : settings[key];
    const cells = stages.map((stage) => {
      const change = stage.changed.find((one) => one.key === key);
      if (!change) return { value: current, before: null, changed: false, refused: false };
      if (stage.refused) return { value: change.after, before: change.before, changed: true, refused: true };
      current = change.after;
      return { value: change.after, before: change.before, changed: true, refused: false };
    });
    const dot = key.indexOf(".");
    return { key, part: dot < 0 ? "run" : key.slice(0, dot), name: dot < 0 ? key : key.slice(dot + 1), cells, now: current, moved: !!first };
  });
  const groups: Group[] = [];
  for (const row of rows) {
    const group = groups.find((one) => one.part === row.part);
    if (group) group.rows.push(row); else groups.push({ part: row.part, rows: [row] });
  }
  return groups;
}

export type Segment = { from: number; to: number; value: string; before: string | null; changed: boolean };

/** A row as bars: each run of stages over which the setting held one value, from the stage that set it to the stage
 *  before the next change (refused stages do not break a bar: the value stayed). changed is false for the value the
 *  row started with. A value of None draws no bar: the setting was off. */
export function segments(row: Row): Segment[] {
  const out: Segment[] = [];
  row.cells.forEach((cell, index) => {
    const starts = cell.changed && !cell.refused;
    if (!out.length || starts) out.push({ from: index, to: index, value: starts ? cell.value : cell.value, before: starts ? cell.before : null, changed: starts });
    else out[out.length - 1].to = index;
  });
  return out.filter((segment) => !(segment.value == null || segment.value === "None" || segment.value === ""));
}
