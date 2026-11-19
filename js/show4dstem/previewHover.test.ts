import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { createPreviewHover } from "./previewHover";

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

function fixture() {
  const group = document.createElement("div");
  group.innerHTML = '<button data-preview-source="COM_row">CoM_r</button>'
    + '<button data-preview-source="COM_col">CoM_c</button><span>Flip</span>';
  const states: (string | null)[] = [];
  let source: string | null = null;
  const hover = createPreviewHover(update => {
    source = update(source);
    states.push(source);
  });
  return { group, hover, states };
}

it("moves between adjacent sources and across the CSS gap without restoring the selected image", () => {
  const { group, hover, states } = fixture();
  hover.show("COM_row");
  hover.leave("COM_row", group, group.firstElementChild!.nextElementSibling);
  hover.show("COM_col");
  hover.leave("COM_col", group, group);
  vi.advanceTimersByTime(25);
  hover.show("COM_row");
  vi.runAllTimers();
  expect(states).toEqual(["COM_row", "COM_col", "COM_row"]);
});

it("clears a parked pointer in the gap and clears immediately on the way out of the group", () => {
  const { group, hover, states } = fixture();
  hover.show("COM_row");
  hover.leave("COM_row", group, group.lastElementChild);
  expect(states).toEqual(["COM_row"]);
  vi.runAllTimers();
  expect(states).toEqual(["COM_row", null]);
  hover.show("COM_col");
  hover.leave("COM_col", group, group);
  hover.leaveGroup();
  hover.show("bf");
  hover.leave("bf", group, document.body);
  vi.runAllTimers();
  expect(states).toEqual(["COM_row", null, "COM_col", null, "bf", null]);
});

it("cancels a pending gap exit on click, blur, and unmount", () => {
  const { group, hover, states } = fixture();
  for (const clear of [() => hover.clear(), () => hover.clear("bf")]) {
    hover.show("bf");
    hover.leave("bf", group, group);
    clear();
    expect(vi.getTimerCount()).toBe(0);
  }
  hover.show("COM_row");
  hover.leave("COM_row", group, group);
  hover.cancel();
  vi.runAllTimers();
  expect(states).toEqual(["bf", null, "bf", null, "COM_row"]);
});
