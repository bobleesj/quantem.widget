import { describe, expect, it } from "vitest";
import { chosenFrameIndex, createScanChoice } from "./scanChoice";

function harness() {
  const sent: string[] = [];
  const timers: (() => void)[] = [];
  const choice = createScanChoice(source => sent.push(source), run => timers.push(run));
  return { sent, timers, choice };
}

describe("scan choice", () => {
  it("sends one choice at a time and keeps only the newest one waiting", () => {
    const { sent, choice } = harness();
    choice.choose("b", "a");
    choice.choose("c", "a");
    choice.choose("d", "a");
    expect(sent).toEqual(["b"]);
    expect(choice.pending()).toBe("d");
    choice.shown("b");
    expect(sent).toEqual(["b", "d"]);
    choice.shown("d");
    expect(choice.pending()).toBeNull();
  });

  it("three quick steps from one scan reach the third scan", () => {
    const { sent, choice } = harness();
    const identities = ["a", "b", "c", "d"];
    let shown = 0;
    for (let step = 0; step < 3; step++) {
      const next = chosenFrameIndex(choice.pending(), identities, shown) + 1;
      choice.choose(identities[next], identities[shown]);
    }
    expect(choice.pending()).toBe("d");
    shown = 1; choice.shown("b");
    shown = 3; choice.shown("d");
    expect(sent).toEqual(["b", "d"]);
    expect(chosenFrameIndex(choice.pending(), identities, shown)).toBe(3);
  });

  it("an answer for another scan does not end the wait", () => {
    const { sent, choice } = harness();
    choice.choose("c", "a");
    choice.shown("a");
    expect(choice.pending()).toBe("c");
    expect(sent).toEqual(["c"]);
  });

  it("does not ask for the scan already shown", () => {
    const { sent, choice } = harness();
    choice.choose("a", "a");
    expect(sent).toEqual([]);
    expect(choice.pending()).toBeNull();
  });

  it("stops waiting for a scan that is never shown and sends the newest choice", () => {
    const { sent, timers, choice } = harness();
    choice.choose("gone", "a");
    choice.choose("b", "a");
    timers[0]();
    expect(sent).toEqual(["gone", "b"]);
    timers[1]();
    expect(choice.pending()).toBeNull();
  });

  it("a late timer of an answered choice changes nothing", () => {
    const { sent, timers, choice } = harness();
    choice.choose("b", "a");
    choice.shown("b");
    choice.choose("c", "b");
    timers[0]();
    expect(choice.pending()).toBe("c");
    expect(sent).toEqual(["b", "c"]);
  });

  it("a chosen scan that was released falls back to the shown one", () => {
    expect(chosenFrameIndex("gone", ["a", "b"], 1)).toBe(1);
    expect(chosenFrameIndex(null, ["a", "b"], 0)).toBe(0);
  });
});
