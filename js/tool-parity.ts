import registryJson from "../src/quantem/widget/tool_parity.json";

type ToolInput = string | string[] | null | undefined;

type WidgetConfig = {
  tool_groups: string[];
  aliases?: Record<string, string>;
};

type ControlPreset = {
  label: string;
  show_groups: string[];
};

type ToolParityRegistry = {
  widgets: Record<string, WidgetConfig>;
  control_presets: Record<string, ControlPreset>;
  viewer_widgets?: string[];
};

const REGISTRY = registryJson as ToolParityRegistry;

function getWidgetConfig(widgetName: string): WidgetConfig {
  const cfg = REGISTRY.widgets[widgetName];
  if (!cfg) {
    const supported = Object.keys(REGISTRY.widgets).sort().join(", ");
    throw new Error(`Unknown widget '${widgetName}'. Supported widgets: ${supported}.`);
  }
  return cfg;
}

function toValues(values: ToolInput): string[] {
  if (values == null) return [];
  if (typeof values === "string") return [values];
  return [...values];
}

function toCanonical(widgetName: string, value: string): string {
  const cfg = getWidgetConfig(widgetName);
  const aliases = cfg.aliases ?? {};
  const key = value.trim().toLowerCase();
  return aliases[key] ?? key;
}

export function getWidgetToolGroups(widgetName: string): string[] {
  return [...getWidgetConfig(widgetName).tool_groups];
}

export function normalizeToolGroups(widgetName: string, values: ToolInput): string[] {
  const groups = getWidgetToolGroups(widgetName);
  const groupSet = new Set(groups);
  const out: string[] = [];
  const seen = new Set<string>();
  for (const raw of toValues(values)) {
    const canonical = toCanonical(widgetName, String(raw));
    if (!canonical) continue;
    if (!groupSet.has(canonical)) {
      const supported = groups.map((g) => `"${g}"`).join(", ");
      throw new Error(`Unknown tool group '${raw}'. Supported values: ${supported}.`);
    }
    if (canonical === "all") return ["all"];
    if (!seen.has(canonical)) {
      seen.add(canonical);
      out.push(canonical);
    }
  }
  return out;
}

function orderedWithoutAll(widgetName: string, values: Set<string>): string[] {
  return getWidgetToolGroups(widgetName).filter((group) => group !== "all" && values.has(group));
}

export function expandToolGroups(widgetName: string, values: ToolInput): string[] {
  const normalized = normalizeToolGroups(widgetName, values);
  if (!normalized.includes("all")) return normalized;
  return getWidgetToolGroups(widgetName).filter((group) => group !== "all");
}

export function compactToolLabel(key: string): string {
  return key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (m) => m.toUpperCase());
}

export function getControlPresetIds(): string[] {
  return Object.keys(REGISTRY.control_presets);
}

export function getControlPresetLabel(presetId: string): string {
  const preset = REGISTRY.control_presets[presetId];
  return preset?.label ?? presetId;
}

export function resolvePresetHiddenTools(widgetName: string, presetId: string): string[] {
  const preset = REGISTRY.control_presets[presetId];
  if (!preset) {
    const supported = Object.keys(REGISTRY.control_presets).sort().join(", ");
    throw new Error(`Unknown control preset '${presetId}'. Supported presets: ${supported}.`);
  }
  const supportedGroups = getWidgetToolGroups(widgetName).filter((group) => group !== "all");
  if (preset.show_groups.includes("*")) return [];
  const show = new Set(preset.show_groups.map((g) => toCanonical(widgetName, g)));
  const hidden = supportedGroups.filter((group) => !show.has(group));
  return normalizeToolGroups(widgetName, hidden);
}

export type ToolVisibilityState = {
  hideAll: boolean;
  lockAll: boolean;
  isHidden: (group: string) => boolean;
  isLocked: (group: string) => boolean;
  hiddenSet: Set<string>;
  disabledSet: Set<string>;
};

export function computeToolVisibility(
  widgetName: string,
  disabledTools: ToolInput,
  hiddenTools: ToolInput,
): ToolVisibilityState {
  const hidden = normalizeToolGroups(widgetName, hiddenTools);
  const disabled = normalizeToolGroups(widgetName, disabledTools);
  const hiddenSet = new Set(hidden);
  const disabledSet = new Set(disabled);
  const hideAll = hiddenSet.has("all");
  const lockAll = hideAll || disabledSet.has("all");

  const isHidden = (group: string): boolean => {
    const canonical = toCanonical(widgetName, group);
    if (canonical === "all") return hideAll;
    return hideAll || hiddenSet.has(canonical);
  };

  const isLocked = (group: string): boolean => {
    const canonical = toCanonical(widgetName, group);
    if (canonical === "all") return lockAll;
    return lockAll || isHidden(canonical) || disabledSet.has(canonical);
  };

  return { hideAll, lockAll, isHidden, isLocked, hiddenSet, disabledSet };
}

export function addToolGroup(widgetName: string, current: ToolInput, group: string): string[] {
  const merged = new Set(expandToolGroups(widgetName, current));
  const canonical = toCanonical(widgetName, group);
  if (canonical === "all") return ["all"];
  merged.add(canonical);
  return orderedWithoutAll(widgetName, merged);
}

export function removeToolGroup(widgetName: string, current: ToolInput, group: string): string[] {
  const merged = new Set(expandToolGroups(widgetName, current));
  merged.delete(toCanonical(widgetName, group));
  return orderedWithoutAll(widgetName, merged);
}
