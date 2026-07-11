import profileJson from "@/data/router-profile.json";

export type RouteLabel = "local_ok" | "escalate";

type RouterProfile = {
  schemaVersion: number;
  revision: string;
  modelType: "hashed-logistic-v1";
  dimension: number;
  bias: number;
  threshold: number;
  trainedExamples: number;
  weights: Record<string, number>;
  categoryPolicy?: {
    always_escalate?: string[];
    trust_local?: string[];
  };
  metrics: {
    accuracy: number | null;
    escalatePrecision: number | null;
    escalateRecall: number | null;
    localOkPrecision?: number | null;
    localOkRecall?: number | null;
    train?: RouterMetrics;
    calibration?: RouterMetrics;
    test?: RouterMetrics;
  };
};

type RouterMetrics = {
  count: number;
  accuracy: number;
  escalatePrecision: number;
  escalateRecall: number;
  localOkPrecision: number;
  localOkRecall: number;
  confusion: { tp: number; fp: number; tn: number; fn: number };
};

export type LocalRouteDecision = {
  label: RouteLabel;
  category: string;
  pEscalate: number;
  threshold: number;
  latencyMs: number;
  revision: string;
  trainedExamples: number;
  policyOverride: boolean;
};

export const routerProfile = profileJson as RouterProfile;

const CODE_HINT = /```|\bdef \w+|\bfunction\b|\breturn\b|\bclass \w+\(|=>|\bprintln\b|\bconsole\.log\b/i;
const DEBUG_HINT = /\bbug(s|gy)?\b|\bdebug\b|\bfix(es|ed|ing)?\b|\bincorrect(ly)?\b|\bdoesn'?t work\b|\bnot work(ing)?\b|\berror\b|\bwrong (output|result|answer)\b/i;
const CODEGEN_HINT = /\b(write|create|implement|develop|code|build)\b[^.?!]{0,60}\b(function|program|script|method|class|snippet|code)\b|\bfunction that\b|\bprogram that\b/i;
const MATH_HINT = /\bhow (many|much|far|long|old)\b|\bcalculate\b|\bcompute\b|\bwhat is \d|\bwhat'?s \d|\bpercent(age)?\b|%|\bsum\b|\bproduct of\b|\bdifference\b|\bremain(s|ing)?\b|\bleft over\b|\btotal\b|\baverage\b|\bper (hour|day|week|month|item|unit)\b|\bcost(s)?\b|\bprice\b|\bdiscount\b|\binterest\b|\bspeed\b|\bdistance\b|\barea\b|\bperimeter\b|\bvolume\b/i;
const LOGIC_HINT = /\beach (own|owns|has|have|like|likes|wear|wears|drink|drinks|play|plays)\b|\bdifferent (pet|color|colour|car|house|drink|sport|hobby|job|instrument|fruit)\b|\bwho (owns|has|likes|plays|wears|drinks|lives|sits)\b|\balways (lies|tells the truth)\b|\bliar\b|\btruth-?teller\b|\bif and only if\b|\bimplies\b|\bdeduce\b|\blogic puzzle\b|\b(sits|seated|stands) (next to|between|left of|right of)\b|\bcan we conclude\b|\bdoes it follow\b|\bnecessarily true\b|\bfinished (before|after)\b|\bwho (finished|came|arrived)\b|\btaller than\b|\bolder than\b|\bfaster than\b.*\bwho\b/i;

export function classifyPrompt(prompt: string): string {
  const normalized = prompt.toLowerCase();
  const hasCode = CODE_HINT.test(prompt);
  if (/\bsentiment\b|\bclassify\b[^.?!]{0,80}\b(review|tweet|comment|feedback|text|statement)\b/i.test(normalized)) return "sentiment";
  if (/\bsummar(y|ize|ise|ies|izing|ising|ization|isation)\b|\btl;?dr\b|\bcondense\b/i.test(normalized)) return "summarization";
  if (/\bentit(y|ies)\b|\bnamed entit/i.test(normalized) || /\bextract\b[^.?!]{0,80}\b(people|persons?|organi[sz]ations?|locations?|dates?|names)\b/i.test(normalized)) return "ner";
  if (hasCode && DEBUG_HINT.test(normalized)) return "debug";
  if (CODEGEN_HINT.test(normalized) || (hasCode && /\bwrite\b|\bimplement\b|\bcomplete\b/i.test(normalized))) return "codegen";
  if (LOGIC_HINT.test(normalized)) return "logic";
  if (/\d/.test(normalized) && MATH_HINT.test(normalized)) return "math";
  if (MATH_HINT.test(normalized) && /\b(twice|half|double|triple|one|two|three|four|five|six|seven|eight|nine|ten|dozen)\b/i.test(normalized)) return "math";
  return "factual";
}

function fnv1a(value: string): number {
  const bytes = new TextEncoder().encode(value);
  let hash = 0x811c9dc5;
  for (const byte of bytes) {
    hash ^= byte;
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash;
}

function featureCounts(prompt: string, category: string): Map<number, number> {
  const tokens = prompt.toLowerCase().match(/[a-z0-9_]+|[^\s\w]/gu)?.slice(0, 256) ?? [];
  const raw = [`category:${category}`, `length:${Math.min(8, Math.floor(Array.from(prompt).length / 80))}`];
  for (let index = 0; index < tokens.length; index += 1) {
    raw.push(`u:${tokens[index]}`);
    if (index > 0) raw.push(`b:${tokens[index - 1]}|${tokens[index]}`);
  }
  if (/\d/.test(prompt)) raw.push("shape:digits");
  if (CODE_HINT.test(prompt)) raw.push("shape:code");
  if (prompt.includes("?")) raw.push("shape:question");

  const counts = new Map<number, number>();
  for (const feature of raw) {
    const index = fnv1a(feature) % routerProfile.dimension;
    counts.set(index, (counts.get(index) ?? 0) + 1);
  }
  return counts;
}

export async function decideLocally(prompt: string): Promise<LocalRouteDecision> {
  const started = performance.now();
  const category = classifyPrompt(prompt);
  let logit = routerProfile.bias;
  for (const [index, count] of featureCounts(prompt, category)) {
    const weight = routerProfile.weights[String(index)] ?? 0;
    logit += weight * (1 + Math.log(count));
  }
  const pEscalate = 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, logit))));
  const policyOverride = routerProfile.categoryPolicy?.always_escalate?.includes(category) ?? false;
  await Promise.resolve();
  return {
    label: policyOverride || pEscalate >= routerProfile.threshold ? "escalate" : "local_ok",
    category,
    pEscalate,
    threshold: routerProfile.threshold,
    latencyMs: performance.now() - started,
    revision: routerProfile.revision,
    trainedExamples: routerProfile.trainedExamples,
    policyOverride,
  };
}
