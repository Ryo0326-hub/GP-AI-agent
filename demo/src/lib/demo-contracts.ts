export const ROUTER_LABELS = ["local_ok", "escalate"] as const;
export type RouterLabel = (typeof ROUTER_LABELS)[number];

export const DEMO_CATEGORIES = [
  "factual",
  "math",
  "sentiment",
  "summarization",
  "ner",
  "debug",
  "logic",
  "codegen",
] as const;
export type DemoCategory = (typeof DEMO_CATEGORIES)[number];

export interface LocalRouterDecision {
  label: RouterLabel;
  pEscalate: number;
  threshold: number;
  latencyMs: number;
  revision: string;
}

export interface DemoRunRequest {
  prompt: string;
  category: DemoCategory;
  localDecision: LocalRouterDecision;
}

/** Fireworks-reported usage. Null means the provider omitted that field. */
export interface TokenUsage {
  promptTokens: number | null;
  completionTokens: number | null;
  reasoningTokens: number | null;
  totalTokens: number | null;
}

export interface PolicyTokenTotal {
  routingTokens: number | null;
  answerTokens: number | null;
  totalTokens: number | null;
}

export type DemoMode = "live" | "simulation";

export type DemoStreamEvent =
  | {
      type: "run.started";
      runId: string;
      mode: DemoMode;
      simulated: boolean;
      comparisonScope: "router-overhead-only";
      localDecision: LocalRouterDecision;
      models: { baseline: string; answer: string };
      note: string;
    }
  | { type: "baseline.started"; runId: string }
  | {
      type: "baseline.completed";
      runId: string;
      simulated: boolean;
      label: RouterLabel;
      model: string;
      latencyMs: number;
      usage: TokenUsage;
    }
  | { type: "answer.started"; runId: string; model: string }
  | {
      type: "answer.completed";
      runId: string;
      simulated: boolean;
      text: string;
      model: string;
      latencyMs: number;
      usage: TokenUsage;
    }
  | {
      type: "comparison.completed";
      runId: string;
      simulated: boolean;
      decisionsDisagree: boolean;
      policyTotals: {
        fineTuned: PolicyTokenTotal;
        promptBaseline: PolicyTokenTotal;
      };
      hypotheticalTokensSaved: number | null;
      hypotheticalSavedPercent: number | null;
      actualComparisonSpend: {
        callCount: number;
        baselineRoutingTokens: number | null;
        answerTokens: number | null;
        totalTokens: number | null;
      };
      note: string;
    }
  | {
      type: "run.error";
      runId: string;
      stage: "configuration" | "baseline" | "answer" | "stream";
      code: string;
      message: string;
      retryable: boolean;
    };
