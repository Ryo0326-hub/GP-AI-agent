import { NextRequest, NextResponse } from "next/server";

import {
  DEMO_CATEGORIES,
  ROUTER_LABELS,
  type DemoCategory,
  type DemoRunRequest,
  type DemoStreamEvent,
  type LocalRouterDecision,
  type PolicyTokenTotal,
  type RouterLabel,
  type TokenUsage,
} from "@/lib/demo-contracts";
import {
  answerWithFireworks,
  classifyWithPromptBaseline,
  DemoProviderError,
  resolveDemoFireworksConfig,
  type DemoFireworksConfig,
} from "@/lib/fireworks.server";
import { decideLocally, routerProfile } from "@/lib/local-router";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

const MAX_PROMPT_CHARS = 8_000;
const MAX_REQUEST_BYTES = 24_000;

const simulatedAnswers: Record<DemoCategory, string> = {
  factual:
    "A stock market index tracks the performance of a selected group of stocks. The S&P 500 is one example.",
  math:
    "The store sells 36 items on Monday and 60 on Tuesday, leaving 144 items.",
  sentiment:
    "Mixed — the review praises battery life while criticizing screen durability.",
  summarization:
    "Local-first routing reduces external AI spend while preserving answer quality through selective escalation.",
  ner: "Maria Sanchez — Person\nFireworks AI — Organization\nBerlin — Location\nlast March — Date",
  debug: "```python\ndef get_max(nums):\n    return max(nums)\n```",
  logic: "Sam owns the cat; Jo owns the dog, so Lee must own the bird.",
  codegen:
    "```python\ndef second_largest(nums):\n    values = sorted(set(nums))\n    return values[-2] if len(values) >= 2 else None\n```",
};

type JsonRecord = Record<string, unknown>;

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isFiniteInRange(value: unknown, min: number, max: number): value is number {
  return (
    typeof value === "number" &&
    Number.isFinite(value) &&
    value >= min &&
    value <= max
  );
}

function validationError(message: string) {
  return NextResponse.json(
    { error: message, code: "invalid_demo_request" },
    {
      status: 400,
      headers: { "Cache-Control": "no-store" },
    },
  );
}

function parseLocalDecision(value: unknown): LocalRouterDecision | null {
  if (!isRecord(value)) return null;
  if (
    typeof value.label !== "string" ||
    !ROUTER_LABELS.includes(value.label as RouterLabel) ||
    !isFiniteInRange(value.pEscalate, 0, 1) ||
    !isFiniteInRange(value.threshold, 0, 1.01) ||
    !isFiniteInRange(value.latencyMs, 0, 60_000)
  ) {
    return null;
  }
  if (
    typeof value.revision !== "string" ||
    !value.revision.trim() ||
    value.revision.length > 128
  ) {
    return null;
  }
  return {
    label: value.label as RouterLabel,
    pEscalate: value.pEscalate,
    threshold: value.threshold,
    latencyMs: Math.round(value.latencyMs),
    revision: value.revision.trim(),
  };
}

function parseDemoRequest(value: unknown): DemoRunRequest | null {
  if (!isRecord(value) || typeof value.prompt !== "string") return null;
  const prompt = value.prompt.trim();
  if (!prompt || prompt.length > MAX_PROMPT_CHARS) return null;

  const localDecision = parseLocalDecision(value.localDecision);
  if (!localDecision) return null;

  if (
    typeof value.category !== "string" ||
    !DEMO_CATEGORIES.includes(value.category as DemoCategory)
  ) {
    return null;
  }

  return {
    prompt,
    category: value.category as DemoCategory,
    localDecision,
  };
}

const ROUTER_SCORE_EPSILON = 1e-12;

async function validateAndRecomputeRouterDecision(
  input: DemoRunRequest,
): Promise<DemoRunRequest> {
  if (
    routerProfile.trainedExamples <= 0 ||
    !routerProfile.revision ||
    /pending/i.test(routerProfile.revision)
  ) {
    throw new DemoProviderError(
      "router_not_ready",
      "The learned router artifact is not ready yet.",
      true,
    );
  }

  const recomputed = await decideLocally(input.prompt);
  if (!DEMO_CATEGORIES.includes(recomputed.category as DemoCategory)) {
    throw new DemoProviderError(
      "router_category_invalid",
      "The learned router returned an unsupported category.",
    );
  }

  const decisionMatches =
    input.category === recomputed.category &&
    input.localDecision.revision === recomputed.revision &&
    input.localDecision.label === recomputed.label &&
    Math.abs(input.localDecision.pEscalate - recomputed.pEscalate) <=
      ROUTER_SCORE_EPSILON &&
    Math.abs(input.localDecision.threshold - recomputed.threshold) <=
      ROUTER_SCORE_EPSILON;
  if (!decisionMatches) {
    throw new DemoProviderError(
      "router_decision_mismatch",
      "The browser router result is stale or invalid. Refresh the page and try again.",
      true,
    );
  }

  return {
    prompt: input.prompt,
    category: recomputed.category as DemoCategory,
    localDecision: {
      label: recomputed.label,
      pEscalate: recomputed.pEscalate,
      threshold: recomputed.threshold,
      latencyMs: recomputed.latencyMs,
      revision: recomputed.revision,
    },
  };
}

function nullableSum(...values: Array<number | null>): number | null {
  return values.every((value): value is number => value !== null)
    ? values.reduce<number>((sum, value) => sum + value, 0)
    : null;
}

function policyTotal(
  routingTokens: number | null,
  answerTokens: number | null,
): PolicyTokenTotal {
  return {
    routingTokens,
    answerTokens,
    totalTokens: nullableSum(routingTokens, answerTokens),
  };
}

function simulatedBaselineLabel(prompt: string): RouterLabel {
  return /```|\b(debug|algorithm|prove|puzzle|edge cases?|multi-step)\b/i.test(
    prompt,
  ) || prompt.length > 280
    ? "escalate"
    : "local_ok";
}

function simulatedUsage(prompt: string, completionTokens: number): TokenUsage {
  const promptTokens = Math.max(8, Math.round(prompt.length / 4));
  return {
    promptTokens,
    completionTokens,
    reasoningTokens: null,
    totalTokens: promptTokens + completionTokens,
  };
}

function safeProviderError(error: unknown): DemoProviderError {
  return error instanceof DemoProviderError
    ? error
    : new DemoProviderError(
        "unexpected_demo_error",
        "The live comparison could not be completed.",
        true,
      );
}

function pause(ms: number, signal: AbortSignal): Promise<void> {
  if (signal.aborted) {
    return Promise.reject(
      new DemoProviderError("demo_cancelled", "The demo request was cancelled."),
    );
  }
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(timer);
      reject(
        new DemoProviderError("demo_cancelled", "The demo request was cancelled."),
      );
    };
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

function ndjsonStream(
  request: NextRequest,
  input: DemoRunRequest,
  config: DemoFireworksConfig,
) {
  const encoder = new TextEncoder();
  const runId = crypto.randomUUID();
  const workController = new AbortController();
  let closed = false;
  let removeRequestAbortListener = () => {};

  const abortWork = () => {
    if (!workController.signal.aborted) workController.abort();
  };

  return new ReadableStream<Uint8Array>({
    start(controller) {
      const close = () => {
        if (closed) return;
        closed = true;
        removeRequestAbortListener();
        try {
          controller.close();
        } catch {
          // The consumer may already have cancelled the stream.
        }
      };
      const onRequestAbort = () => {
        abortWork();
        close();
      };
      if (request.signal.aborted) {
        onRequestAbort();
      } else {
        request.signal.addEventListener("abort", onRequestAbort, { once: true });
        removeRequestAbortListener = () =>
          request.signal.removeEventListener("abort", onRequestAbort);
      }

      const emit = (event: DemoStreamEvent) => {
        if (!closed && !workController.signal.aborted) {
          try {
            controller.enqueue(encoder.encode(`${JSON.stringify(event)}\n`));
          } catch {
            abortWork();
            closed = true;
            removeRequestAbortListener();
          }
        }
      };
      const fail = (
        stage: Extract<DemoStreamEvent, { type: "run.error" }>["stage"],
        error: unknown,
      ) => {
        const safe = safeProviderError(error);
        emit({
          type: "run.error",
          runId,
          stage,
          code: safe.code,
          message: safe.message,
          retryable: safe.retryable,
        });
        close();
      };

      void (async () => {
        emit({
          type: "run.started",
          runId,
          mode: config.mode,
          simulated: config.mode === "simulation",
          comparisonScope: "router-overhead-only",
          localDecision: input.localDecision,
          models: {
            baseline: config.baselineModel,
            answer: config.answerModel,
          },
          note:
            config.mode === "live"
              ? "The learned compact and prompt-based policy totals reuse one answer call so this comparison isolates routing overhead."
              : `Simulation mode: ${config.reason ?? "Fireworks is not configured."} No external API calls will be made.`,
        });

        emit({ type: "baseline.started", runId });

        let baseline: {
          label: RouterLabel;
          model: string;
          latencyMs: number;
          usage: TokenUsage;
        };
        if (config.mode === "simulation") {
          await pause(520, workController.signal);
          baseline = {
            label: simulatedBaselineLabel(input.prompt),
            model: config.baselineModel,
            latencyMs: 640,
            usage: simulatedUsage(input.prompt, 8),
          };
        } else {
          try {
            baseline = await classifyWithPromptBaseline(
              config,
              input.prompt,
              workController.signal,
            );
          } catch (error) {
            fail("baseline", error);
            return;
          }
        }
        emit({
          type: "baseline.completed",
          runId,
          simulated: config.mode === "simulation",
          label: baseline.label,
          model: baseline.model,
          latencyMs: baseline.latencyMs,
          usage: baseline.usage,
        });

        emit({ type: "answer.started", runId, model: config.answerModel });

        let answer: {
          text: string;
          model: string;
          latencyMs: number;
          usage: TokenUsage;
        };
        if (config.mode === "simulation") {
          await pause(760, workController.signal);
          const text = simulatedAnswers[input.category];
          answer = {
            text,
            model: config.answerModel,
            latencyMs: 1_180,
            usage: simulatedUsage(input.prompt, Math.max(16, Math.round(text.length / 4))),
          };
        } else {
          try {
            answer = await answerWithFireworks(
              config,
              input.prompt,
              input.category,
              workController.signal,
            );
          } catch (error) {
            fail("answer", error);
            return;
          }
        }
        emit({
          type: "answer.completed",
          runId,
          simulated: config.mode === "simulation",
          text: answer.text,
          model: answer.model,
          latencyMs: answer.latencyMs,
          usage: answer.usage,
        });

        const baselineRoutingTokens = baseline.usage.totalTokens;
        const answerTokens = answer.usage.totalTokens;
        const fineTuned = policyTotal(0, answerTokens);
        const promptBaseline = policyTotal(baselineRoutingTokens, answerTokens);
        const hypotheticalTokensSaved =
          fineTuned.totalTokens !== null && promptBaseline.totalTokens !== null
            ? promptBaseline.totalTokens - fineTuned.totalTokens
            : null;
        const hypotheticalSavedPercent =
          hypotheticalTokensSaved !== null &&
          promptBaseline.totalTokens !== null &&
          promptBaseline.totalTokens > 0
            ? Math.round(
                (hypotheticalTokensSaved / promptBaseline.totalTokens) * 10_000,
              ) / 100
            : null;

        emit({
          type: "comparison.completed",
          runId,
          simulated: config.mode === "simulation",
          decisionsDisagree: baseline.label !== input.localDecision.label,
          policyTotals: { fineTuned, promptBaseline },
          hypotheticalTokensSaved,
          hypotheticalSavedPercent,
          actualComparisonSpend:
            config.mode === "live"
              ? {
                  callCount: 2,
                  baselineRoutingTokens,
                  answerTokens,
                  totalTokens: nullableSum(baselineRoutingTokens, answerTokens),
                }
              : {
                  callCount: 0,
                  baselineRoutingTokens: 0,
                  answerTokens: 0,
                  totalTokens: 0,
                },
          note:
            config.mode === "live"
              ? "Policy totals are hypothetical. Actual comparison spend includes both the baseline classification call and the shared answer call."
              : "All policy token figures are simulated; actual external spend is zero.",
        });
        close();
      })().catch((error: unknown) => fail("stream", error));
    },
    cancel() {
      closed = true;
      removeRequestAbortListener();
      abortWork();
    },
  });
}

export async function POST(request: NextRequest) {
  const contentType = request.headers.get("content-type") ?? "";
  if (!contentType.toLowerCase().includes("application/json")) {
    return NextResponse.json(
      { error: "Content-Type must be application/json.", code: "unsupported_media_type" },
      { status: 415, headers: { "Cache-Control": "no-store" } },
    );
  }

  const declaredLength = Number(request.headers.get("content-length") ?? 0);
  if (Number.isFinite(declaredLength) && declaredLength > MAX_REQUEST_BYTES) {
    return NextResponse.json(
      { error: "The demo request is too large.", code: "request_too_large" },
      { status: 413, headers: { "Cache-Control": "no-store" } },
    );
  }

  let rawBody: string;
  try {
    rawBody = await request.text();
  } catch {
    return validationError("The request body could not be read.");
  }
  if (new TextEncoder().encode(rawBody).byteLength > MAX_REQUEST_BYTES) {
    return NextResponse.json(
      { error: "The demo request is too large.", code: "request_too_large" },
      { status: 413, headers: { "Cache-Control": "no-store" } },
    );
  }

  let body: unknown;
  try {
    body = JSON.parse(rawBody);
  } catch {
    return validationError("The request body must be valid JSON.");
  }
  const parsedInput = parseDemoRequest(body);
  if (!parsedInput) {
    return validationError(
      "Provide a non-empty prompt (up to 8,000 characters), category, and valid versioned localDecision.",
    );
  }

  let input: DemoRunRequest;
  try {
    input = await validateAndRecomputeRouterDecision(parsedInput);
  } catch (error) {
    const safe = safeProviderError(error);
    return NextResponse.json(
      { error: safe.message, code: safe.code, retryable: safe.retryable },
      {
        status: safe.code === "router_decision_mismatch" ? 409 : 503,
        headers: { "Cache-Control": "no-store" },
      },
    );
  }

  let config: DemoFireworksConfig;
  try {
    config = resolveDemoFireworksConfig();
  } catch (error) {
    const safe = safeProviderError(error);
    return NextResponse.json(
      { error: safe.message, code: safe.code, retryable: safe.retryable },
      { status: 503, headers: { "Cache-Control": "no-store" } },
    );
  }

  return new Response(ndjsonStream(request, input, config), {
    status: 200,
    headers: {
      "Content-Type": "application/x-ndjson; charset=utf-8",
      "Cache-Control": "no-store, no-cache, must-revalidate",
      "X-Accel-Buffering": "no",
      "X-Content-Type-Options": "nosniff",
    },
  });
}
