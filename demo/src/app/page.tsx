"use client";

import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";
import type { DemoStreamEvent } from "@/lib/demo-contracts";
import { decideLocally, type LocalRouteDecision, routerProfile } from "@/lib/local-router";

const categories = [
  ["factual", "Factual knowledge"], ["math", "Mathematical reasoning"],
  ["sentiment", "Sentiment"], ["summarization", "Summarization"],
  ["ner", "Entity extraction"], ["debug", "Code debugging"],
  ["logic", "Logical reasoning"], ["codegen", "Code generation"],
] as const;

const examples: Record<string, string> = {
  factual: "Explain what a stock market index is, giving one example.",
  math: "A store has 240 items. It sells 15% on Monday and 60 more on Tuesday. How many remain?",
  sentiment: "Classify the sentiment: The battery life is great, but the screen scratches too easily.",
  summarization: "Summarize in one sentence: Enterprises want to reduce AI costs without sacrificing answer quality.",
  ner: "Extract named entities: Maria Sanchez joined Fireworks AI in Berlin last March.",
  debug: "Find and fix the bug: def get_max(nums): return nums[0]",
  logic: "Sam does not own the bird. Jo owns the dog. Who owns the cat?",
  codegen: "Write a Python function that returns the second-largest distinct number in a list.",
};

type BaselineResult = Extract<DemoStreamEvent, { type: "baseline.completed" }>;
type AnswerResult = Extract<DemoStreamEvent, { type: "answer.completed" }>;
type ComparisonResult = Extract<DemoStreamEvent, { type: "comparison.completed" }>;

type StageStatus = "idle" | "running" | "done" | "error";

type RunView = {
  mode?: "live" | "simulation";
  localStatus: StageStatus;
  baselineStatus: StageStatus;
  answerStatus: StageStatus;
  local?: LocalRouteDecision;
  baseline?: BaselineResult;
  answer?: AnswerResult;
  comparison?: ComparisonResult;
  note?: string;
  error?: string;
};

type HistoryRow = {
  id: string;
  prompt: string;
  localLabel: string;
  baselineLabel: string;
  fineTokens: number | null;
  baselineTokens: number | null;
  saved: number | null;
  simulated: boolean;
};

type SessionTotals = {
  queries: number;
  fine: number;
  baseline: number;
  saved: number;
  fineUnknown: boolean;
  baselineUnknown: boolean;
  savedUnknown: boolean;
};

const EMPTY_RUN: RunView = {
  localStatus: "idle",
  baselineStatus: "idle",
  answerStatus: "idle",
};

const EMPTY_SESSION: SessionTotals = {
  queries: 0,
  fine: 0,
  baseline: 0,
  saved: 0,
  fineUnknown: false,
  baselineUnknown: false,
  savedUnknown: false,
};

function token(value: number | null | undefined): string {
  return value == null ? "N/A" : value.toLocaleString();
}

function label(value: string | undefined): string {
  if (!value) return "Waiting";
  return value === "local_ok" ? "KEEP LOCAL" : "ESCALATE";
}

function percent(value: number | null | undefined): string {
  return value == null ? "N/A" : `${value.toFixed(1)}%`;
}

function StageDot({ status }: { status: StageStatus }) {
  return <span className={`stage-dot ${status}`} aria-hidden="true" />;
}

function UsageLine({ usage }: { usage?: BaselineResult["usage"] }) {
  return (
    <div className="usage-line">
      <span>prompt {token(usage?.promptTokens)}</span>
      <span>completion {token(usage?.completionTokens)}</span>
      {usage?.reasoningTokens != null && <span>reasoning {token(usage.reasoningTokens)}</span>}
      <strong>total {token(usage?.totalTokens)}</strong>
    </div>
  );
}

export default function Home() {
  const [category, setCategory] = useState("math");
  const [prompt, setPrompt] = useState(examples.math);
  const [run, setRun] = useState<RunView>(EMPTY_RUN);
  const [history, setHistory] = useState<HistoryRow[]>([]);
  const [session, setSession] = useState<SessionTotals>(EMPTY_SESSION);
  const [loading, setLoading] = useState(false);
  const runSequence = useRef(0);
  const activeController = useRef<AbortController | null>(null);
  const routerReady = routerProfile.trainedExamples > 0;
  const routerTestMetrics = routerProfile.metrics.test;

  useEffect(() => () => activeController.current?.abort(), []);

  function selectCategory(id: string) {
    if (loading) return;
    setCategory(id);
    setPrompt(examples[id]);
    setRun(EMPTY_RUN);
  }

  function cancelRun() {
    const controller = activeController.current;
    if (!loading || !controller) return;
    runSequence.current += 1;
    controller.abort();
    activeController.current = null;
    setLoading(false);
    setRun((current) => ({
      ...current,
      localStatus: current.localStatus === "running" ? "error" : current.localStatus,
      baselineStatus:
        current.baselineStatus === "running" ? "error" : current.baselineStatus,
      answerStatus: current.answerStatus === "running" ? "error" : current.answerStatus,
      error: "Comparison cancelled.",
    }));
  }

  function resetSession() {
    setHistory([]);
    setSession({ ...EMPTY_SESSION });
  }

  function applyEvent(event: DemoStreamEvent) {
    if (event.type === "run.started") {
      setRun((current) => ({ ...current, mode: event.mode, note: event.note }));
    } else if (event.type === "baseline.started") {
      setRun((current) => ({ ...current, baselineStatus: "running" }));
    } else if (event.type === "baseline.completed") {
      setRun((current) => ({ ...current, baselineStatus: "done", baseline: event }));
    } else if (event.type === "answer.started") {
      setRun((current) => ({ ...current, answerStatus: "running" }));
    } else if (event.type === "answer.completed") {
      setRun((current) => ({ ...current, answerStatus: "done", answer: event }));
    } else if (event.type === "comparison.completed") {
      setRun((current) => ({ ...current, comparison: event }));
    } else if (event.type === "run.error") {
      setRun((current) => ({
        ...current,
        baselineStatus: event.stage === "baseline" || event.stage === "configuration" || current.baselineStatus === "running" ? "error" : current.baselineStatus,
        answerStatus: event.stage === "answer" || current.answerStatus === "running" ? "error" : current.answerStatus,
        error: event.message,
      }));
    }
  }

  async function runAgent(event?: FormEvent) {
    event?.preventDefault();
    if (loading || !prompt.trim() || !routerReady) return;
    const sequence = runSequence.current + 1;
    runSequence.current = sequence;
    activeController.current?.abort();
    const controller = new AbortController();
    activeController.current = controller;
    const submittedPrompt = prompt.trim();
    setLoading(true);
    setRun({ ...EMPTY_RUN, localStatus: "running" });
    let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;

    try {
      const local = await decideLocally(submittedPrompt);
      if (controller.signal.aborted || runSequence.current !== sequence) return;
      setRun((current) => ({ ...current, localStatus: "done", local, baselineStatus: "running" }));

      const response = await fetch("/api/demo", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/x-ndjson" },
        body: JSON.stringify({
          prompt: submittedPrompt,
          category: local.category,
          localDecision: {
            label: local.label,
            pEscalate: local.pEscalate,
            threshold: local.threshold,
            latencyMs: local.latencyMs,
            revision: local.revision,
          },
        }),
        signal: controller.signal,
      });
      if (!response.ok) {
        let message = "The comparison stream could not start.";
        try {
          const payload = (await response.json()) as { error?: unknown };
          if (typeof payload.error === "string") message = payload.error;
        } catch {
          // Keep the safe generic message for non-JSON platform errors.
        }
        throw new Error(message);
      }
      if (!response.body) throw new Error("The comparison stream returned no body.");

      reader = response.body.getReader();
      const decoder = new TextDecoder();
      let pending = "";
      let baselineResult: BaselineResult | undefined;
      let answerResult: AnswerResult | undefined;
      let comparisonResult: ComparisonResult | undefined;
      let streamError: Extract<DemoStreamEvent, { type: "run.error" }> | undefined;
      while (true) {
        const { done, value } = await reader.read();
        pending += decoder.decode(value, { stream: !done });
        const lines = pending.split("\n");
        pending = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.trim()) continue;
          const streamEvent = JSON.parse(line) as DemoStreamEvent;
          if (streamEvent.type === "baseline.completed") baselineResult = streamEvent;
          if (streamEvent.type === "answer.completed") answerResult = streamEvent;
          if (streamEvent.type === "comparison.completed") comparisonResult = streamEvent;
          if (streamEvent.type === "run.error") streamError = streamEvent;
          if (runSequence.current === sequence) applyEvent(streamEvent);
        }
        if (done) break;
      }
      if (pending.trim()) {
        const streamEvent = JSON.parse(pending) as DemoStreamEvent;
        if (streamEvent.type === "baseline.completed") baselineResult = streamEvent;
        if (streamEvent.type === "answer.completed") answerResult = streamEvent;
        if (streamEvent.type === "comparison.completed") comparisonResult = streamEvent;
        if (streamEvent.type === "run.error") streamError = streamEvent;
        if (runSequence.current === sequence) applyEvent(streamEvent);
      }
      if (streamError) return;
      if (!baselineResult || !answerResult || !comparisonResult) {
        throw new Error("The comparison stream ended before every stage completed.");
      }
      if (comparisonResult && runSequence.current === sequence) {
        const completed = comparisonResult;
        const historyRow: HistoryRow = {
          id: completed.runId,
          prompt: submittedPrompt,
          localLabel: local.label,
          baselineLabel: baselineResult.label,
          fineTokens: completed.policyTotals.fineTuned.totalTokens,
          baselineTokens: completed.policyTotals.promptBaseline.totalTokens,
          saved: completed.hypotheticalTokensSaved,
          simulated: completed.simulated,
        };
        setSession((current) => ({
          queries: current.queries + 1,
          fine: current.fine + (historyRow.fineTokens ?? 0),
          baseline: current.baseline + (historyRow.baselineTokens ?? 0),
          saved: current.saved + (historyRow.saved ?? 0),
          fineUnknown: current.fineUnknown || historyRow.fineTokens == null,
          baselineUnknown:
            current.baselineUnknown || historyRow.baselineTokens == null,
          savedUnknown: current.savedUnknown || historyRow.saved == null,
        }));
        setHistory((rows) => [historyRow, ...rows].slice(0, 8));
      }
    } catch (error) {
      const wasCancelled = controller.signal.aborted;
      if (wasCancelled) return;
      controller.abort();
      if (reader) {
        try {
          await reader.cancel();
        } catch {
          // The stream may already be closed after a platform error.
        }
      }
      setRun((current) => ({
        ...current,
        localStatus: current.localStatus === "running" ? "error" : current.localStatus,
        baselineStatus: current.baselineStatus === "running" ? "error" : current.baselineStatus,
        answerStatus: current.answerStatus === "running" ? "error" : current.answerStatus,
        error: error instanceof Error ? error.message : "The live walkthrough failed.",
      }));
    } finally {
      try {
        reader?.releaseLock();
      } catch {
        // The browser may release a cancelled stream automatically.
      }
      if (runSequence.current === sequence) {
        activeController.current = null;
        setLoading(false);
      }
    }
  }

  function handlePromptKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      void runAgent();
    }
  }

  const comparison = run.comparison;
  const fineTotal = comparison?.policyTotals.fineTuned.totalTokens;
  const baselineTotal = comparison?.policyTotals.promptBaseline.totalTokens;
  const maxTotal = Math.max(fineTotal ?? 0, baselineTotal ?? 0, 1);

  return (
    <main>
      <nav className="nav shell">
        <a className="brand" href="#top"><span className="brand-mark">GP</span><span>Agent</span></a>
        <div className="nav-links"><a href="#lab">Live lab</a><a href="#architecture">Architecture</a><a className="nav-cta" href="https://github.com/Ryo0326-hub/GP-AI-agent">View source ↗</a></div>
      </nav>

      <section className="hero shell" id="top">
        <div className="eyebrow"><span className="pulse" /> AMD Developer Hackathon · Track 1</div>
        <h1>Route locally.<br /><span>Spend only when needed.</span></h1>
        <p className="hero-copy">Watch a learned router make a zero-token decision in this browser, compare it with a paid prompt-based decision, and inspect provider-reported Fireworks usage in live mode.</p>
        <div className="hero-actions"><a className="button primary" href="#lab">Run the walkthrough <span>→</span></a><a className="button ghost" href="#architecture">See the system</a></div>
        <div className="metrics">
          <div><strong>18/19</strong><span>previous accuracy</span></div>
          <div><strong>97.14%</strong><span>v11 projected accuracy</span></div>
          <div><strong>9,685</strong><span>previous API tokens</span></div>
          <div><strong>1,824</strong><span>v11 expected API tokens</span></div>
          <div><strong>0</strong><span>v11 routing tokens</span></div>
          <div><strong>130/130</strong><span>tests passing</span></div>
        </div>
      </section>

      <section className="lab shell" id="lab">
        <div className="section-heading">
          <div><span className="kicker">Live routing lab</span><h2>See the decision cost.</h2></div>
          <p>Each run reveals the local route first, then performs the paid baseline classification, and finally returns an answer with exact provider usage when live.</p>
        </div>

        <div className={`router-readiness ${routerReady ? "ready" : "pending"}`}>
          <span className="status-dot" />
          <div><strong>{routerReady ? "Browser router loaded" : "Router training artifact pending"}</strong><small>{routerProfile.modelType} · {routerProfile.revision} · {routerProfile.trainedExamples} training labels{routerTestMetrics ? ` · held-out ${(routerTestMetrics.accuracy * 100).toFixed(1)}% acc / ${(routerTestMetrics.escalateRecall * 100).toFixed(1)}% escalation recall` : ""}</small></div>
          <b>{routerReady ? "ZERO FIREWORKS TOKENS" : "RUN DISABLED"}</b>
        </div>

        <div className="console-grid">
          <aside className="category-list" aria-label="Prompt examples">
            {categories.map(([id, categoryLabel], index) => (
              <button type="button" disabled={loading} aria-pressed={id === category} className={id === category ? "active" : ""} key={id} onClick={() => selectCategory(id)}>
                <span>0{index + 1}</span>{categoryLabel}<i>→</i>
              </button>
            ))}
          </aside>

          <form className="agent-console" onSubmit={runAgent}>
            <div className="console-top"><span className="status-dot" /> Walkthrough console <small>{run.local?.category ?? category}</small></div>
            <label htmlFor="prompt">Task prompt</label>
            <textarea id="prompt" disabled={loading} value={prompt} onChange={(event) => setPrompt(event.target.value)} onKeyDown={handlePromptKeyDown} rows={6} maxLength={8000} />
            {loading ? (
              <button type="button" className="run-button cancel-button" onClick={cancelRun}>
                <i className="spinner" aria-hidden="true" /> Cancel comparison
              </button>
            ) : (
              <button type="submit" className="run-button" disabled={!prompt.trim() || !routerReady}>
                Run through both routers <span>⌘/Ctrl ↵</span>
              </button>
            )}
            {run.error && <div className="run-error" role="alert">{run.error}</div>}
            {run.mode && <div className={`mode-banner ${run.mode}`}>{run.mode === "simulation" ? "Simulation mode — explicitly enable live mode after configuring Vercel credentials and the firewall." : "Live Fireworks calls — usage comes from the provider response."}</div>}
          </form>
        </div>

        <div className="timeline">
          <article className={`stage-card ${run.localStatus}`}>
            <div className="stage-head"><span>01</span><StageDot status={run.localStatus} /></div>
            <p className="stage-kicker">Learned compact router</p>
            <h3>{label(run.local?.label)}</h3>
            <div className="score-track"><i style={{ width: `${(run.local?.pEscalate ?? 0) * 100}%` }} /><b style={{ left: `${Math.min(100, (run.local?.threshold ?? .5) * 100)}%` }} /></div>
            <div className="stage-stats"><span>P(escalate) <strong>{run.local ? run.local.pEscalate.toFixed(3) : "—"}</strong></span><span>latency <strong>{run.local ? `${run.local.latencyMs.toFixed(2)} ms` : "—"}</strong></span></div>
            {run.local?.policyOverride && <div className="policy-note">Train-only category policy forces escalation for {run.local.category}.</div>}
            <div className="token-zero">0 <small>billed routing tokens</small></div>
          </article>

          <article className={`stage-card ${run.baselineStatus}`}>
            <div className="stage-head"><span>02</span><StageDot status={run.baselineStatus} /></div>
            <p className="stage-kicker">Prompt-based baseline</p>
            <h3>{label(run.baseline?.label)}</h3>
            <p className="stage-copy">{run.mode === "simulation" ? "A timed simulation demonstrates where the paid routing call appears." : run.mode === "live" ? "A real Fireworks call pays to answer the same routing question." : "The paid comparison starts after the local decision."}</p>
            <UsageLine usage={run.baseline?.usage} />
            <div className="model-line"><span>{run.baseline?.model ?? "Waiting for baseline"}</span><b>{run.baseline ? `${Math.round(run.baseline.latencyMs)} ms` : "—"}</b></div>
          </article>

          <article className={`stage-card answer-stage ${run.answerStatus}`}>
            <div className="stage-head"><span>03</span><StageDot status={run.answerStatus} /></div>
            <p className="stage-kicker">{run.mode === "simulation" ? "Simulated answer" : run.mode === "live" ? "Real Fireworks answer" : "Answer stage"}</p>
            <h3>{run.answerStatus === "error" ? "ERROR" : run.answerStatus === "done" ? (run.mode === "simulation" ? "SIMULATED" : "ANSWERED") : "WAITING"}</h3>
            <pre>{run.answer?.text ?? "The answer appears after both routing decisions are visible."}</pre>
            <UsageLine usage={run.answer?.usage} />
            <div className="model-line"><span>{run.answer?.model ?? "Answer model"}</span><b>{run.answer ? `${Math.round(run.answer.latencyMs)} ms` : "—"}</b></div>
          </article>
        </div>

        <div className={`comparison ${comparison ? "visible" : ""}`}>
          <div className="comparison-copy">
            <span className="kicker">Per-query policy comparison</span>
            {comparison && <div className={`agreement ${comparison.decisionsDisagree ? "disagree" : "agree"}`}>{comparison.decisionsDisagree ? "ROUTERS DISAGREE" : "ROUTERS AGREE"}</div>}
            <h3>{comparison ? `${token(comparison.hypotheticalTokensSaved)} tokens avoided` : "Run a query to compare"}</h3>
            <p>{comparison?.note ?? "Both bars share the same answer usage, isolating the cost of paying an LLM to route."}</p>
            <div className="saving"><strong>{percent(comparison?.hypotheticalSavedPercent)}</strong><span>fewer policy tokens</span></div>
          </div>
          <div className="bars">
            <div><header><span>Learned policy · shared answer</span><strong>{token(fineTotal)}</strong></header><i><b className="fine" style={{ width: `${((fineTotal ?? 0) / maxTotal) * 100}%` }} /></i><small>routing {token(comparison?.policyTotals.fineTuned.routingTokens)} · answer {token(comparison?.policyTotals.fineTuned.answerTokens)}</small></div>
            <div><header><span>Prompt-routed policy</span><strong>{token(baselineTotal)}</strong></header><i><b className="baseline" style={{ width: `${((baselineTotal ?? 0) / maxTotal) * 100}%` }} /></i><small>routing {token(comparison?.policyTotals.promptBaseline.routingTokens)} · answer {token(comparison?.policyTotals.promptBaseline.answerTokens)}</small></div>
            <div className="actual-spend"><span>Actual walkthrough API spend</span><strong>{token(comparison?.actualComparisonSpend.totalTokens)} tokens across {comparison?.actualComparisonSpend.callCount ?? 0} calls</strong></div>
          </div>
        </div>

        <div className="session-panel">
          <div className="session-summary">
            <span className="kicker">Session totals</span>
            <div><strong>{session.queries}</strong><span>queries</span></div>
            <div><strong>{token(session.fineUnknown ? null : session.fine)}</strong><span>learned policy</span></div>
            <div><strong>{token(session.baselineUnknown ? null : session.baseline)}</strong><span>prompt baseline</span></div>
            <div className="saved"><strong>{token(session.savedUnknown ? null : session.saved)}</strong><span>tokens avoided</span></div>
            <button type="button" onClick={resetSession} disabled={!session.queries}>Reset session</button>
          </div>
          <div className="history-table">
            <div className="history-head"><span>Query</span><span>Local</span><span>Baseline</span><span>Saved</span></div>
            {history.length ? history.map((item) => (
              <div className="history-row" key={item.id}><span title={item.prompt}>{item.prompt}</span><span>{label(item.localLabel)}</span><span>{label(item.baselineLabel)}</span><strong>{token(item.saved)}{item.simulated ? " sim" : ""}</strong></div>
            )) : <div className="history-empty">Your narrated walkthrough log will build here.</div>}
          </div>
        </div>
      </section>

      <section className="architecture shell" id="architecture">
        <div className="section-heading"><div><span className="kicker">Submission architecture</span><h2>Learn once. Route free.</h2></div><p>The router is trained on empirical local-model outcomes. At judge time it spends zero Fireworks tokens and keeps deterministic verification as a second quality gate.</p></div>
        <div className="flow">
          <article><span>01</span><div className="flow-icon">⌁</div><h3>Score locally</h3><p>The learned classifier estimates whether the local pipeline will succeed.</p></article>
          <div className="connector">→</div>
          <article><span>02</span><div className="flow-icon">◈</div><h3>Generate locally</h3><p>A quantized Qwen model handles the zero-token path inside the container.</p></article>
          <div className="connector">→</div>
          <article><span>03</span><div className="flow-icon">✓</div><h3>Verify</h3><p>Arithmetic, code, logic, and output constraints are checked deterministically.</p></article>
          <div className="connector branch">↗<small>only if needed</small></div>
          <article className="accent-card"><span>04</span><div className="flow-icon">✦</div><h3>Escalate</h3><p>An allowed Fireworks model repairs the small subset that needs more capability.</p></article>
        </div>
      </section>

      <section className="proof shell">
        <div><span className="kicker">Evidence before resubmission</span><h2>Measured under the real limits.</h2><p>The new image is accepted only after a disjoint accuracy check, 2-vCPU timing rehearsal, router class metrics, and provider-reported token comparison against the previous 9,685-token run.</p><a href="https://github.com/Ryo0326-hub/GP-AI-agent" className="text-link">Inspect the benchmark code →</a></div>
        <div className="proof-grid"><article><b>linux/amd64</b><span>Judge-compatible image</span></article><article><b>&lt; 10 GB</b><span>Compressed image gate</span></article><article><b>4 GB / 2 CPU</b><span>Matched label + rehearsal limits</span></article><article><b>&lt; 30 s</b><span>Per-task call deadline</span></article></div>
      </section>

      <footer className="shell"><div className="brand"><span className="brand-mark">GP</span><span>Agent</span></div><p>Built for AMD Developer Hackathon · Track 1</p><a href="#top">Back to top ↑</a></footer>
    </main>
  );
}
