"use client";

import { FormEvent, useState } from "react";

const categories = [
  ["factual", "Factual knowledge"], ["math", "Mathematical reasoning"],
  ["sentiment", "Sentiment"], ["summarization", "Summarization"],
  ["ner", "Entity extraction"], ["debug", "Code debugging"],
  ["logic", "Logical reasoning"], ["codegen", "Code generation"],
];

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

export default function Home() {
  const [category, setCategory] = useState("math");
  const [prompt, setPrompt] = useState(examples.math);
  const [answer, setAnswer] = useState("");
  const [source, setSource] = useState("");
  const [loading, setLoading] = useState(false);
  const selectedLabel = categories.find(([id]) => id === category)?.[1];

  function selectCategory(id: string) {
    setCategory(id);
    setPrompt(examples[id]);
    setAnswer("");
  }

  async function runAgent(event: FormEvent) {
    event.preventDefault();
    setLoading(true);
    setAnswer("");
    try {
      const response = await fetch("/api/demo", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, category }),
      });
      const data = await response.json();
      setAnswer(data.answer ?? data.error ?? "No answer returned.");
      setSource(`${data.source ?? "Agent"} · ${data.model ?? selectedLabel}`);
    } catch {
      setAnswer("The demo endpoint is unavailable. Please try again.");
      setSource("Connection error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <nav className="nav shell">
        <a className="brand" href="#top"><span className="brand-mark">GP</span><span>Agent</span></a>
        <div className="nav-links"><a href="#architecture">Architecture</a><a href="#benchmarks">Benchmarks</a><a className="nav-cta" href="https://github.com/Ryo0326-hub/GP-AI-agent">View source ↗</a></div>
      </nav>

      <section className="hero shell" id="top">
        <div className="eyebrow"><span className="pulse" /> AMD Developer Hackathon · Track 1</div>
        <h1>Reason locally.<br /><span>Escalate intelligently.</span></h1>
        <p className="hero-copy">A general-purpose AI agent that protects answer quality while minimizing metered model tokens through local inference, deterministic verification, and adaptive routing.</p>
        <div className="hero-actions"><a className="button primary" href="#playground">Try the agent <span>→</span></a><a className="button ghost" href="#architecture">See how it works</a></div>
        <div className="metrics" id="benchmarks">
          <div><strong>94.7%</strong><span>acceptance score</span></div>
          <div><strong>8</strong><span>capability domains</span></div>
          <div><strong>49s</strong><span>19-task run</span></div>
          <div><strong>54/54</strong><span>tests passing</span></div>
        </div>
      </section>

      <section className="playground shell" id="playground">
        <div className="section-heading"><div><span className="kicker">Interactive demo</span><h2>One agent. Eight kinds of work.</h2></div><p>Choose a capability, edit the prompt, and see how the routing layer responds.</p></div>
        <div className="console-grid">
          <aside className="category-list">
            {categories.map(([id, label], index) => <button className={id === category ? "active" : ""} key={id} onClick={() => selectCategory(id)}><span>0{index + 1}</span>{label}<i>→</i></button>)}
          </aside>
          <form className="agent-console" onSubmit={runAgent}>
            <div className="console-top"><span className="status-dot" /> Agent console <small>{selectedLabel}</small></div>
            <label htmlFor="prompt">Task prompt</label>
            <textarea id="prompt" value={prompt} onChange={(event) => setPrompt(event.target.value)} rows={6} />
            <div className="route"><span>Classifier</span><b>→</b><span>Local model</span><b>→</b><span>Verifier</span><b>→</b><span className="route-accent">Selective escalation</span></div>
            <button className="run-button" disabled={loading || !prompt.trim()}>{loading ? <><i className="spinner" /> Running agent…</> : <>Run agent <span>⌘ ↵</span></>}</button>
            <div className={`answer ${answer ? "visible" : ""}`}><div><span>Answer</span><small>{source}</small></div><pre>{answer || "Your result will appear here."}</pre></div>
          </form>
        </div>
      </section>

      <section className="architecture shell" id="architecture">
        <div className="section-heading"><div><span className="kicker">System design</span><h2>Accuracy first. Tokens second.</h2></div><p>The router only trusts answers it can verify, keeping local inference useful without gambling on the accuracy gate.</p></div>
        <div className="flow">
          <article><span>01</span><div className="flow-icon">⌁</div><h3>Classify</h3><p>Route each prompt into one of eight specialized capability paths.</p></article>
          <div className="connector">→</div>
          <article><span>02</span><div className="flow-icon">◈</div><h3>Generate locally</h3><p>Run a quantized 3B model inside the container at zero scored tokens.</p></article>
          <div className="connector">→</div>
          <article><span>03</span><div className="flow-icon">✓</div><h3>Verify</h3><p>Check arithmetic, code, constraints, entities, and confidence deterministically.</p></article>
          <div className="connector branch">↗<small>unverified only</small></div>
          <article className="accent-card"><span>04</span><div className="flow-icon">✦</div><h3>Escalate</h3><p>Call an allowed Fireworks model only when quality needs reinforcement.</p></article>
        </div>
      </section>

      <section className="proof shell">
        <div><span className="kicker">Built for the harness</span><h2>Not a prototype.<br />A measured system.</h2><p>The same container contract used by the evaluator: strict resource limits, atomic output, runtime-only credentials, and adaptive pacing.</p><a href="https://hub.docker.com/r/rkitano/gp-agent/tags" className="text-link">Inspect the public image →</a></div>
        <div className="proof-grid"><article><b>linux/amd64</b><span>Judge-compatible image</span></article><article><b>&lt; 10 GB</b><span>Compressed image gate</span></article><article><b>4 GB / 2 CPU</b><span>Rehearsed resource limit</span></article><article><b>0 fallbacks</b><span>Acceptance run</span></article></div>
      </section>

      <footer className="shell"><div className="brand"><span className="brand-mark">GP</span><span>Agent</span></div><p>Built for AMD Developer Hackathon · Track 1</p><a href="#top">Back to top ↑</a></footer>
    </main>
  );
}
