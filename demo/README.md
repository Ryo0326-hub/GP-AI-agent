# GP-AI-Agent live routing demo

This Next.js application is a Vercel-ready walkthrough of the Track 1 routing
strategy. For each query it reveals:

1. the compact learned router's local browser decision;
2. a prompt-based routing decision that consumes Fireworks tokens;
3. a real Fireworks answer in live mode; and
4. provider-reported token usage, per-policy bars, and cumulative session
   totals.

The browser scorer uses the byte-identical JSON profile shipped in the Python
container. It is a deterministic hashed-logistic classifier, not a mocked API
response and not the optional DistilBERT research model.

## Prepare the learned-router artifact

Run this once from the repository root after the 360 local labels are complete:

```bash
make train-router
make router-artifact-check
```

The first command writes both `router_model/compact_router.json` and
`demo/src/data/router-profile.json`. The second verifies exact label coverage,
non-placeholder held-out metrics, and byte-for-byte parity. The demo run button
stays disabled while the bundled profile has a pending revision or zero
training examples.

## Run locally

```bash
cd demo
npm install
cp .env.example .env.local
npm run dev
```

Use placeholder values to inspect the UI in clearly labeled simulation mode,
which makes zero external calls. Supply valid server-side credentials to run
the two real Fireworks calls, then explicitly set `DEMO_LIVE_ENABLED=true` only
after the public firewall rule is active.

## Environment variables

| Variable | Required for live mode | Purpose |
|---|---:|---|
| `DEMO_LIVE_ENABLED` | Yes | Fail-closed kill switch. Only the exact value `true` enables external calls; it defaults to simulation. |
| `FIREWORKS_API_KEY` | Yes | Server-only Fireworks credential. Never use a `NEXT_PUBLIC_` prefix. |
| `FIREWORKS_BASE_URL` | Yes | Must be exactly `https://api.fireworks.ai/inference/v1`. Other hosts, HTTP, credentials, query strings, and redirects are rejected. |
| `ALLOWED_MODELS` | Yes | Comma-separated exact model IDs that the demo server may call. |
| `DEMO_BASELINE_MODEL` | No | Exact member of `ALLOWED_MODELS` used for the paid prompt-router decision. A conservative allowlisted model is selected when omitted. |
| `DEMO_ANSWER_MODEL` | No | Exact member of `ALLOWED_MODELS` used for the final answer. A capable allowlisted model is selected when omitted. |
| `DEMO_BASELINE_MAX_TOKENS` | No | Prompt-router completion cap; default `16`, maximum `1000`. |
| `DEMO_ANSWER_MAX_TOKENS` | No | Answer completion cap; default `500`, maximum `1000`. |

For a reproducible recording, set both optional model variables explicitly
instead of relying on name-pattern selection. The submission runtime currently
prefers `deepseek-v4-flash`, adds the task category's system prompt, and sends
`reasoning_effort: "none"`. The demo prefers DeepSeek Flash for the paid prompt
router, uses a system instruction plus a delimited JSON-encoded query, and adds
the same category-specific system instruction to the answer call. It sends
`reasoning_effort: "none"` only for allowlisted DeepSeek and Kimi model IDs,
where that option is supported. Other hosted models remain independently
configurable; the shared browser router artifact is the part that exactly
matches the container.

Next.js reads `.env.local` and `.env` during local development. Vercel does not
upload either local file: add each live value in the project's **Environment
Variables** settings for the Production, Preview, or Development environments
where it is needed.

## Walkthrough and token semantics

The browser computes `P(escalate)`, applies the calibrated threshold and any
train-derived `always_escalate` category policy, and displays that result
immediately. The server recomputes the category, score, label, threshold, and
revision from its byte-identical profile before making any external call. A
stale or forged browser decision is rejected. It then streams:

```text
run.started
baseline.started
baseline.completed
answer.started
answer.completed
comparison.completed
```

In live mode the Fireworks `usage` object is displayed as returned by the
provider. Missing usage fields remain `null`/`N/A`; the app does not invent a
number. In simulation mode the example usage is visibly labeled simulated,
and actual external spend is reported as zero.

The session counters remain cumulative until **Reset session** is selected;
only the eight newest rows are retained in the visible walkthrough log. A
running comparison can be cancelled from the main action button, which aborts
the browser stream and in-flight provider request.

The final comparison separates two meanings:

- **Policy totals** reuse the single answer call for both policies so the bars
  isolate routing overhead. The learned route contributes zero billed routing
  tokens; the prompt-based route adds its classification call.
- **Actual comparison spend** is what the live walkthrough itself paid:
  exactly two calls, the baseline classification plus the shared answer. In
  simulation mode it is zero calls and zero billed tokens.

This is an explanatory per-query comparison, not a new end-to-end submission
benchmark. Final v11 accuracy and token claims belong in the root README only
after the container rehearsal and judge pass are measured.

## Streaming API

`POST /api/demo` accepts up to an 8,000-character prompt plus the browser's
already-computed category and versioned local decision:

```json
{
  "prompt": "Explain why binary search is O(log n).",
  "category": "factual",
  "localDecision": {
    "label": "local_ok",
    "pEscalate": 0.18,
    "threshold": 0.62,
    "latencyMs": 0.41,
    "revision": "compact-example"
  }
}
```

The response uses `application/x-ndjson`. Requests and responses are marked
`no-store`. Model IDs are selected only on the server from `ALLOWED_MODELS`;
the browser cannot inject a model or API credential. Provider calls are
bounded by separate 20-second classification and 25-second answer timeouts
inside the route's 60-second maximum duration. Fireworks fetches reject HTTP
redirects so an authorization header cannot be forwarded to another origin.

## Deploy safely on Vercel

1. Import the GitHub repository and set **Root Directory** to `demo`.
2. Add the live variables above in Vercel with `DEMO_LIVE_ENABLED=false`; do not
   commit them to the repository.
3. Add a Vercel Firewall rate-limit rule for
   `POST /api/demo`: fixed window, **60 seconds**, **5 requests per IP**, action
   **deny**.
4. After the rule is active, set `DEMO_LIVE_ENABLED=true` and redeploy.
5. Verify one live request and confirm the stream,
   answer, token bars, session log, and browser console are clean.

The rate limit is a Vercel Firewall setting, not a `vercel.json` rule. Keep the
public deployment in simulation mode until it is active; otherwise anonymous
visitors can consume the server-side Fireworks key.

You can also deploy from the repository root with:

```bash
vercel --cwd demo
```

If any real key was previously committed, revoke and rotate it before adding a
replacement to Vercel. Removing the current plaintext does not remove the old
credential from Git history.
