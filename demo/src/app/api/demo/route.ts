import { NextRequest, NextResponse } from "next/server";

const fallbackAnswers: Record<string, string> = {
  factual: "A stock market index tracks the performance of a selected group of stocks. The S&P 500 is one example.",
  math: "The store sells 36 items on Monday and 60 on Tuesday, leaving 144 items.",
  sentiment: "Mixed — the review praises battery life while criticizing screen durability.",
  summarization: "Local-first routing reduces external AI spend while preserving answer quality through verification and selective escalation.",
  ner: "Maria Sanchez — Person\nFireworks AI — Organization\nBerlin — Location\nlast March — Date",
  debug: "```python\ndef get_max(nums):\n    return max(nums)\n```",
  logic: "Sam owns the cat; Jo owns the dog, so Lee must own the bird.",
  codegen: "```python\ndef second_largest(nums):\n    values = sorted(set(nums))\n    return values[-2] if len(values) >= 2 else None\n```",
};

export async function POST(request: NextRequest) {
  const { prompt, category = "factual" } = await request.json();
  if (!prompt || typeof prompt !== "string") {
    return NextResponse.json({ error: "A prompt is required." }, { status: 400 });
  }

  const baseUrl = process.env.FIREWORKS_BASE_URL?.replace(/\/$/, "");
  const apiKey = process.env.FIREWORKS_API_KEY;
  const models = process.env.ALLOWED_MODELS?.split(",").map((model) => model.trim()).filter(Boolean);

  if (baseUrl && apiKey && models?.length) {
    try {
      const preferred = models.find((model) => /gpt[-_]?oss/i.test(model)) ?? models[0];
      const response = await fetch(`${baseUrl}/chat/completions`, {
        method: "POST",
        headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
        body: JSON.stringify({
          model: preferred,
          temperature: 0,
          max_tokens: 500,
          messages: [
            { role: "system", content: "Return only the final answer. Be concise and complete." },
            { role: "user", content: prompt },
          ],
        }),
        signal: AbortSignal.timeout(20_000),
      });
      if (response.ok) {
        const data = await response.json();
        const answer = data?.choices?.[0]?.message?.content?.trim();
        if (answer) return NextResponse.json({ answer, source: "Fireworks", model: preferred });
      }
    } catch {
      // The public demo remains usable if runtime credentials or the provider are unavailable.
    }
  }

  return NextResponse.json({
    answer: fallbackAnswers[category] ?? fallbackAnswers.factual,
    source: "Demo simulation",
    model: "GP Agent routing preview",
  });
}
