# GP-AI-Agent Hackathon Pitch Transcript

## Slide 1 - GP-AI-Agent

Hello, I am presenting GP-AI-Agent, my project for the AMD Developer Hackathon Track 1.

The idea is simple: not every AI task needs an expensive cloud model. GP-AI-Agent first decides whether a task can be answered safely on the local machine. If it can, it answers locally. If the task is difficult or the local answer does not pass safety checks, it escalates to a stronger Fireworks model.

So the project is local-first, but not local-only. The goal is to spend tokens only when they create real value.

## Slide 2 - The problem

Using a strong model for every prompt can be costly and slow. A short factual question may take the same paid path as a complex programming request, even though the two tasks have very different needs.

This is especially important in Track 1 because the agent has real limits: 4 gigabytes of memory, 2 CPU cores, and less than 10 minutes to complete the full task set.

My challenge was to build an agent that works within those limits without sacrificing reliability.

## Slide 3 - The solution

GP-AI-Agent solves this with a learned router. Before calling any external model, the router looks at the prompt and makes a small, local prediction.

First, it classifies the task. Next, it estimates whether the local path is likely to need escalation. Then the agent verifies local work whenever possible. Finally, it calls Fireworks only for tasks that need a stronger model.

This routing decision is local, fast, and uses zero Fireworks tokens.

## Slide 4 - Technical architecture

Technically, the router is a compact logistic classifier using hashed word and word-pair features. It was trained from 360 measured outcomes from the local pipeline, rather than from generic examples.

For local generation, I use Qwen 2.5 1.5B in a compressed Q4 format through llama.cpp. This is small enough to fit the required CPU-only environment.

The important safety layer is verification. The agent can recompute arithmetic, run code tests, check some logic problems, and validate output format. If the local result is not trustworthy, the system escalates instead of guessing.

## Slide 5 - Results

The earlier API-first approach used 9,685 API tokens. The new v11 learned-router profile estimates 1,824 API tokens for the comparable task mix, while projecting 97.14 percent accuracy.

I also tested the real end-to-end system. The 19-task rehearsal completed all 19 tasks in 70.5 seconds. The 80-task pass answered every task in 207.4 seconds, with 45 successful remote calls and no fallbacks. The project also has 130 passing automated tests.

The key point is that I am not claiming that all tasks should be local. I am using data and verification to choose the right path for each task.

## Slide 6 - Why it matters

GP-AI-Agent shows that a practical AI agent can be efficient and dependable at the same time.

The local model handles work when it is safe. The stronger hosted model is still available when the task requires it. That makes the system faster, more budget-aware, and safer than sending every prompt through one expensive route.

The live walkthrough shows the local routing decision, the prompt-based baseline, the final answer, and the token comparison. Thank you for your time.
