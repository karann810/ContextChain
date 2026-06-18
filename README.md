# ContextChain

**Episodic Memory Objects as a handoff primitive in multi-agent enterprise workflows.**

> Built for the Band of Agents Hackathon 2026.

## The problem

Every multi-agent system today hands off context as either:
- **(a) a massive text dump** — overwhelms the next agent
- **(b) a compressed summary** — strips the reasoning

Both are wrong. ContextChain defines a third primitive: the **Episodic Memory Object (EMO)**.

## What's an EMO?

A structured, queryable, versioned object containing:

| Field | Description |
|---|---|
| `decision` | What was concluded |
| `evidence[]` | Raw source refs, NOT summaries. With Ebbinghaus decay weights. |
| `rejected_alternatives[]` | What was dismissed + why + `compliance_gap` flag |
| `confidence_score` | 0.0–1.0. Below 0.70 triggers automatic risk audit. |
| `reasoning_chain[]` | Step-by-step inference trace |
| `revision` | If overturned: git-style diff, original preserved |
| `hallucination_flags[]` | Claims in `decision` with no evidence backing |
| `parent_emo_id` | Links to prior EMO for chain graph |
| `emo_version` | Increments on every agent write |

Agents call `emo.query("rejected_alternatives", "compliance_gap")` — they query what they need, not read the whole object.

## The 4-agent procurement pipeline

```
Raw request → Agent 1 (Needs Analyzer)
                   ↓ EMO v1
            Agent 2 (Vendor Intelligence)
                   ↓ EMO v2
            [Router: confidence < 0.70?]
                   ↓ YES              → NO
            Agent 3 (Risk Auditor)    ↓
            [can overturn Agent 2]    ↓
                   ↓ EMO v3           ↓
            Agent 4 (Approval Packager)
                   ↓
            Decision archaeology trace
```

## Key innovations

1. **EMO is queryable, not readable** — `.query(field, condition)` API
2. **Confidence-gated routing** — pipeline adapts based on agent certainty
3. **Reopenable decisions** — Agent 3 can overturn Agent 2, original preserved
4. **Hallucination guard** — claims without evidence backing are flagged
5. **Ebbinghaus decay** — evidence loses weight over time (critical evidence decays slower)
6. **Revision diff** — git-style diff when a decision is overturned
7. **Live streaming UI** — agents appear one by one via SSE

## Quickstart

```bash
cp .env.example .env
# Add your OPENAI_API_KEY or configure LITELLM_MODEL

pip install -r requirements.txt
uvicorn app:app --reload
# Open http://localhost:8000
```

## Deployment / Production

Run the watcher service locally:

```bash
python -u run_agents.py
```
Notes:
- Adapters prefer `PlatformMessage.metadata['emo']` when present; they fall back to parsing `PlatformMessage.content` as JSON or raw text.
- If you see `session.already_connected`, stop other processes using the same Band agent keys before starting this service.

Session diagnostics and key rotation helpers:

- Validate your agent API keys:

```bash
python scripts/band_session_diagnostics.py
```

- Rotate/update keys interactively:

```bash
python scripts/update_agent_keys.py
```


## Research paper

*"Episodic Memory Objects as a Handoff Primitive in Multi-Agent Enterprise Workflows"*

Target: arXiv cs.AI / cs.MA

Extends: ActMem (Zhang et al., 2026) — agents reason better with original evidence, not summaries.
