"""
ContextChain — Agent 1: Needs Analyzer
=======================================
Reads raw procurement text → produces EMO v1 with structured requirements.

Model: DeepSeek-R1 via AIML API
Why:   Ambiguous procurement requests need genuine reasoning to extract
       implicit constraints. DeepSeek-R1's chain-of-thought becomes
       the EMO reasoning_chain — visible in the archaeology trace.

Fallback: gpt-4o-mini via OpenAI if AIMLAPI_KEY not set.
"""

import json
from core.emo import EpisodicMemoryObject, AgentEntry, EvidenceItem
from core.model_config import call_llm

AGENT_ID = "needs_analyzer_v1"

SYSTEM_PROMPT = """You are a procurement requirements analyst.
Extract ALL requirements from the user's raw procurement request —
both explicit ones stated directly and implicit ones you can infer.

Return ONLY valid JSON with this exact structure:
{
  "decision": "one-sentence summary of what is needed",
  "evidence": [
    {"content": "specific requirement or constraint", "critical": true/false}
  ],
  "reasoning_chain": [
    "step 1 of your analysis",
    "step 2..."
  ],
  "confidence_score": 0.0-1.0,
  "implicit_constraints": ["any inferred constraints not explicitly stated"]
}

Rules:
- evidence items must be atomic (one fact per item)
- mark evidence as critical=true if missing it would cause a compliance failure
- confidence > 0.85 only when request is clear, complete, unambiguous
- confidence < 0.70 when key info is missing or contradictory
- ALWAYS infer implicit constraints:
    * "India" → data residency requirement (ap-south-1 or centralindia)
    * "compliance" without specifics → flag as ambiguous, lower confidence
    * "3 weeks" → urgency constraint, affects vendor SLA requirement
    * budget stated → infer per-TB cost ceiling from storage size
    * "secure" without cert → flag: which cert? SOC2? ISO27001? HIPAA?
"""


def run(emo: EpisodicMemoryObject) -> EpisodicMemoryObject:
    """Run Agent 1 on the EMO's raw_input. Appends entry to EMO."""

    raw = call_llm(
        agent_id=AGENT_ID,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": emo.raw_input},
        ],
    )

    data = json.loads(raw)

    evidence_items = []
    for e in data.get("evidence", []):
        evidence_items.append(EvidenceItem(
            content=e["content"],
            source="needs_analyzer",
            critical=e.get("critical", False),
        ))
    for ic in data.get("implicit_constraints", []):
        evidence_items.append(EvidenceItem(
            content=f"[implicit] {ic}",
            source="needs_analyzer",
            critical=False,
        ))

    entry = AgentEntry(
        agent_id=AGENT_ID,
        decision=data["decision"],
        evidence=evidence_items,
        reasoning_chain=data.get("reasoning_chain", []),
        confidence_score=float(data.get("confidence_score", 0.85)),
    )

    emo.append(entry)
    return emo
