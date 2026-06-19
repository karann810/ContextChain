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
import os
import re
from emo import EpisodicMemoryObject, AgentEntry, EvidenceItem
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


def _fallback_analysis(raw_input: str) -> dict:
    text = raw_input.lower()
    evidence = []
    implicit_constraints = []

    storage = re.search(r"(\d+(?:\.\d+)?)\s*tb", text)
    if storage:
        evidence.append({"content": f"{storage.group(1)}TB cloud object storage required", "critical": True})

    budget = re.search(r"\$?\s*([\d,]+)\s*/?\s*(?:month|mo)", text)
    if budget:
        evidence.append({"content": f"Monthly budget must be under ${budget.group(1).replace(',', '')}", "critical": True})

    if "soc2" in text or "soc 2" in text:
        evidence.append({"content": "SOC2 Type II compliance is required", "critical": True})

    if "india" in text or "ap-south-1" in text:
        evidence.append({"content": "Data must stay in India", "critical": True})

    if "ap-south-1" in text:
        evidence.append({"content": "The storage service must support the ap-south-1 region", "critical": True})

    if "object" in text and "log" in text:
        evidence.append({"content": "SOC2 scope must explicitly cover object-level logging", "critical": True})

    if not evidence:
        evidence.append({"content": raw_input.strip() or "Procurement request provided", "critical": False})

    if "secure" in text and "soc2" not in text and "soc 2" not in text:
        implicit_constraints.append("Security requirement is ambiguous; specific certifications should be confirmed")

    return {
        "decision": "Procure cloud object storage that satisfies the stated compliance, residency, logging, and budget constraints",
        "evidence": evidence,
        "reasoning_chain": [
            "Model output was not valid JSON, so requirements were extracted deterministically from the request text.",
            "Critical constraints were identified from explicit procurement language.",
        ],
        "confidence_score": 0.82 if len(evidence) >= 3 else 0.65,
        "implicit_constraints": implicit_constraints,
    }


def run(emo: EpisodicMemoryObject) -> EpisodicMemoryObject:
    """Run Agent 1 on the EMO's raw_input. Appends entry to EMO."""

    if os.getenv("FAST_NEEDS", "0") == "1":
        data = _fallback_analysis(emo.raw_input)
    else:
        raw = call_llm(
            agent_id=AGENT_ID,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": emo.raw_input},
            ],
        )

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = _fallback_analysis(emo.raw_input)

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
