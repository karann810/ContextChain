"""
ContextChain — Agent 3: Risk Auditor (RAG-augmented)
=====================================================
Called ONLY when Agent 2 confidence < 0.70.
Verifies chosen vendor via KB, re-checks rejected alternatives,
and can OVERTURN the vendor recommendation.

Model: DeepSeek-R1 via AIML API
Why:   Adversarial auditing benefits from a reasoning model that
       explicitly explores counterarguments. The chain-of-thought
       trace is surfaced in the UI archaeology panel — judges and
       approvers can see exactly WHY a decision was overturned.

Fallback: gpt-4o-mini via OpenAI if AIMLAPI_KEY not set.
"""

import json
from core.emo import EpisodicMemoryObject, AgentEntry, EvidenceItem, DecisionRevision
from core.model_config import call_llm
from rag.retriever import get_retriever

AGENT_ID = "risk_auditor_v1"

SYSTEM_PROMPT = """You are a senior procurement risk auditor.
You are called ONLY because vendor intelligence had LOW CONFIDENCE.
You have KB-verified compliance facts for all vendors.

Your job: find gaps, verify critical requirements, CONFIRM or OVERTURN.

Return ONLY valid JSON:
{
  "audit_verdict": "CONFIRM" or "OVERTURN",
  "new_recommendation": "vendor name (same if CONFIRM)",
  "risk_findings": ["specific KB-backed finding", ...],
  "evidence_added": ["new fact from KB that changes the picture", ...],
  "reasoning_chain": ["step 1 with KB citation", "step 2...", ...],
  "overturn_reason": null or "detailed reason citing KB source",
  "confidence_score": 0.0-1.0
}

RULES:
- Every finding MUST cite a KB source (e.g. "KB::AWS S3::compliance")
- Do NOT state compliance facts from memory — only use the KB context below
- If the chosen vendor FAILS a critical requirement in the KB → OVERTURN
- If an alternative passes all critical requirements better → OVERTURN
- confidence after audit should be > 0.80 (you're resolving uncertainty)
"""


def run(emo: EpisodicMemoryObject) -> EpisodicMemoryObject:
    retriever = get_retriever()

    # ── Pull context from EMO ────────────────────────────────────────────────
    all_evidence   = emo.query("evidence")
    critical_evs   = emo.query("evidence", "critical")
    rejected_alts  = emo.query("rejected_alternatives")
    uncertainties  = [e for e in all_evidence if "[UNCERTAINTY]" in e.content]

    vendor_entry   = next((e for e in emo.entries if e.agent_id == "vendor_intelligence_v1"), None)
    vendor_decision  = vendor_entry.decision if vendor_entry else "unknown"
    vendor_confidence = vendor_entry.confidence_score if vendor_entry else 0.5

    # Identify chosen vendor name
    chosen_vendor = ""
    for vname in retriever.all_vendor_names():
        if vname.lower() in vendor_decision.lower():
            chosen_vendor = vname
            break

    # ── RAG: deep-verify chosen vendor on all compliance axes ────────────────
    chosen_verifs = []
    if chosen_vendor:
        for claim in ["SOC2_Type_II","data_residency_india","object_level_logging","hipaa"]:
            r = retriever.verify_claim(chosen_vendor, claim, True)
            chosen_verifs.append({"claim": claim, "vendor": chosen_vendor, **r})

    # ── RAG: re-check every rejected alternative ──────────────────────────────
    alt_verifs = {}
    for alt in rejected_alts:
        if alt.name == chosen_vendor: continue
        v = []
        for claim in ["SOC2_Type_II","data_residency_india","object_level_logging"]:
            v.append({"claim": claim, **retriever.verify_claim(alt.name, claim, True)})
        alt_verifs[alt.name] = v

    # ── Build KB context ──────────────────────────────────────────────────────
    lines = ["=== KB VERIFICATION RESULTS (ground truth) ===\n"]
    lines.append(f"CHOSEN VENDOR: {chosen_vendor}")
    for v in chosen_verifs:
        s = "✓ PASS" if v["verified"] else "✗ FAIL ← COMPLIANCE GAP"
        lines.append(f"  {s} | {v['claim']}: actual={v['actual_value']} | {v['note']}")

    comp_chunk = retriever.search_vendor_category(chosen_vendor, "compliance") if chosen_vendor else None
    feat_chunk = retriever.search_vendor_category(chosen_vendor, "features")  if chosen_vendor else None
    if comp_chunk: lines.append(f"\n  Compliance detail: {comp_chunk.text}")
    if feat_chunk: lines.append(f"  Feature detail:    {feat_chunk.text}")

    lines.append("\nREJECTED ALTERNATIVES re-verification:")
    for aname, avlist in alt_verifs.items():
        lines.append(f"\n  {aname}:")
        for v in avlist:
            s = "✓ PASS" if v["verified"] else "✗ FAIL"
            lines.append(f"    {s} | {v['claim']}: actual={v['actual_value']} | {v['note']}")

    kb_context = "\n".join(lines)
    critical_text  = "\n".join(f"- [CRITICAL] {e.content}" for e in critical_evs) or "  none"
    uncertainty_text = "\n".join(f"- {e.content}" for e in uncertainties) or "  none"
    rejected_text  = "\n".join(
        f"- {r.name}: {r.reason} (risk={r.risk_score}, compliance_gap={r.compliance_gap})"
        for r in rejected_alts
    ) or "  none"

    user_msg = f"""{kb_context}

AGENT 2 RECOMMENDED: {vendor_decision}
AGENT 2 CONFIDENCE:  {vendor_confidence} (below threshold — that's why you're here)

Critical requirements from EMO:
{critical_text}

Agent 2 uncertainty flags:
{uncertainty_text}

Rejected alternatives from Agent 2:
{rejected_text}

Using ONLY the KB verification results, confirm or overturn the recommendation.
Cite KB sources for every finding."""

    # ── LLM audit ────────────────────────────────────────────────────────────
    raw  = call_llm(
        agent_id=AGENT_ID,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
    )
    data = json.loads(raw)

    # ── Build grounded evidence ───────────────────────────────────────────────
    new_evidence = []
    for v in chosen_verifs:
        new_evidence.append(EvidenceItem(
            content=f"[KB-AUDIT] {chosen_vendor} {v['claim']}: {v['actual_value']} — {v['note']}",
            source=v["source"],
            critical=not v["verified"],
            grounded=True,
        ))
    for ev_text in data.get("evidence_added", []):
        new_evidence.append(EvidenceItem(
            content=ev_text, source="risk_auditor_rag",
            critical=True, grounded=True,
        ))

    emo.append(AgentEntry(
        agent_id=AGENT_ID,
        decision=f"Risk audit: {data['audit_verdict']} → {data['new_recommendation']}",
        evidence=new_evidence,
        reasoning_chain=data.get("reasoning_chain", []),
        confidence_score=float(data.get("confidence_score", 0.85)),
        risk_findings=data.get("risk_findings", []),
    ))

    # If overturned, apply revision — original decision preserved in EMO
    if data["audit_verdict"] == "OVERTURN":
        emo.apply_revision("vendor_intelligence_v1", DecisionRevision(
            overturned_by=AGENT_ID,
            original_decision=vendor_decision,
            new_decision=data["new_recommendation"],
            reason=data.get("overturn_reason", "Compliance gap found (KB-verified)."),
            evidence_added=data.get("evidence_added", []),
        ))

    return emo
