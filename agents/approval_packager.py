"""
ContextChain — Agent 4: Approval Packager
Assembles the final human-readable approval request.
Includes the full decision archaeology trace.
"""

from core.emo import EpisodicMemoryObject


def run(emo: EpisodicMemoryObject) -> dict:
    """
    Reads the full EMO and returns a structured approval packet.
    No LLM call needed — this is pure EMO assembly.
    """

    final_recommendation = emo.latest_decision()
    chain_confidence = emo.chain_confidence()
    overturned = emo.was_overturned()

    archaeology = []
    for entry in emo.entries:
        node = {
            "agent_id": entry.agent_id,
            "decision": entry.decision,
            "confidence": entry.confidence_score,
            "reasoning_chain": entry.reasoning_chain,
            "evidence": [
                {"content": e.content, "critical": e.critical}
                for e in entry.evidence
            ],
            "rejected_alternatives": [
                {"name": r.name, "reason": r.reason, "risk_score": r.risk_score}
                for r in entry.rejected_alternatives
            ],
            "risk_findings": entry.risk_findings,
            "revision": None,
            "timestamp": entry.timestamp,
        }
        if entry.revision:
            node["revision"] = {
                "overturned_by": entry.revision.overturned_by,
                "original_decision": entry.revision.original_decision,
                "new_decision": entry.revision.new_decision,
                "reason": entry.revision.reason,
                "evidence_added": entry.revision.evidence_added,
            }
        archaeology.append(node)

    critical_items = emo.query("evidence", "critical")
    risk_findings = emo.query("risk_findings")
    all_risk = [f for sublist in risk_findings if sublist for f in (sublist if isinstance(sublist, list) else [sublist])]

    return {
        "task_id": emo.task_id,
        "status": "READY_FOR_APPROVAL",
        "final_recommendation": final_recommendation,
        "chain_confidence": chain_confidence,
        "was_overturned": overturned,
        "critical_requirements_count": len(critical_items),
        "risk_findings": all_risk,
        "decision_archaeology": archaeology,
        "raw_input": emo.raw_input,
    }
