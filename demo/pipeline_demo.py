"""
Demo: procurement pipeline (streaming-capable).
Frame for demo use; production systems should import EMO from `emo` and reuse agents.
"""

import uuid
from time import perf_counter
from emo import EpisodicMemoryObject
from core.confidence_router import route
from agents import needs_analyzer, vendor_intelligence, risk_auditor, approval_packager
from typing import Callable, Optional


def run_pipeline_streaming(
    raw_input: str,
    task_id: str = None,
    emit: Optional[Callable] = None,
    confidence_threshold: float | None = None,
    report_timings: bool = False,
) -> dict:
    """
    Full ContextChain pipeline with optional live event emission.

    emit(event_type, data) is called after each agent completes.
    If emit is None, runs silently (used by /api/run non-streaming).
    """

    def _emit(event_type: str, data: dict):
        if emit:
            emit(event_type, data)

    task_id = task_id or f"PROC-{uuid.uuid4().hex[:8].upper()}"

    _emit("pipeline_start", {"task_id": task_id, "message": "Pipeline initialised"})

    emo = EpisodicMemoryObject(task_id=task_id, raw_input=raw_input)
    history = []
    timings: dict = {}

    # ── Agent 1 ──────────────────────────────────────────────────────────
    _emit("agent_start", {"agent": "needs_analyzer_v1", "label": "Agent 1 — Needs Analyzer", "message": "Extracting requirements..."})
    t0 = perf_counter()
    emo = needs_analyzer.run(emo)
    t1 = perf_counter()
    timings["needs_analyzer_v1"] = t1 - t0
    entry1 = emo.entries[-1]
    _emit("agent_complete", {
        "agent": entry1.agent_id,
        "label": "Agent 1 — Needs Analyzer",
        "decision": entry1.decision,
        "confidence": entry1.confidence_score,
        "evidence": [{"content": e.content, "critical": e.critical} for e in entry1.evidence],
        "reasoning_chain": entry1.reasoning_chain,
        "rejected_alternatives": [],
        "risk_findings": [],
        "hallucination_flags": entry1.hallucination_flags,
        "revision": None,
        "timestamp": entry1.timestamp,
        "emo_version": emo.emo_version,
        "emo": emo.to_dict(),
    })
    # record snapshot after agent 1
    history.append({"agent": entry1.agent_id, "entry": entry1, "emo": emo.to_dict()})

    # ── Agent 2 ──────────────────────────────────────────────────────────
    _emit("agent_start", {"agent": "vendor_intelligence_v1", "label": "Agent 2 — Vendor Intelligence", "message": "Scoring vendors against EMO requirements..."})
    t0 = perf_counter()
    emo = vendor_intelligence.run(emo)
    t1 = perf_counter()
    timings["vendor_intelligence_v1"] = t1 - t0
    entry2 = emo.entries[-1]
    routing = route(emo, confidence_threshold)
    _emit("agent_complete", {
        "agent": entry2.agent_id,
        "label": "Agent 2 — Vendor Intelligence",
        "decision": entry2.decision,
        "confidence": entry2.confidence_score,
        "evidence": [{"content": e.content, "critical": e.critical} for e in entry2.evidence],
        "reasoning_chain": entry2.reasoning_chain,
        "rejected_alternatives": [{"name": r.name, "reason": r.reason, "risk_score": r.risk_score, "compliance_gap": r.compliance_gap} for r in entry2.rejected_alternatives],
        "risk_findings": [],
        "hallucination_flags": entry2.hallucination_flags,
        "revision": None,
        "timestamp": entry2.timestamp,
        "emo_version": emo.emo_version,
        "emo": emo.to_dict(),
    })
    # record snapshot after agent 2
    history.append({"agent": entry2.agent_id, "entry": entry2, "emo": emo.to_dict()})

    # ── Router ───────────────────────────────────────────────────────────
    _emit("router_decision", {
        "next_agent": routing["next_agent"],
        "confidence": routing["confidence"],
        "reason": routing.get("trigger_reason") or routing.get("skip_reason"),
        "threshold": confidence_threshold if confidence_threshold is not None else 0.70,
    })

    # ── Agent 3 (conditional) ────────────────────────────────────────────
    if routing["next_agent"] == "risk_auditor":
        _emit("agent_start", {"agent": "risk_auditor_v1", "label": "Agent 3 — Risk Auditor", "message": "Confidence below threshold — auditing rejected alternatives..."})
        t0 = perf_counter()
        emo = risk_auditor.run(emo)
        t1 = perf_counter()
        timings["risk_auditor_v1"] = t1 - t0
        entry3 = emo.entries[-1]
        revision_data = None
        if emo.was_overturned():
            diff = emo.revision_diff()
            revision_data = diff
        _emit("agent_complete", {
            "agent": entry3.agent_id,
            "label": "Agent 3 — Risk Auditor",
            "decision": entry3.decision,
            "confidence": entry3.confidence_score,
            "evidence": [{"content": e.content, "critical": e.critical} for e in entry3.evidence],
            "reasoning_chain": entry3.reasoning_chain,
            "rejected_alternatives": [],
            "risk_findings": entry3.risk_findings,
            "hallucination_flags": entry3.hallucination_flags,
            "revision": revision_data,
            "timestamp": entry3.timestamp,
            "emo_version": emo.emo_version,
            "emo": emo.to_dict(),
        })
        # record snapshot after agent 3
        history.append({"agent": entry3.agent_id, "entry": entry3, "emo": emo.to_dict()})
    else:
        _emit("agent_skipped", {"agent": "risk_auditor_v1", "label": "Agent 3 — Risk Auditor", "reason": "Confidence above threshold"})
        
        # include EMO snapshot for skipped agent
        _emit("agent_snapshot", {"agent": "risk_auditor_v1", "emo": emo.to_dict()})
        # record snapshot for skipped agent (no change)
        history.append({"agent": "risk_auditor_v1", "entry": None, "emo": emo.to_dict()})

    # ── Agent 4 ──────────────────────────────────────────────────────────
    _emit("agent_start", {"agent": "approval_packager_v1", "label": "Agent 4 — Approval Packager", "message": "Assembling decision archaeology..."})
    t0 = perf_counter()
    packet = approval_packager.run(emo)
    t1 = perf_counter()
    timings["approval_packager_v1"] = t1 - t0

    _emit("pipeline_complete", {
        "task_id": packet["task_id"],
        "final_recommendation": packet["final_recommendation"],
        "chain_confidence": packet["chain_confidence"],
        "was_overturned": packet["was_overturned"],
        "critical_requirements_count": packet["critical_requirements_count"],
        "risk_findings": packet["risk_findings"],
        "decision_archaeology": packet["decision_archaeology"],
        "hallucination_flags": emo.all_hallucination_flags(),
        "emo_version": emo.emo_version,
        "emo": emo.to_dict(),
        "revision_diff": emo.revision_diff(),
        "weighted_evidence_count": len(emo.weighted_evidence()),
    })

    # final snapshot after approval packager
    history.append({"agent": "approval_packager_v1", "entry": None, "emo": emo.to_dict(), "packet": packet})

    result = {"approval_packet": packet, "emo": emo, "history": history}
    if report_timings:
        result["timings"] = timings
    return result


def run_pipeline(raw_input: str, task_id: str = None, confidence_threshold: float | None = None) -> dict:
    """Backward-compatible wrapper used by tests and simple demos.

    Calls the streaming pipeline in non-streaming mode (no `emit`).
    """
    return run_pipeline_streaming(raw_input=raw_input, task_id=task_id, emit=None, confidence_threshold=confidence_threshold)
