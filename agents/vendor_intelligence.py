"""
ContextChain — Agent 2: Vendor Intelligence (RAG-powered)
==========================================================
Queries EMO for requirements → RAG-retrieves vendor facts →
LLM reasons ONLY over retrieved facts → appends grounded decision.

Model: Qwen2.5-72B via Featherless
Why:   - Open-source: sensitive vendor/contract data stays off closed APIs
       - Featherless hosts 6,700+ HuggingFace models at flat cost
       - Qwen2.5-72B is excellent at structured JSON output
       - Can be fine-tuned on proprietary procurement data in future

RAG layer (rag/retriever.py):
  - semantic search (sentence-transformers + FAISS) or keyword fallback
  - verify_claim() gives hard boolean ground-truth per vendor per requirement
  - every EvidenceItem gets grounded=True + KB source citation

Fallback: gpt-4o-mini via OpenAI if FEATHERLESS_API_KEY not set.
"""

import json
import re
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from emo import EpisodicMemoryObject, AgentEntry, EvidenceItem, RejectedOption
from core.model_config import call_llm
from rag.retriever import get_retriever

AGENT_ID = "vendor_intelligence_v1"

SYSTEM_PROMPT = """You are an enterprise vendor analyst.
You have been given VERIFIED FACTS retrieved from a knowledge base.
Reason ONLY from these facts. Do not add any information not present below.

Score each vendor against the requirements and recommend one.

Return ONLY valid JSON:
{
  "chosen_vendor": "exact vendor name from the facts provided",
  "decision": "Recommend [Vendor] because [specific reason from facts]",
  "vendor_scores": [
    {"vendor": "name", "score": 0.0-1.0, "meets_all_critical": true/false, "notes": "brief"}
  ],
  "rejected_alternatives": [
    {
      "name": "vendor name",
      "reason": "specific fact-based rejection reason",
      "risk_score": 0.0-1.0,
      "compliance_gap": true/false
    }
  ],
  "reasoning_chain": ["step citing KB source", "step 2..."],
  "confidence_score": 0.0-1.0,
  "uncertainty_flags": ["anything the KB facts did not cover clearly"]
}

CRITICAL RULES:
- compliance_gap=true if vendor rejected for compliance/certification reason
- confidence > 0.80 ONLY when all critical requirements are KB-verified
- confidence < 0.70 if any critical requirement cannot be verified from facts
- Do NOT invent pricing, compliance status, or features not in the KB context
"""


def _extract_requirements(emo: EpisodicMemoryObject) -> dict:
    """Parse EMO evidence into structured requirement flags for RAG queries."""
    all_evidence = emo.query("evidence")
    decision = emo.query("decision")

    reqs = {
        "raw_text": decision[0] if decision else "",
        "needs_india_region": False,
        "needs_soc2": False,
        "needs_object_logging": False,
        "needs_hipaa": False,
        "needs_ap_south_1": False,
        "budget_total": None,
        "storage_tb": None,
        "critical_flags": [],
    }

    for ev in all_evidence:
        t = ev.content.lower()
        if any(k in t for k in ["india", "ap-south", "mumbai", "centralindia"]):
            reqs["needs_india_region"] = True
            if ev.critical: reqs["critical_flags"].append("india_region")
        if "ap-south-1" in t:
            reqs["needs_ap_south_1"] = True
            if ev.critical: reqs["critical_flags"].append("ap_south_1")
        if "soc2" in t or "soc 2" in t:
            reqs["needs_soc2"] = True
            if ev.critical: reqs["critical_flags"].append("soc2")
        if "object" in t and "log" in t:
            reqs["needs_object_logging"] = True
            if ev.critical: reqs["critical_flags"].append("object_level_logging")
        if "hipaa" in t:
            reqs["needs_hipaa"] = True
        m = re.search(r'\$?([\d,]+)\s*/?\s*(?:month|mo)', t)
        if m and reqs["budget_total"] is None:
            try: reqs["budget_total"] = float(m.group(1).replace(",", ""))
            except: pass
        m2 = re.search(r'(\d+)\s*tb', t)
        if m2 and reqs["storage_tb"] is None:
            reqs["storage_tb"] = int(m2.group(1))

    return reqs


def _score_vendors_from_kb(retriever, reqs: dict, verifications: dict, candidate_vendors: list[str]) -> dict:
    critical_claims = set()
    if reqs["needs_soc2"]:
        critical_claims.add("SOC2_Type_II")
    if reqs["needs_india_region"]:
        critical_claims.add("data_residency_india")
    if reqs["needs_ap_south_1"]:
        critical_claims.add("ap_south_1_region")
    if reqs["needs_object_logging"]:
        critical_claims.add("object_level_logging")
    if reqs["needs_hipaa"]:
        critical_claims.add("HIPAA")

    scores = []
    for vname in candidate_vendors:
        checks = verifications.get(vname, [])
        verified_count = sum(1 for check in checks if check.get("verified"))
        score = verified_count / len(checks) if checks else 0.5
        critical_checks = [check for check in checks if check.get("claim") in critical_claims]
        meets_all_critical = all(check.get("verified", False) for check in critical_checks)
        vendor = retriever.get_vendor(vname) or {}
        scores.append({
            "vendor": vname,
            "score": round(score, 2),
            "meets_all_critical": meets_all_critical,
            "price_per_tb": vendor.get("pricing_per_tb_month_usd", 999999),
            "notes": "KB deterministic scoring",
        })

    scores.sort(key=lambda item: (not item["meets_all_critical"], -item["score"], item["price_per_tb"]))
    chosen = scores[0]["vendor"] if scores else ""

    rejected_alts = []
    for item in scores[1:]:
        failed = [
            check for check in verifications.get(item["vendor"], [])
            if check.get("claim") in critical_claims and not check.get("verified")
        ]
        if failed:
            reason = "; ".join(f"{check['claim']} failed: {check.get('note', '')}" for check in failed)
        else:
            reason = "Lower KB score or higher cost than the selected vendor"
        rejected_alts.append({
            "name": item["vendor"],
            "reason": reason,
            "risk_score": round(1.0 - item["score"], 2),
            "compliance_gap": bool(failed),
        })

    chosen_checks = verifications.get(chosen, [])
    chosen_failures = [
        check for check in chosen_checks
        if check.get("claim") in critical_claims and not check.get("verified")
    ]
    confidence = 0.9 if not chosen_failures else 0.62

    return {
        "chosen_vendor": chosen,
        "decision": f"Recommend {chosen} because KB checks show it satisfies the critical procurement constraints",
        "vendor_scores": [{k: v for k, v in item.items() if k != "price_per_tb"} for item in scores],
        "rejected_alternatives": rejected_alts,
        "reasoning_chain": [
            "Model output was not valid JSON, so vendors were scored deterministically from KB verification results.",
            f"Selected {chosen} by prioritizing critical compliance, exact region support, logging, and budget checks.",
        ],
        "confidence_score": confidence,
        "uncertainty_flags": [f"{check['claim']} could not be verified for {chosen}" for check in chosen_failures],
    }


def run(emo: EpisodicMemoryObject) -> EpisodicMemoryObject:
    retriever = get_retriever()
    reqs = _extract_requirements(emo)

    # ── Step 1: RAG — semantic search for relevant KB chunks ────────────────
    rag_queries = [reqs["raw_text"]]
    if reqs["needs_soc2"]:        rag_queries.append("SOC2 Type II compliance certification")
    if reqs["needs_india_region"]: rag_queries.append("India region data residency ap-south-1")
    if reqs["needs_object_logging"]: rag_queries.append("object level logging audit trail")
    if reqs["budget_total"]:      rag_queries.append("pricing cost per TB cloud storage")

    seen, chunks = set(), []
    for q in rag_queries:
        for c in retriever.search(q, top_k=2):
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                chunks.append(c)

    # ── Step 2: RAG — hard verify each vendor against each requirement ───────
    verifications = {}
    # Restrict verification to vendors present in retrieved chunks to reduce work
    candidate_vendors = retriever.all_vendor_names()
    if not candidate_vendors:
        candidate_vendors = retriever.all_vendor_names()[:5]

    for vname in candidate_vendors:
        v = []
        if reqs["needs_soc2"]:
            v.append({"claim": "SOC2_Type_II",         **retriever.verify_claim(vname, "SOC2_Type_II", True)})
        if reqs["needs_india_region"]:
            v.append({"claim": "data_residency_india",  **retriever.verify_claim(vname, "data_residency_india", True)})
        if reqs["needs_ap_south_1"]:
            vendor = retriever.get_vendor(vname) or {}
            regions = vendor.get("regions", [])
            has_ap_south_1 = "ap-south-1" in regions
            v.append({
                "claim": "ap_south_1_region",
                "verified": has_ap_south_1,
                "actual_value": regions,
                "source": f"KB::{vname}::regions",
                "note": "ap-south-1 is available" if has_ap_south_1 else f"ap-south-1 not available. Regions: {regions}",
            })
        if reqs["needs_object_logging"]:
            v.append({"claim": "object_level_logging",  **retriever.verify_claim(vname, "object_level_logging", True)})
        if reqs["needs_hipaa"]:
            v.append({"claim": "HIPAA",                 **retriever.verify_claim(vname, "hipaa", True)})
        if reqs["budget_total"] and reqs["storage_tb"]:
            limit = reqs["budget_total"] / reqs["storage_tb"]
            v.append({"claim": f"price_under_{limit:.1f}_per_tb", **retriever.verify_claim(vname, f"price_under_{limit:.1f}", None)})
        verifications[vname] = v

    # ── Step 3: Build KB context string for LLM ──────────────────────────────
    ctx = ["=== VERIFIED KB FACTS (use ONLY these) ===\n"]
    ctx.append("--- Retrieved chunks ---")
    for c in chunks:
        ctx.append(f"[{c.chunk_id}] {c.text}")
    ctx.append("\n--- Per-vendor requirement verification ---")
    for vname, vlist in verifications.items():
        if not vlist: continue
        ctx.append(f"\n{vname}:")
        for v in vlist:
            s = "✓ PASS" if v["verified"] else "✗ FAIL"
            ctx.append(f"  {s} | {v['claim']}: actual={v['actual_value']} | {v['note']}")

    tb   = reqs["storage_tb"]   or "unspecified"
    bud  = reqs["budget_total"] or "unspecified"
    user_msg = "\n".join(ctx) + f"""

Requirements summary:
- Storage: {tb} TB
- Budget: ${bud}/month total
- India region: {'CRITICAL' if 'india_region' in reqs['critical_flags'] else ('required' if reqs['needs_india_region'] else 'not required')}
- SOC2 Type II: {'CRITICAL' if 'soc2' in reqs['critical_flags'] else ('required' if reqs['needs_soc2'] else 'not required')}
- Object-level logging: {'CRITICAL' if 'object_level_logging' in reqs['critical_flags'] else ('required' if reqs['needs_object_logging'] else 'not required')}
- HIPAA: {'required' if reqs['needs_hipaa'] else 'not required'}

Using ONLY the facts above, score all vendors and recommend one.
For rejected vendors, cite the specific KB fact that caused rejection."""

    # ── Step 4: LLM reasons over KB facts ────────────────────────────────────
    # Quick-mode: deterministic vendor scoring without LLM (useful for low-latency deployments)
    if os.getenv("FAST_VENDOR", "0") == "1":
        data = _score_vendors_from_kb(retriever, reqs, verifications, candidate_vendors)

    else:
        raw = ""
        try:
            raw = call_llm(
            agent_id=AGENT_ID,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            )
        except Exception as call_exc:
            # surface LLM call errors clearly
            raise RuntimeError(f"LLM call failed for {AGENT_ID}: {call_exc}")

    # Parse model output as JSON; if parsing fails, try AIML API fallback once and surface helpful error
    if os.getenv("FAST_VENDOR", "0") != "1":
        try:
            data = json.loads(raw)
        except Exception as e:
            print(f"[vendor_intelligence] JSON parse failed from {AGENT_ID} output: {e}")
            print("---- model raw output START ----")
            print((raw or '')[:2000])
            print("---- model raw output END ----")
            # Attempt fallback via AIML API (first available aiml provider)
            try:
                fallback_raw = ""
                fallback_raw = call_llm(agent_id="needs_analyzer_v1", messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_msg}])
                try:
                    data = json.loads(fallback_raw)
                    print("[vendor_intelligence] AIML fallback succeeded and returned valid JSON")
                except Exception as e2:
                    print(f"[vendor_intelligence] AIML fallback also produced invalid JSON: {e2}")
                    print("---- fallback raw output START ----")
                    print((fallback_raw or '')[:2000])
                    print("---- fallback raw output END ----")
                    data = _score_vendors_from_kb(retriever, reqs, verifications, candidate_vendors)
            except Exception:
                data = _score_vendors_from_kb(retriever, reqs, verifications, candidate_vendors)

    # ── Step 5: Build grounded EvidenceItems ─────────────────────────────────
    evidence = []
    for c in chunks[:4]:
        evidence.append(EvidenceItem(
            content=f"[KB:{c.chunk_id}] {c.text[:200]}",
            source=f"RAG::{c.chunk_id}",
            critical=(c.category == "compliance"),
            grounded=True,
        ))
    chosen = data.get("chosen_vendor", "")
    for v in verifications.get(chosen, []):
        evidence.append(EvidenceItem(
            content=f"[VERIFIED] {chosen} — {v['claim']}: {v['actual_value']} | {v['note']}",
            source=v["source"],
            critical=not v["verified"],
            grounded=True,
        ))
    for u in data.get("uncertainty_flags", []):
        evidence.append(EvidenceItem(
            content=f"[UNCERTAINTY] {u}",
            source="vendor_intelligence_rag",
            critical=True,
            grounded=False,
        ))

    # ── Step 6: Build rejected alternatives with KB-backed compliance_gap ────
    rejected = []
    for r in data.get("rejected_alternatives", []):
        is_comp_gap = r.get("compliance_gap", False)
        for v in verifications.get(r["name"], []):
            if not v["verified"] and v["claim"] in ["SOC2_Type_II","data_residency_india","object_level_logging","HIPAA"]:
                is_comp_gap = True
        rejected.append(RejectedOption(
            name=r["name"],
            reason=r["reason"],
            risk_score=float(r.get("risk_score", 0.5)),
            compliance_gap=is_comp_gap,
        ))

    emo.append(AgentEntry(
        agent_id=AGENT_ID,
        decision=data["decision"],
        evidence=evidence,
        rejected_alternatives=rejected,
        reasoning_chain=data.get("reasoning_chain", []),
        confidence_score=float(data.get("confidence_score", 0.75)),
    ))
    return emo
