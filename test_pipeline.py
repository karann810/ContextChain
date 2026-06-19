"""
Quick test — run this to verify the pipeline works before demo.
Usage: python test_pipeline.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from demo.pipeline_demo import run_pipeline

TEST_CASES = [
    {
        "name": "Clean flow (should skip risk audit)",
        "input": "We need 10TB of cloud storage for backups. AWS is preferred. Budget $200/month. No special compliance needed.",
    },
    {
        "name": "Complex compliance (should trigger risk audit)",
        "input": "50TB cloud object storage. SOC2 Type II required. India region only (ap-south-1). Critical: SOC2 scope must explicitly cover object-level logging. Budget $2000/month. 3-week deadline.",
    },
]

for tc in TEST_CASES:
    print(f"\n{'#'*60}")
    print(f"TEST: {tc['name']}")
    print(f"{'#'*60}")

    result = run_pipeline(tc["input"])
    packet = result["approval_packet"]
    emo = result["emo"]
    history = result.get("history", [])

    print(f"\nFinal recommendation : {packet['final_recommendation']}")
    print(f"Chain confidence     : {packet['chain_confidence']}")
    print(f"Was overturned       : {packet['was_overturned']}")
    print(f"Agents ran           : {[e.agent_id for e in emo.entries]}")

    print("\nPipeline history snapshots:")
    for h in history:
        agent = h.get("agent")
        entry = h.get("entry")
        print(f"- After {agent}:")
        if entry:
            print(f"    decision: {entry.decision}")
            print(f"    confidence: {entry.confidence_score}")
            print(f"    evidence count: {len(entry.evidence)}")
        else:
            # show summary of emo at this point
            ecount = len(h.get("emo", {}).get("entries", []))
            print(f"    emo entries: {ecount}")

    if packet["risk_findings"]:
        print(f"Risk findings        :")
        for r in packet["risk_findings"]:
            print(f"  - {r}")

print(f"\n{'='*60}")
print("All tests complete.")
