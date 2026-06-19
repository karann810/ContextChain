"""
ContextChain — Confidence Router
Reads the EMO's current confidence and decides next agent.
This is what makes the pipeline adaptive, not just sequential.
"""

from emo import EpisodicMemoryObject

CONFIDENCE_THRESHOLD = 0.70


def route(emo: EpisodicMemoryObject, threshold: float | None = None) -> dict:
    """
    Returns a routing decision based on EMO's latest confidence.

    Returns:
        {
            "next_agent": "risk_auditor" | "approval_packager",
            "confidence": float,
            "trigger_reason": str | None,
            "skip_reason": str | None,
        }
    """
    confidence = emo.latest_confidence()
    th = threshold if threshold is not None else CONFIDENCE_THRESHOLD

    if confidence < th:
        return {
            "next_agent": "risk_auditor",
            "confidence": confidence,
            "trigger_reason": (
                f"Confidence {confidence:.2f} is below threshold {th}. "
                f"Risk audit required before proceeding to approval."
            ),
            "skip_reason": None,
        }

    return {
        "next_agent": "approval_packager",
        "confidence": confidence,
        "trigger_reason": None,
        "skip_reason": (
            f"Confidence {confidence:.2f} is above threshold {th}. "
            f"Risk audit skipped."
        ),
    }
