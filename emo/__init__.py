"""
Standalone EMO package extracted from core — importable as `emo`.
"""

# Copied from core/emo.py

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional
import json
import uuid
import math


@dataclass
class EvidenceItem:
    content: str
    source: str = "agent"
    critical: bool = False
    grounded: bool = True          # hallucination guard: False = unverified claim
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def decay_weight(self, now: Optional[datetime] = None, half_life_hours: float = 48.0) -> float:
        """Ebbinghaus-style decay: critical evidence decays slower (2x half-life)."""
        try:
            created = datetime.fromisoformat(self.timestamp)
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            now = now or datetime.now(timezone.utc)
            hours_elapsed = (now - created).total_seconds() / 3600
            hl = half_life_hours * (2.0 if self.critical else 1.0)
            return round(math.exp(-0.693 * hours_elapsed / hl), 4)
        except Exception:
            return 1.0


@dataclass
class RejectedOption:
    name: str
    reason: str
    risk_score: float = 0.0
    compliance_gap: bool = False   # new: flag if rejection was compliance-related


@dataclass
class DecisionRevision:
    overturned_by: str
    original_decision: str
    new_decision: str
    reason: str
    evidence_added: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def as_diff(self) -> dict:
        """Git-style diff of the revision."""
        return {
            "removed": self.original_decision,
            "added": self.new_decision,
            "reason": self.reason,
            "new_evidence": self.evidence_added,
            "by": self.overturned_by,
            "at": self.timestamp,
        }


@dataclass
class AgentEntry:
    agent_id: str
    decision: str
    evidence: list[EvidenceItem] = field(default_factory=list)
    rejected_alternatives: list[RejectedOption] = field(default_factory=list)
    confidence_score: float = 1.0
    reasoning_chain: list[str] = field(default_factory=list)
    revision: Optional[DecisionRevision] = None
    risk_findings: list[str] = field(default_factory=list)
    hallucination_flags: list[str] = field(default_factory=list)  # new
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class EpisodicMemoryObject:
    """
    One object that grows as agents append to it.
    Never replaced — only extended.
    v2: versioned, chainable, hallucination-guarded.
    """

    def __init__(self, task_id: str, raw_input: str, parent_emo_id: Optional[str] = None):
        self.task_id = task_id
        self.raw_input = raw_input
        self.parent_emo_id = parent_emo_id          # links to prior EMO for chain graph
        self.emo_version: int = 1
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.entries: list[AgentEntry] = []

    # ── write ──────────────────────────────────────────────────────────────

    def append(self, entry: AgentEntry) -> None:
        """Each agent calls this once. Runs hallucination guard before appending."""
        flags = self._hallucination_check(entry)
        entry.hallucination_flags = flags
        if flags:
            # penalise confidence for ungrounded claims
            penalty = 0.05 * len(flags)
            entry.confidence_score = max(0.0, round(entry.confidence_score - penalty, 2))
        self.entries.append(entry)
        self.emo_version += 1

    def apply_revision(self, agent_id: str, revision: DecisionRevision) -> None:
        """Downstream agent overturns an upstream decision. Original preserved."""
        for entry in self.entries:
            if entry.agent_id == agent_id:
                entry.revision = revision
                self.emo_version += 1
                return
        raise ValueError(f"No entry found for agent_id='{agent_id}' to revise.")

    # ── hallucination guard ────────────────────────────────────────────────

    def _hallucination_check(self, entry: AgentEntry) -> list[str]:
        """
        Verify that claims in `decision` are grounded in evidence.
        Returns list of ungrounded claim fragments (empty = all good).
        Simple heuristic: decision words not found in any evidence content.
        Extend with LLM-based verification in production.
        """
        flags = []
        decision_words = set(entry.decision.lower().split())
        all_evidence_text = " ".join(e.content.lower() for e in entry.evidence)
        # Check for vendor/product names in decision that appear in no evidence
        import re
        named_entities = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', entry.decision)
        for entity in named_entities:
            if entity.lower() not in all_evidence_text and len(entity) > 3:
                flags.append(f"'{entity}' in decision has no evidence backing")
        return flags

    # ── read / query ────────────────────────────────────────────────────────

    def query(self, field: str, filter_str: Optional[str] = None) -> Any:
        """
        Query a specific field across all entries.
        Examples:
            emo.query("confidence_score")
            emo.query("evidence", "critical")
            emo.query("rejected_alternatives", "risk_score > 0.7")
            emo.query("rejected_alternatives", "compliance_gap")
            emo.query("revision")
            emo.query("hallucination_flags")
        """
        results = []

        for entry in self.entries:
            val = getattr(entry, field, None)
            if val is None:
                continue

            if isinstance(val, list):
                for item in val:
                    if filter_str is None:
                        results.append(item)
                    elif _matches(item, filter_str):
                        results.append(item)
            else:
                if filter_str is None:
                    results.append(val)
                elif _matches(val, filter_str):
                    results.append(val)

        if field == "confidence_score" and results:
            return results[-1]

        if field == "revision":
            revisions = [e.revision for e in self.entries if e.revision]
            return revisions if revisions else None

        return results

    def weighted_evidence(self) -> list[dict]:
        """Return all evidence with current Ebbinghaus decay weights."""
        now = datetime.now(timezone.utc)
        result = []
        for entry in self.entries:
            for ev in entry.evidence:
                result.append({
                    "content": ev.content,
                    "source": ev.source,
                    "critical": ev.critical,
                    "grounded": ev.grounded,
                    "decay_weight": ev.decay_weight(now),
                })
        return sorted(result, key=lambda x: -x["decay_weight"])

    def latest_decision(self) -> str:
        for entry in reversed(self.entries):
            if entry.revision:
                return entry.revision.new_decision
        for entry in reversed(self.entries):
            if entry.decision:
                return entry.decision
        return ""

    def latest_confidence(self) -> float:
        if not self.entries:
            return 0.0
        return self.entries[-1].confidence_score

    def chain_confidence(self) -> float:
        if not self.entries:
            return 0.0
        return round(sum(e.confidence_score for e in self.entries) / len(self.entries), 2)

    def was_overturned(self) -> bool:
        return any(e.revision for e in self.entries)

    def has_hallucination_flags(self) -> bool:
        return any(e.hallucination_flags for e in self.entries)

    def all_hallucination_flags(self) -> list[str]:
        flags = []
        for e in self.entries:
            flags.extend(e.hallucination_flags)
        return flags

    def revision_diff(self) -> Optional[dict]:
        """Return git-style diff if any decision was overturned."""
        for e in self.entries:
            if e.revision:
                return e.revision.as_diff()
        return None

    # ── serialization ───────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "raw_input": self.raw_input,
            "parent_emo_id": self.parent_emo_id,
            "emo_version": self.emo_version,
            "created_at": self.created_at,
            "entries": [asdict(e) for e in self.entries],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_dict(cls, data: dict) -> "EpisodicMemoryObject":
        emo = cls(
            task_id=data["task_id"],
            raw_input=data["raw_input"],
            parent_emo_id=data.get("parent_emo_id"),
        )
        emo.created_at = data.get("created_at", emo.created_at)
        emo.emo_version = data.get("emo_version", 1)
        for e in data.get("entries", []):
            revision = None
            if e.get("revision"):
                r = e["revision"]
                revision = DecisionRevision(
                    overturned_by=r["overturned_by"],
                    original_decision=r["original_decision"],
                    new_decision=r["new_decision"],
                    reason=r["reason"],
                    evidence_added=r.get("evidence_added", []),
                    timestamp=r.get("timestamp", ""),
                )
            entry = AgentEntry(
                agent_id=e["agent_id"],
                decision=e["decision"],
                evidence=[EvidenceItem(**i) for i in e.get("evidence", [])],
                rejected_alternatives=[RejectedOption(**r) for r in e.get("rejected_alternatives", [])],
                confidence_score=e.get("confidence_score", 1.0),
                reasoning_chain=e.get("reasoning_chain", []),
                revision=revision,
                risk_findings=e.get("risk_findings", []),
                hallucination_flags=e.get("hallucination_flags", []),
                timestamp=e.get("timestamp", ""),
            )
            emo.entries.append(entry)
        return emo

    @classmethod
    def from_json(cls, json_str: str) -> "EpisodicMemoryObject":
        return cls.from_dict(json.loads(json_str))

    def __repr__(self) -> str:
        return (
            f"<EMO task_id={self.task_id!r} v{self.emo_version} "
            f"agents={[e.agent_id for e in self.entries]} "
            f"confidence={self.latest_confidence()} "
            f"overturned={self.was_overturned()} "
            f"hallucination_flags={self.has_hallucination_flags()}>"
        )


# ── internal filter helper ──────────────────────────────────────────────────


def _matches(item: Any, filter_str: str) -> bool:
    if filter_str == "critical":
        return getattr(item, "critical", False)

    if filter_str == "compliance_gap":
        return getattr(item, "compliance_gap", False)

    # equality / inequality checks for boolean or simple values
    m = None
    try:
        import re
        m = re.match(r"^(?P<field>\w+)\s*(?P<op>==|!=)\s*(?P<val>.+)$", filter_str)
    except Exception:
        m = None

    if m:
        field = m.group('field')
        op = m.group('op')
        raw_val = m.group('val').strip()
        # normalize boolean tokens
        if raw_val.lower() in ("true", "false"):
            val = raw_val.lower() == "true"
        else:
            # try numeric
            try:
                if '.' in raw_val:
                    val = float(raw_val)
                else:
                    val = int(raw_val)
            except Exception:
                # strip quotes if present
                val = raw_val.strip('"\'')

        item_val = getattr(item, field, None)
        if op == '==':
            return item_val == val
        else:
            return item_val != val

    if filter_str == "ungrounded":
        return getattr(item, "grounded", True) is False

    if "risk_score >" in filter_str:
        threshold = float(filter_str.split(">")[1].strip())
        return getattr(item, "risk_score", 0.0) > threshold

    if "risk_score <" in filter_str:
        threshold = float(filter_str.split("<")[1].strip())
        return getattr(item, "risk_score", 0.0) < threshold

    item_str = str(item).lower()
    return filter_str.lower() in item_str
