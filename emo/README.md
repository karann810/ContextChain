EMO — Episodic Memory Object (standalone)

This package provides the `EpisodicMemoryObject` primitive used across the ContextChain project.

Public API
- `EpisodicMemoryObject(task_id, raw_input, parent_emo_id=None)` — create a new EMO
- `EpisodicMemoryObject.append(entry: AgentEntry)` — append an `AgentEntry`
- `EpisodicMemoryObject.from_dict(d)` / `from_json(str)` / `to_dict()` / `to_json()` — serialization helpers

Key dataclasses
- `EvidenceItem` — evidence item with decay weighting
- `AgentEntry` — agent-produced entry (decision + evidence)
- `RejectedOption`, `DecisionRevision`

Notes
- The package is intentionally lightweight and dependency-free.
- For packaging: include `emo` as a local package or install via `pip install -e .` from repository root.
