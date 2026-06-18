"""
ContextChain RAG — Vendor Knowledge Retriever
Converts the vendor KB into searchable chunks, embeds them with
sentence-transformers, stores in FAISS, and returns grounded
evidence items with source attribution.

Why this matters:
  Agent 2 was using LLM parametric memory for vendor facts — meaning
  it could hallucinate pricing, compliance scope, or feature support.
  This RAG layer gives it a ground-truth KB to query instead.
  Every EvidenceItem it returns has grounded=True and a cited source.
"""

from __future__ import annotations
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

_KB_PATH = Path(__file__).parent / "vendor_kb.json"

# ── chunk schema ────────────────────────────────────────────────────────────

@dataclass
class KBChunk:
    chunk_id: str
    vendor: str
    category: str          # pricing | compliance | features | risk | regions
    text: str              # natural-language sentence(s) for embedding
    raw_data: dict         # original structured data for exact facts


# ── build chunks from KB ────────────────────────────────────────────────────

def _build_chunks(kb: dict) -> list[KBChunk]:
    chunks = []
    for v in kb["vendors"]:
        name = v["name"]

        # ── pricing chunk ────────────────────────────────────────────────
        chunks.append(KBChunk(
            chunk_id=f"{name}::pricing",
            vendor=name,
            category="pricing",
            text=(
                f"{name} costs approximately ${v['pricing_per_tb_month_usd']:.0f} per TB per month. "
                f"{v['min_pricing_note']}"
            ),
            raw_data={"price_per_tb": v["pricing_per_tb_month_usd"], "note": v["min_pricing_note"]},
        ))

        # ── regions chunk ───────────────────────────────────────────────
        chunks.append(KBChunk(
            chunk_id=f"{name}::regions",
            vendor=name,
            category="regions",
            text=(
                f"{name} is available in regions: {', '.join(v['regions'])}. "
                f"Data residency supported: {v['compliance']['data_residency_supported']}. "
                f"Residency regions: {', '.join(v['compliance'].get('data_residency_regions', []))}."
            ),
            raw_data={
                "regions": v["regions"],
                "data_residency": v["compliance"]["data_residency_supported"],
                "residency_regions": v["compliance"].get("data_residency_regions", []),
            },
        ))

        # ── compliance chunk (one per compliance item) ──────────────────
        comp = v["compliance"]
        compliance_items = []
        for key in ["SOC2_Type_II", "ISO_27001", "GDPR", "HIPAA", "PCI_DSS", "FedRAMP"]:
            val = comp.get(key)
            if val is not None:
                compliance_items.append(f"{key}: {'YES' if val else 'NO'}")

        scope_note = comp.get("SOC2_scope_notes", "")
        chunks.append(KBChunk(
            chunk_id=f"{name}::compliance",
            vendor=name,
            category="compliance",
            text=(
                f"{name} compliance certifications — {'; '.join(compliance_items)}. "
                f"SOC2 scope detail: {scope_note}"
            ),
            raw_data={k: comp.get(k) for k in ["SOC2_Type_II","ISO_27001","GDPR","HIPAA","PCI_DSS","FedRAMP","SOC2_scope_notes","data_residency_supported"]},
        ))

        # ── features chunk ──────────────────────────────────────────────
        feat = v["features"]
        log_note = feat.get("object_level_logging_note", "")
        chunks.append(KBChunk(
            chunk_id=f"{name}::features",
            vendor=name,
            category="features",
            text=(
                f"{name} features — "
                f"Object-level logging: {'YES' if feat.get('object_level_logging') else 'NO'}. {log_note} "
                f"Versioning: {'YES' if feat.get('versioning') else 'NO'}. "
                f"Encryption at rest: {'YES' if feat.get('encryption_at_rest') else 'NO'}. "
                f"Cross-region replication: {'YES' if feat.get('cross_region_replication') else 'NO'}. "
                f"SLA uptime: {v['sla_uptime_percent']}%."
            ),
            raw_data={
                "object_level_logging": feat.get("object_level_logging"),
                "object_level_logging_note": log_note,
                "versioning": feat.get("versioning"),
                "sla_uptime": v["sla_uptime_percent"],
                "cross_region_replication": feat.get("cross_region_replication"),
            },
        ))

        # ── vendor risk chunk ───────────────────────────────────────────
        risk = v["vendor_risk"]
        chunks.append(KBChunk(
            chunk_id=f"{name}::risk",
            vendor=name,
            category="risk",
            text=(
                f"{name} vendor risk — "
                f"Financial stability: {risk['financial_stability']}. "
                f"Market position: {risk['market_position']}. "
                f"Lock-in risk: {risk['lock_in_risk']}. "
                f"Known outages 2023-2024: {risk['known_outages_2023_2024']}."
            ),
            raw_data=risk,
        ))

    return chunks


# ── RAG retriever ────────────────────────────────────────────────────────────

class VendorRetriever:
    """
    Embeds KB chunks once on init, then supports semantic search.
    Uses sentence-transformers (all-MiniLM-L6-v2) + FAISS.
    Falls back to keyword search if sentence-transformers unavailable.
    """

    def __init__(self, kb_path: Path = _KB_PATH):
        with open(kb_path) as f:
            self._kb = json.load(f)

        self._chunks = _build_chunks(self._kb)
        self._vendor_names = [v["name"] for v in self._kb["vendors"]]
        self._index = None
        self._embeddings = None
        self._model = None
        self._use_semantic = False

        self._init_semantic()

    def _init_semantic(self):
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            texts = [c.text for c in self._chunks]
            embs = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            self._embeddings = embs.astype("float32")
            dim = embs.shape[1]
            self._index = faiss.IndexFlatIP(dim)   # inner product = cosine on normalised vecs
            self._index.add(self._embeddings)
            self._use_semantic = True
        except Exception as e:
            # Graceful fallback — keyword BM25-lite
            self._use_semantic = False

    # ── public API ──────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> list[KBChunk]:
        """
        Semantic (or keyword fallback) search over KB chunks.
        Returns top_k most relevant chunks.
        """
        if self._use_semantic:
            return self._semantic_search(query, top_k)
        return self._keyword_search(query, top_k)

    def get_vendor(self, vendor_name: str) -> Optional[dict]:
        """Return full vendor dict by name (fuzzy match)."""
        name_lower = vendor_name.lower()
        for v in self._kb["vendors"]:
            if name_lower in v["name"].lower() or v["name"].lower() in name_lower:
                return v
        return None

    def search_vendor_category(self, vendor_name: str, category: str) -> Optional[KBChunk]:
        """Directly fetch a specific vendor+category chunk."""
        for chunk in self._chunks:
            if chunk.vendor.lower() in vendor_name.lower() or vendor_name.lower() in chunk.vendor.lower():
                if chunk.category == category:
                    return chunk
        return None

    def all_vendor_names(self) -> list[str]:
        return self._vendor_names

    def verify_claim(self, vendor_name: str, claim_type: str, claim_value) -> dict:
        """
        Ground-truth verification of a specific claim.
        Returns {verified: bool, actual_value: any, source: str, note: str}

        claim_type: 'SOC2_Type_II' | 'object_level_logging' | 'data_residency_india' |
                    'price_under_X' | 'sla_uptime' | 'hipaa'
        """
        vendor = self.get_vendor(vendor_name)
        if not vendor:
            return {"verified": False, "actual_value": None, "source": "KB", "note": f"Vendor '{vendor_name}' not in KB"}

        comp = vendor.get("compliance", {})
        feat = vendor.get("features", {})

        if claim_type == "SOC2_Type_II":
            val = comp.get("SOC2_Type_II", False)
            return {
                "verified": bool(val) == bool(claim_value),
                "actual_value": val,
                "source": f"KB::{vendor['name']}::compliance",
                "note": comp.get("SOC2_scope_notes", ""),
            }

        if claim_type == "object_level_logging":
            val = feat.get("object_level_logging", False)
            return {
                "verified": bool(val) == bool(claim_value),
                "actual_value": val,
                "source": f"KB::{vendor['name']}::features",
                "note": feat.get("object_level_logging_note", ""),
            }

        if claim_type == "data_residency_india":
            regions = comp.get("data_residency_regions", [])
            india_supported = any("india" in r.lower() or "ap-south" in r.lower() for r in regions)
            return {
                "verified": india_supported == bool(claim_value),
                "actual_value": india_supported,
                "source": f"KB::{vendor['name']}::compliance",
                "note": f"India-region data residency: {'supported' if india_supported else 'NOT supported'}. Regions: {regions}",
            }

        if claim_type.startswith("price_under_"):
            threshold = float(claim_type.split("_")[-1])
            price = vendor["pricing_per_tb_month_usd"]
            return {
                "verified": price < threshold,
                "actual_value": price,
                "source": f"KB::{vendor['name']}::pricing",
                "note": vendor.get("min_pricing_note", ""),
            }

        if claim_type == "hipaa":
            val = comp.get("HIPAA", False)
            return {
                "verified": bool(val) == bool(claim_value),
                "actual_value": val,
                "source": f"KB::{vendor['name']}::compliance",
                "note": "",
            }

        if claim_type == "sla_uptime":
            val = vendor.get("sla_uptime_percent", 0.0)
            return {
                "verified": val >= float(claim_value),
                "actual_value": val,
                "source": f"KB::{vendor['name']}::features",
                "note": f"Actual SLA: {val}%",
            }

        return {"verified": False, "actual_value": None, "source": "KB", "note": f"Unknown claim_type: {claim_type}"}

    # ── internal search ──────────────────────────────────────────────────────

    def _semantic_search(self, query: str, top_k: int) -> list[KBChunk]:
        import faiss
        q_emb = self._model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self._index.search(q_emb, min(top_k, len(self._chunks)))
        return [self._chunks[i] for i in indices[0] if i >= 0]

    def _keyword_search(self, query: str, top_k: int) -> list[KBChunk]:
        """Simple TF-based fallback."""
        query_terms = set(re.findall(r'\w+', query.lower()))
        scored = []
        for chunk in self._chunks:
            chunk_terms = set(re.findall(r'\w+', chunk.text.lower()))
            overlap = len(query_terms & chunk_terms)
            scored.append((overlap, chunk))
        scored.sort(key=lambda x: -x[0])
        return [c for _, c in scored[:top_k]]


# ── singleton ────────────────────────────────────────────────────────────────

_retriever: Optional[VendorRetriever] = None

def get_retriever() -> VendorRetriever:
    global _retriever
    if _retriever is None:
        _retriever = VendorRetriever()
    return _retriever
