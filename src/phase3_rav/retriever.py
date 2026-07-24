"""
retriever.py — Retrieval-Augmented Verification (RAV) pipeline.

For each claim:
  1. Encode the claim with MiniLM (CPU)
  2. Retrieve top-5 most similar Wikipedia chunks (FAISS, CPU)
  3. Run NLI on each (claim, evidence) pair (GPU via NLIPredictor)
  4. Take max entailment score across all evidence
  5. If max_entailment < threshold → hallucination candidate
"""

import os
import json
import pickle
from typing import Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils import get_logger, get_project_root
from src.phase3_rav.build_index import INDEX_PATH, CHUNKS_PATH, ENCODER_MODEL

log = get_logger("rav.retriever")


class LightweightRAV:
    """
    Retrieval-Augmented Verification system using MiniLM + CPU FAISS.

    This is the lightweight alternative to the full Wikipedia index.
    Uses 500K domain-scoped chunks, CPU FAISS, and MiniLM for encoding.
    Zero VRAM used for retrieval — GPU only used for NLI verification.

    Usage:
        from src.phase1_nli.predict import NLIPredictor
        from src.phase3_rav.retriever import LightweightRAV

        nli = NLIPredictor()
        rav = LightweightRAV(nli_predictor=nli)

        result = rav.verify_claim("Einstein won the Nobel Prize in 1921.")
        print(result['supported'])           # True
        print(result['best_doc_title'])      # 'Albert Einstein'
        print(result['rav_hal_score'])       # 0.08
    """

    ENTAILMENT_THRESHOLD = 0.55  # Below this = claim is not supported

    def __init__(
        self,
        nli_predictor=None,
        index_path: Optional[str] = None,
        chunks_path: Optional[str] = None,
        encoder_model: str = ENCODER_MODEL,
        top_k: int = 5,
    ):
        """
        Args:
            nli_predictor: NLIPredictor instance from Phase 1.
                           If None, will try to load from default path.
            index_path:    Path to FAISS index file.
            chunks_path:   Path to chunks metadata JSON.
            encoder_model: Sentence encoder for query encoding (CPU).
            top_k:         Number of evidence passages to retrieve per claim.
        """
        # ── Load FAISS index ─────────────────────────────────────────────────
        if index_path is None:
            index_path = INDEX_PATH
        if chunks_path is None:
            chunks_path = CHUNKS_PATH

        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FAISS index not found at '{index_path}'. "
                "Run Phase 3 first: python scripts/run_phase3.py"
            )

        log.info(f"Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(index_path)
        log.info(f"FAISS index loaded: {self.index.ntotal:,} vectors")

        # ── Load chunks metadata ──────────────────────────────────────────────
        log.info(f"Loading chunks metadata from {chunks_path}...")
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        log.info(f"Chunks loaded: {len(self.chunks):,}")

        # ── Load query encoder (CPU) ──────────────────────────────────────────
        log.info(f"Loading query encoder: {encoder_model} (CPU)")
        self.encoder = SentenceTransformer(encoder_model, device="cpu")

        # ── NLI predictor ─────────────────────────────────────────────────────
        if nli_predictor is not None:
            self.nli = nli_predictor
        else:
            log.info("Loading NLI predictor...")
            from src.phase1_nli.predict import NLIPredictor
            self.nli = NLIPredictor()

        self.top_k = top_k
        log.info("LightweightRAV ready.")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Retrieve the top-K most relevant Wikipedia passages for a query.

        Args:
            query:  The claim text to search for.
            top_k:  Number of results (defaults to self.top_k).

        Returns:
            List of dicts: {title, text, chunk_id, retrieval_score}
        """
        k = top_k or self.top_k
        q_emb = self.encoder.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(q_emb.astype("float32"), k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = dict(self.chunks[idx])
            chunk["retrieval_score"] = float(score)
            results.append(chunk)

        return results

    def verify_claim(self, claim: str) -> Dict:
        """
        Verify a single atomic claim against the knowledge base.

        Args:
            claim: An atomic factual claim.

        Returns:
            Dict with:
                claim:           input claim
                supported:       bool — True if evidence supports the claim
                max_ent:         max entailment probability across evidence docs
                rav_hal_score:   1 - max_ent (hallucination risk)
                best_doc_title:  title of most supporting Wikipedia article
                best_doc:        text of most supporting passage
                evidence_used:   full evidence list with NLI scores
        """
        docs = self.retrieve(claim)

        if not docs:
            return {
                "claim":          claim,
                "supported":      False,
                "max_ent":        0.0,
                "rav_hal_score":  1.0,
                "best_doc_title": "No evidence found",
                "best_doc":       "",
                "evidence_used":  [],
            }

        evidence_results = []
        for doc in docs:
            # Truncate doc to 400 words for NLI speed
            doc_text = " ".join(doc["text"].split()[:400])
            nli_result = self.nli.score_claim(doc_text, claim)
            evidence_results.append({
                "title":           doc["title"],
                "text":            doc_text[:300],
                "retrieval_score": doc["retrieval_score"],
                "entailment":      nli_result["probs"]["entailment"],
                "contradiction":   nli_result["probs"]["contradiction"],
                "verdict":         nli_result["verdict"],
            })

        max_ent  = max(e["entailment"] for e in evidence_results)
        best_idx = max(range(len(evidence_results)),
                       key=lambda i: evidence_results[i]["entailment"])
        best_doc = evidence_results[best_idx]

        return {
            "claim":          claim,
            "supported":      max_ent >= self.ENTAILMENT_THRESHOLD,
            "max_ent":        float(max_ent),
            "rav_hal_score":  float(1.0 - max_ent),
            "best_doc_title": best_doc["title"],
            "best_doc":       best_doc["text"],
            "evidence_used":  evidence_results,
        }

    def verify_response(self, response: str) -> Dict:
        """
        Verify all claims in an LLM response.

        Args:
            response: Full LLM response text.

        Returns:
            Dict with per-claim results and aggregate statistics.
        """
        from src.phase1_nli.claim_splitter import split_claims
        claims = split_claims(response)

        if not claims:
            return {"claims": [], "results": [], "unsupported_rate": 0.0}

        results   = [self.verify_claim(c) for c in claims]
        unsupported = [r for r in results if not r["supported"]]

        return {
            "claims":           claims,
            "results":          results,
            "unsupported":      unsupported,
            "unsupported_rate": len(unsupported) / len(results),
            "avg_rav_score":    sum(r["rav_hal_score"] for r in results) / len(results),
        }
