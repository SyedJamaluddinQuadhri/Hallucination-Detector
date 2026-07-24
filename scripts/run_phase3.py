"""
run_phase3.py — Build the Wikipedia FAISS knowledge base.

Downloads 100K domain-relevant Wikipedia articles via streaming,
chunks them into 150-word passages, encodes with MiniLM-L6-v2,
and saves a CPU FAISS index.

Runtime:  ~90 minutes on Ryzen 7 (CPU encoding)
Storage:  ~2GB (chunks JSON + FAISS index)
RAM:      ~4GB peak during build

Usage:
    python scripts/run_phase3.py
    python scripts/run_phase3.py --articles 10000  # quick test with 10K articles
    python scripts/run_phase3.py --force           # rebuild if index exists
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import get_logger, get_project_root
from src.phase3_rav.build_index import build_knowledge_base

log = get_logger("script.phase3")


def test_retrieval(index, chunks):
    """Quick smoke test of the FAISS index."""
    from sentence_transformers import SentenceTransformer
    import numpy as np

    log.info("Testing retrieval...")
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    test_queries = [
        "Einstein Nobel Prize physics",
        "Eiffel Tower Paris France construction",
        "DNA deoxyribonucleic acid structure",
    ]

    for query in test_queries:
        q_emb = encoder.encode([query], normalize_embeddings=True)
        scores, idxs = index.search(q_emb.astype("float32"), 3)
        log.info(f"\n  Query: '{query}'")
        for s, i in zip(scores[0], idxs[0]):
            if 0 <= i < len(chunks):
                log.info(f"    [{s:.3f}] {chunks[i]['title']}: {chunks[i]['text'][:80]}...")


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Build knowledge base")
    parser.add_argument("--articles",  type=int,  default=100_000, help="Number of Wikipedia articles")
    parser.add_argument("--force",     action="store_true",        help="Rebuild even if index exists")
    parser.add_argument("--test-only", action="store_true",        help="Only test existing index")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("HalluDet-Lite — Phase 3: Knowledge Base Build")
    log.info("=" * 60)
    log.info(f"Target articles: {args.articles:,}")
    log.info(f"Output: data/knowledge_base/")
    log.info("TIP: Run overnight with: nohup python scripts/run_phase3.py > logs/phase3.log &")
    log.info("=" * 60)

    if args.test_only:
        # Test existing index
        import faiss, json
        from src.phase3_rav.build_index import INDEX_PATH, CHUNKS_PATH
        index  = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH) as f:
            chunks = json.load(f)
        test_retrieval(index, chunks)
        return

    index, chunks = build_knowledge_base(
        target_articles=args.articles,
        force_rebuild=args.force,
    )

    log.info(f"\nIndex built: {index.ntotal:,} vectors from {len(chunks):,} chunks")
    test_retrieval(index, chunks)

    log.info("\n✓ Phase 3 complete!")
    log.info("Next step: python scripts/run_phase4.py")


if __name__ == "__main__":
    main()
