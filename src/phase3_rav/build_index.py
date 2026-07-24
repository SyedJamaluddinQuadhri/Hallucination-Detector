"""
build_index.py — Build the domain-scoped Wikipedia FAISS index.

Downloads 100K Wikipedia articles via streaming (no full dump needed),
chunks them into 150-word passages, encodes with all-MiniLM-L6-v2,
and builds a CPU FAISS index.

Resources:
    - Storage: ~2GB (chunks JSON + FAISS index)
    - RAM:     ~4GB during build
    - Time:    ~90 minutes on Ryzen 7

Run: python scripts/run_phase3.py
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

from src.utils import get_logger, get_project_root, save_json

log = get_logger("rav.build_index")

# ── Paths ─────────────────────────────────────────────────────────────────────
KB_DIR         = get_project_root() / "data" / "knowledge_base"
CHUNKS_PATH    = str(KB_DIR / "wiki_chunks.json")
INDEX_PATH     = str(KB_DIR / "minilm_wiki.faiss")
ENCODER_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"

# ── Domain keywords — we only keep articles matching these ────────────────────
TARGET_KEYWORDS = [
    # Science & Technology
    "physics", "chemistry", "biology", "medicine", "mathematics",
    "astronomy", "technology", "computer", "software", "algorithm",
    "engineering", "quantum", "molecule", "species", "evolution",
    # History & Politics
    "war", "revolution", "empire", "century", "ancient", "medieval",
    "president", "prime minister", "government", "parliament", "treaty",
    "battle", "independence", "colonial", "dynasty",
    # Geography
    "country", "capital", "continent", "river", "mountain", "ocean",
    "city", "population", "located", "bordered",
    # People (common hallucination targets)
    "born", "died", "awarded", "scientist", "author", "philosopher",
    "musician", "athlete", "inventor", "politician", "economist",
    # Culture & Arts
    "novel", "film", "music", "painting", "architecture", "literature",
    # General knowledge
    "founded", "established", "discovered", "invented", "published",
]


# ── Chunking ─────────────────────────────────────────────────────────────────

def chunk_article(
    title: str,
    text: str,
    max_words: int = 150,
    overlap: int = 30,
) -> List[Dict]:
    """
    Split one Wikipedia article into overlapping 150-word chunks.

    Args:
        title:     Article title (used for provenance).
        text:      Full article text.
        max_words: Chunk size in words.
        overlap:   Overlap between consecutive chunks.

    Returns:
        List of chunk dicts: {title, text, chunk_id, start_word}
    """
    words = text.split()
    chunks = []
    start = 0
    chunk_num = 0

    while start < len(words):
        end = min(start + max_words, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append({
            "title":      title,
            "text":       chunk_text,
            "chunk_id":   f"{title[:50]}_{chunk_num}",
            "start_word": start,
        })
        chunk_num += 1
        start += max_words - overlap

    return chunks


# ── Article Download ──────────────────────────────────────────────────────────

def download_wiki_subset(
    target_articles: int = 100_000,
    max_words_per_article: int = 600,
) -> List[Dict]:
    """
    Stream Wikipedia and collect domain-relevant articles.

    Args:
        target_articles:        Stop after this many matched articles.
        max_words_per_article:  Truncate very long articles (saves storage).

    Returns:
        List of {title, text} dicts.
    """
    log.info(f"Streaming Wikipedia — target {target_articles:,} articles...")
    log.info("This may take 30–60 min depending on internet speed.")

    wiki = load_dataset(
        "wikipedia",
        "20220301.en",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    selected = []
    scanned = 0

    for article in wiki:
        scanned += 1
        title = article["title"]
        text  = article["text"]

        # Quick domain filter — check first 600 chars
        preview = (title + " " + text[:600]).lower()
        if any(kw in preview for kw in TARGET_KEYWORDS):
            # Truncate to save storage
            words = text.split()
            if len(words) > max_words_per_article:
                text = " ".join(words[:max_words_per_article])
            selected.append({"title": title, "text": text})

        if len(selected) >= target_articles:
            break

        if scanned % 50_000 == 0:
            log.info(f"  Scanned {scanned:,} articles, matched {len(selected):,}")

    log.info(f"Downloaded {len(selected):,} articles (scanned {scanned:,})")
    return selected


# ── Index Builder ─────────────────────────────────────────────────────────────

def build_knowledge_base(
    target_articles:     int = 100_000,
    max_words_per_chunk: int = 150,
    overlap_words:       int = 30,
    encoding_batch_size: int = 256,
    force_rebuild:       bool = False,
) -> Tuple[faiss.Index, List[Dict]]:
    """
    Full pipeline: download Wikipedia → chunk → encode → build FAISS index.

    Args:
        target_articles:     Number of Wikipedia articles to download.
        max_words_per_chunk: Words per passage chunk.
        overlap_words:       Overlap between chunks.
        encoding_batch_size: Sentences to encode per batch.
        force_rebuild:       Re-build even if index already exists.

    Returns:
        (faiss_index, chunk_metadata_list)
    """
    os.makedirs(str(KB_DIR), exist_ok=True)

    # ── Step 1: Check if already built ───────────────────────────────────────
    if not force_rebuild and os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        log.info("FAISS index already exists. Loading...")
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        log.info(f"Loaded existing index: {index.ntotal:,} vectors, {len(chunks):,} chunks")
        return index, chunks

    # ── Step 2: Download articles ─────────────────────────────────────────────
    articles = download_wiki_subset(target_articles)
    log.info(f"Saving articles to disk...")
    save_json(articles, str(KB_DIR / "wiki_articles.json"))

    # ── Step 3: Chunk all articles ────────────────────────────────────────────
    log.info("Chunking articles...")
    all_chunks = []
    for art in articles:
        all_chunks.extend(chunk_article(
            art["title"], art["text"],
            max_words=max_words_per_chunk,
            overlap=overlap_words,
        ))

    log.info(f"Total chunks: {len(all_chunks):,}")

    # Save chunks metadata
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f)
    log.info(f"Chunks saved to {CHUNKS_PATH}")

    # ── Step 4: Encode with MiniLM ────────────────────────────────────────────
    log.info(f"Loading encoder: {ENCODER_MODEL}")
    log.info("Encoding runs on CPU — no VRAM used.")
    encoder = SentenceTransformer(ENCODER_MODEL)

    DIM = 384  # MiniLM-L6-v2 output dimension
    index = faiss.IndexFlatIP(DIM)  # Exact inner product (= cosine after L2 norm)

    chunk_texts = [c["text"] for c in all_chunks]
    n_batches   = (len(chunk_texts) + encoding_batch_size - 1) // encoding_batch_size

    log.info(f"Encoding {len(chunk_texts):,} chunks in {n_batches:,} batches...")
    log.info("Expected time: ~90 minutes on Ryzen 7")

    for i in range(0, len(chunk_texts), encoding_batch_size):
        batch = chunk_texts[i:i + encoding_batch_size]
        embs  = encoder.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        index.add(embs.astype("float32"))

        if i % 10_000 == 0:
            progress = (i / len(chunk_texts)) * 100
            log.info(f"  {i:,} / {len(chunk_texts):,} ({progress:.1f}%)")

    # ── Step 5: Save index ────────────────────────────────────────────────────
    faiss.write_index(index, INDEX_PATH)
    log.info(f"FAISS index saved to {INDEX_PATH}")
    log.info(f"Index size: {index.ntotal:,} vectors | ~{index.ntotal * DIM * 4 / 1e9:.2f}GB")

    return index, all_chunks


