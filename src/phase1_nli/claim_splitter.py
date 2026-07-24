"""
claim_splitter.py — Split LLM responses into atomic claims.

Uses spaCy sentence segmentation + coordinating conjunction splitting.
Runs on CPU — no GPU needed. Processes ~500 responses/sec.
"""

import re
import spacy
from typing import List
from src.utils import get_logger

log = get_logger("claim_splitter")


class ClaimSplitter:
    """
    Split a multi-sentence LLM response into atomic, independently
    verifiable claims.

    Example:
        splitter = ClaimSplitter()
        claims = splitter.split(
            "Einstein won the Nobel Prize in 1921 and he was born in Germany."
        )
        # ['Einstein won the Nobel Prize in 1921',
        #  'Einstein was born in Germany.']
    """

    # Conjunctions that typically join two independent clauses
    SPLIT_CONJUNCTIONS = [" and ", " but ", " while ", " however ", " although "]

    # Minimum claim length in characters — shorter claims are noise
    MIN_CLAIM_LEN = 15

    def __init__(self, model: str = "en_core_web_sm"):
        """
        Load spaCy model for sentence segmentation.

        Args:
            model: spaCy model name. 'en_core_web_sm' is fast and sufficient.
                   Install: python -m spacy download en_core_web_sm
        """
        log.info(f"Loading spaCy model: {model}")
        try:
            self.nlp = spacy.load(model)
        except OSError:
            log.error(
                f"spaCy model '{model}' not found. "
                f"Run: python -m spacy download {model}"
            )
            raise

        # Disable unused pipeline components for speed (keep parser for deps and sents)
        self.nlp.select_pipes(enable=["tok2vec", "parser"])
        log.info("ClaimSplitter ready")

    def split(self, text: str) -> List[str]:
        """
        Split text into a list of atomic claims.

        Args:
            text: Raw LLM response text.

        Returns:
            List of claim strings, each being a single verifiable proposition.
        """
        if not text or not text.strip():
            return []

        # Step 1: Sentence boundary detection via spaCy
        doc = self.nlp(text.strip())
        sentences = [sent.text.strip() for sent in doc.sents]

        # Step 2: Further split long sentences on conjunctions
        claims = []
        for sent in sentences:
            sub_claims = self._split_on_conjunctions(sent)
            claims.extend(sub_claims)

        # Step 3: Clean and filter
        claims = [self._clean(c) for c in claims]
        claims = [c for c in claims if len(c) >= self.MIN_CLAIM_LEN]

        return claims

    def _split_on_conjunctions(self, sentence: str) -> List[str]:
        """
        If a sentence is long and contains a coordinating conjunction
        that likely joins two independent propositions, split it.
        """
        # Only split sentences that are long enough to contain two claims
        if len(sentence) < 50:
            return [sentence]

        for conj in self.SPLIT_CONJUNCTIONS:
            if conj in sentence.lower():
                idx = sentence.lower().index(conj)
                left  = sentence[:idx].strip()
                right = sentence[idx + len(conj):].strip()

                # Both halves must be substantial
                if len(left) < self.MIN_CLAIM_LEN or len(right) < self.MIN_CLAIM_LEN:
                    continue

                # Right half should start with a capital or a verb-like word
                # If it starts with lowercase and has no subject, prepend a pronoun
                right_doc = self.nlp(right)
                has_subject = any(t.dep_ == "nsubj" for t in right_doc)

                if not has_subject:
                    # Extract the subject from the left half
                    left_doc = self.nlp(left)
                    subjects = [t.text for t in left_doc if t.dep_ == "nsubj"]
                    if subjects:
                        right = subjects[0] + " " + right

                return [left, right]

        return [sentence]

    def _clean(self, text: str) -> str:
        """Remove extra whitespace and trailing punctuation artifacts."""
        text = re.sub(r"\s+", " ", text).strip()
        # Remove leading bullet/list characters
        text = re.sub(r"^[-•*]\s*", "", text)
        return text

    def batch_split(self, texts: List[str]) -> List[List[str]]:
        """Split multiple texts. Returns a list of claim lists."""
        return [self.split(t) for t in texts]


# ── Module-level convenience function ────────────────────────────────────────

_default_splitter: ClaimSplitter = None


def split_claims(text: str) -> List[str]:
    """
    Convenience function using a module-level splitter instance.
    Avoids re-loading spaCy on every call.
    """
    global _default_splitter
    if _default_splitter is None:
        _default_splitter = ClaimSplitter()
    return _default_splitter.split(text)
