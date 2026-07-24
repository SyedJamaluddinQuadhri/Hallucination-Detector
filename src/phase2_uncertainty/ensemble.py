"""
ensemble.py — Multi-model NLI ensemble for uncertainty estimation.

Instead of sampling one large LLM 15 times (impossible on 4GB VRAM),
we use THREE different NLI models and measure their DISAGREEMENT.
High disagreement = the models are uncertain = hallucination risk.

Models used (all fit in 4GB total, loaded sequentially):
  1. Your fine-tuned DeBERTa-base (primary, from Phase 1)
  2. cross-encoder/nli-deberta-v3-small (22M params, 0.2GB)
  3. facebook/bart-large-mnli (400M, run on CPU for diversity)

This is the 4GB-safe alternative to semantic entropy.
"""

from typing import Dict, List, Optional, Tuple

import torch
from transformers import pipeline as hf_pipeline

from src.utils import get_logger, free_gpu_memory, get_project_root
from src.phase1_nli.predict import NLIPredictor

log = get_logger("uncertainty.ensemble")


class EnsembleDisagreement:
    """
    3-model NLI ensemble for uncertainty-based hallucination detection.

    The key metric is disagreement (variance in entailment predictions):
    - All 3 models agree claim is supported → low uncertainty → not hallucinated
    - Models disagree → high uncertainty → hallucination risk

    Usage:
        ens = EnsembleDisagreement(nli_predictor=your_phase1_model)
        result = ens.score(
            premise="Einstein was born in Germany.",
            claim="Einstein won the Nobel Prize in 1921."
        )
        print(result['disagreement'])  # Low = certain, High = uncertain
    """

    def __init__(
        self,
        nli_predictor: Optional[NLIPredictor] = None,
        load_small: bool = True,
        load_bart: bool = True,
        device: Optional[str] = None,
    ):
        """
        Args:
            nli_predictor: Your Phase 1 fine-tuned NLIPredictor.
                           If None, will try to load from default path.
            load_small:    Load cross-encoder/nli-deberta-v3-small (GPU).
            load_bart:     Load facebook/bart-large-mnli (CPU).
            device:        'cuda' or 'cpu'.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # ── Model 1: Fine-tuned DeBERTa-base (from Phase 1) ──────────────────
        if nli_predictor is not None:
            self.m1 = nli_predictor
        else:
            log.info("Loading Phase 1 NLI model...")
            try:
                self.m1 = NLIPredictor()
            except FileNotFoundError:
                log.warning("Phase 1 model not found — using only models 2 and 3.")
                self.m1 = None

        # ── Model 2: Small cross-encoder (GPU, 0.2GB VRAM) ───────────────────
        self.m2 = None
        if load_small:
            log.info("Loading cross-encoder/nli-deberta-v3-small on GPU...")
            try:
                self.m2 = hf_pipeline(
                    "text-classification",
                    model="cross-encoder/nli-deberta-v3-small",
                    device=0 if device == "cuda" else -1,
                    truncation=True,
                    max_length=512,
                )
                log.info("Model 2 (small cross-encoder) loaded.")
            except Exception as e:
                log.warning(f"Could not load small model: {e}")

        # ── Model 3: BART-large NLI on CPU (big, but CPU frees GPU for 1+2) ──
        self.m3 = None
        if load_bart:
            log.info("Loading facebook/bart-large-mnli on CPU...")
            try:
                self.m3 = hf_pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",
                    device=-1,  # Always CPU — too big for 4GB alongside others
                )
                log.info("Model 3 (BART-large NLI) loaded on CPU.")
            except Exception as e:
                log.warning(f"Could not load BART: {e}")

        loaded = sum([
            self.m1 is not None,
            self.m2 is not None,
            self.m3 is not None
        ])
        log.info(f"Ensemble ready: {loaded}/3 models loaded.")

    def _score_m1(self, premise: str, claim: str) -> float:
        """Get entailment probability from Model 1 (fine-tuned DeBERTa-base)."""
        if self.m1 is None:
            return 0.5  # neutral fallback
        result = self.m1.score_claim(premise, claim)
        return result["probs"]["entailment"]

    def _score_m2(self, premise: str, claim: str) -> float:
        """Get entailment probability from Model 2 (small cross-encoder)."""
        if self.m2 is None:
            return 0.5
        try:
            text = f"{premise} [SEP] {claim}"
            result = self.m2(text, truncation=True, max_length=512)
            label = result[0]["label"].lower()
            score = result[0]["score"]
            # Map to entailment probability
            if label == "entailment":
                return float(score)
            elif label == "contradiction":
                return float(1.0 - score)
            else:
                return 0.5
        except Exception as e:
            log.debug(f"Model 2 error: {e}")
            return 0.5

    def _score_m3(self, premise: str, claim: str) -> float:
        """Get entailment probability from Model 3 (BART zero-shot NLI)."""
        if self.m3 is None:
            return 0.5
        try:
            result = self.m3(
                premise,
                candidate_labels=["true", "false"],
                hypothesis_template=f"According to the context, this is true: {claim}",
                multi_label=False,
            )
            labels = result["labels"]
            scores = result["scores"]
            true_idx = labels.index("true")
            return float(scores[true_idx])
        except Exception as e:
            log.debug(f"Model 3 error: {e}")
            return 0.5

    def score(self, premise: str, claim: str) -> Dict:
        """
        Score a (premise, claim) pair using all 3 models.

        Returns:
            Dict with:
                model_scores:       entailment prob from each model
                mean_entailment:    average across models
                disagreement:       variance (proxy for semantic entropy)
                ensemble_hal_score: combined hallucination signal
                n_models_used:      how many models contributed
        """
        scores = []

        e1 = self._score_m1(premise, claim)
        scores.append(("deberta_finetuned", e1))

        e2 = self._score_m2(premise, claim)
        scores.append(("deberta_small", e2))

        e3 = self._score_m3(premise, claim)
        scores.append(("bart_large", e3))

        entailments = [s for _, s in scores]
        mean_e = sum(entailments) / len(entailments)

        # Variance = disagreement signal
        variance = sum((e - mean_e) ** 2 for e in entailments) / len(entailments)

        # Combined score: low entailment + high disagreement = hallucination
        # Variance is weighted by 2x because disagreement is a strong signal
        ensemble_hal_score = (1.0 - mean_e) * 0.7 + variance * 2.0 * 0.3
        ensemble_hal_score = min(1.0, max(0.0, ensemble_hal_score))

        return {
            "claim":               claim,
            "model_scores":        {name: float(s) for name, s in scores},
            "mean_entailment":     float(mean_e),
            "disagreement":        float(variance),
            "ensemble_hal_score":  float(ensemble_hal_score),
            "n_models_used":       len(entailments),
        }

    def score_response(self, evidence: str, response: str) -> List[Dict]:
        """Score all claims in a response."""
        from src.phase1_nli.claim_splitter import split_claims
        claims = split_claims(response)
        return [self.score(evidence, c) for c in claims]
