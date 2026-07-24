"""
predict.py — NLI inference for hallucination detection.

The NLIPredictor takes a (premise, hypothesis) pair and returns
entailment/neutral/contradiction probabilities.

For hallucination detection:
  hallucination_score = 1 - P(entailment)
  A claim is hallucinated if the evidence does NOT entail it.
"""

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.utils import get_logger, get_project_root, free_gpu_memory
from src.phase1_nli.dataset import LABEL2ID, ID2LABEL
from src.phase1_nli.claim_splitter import split_claims

log = get_logger("nli.predict")


class NLIPredictor:
    """
    NLI-based hallucination detector using fine-tuned DeBERTa-base.

    Usage:
        predictor = NLIPredictor()
        result = predictor.score_claim(
            premise="The Eiffel Tower is in Paris, France.",
            claim="The Eiffel Tower was built in 1889."
        )
        print(result['hallucination_score'])  # 0.12 (low — likely not hallucinated)
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 256,
    ):
        """
        Load the fine-tuned NLI model.

        Args:
            model_dir: Path to saved model directory.
                       Defaults to models/nli_detector/best
            device:    'cuda' or 'cpu'. Auto-detected if None.
            max_length: Max token length for inference.
        """
        if model_dir is None:
            model_dir = str(get_project_root() / "models" / "nli_detector" / "best")

        if not os.path.exists(model_dir):
            raise FileNotFoundError(
                f"NLI model not found at '{model_dir}'. "
                "Run Phase 1 training first: python scripts/run_phase1.py"
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.max_length = max_length

        log.info(f"Loading NLI model from {model_dir} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(device)
        self.model.eval()

        vram = torch.cuda.memory_allocated(0) / 1e9 if device == "cuda" else 0
        log.info(f"NLI model loaded. VRAM used: {vram:.2f}GB")

    def score_claim(self, premise: str, claim: str) -> Dict:
        """
        Score a single (premise, claim) pair.

        Args:
            premise: Evidence document / source text.
            claim:   The atomic claim to verify.

        Returns:
            Dict with keys:
                verdict:             'entailment' | 'neutral' | 'contradiction'
                hallucination_score: float in [0, 1] — 1 = definitely hallucinated
                probs:               dict of {label: probability}
                claim:               the input claim (for convenience)
        """
        encoding = self.tokenizer(
            premise,
            claim,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            # Use autocast for fp16 speed on RTX 2050
            if self.device == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = self.model(**encoding).logits
            else:
                logits = self.model(**encoding).logits

        probs = F.softmax(logits, dim=-1).squeeze().cpu().tolist()

        verdict_idx = int(torch.argmax(torch.tensor(probs)).item())
        verdict     = ID2LABEL[verdict_idx]

        # hallucination_score = 1 - P(entailment)
        # High score = the evidence does NOT support the claim
        hal_score = 1.0 - probs[LABEL2ID["entailment"]]

        return {
            "claim":               claim,
            "verdict":             verdict,
            "hallucination_score": float(hal_score),
            "probs": {
                "entailment":    float(probs[LABEL2ID["entailment"]]),
                "neutral":       float(probs[LABEL2ID["neutral"]]),
                "contradiction": float(probs[LABEL2ID["contradiction"]]),
            },
        }

    def score_response(
        self,
        evidence: str,
        response: str,
    ) -> Dict:
        """
        Score a full LLM response by splitting it into claims and
        scoring each against the evidence.

        Args:
            evidence: The source document or ground-truth context.
            response: The full LLM response text.

        Returns:
            Dict with:
                claims:               list of atomic claim strings
                claim_results:        list of score_claim results
                hallucination_rate:   fraction of claims not entailed
                overall_hal_score:    max hallucination score across claims
                flagged_claims:       claims with hal_score > 0.5
        """
        claims = split_claims(response)
        if not claims:
            return {
                "claims": [],
                "claim_results": [],
                "hallucination_rate": 0.0,
                "overall_hal_score": 0.0,
                "flagged_claims": [],
            }

        results = [self.score_claim(evidence, c) for c in claims]

        hal_count   = sum(1 for r in results if r["hallucination_score"] > 0.5)
        hal_rate    = hal_count / len(results)
        overall_hal = max(r["hallucination_score"] for r in results)
        flagged     = [r for r in results if r["hallucination_score"] > 0.5]

        return {
            "claims":               claims,
            "claim_results":        results,
            "hallucination_rate":   float(hal_rate),
            "overall_hal_score":    float(overall_hal),
            "flagged_claims":       flagged,
        }

    def score_batch(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: int = 16,
    ) -> List[Dict]:
        """
        Score a batch of (premise, claim) pairs efficiently.

        Args:
            pairs:      List of (premise, claim) tuples.
            batch_size: Inference batch size.

        Returns:
            List of score_claim result dicts.
        """
        results = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            premises = [p for p, _ in batch]
            claims   = [c for _, c in batch]

            encoding = self.tokenizer(
                premises,
                claims,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            encoding = {k: v.to(self.device) for k, v in encoding.items()}

            with torch.no_grad():
                if self.device == "cuda":
                    with torch.autocast("cuda", dtype=torch.float16):
                        logits = self.model(**encoding).logits
                else:
                    logits = self.model(**encoding).logits

            batch_probs = F.softmax(logits, dim=-1).cpu().tolist()

            for claim, probs in zip(claims, batch_probs):
                verdict_idx = int(probs.index(max(probs)))
                results.append({
                    "claim":               claim,
                    "verdict":             ID2LABEL[verdict_idx],
                    "hallucination_score": 1.0 - probs[LABEL2ID["entailment"]],
                    "probs": {
                        "entailment":    probs[LABEL2ID["entailment"]],
                        "neutral":       probs[LABEL2ID["neutral"]],
                        "contradiction": probs[LABEL2ID["contradiction"]],
                    },
                })

        return results

    def to_cpu(self):
        """Move model to CPU to free VRAM for the next model."""
        self.model.to("cpu")
        self.device = "cpu"
        free_gpu_memory()
        log.info("NLIPredictor moved to CPU. GPU memory freed.")
