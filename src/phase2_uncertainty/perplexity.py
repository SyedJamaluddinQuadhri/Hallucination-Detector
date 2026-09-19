"""
perplexity.py — Token-level perplexity using TinyLLaMA-1.1B.

TinyLLaMA-1.1B in 4-bit uses only ~0.8GB VRAM.
High perplexity on a named entity = the model was uncertain about it
= hallucination signal.

IMPORTANT FOR RTX 2050: Unload NLIPredictor before loading this.
    nli.to_cpu()
    free_gpu_memory()
    ppl = TinyLLaMaPerplexity()
"""

import os
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
import spacy

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.utils import get_logger, free_gpu_memory, get_project_root

log = get_logger("uncertainty.perplexity")


class TinyLLaMaPerplexity:
    """
    Token-level perplexity scorer using TinyLLaMA-1.1B (4-bit quantized).

    Perplexity measures how "surprised" the model is by its own output.
    High perplexity on a claim = the model was guessing = hallucination risk.

    Usage:
        scorer = TinyLLaMaPerplexity()
        result = scorer.score_claim("The Eiffel Tower was built in 1889.")
        print(result['perplexity'])  # e.g., 8.3 (low = model knows this)

        result2 = scorer.score_claim("Einstein won the prize in 1953.")
        print(result2['perplexity'])  # e.g., 89.2 (high = uncertain)
    """

    MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    def __init__(
        self,
        device: Optional[str] = None,
        use_4bit: bool = True,
    ):
        """
        Load TinyLLaMA in 4-bit quantization.

        Args:
            device:   'cuda' or 'cpu'. 4-bit only works on CUDA.
            use_4bit: Use BitsAndBytes 4-bit quantization (0.8GB VRAM).
                      Set False to use fp32 on CPU (~4.4GB RAM).
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        log.info(f"Loading TinyLLaMA on {device} (4bit={use_4bit})...")
        log.warning("Ensure NLI model is on CPU before loading this!")

        try:
            import gc
            gc.collect()
            if use_4bit and device == "cuda":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.MODEL_ID,
                    quantization_config=bnb_config,
                    device_map={"": 0},
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.MODEL_ID,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                )
                if device != "cpu":
                    self.model = self.model.to(device)
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        except (OSError, MemoryError, RuntimeError) as e:
            if "1455" in str(e) or "paging file" in str(e) or "out of memory" in str(e).lower() or isinstance(e, MemoryError):
                log.error(f"Failed to load TinyLLaMA due to memory/paging limits! Gracefully degrading Perplexity features. Details: {e}")
                self.model = None
                self.tokenizer = None
            else:
                raise e

        vram = torch.cuda.memory_allocated(0) / 1e9 if device == "cuda" else 0
        log.info(f"TinyLLaMA loaded. VRAM: {vram:.2f}GB")

        # Load spaCy for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            log.warning("spaCy en_core_web_sm not found. Entity scoring disabled.")
            self.nlp = None

    def score_claim(self, claim: str) -> Dict:
        """
        Compute perplexity score for a claim.

        Args:
            claim: The atomic claim text.

        Returns:
            Dict with:
                claim:          input text
                perplexity:     float — lower=model knows this, higher=uncertain
                log_perplexity: log-normalized perplexity
                confidence:     1 / perplexity (higher = more confident)
                token_info:     per-token log-probs (for debugging)
        """
        if self.model is None or self.tokenizer is None:
            return {
                "claim":          claim,
                "perplexity":     50.0,
                "log_perplexity": 5.0,
                "confidence":     0.02,
                "nll_loss":       5.0,
            }

        tokens = self.tokenizer(claim, return_tensors="pt")
        input_ids = tokens.input_ids

        if self.device == "cuda":
            input_ids = input_ids.to("cuda")

        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)

        # Loss = mean negative log-likelihood per token
        # Perplexity = exp(loss)
        loss = outputs.loss.item()
        perplexity = float(np.exp(loss))
        log_ppl = float(np.log1p(perplexity))  # log-normalised for feature use

        return {
            "claim":          claim,
            "perplexity":     perplexity,
            "log_perplexity": log_ppl,
            "confidence":     1.0 / max(perplexity, 1.0),
            "nll_loss":       loss,
        }

    def score_entities(self, claim: str) -> Dict:
        """
        Score individual named entities in a claim by perplexity.
        High-perplexity entities are hallucination suspects.

        Args:
            claim: The atomic claim text.

        Returns:
            Dict with:
                claim_perplexity:   overall claim perplexity
                entities:           {entity_text: {type, perplexity, suspicious}}
                suspect_entities:   list of entities with perplexity > threshold
        """
        overall = self.score_claim(claim)

        entity_scores = {}
        if self.nlp:
            doc = self.nlp(claim)
            for ent in doc.ents:
                ent_result = self.score_claim(ent.text)
                entity_scores[ent.text] = {
                    "type":       ent.label_,
                    "perplexity": ent_result["perplexity"],
                    # Threshold: perplexity > 50 is suspicious for short entity names
                    "suspicious": ent_result["perplexity"] > 50,
                }

        suspects = [
            {"entity": e, **info}
            for e, info in entity_scores.items()
            if info["suspicious"]
        ]

        return {
            "claim_perplexity": overall["perplexity"],
            "log_perplexity":   overall["log_perplexity"],
            "entities":         entity_scores,
            "suspect_entities": suspects,
            "has_suspects":     len(suspects) > 0,
        }

    def to_cpu(self):
        """Move model to CPU to free VRAM or delete if quantized."""
        try:
            if hasattr(self.model, 'to'):
                self.model.to("cpu")
        except ValueError as e:
            if "4-bit" in str(e) or "8-bit" in str(e):
                log.info("Quantized model cannot be moved to CPU via .to(). Deleting model to free VRAM.")
                try:
                    del self.model
                except AttributeError:
                    pass
                self.model = None
            else:
                raise e
                
        self.device = "cpu"
        free_gpu_memory()
        log.info("TinyLLaMA moved to CPU (or deleted to free memory).")
