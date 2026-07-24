"""
utils.py — Shared utilities for HalluDet-Lite
Handles GPU management, logging, config loading, and caching.
"""

import os
import gc
import yaml
import json
import pickle
import logging
from pathlib import Path
from typing import Any, Optional

import torch
import numpy as np

# ── Logging setup ────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s — %(message)s",
            datefmt="%H:%M:%S"
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

log = get_logger("halludet")


# ── GPU Memory Management ────────────────────────────────────────────────────

def get_device() -> str:
    """Return 'cuda' if available and has VRAM, else 'cpu'."""
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.1f}GB VRAM)")
        return "cuda"
    log.warning("No CUDA GPU found — running on CPU")
    return "cpu"


def free_gpu_memory():
    """
    Aggressively free GPU memory.
    Call this between loading different models on RTX 2050 4GB.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    log.debug("GPU memory freed")


def move_model_to_cpu(model):
    """
    Move a model to CPU and free GPU memory.
    Use this before loading the next model on RTX 2050.

    Example:
        nli_model = move_model_to_cpu(nli_model)
        free_gpu_memory()
        # Now safe to load TinyLLaMA
    """
    if hasattr(model, 'to'):
        model.to('cpu')
    free_gpu_memory()
    return model


def vram_usage_gb() -> float:
    """Return current VRAM usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(0) / 1e9
    return 0.0


def check_vram_safe(required_gb: float, buffer_gb: float = 0.5) -> bool:
    """
    Check if there is enough free VRAM to load a model.

    Args:
        required_gb: How much VRAM the model needs.
        buffer_gb: Safety buffer (default 0.5GB).

    Returns:
        True if safe to load, False otherwise.
    """
    if not torch.cuda.is_available():
        return True  # CPU — no VRAM limit
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    used = vram_usage_gb()
    free = total - used
    ok = free >= (required_gb + buffer_gb)
    if not ok:
        log.warning(
            f"Insufficient VRAM: need {required_gb:.1f}GB + {buffer_gb:.1f}GB buffer, "
            f"have {free:.1f}GB free ({used:.1f}/{total:.1f}GB used)"
        )
    return ok


# ── Config Loading ────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    """Load a YAML config file and return as dict."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    log.info(f"Config loaded: {config_path}")
    return cfg


def get_project_root() -> Path:
    """Return the project root directory (where this file's parent sits)."""
    return Path(__file__).parent.parent


# ── Caching Utilities ─────────────────────────────────────────────────────────

def save_cache(data: Any, path: str):
    """Save any Python object to a pickle file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    log.debug(f"Cached to {path}")


def load_cache(path: str) -> Optional[Any]:
    """Load a pickle cache file. Returns None if file doesn't exist."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
    log.debug(f"Loaded cache from {path}")
    return data


def save_json(data: Any, path: str):
    """Save data as a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Text Utilities ────────────────────────────────────────────────────────────

def truncate_text(text: str, max_words: int = 100) -> str:
    """Truncate text to at most max_words words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def batch_list(lst: list, batch_size: int):
    """Split a list into batches of batch_size."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


# ── Env Setup ─────────────────────────────────────────────────────────────────

def setup_environment():
    """
    Apply recommended environment settings for RTX 2050.
    Call this at the start of every script.
    """
    # Reduce VRAM fragmentation
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

    # Deterministic (optional — slows training slightly)
    # torch.use_deterministic_algorithms(True)

    # Suppress tokenizer parallelism warnings
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # HuggingFace cache directory
    cache_dir = get_project_root() / ".hf_cache"
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))
    os.environ.setdefault("HF_HOME", str(cache_dir))

    log.info("Environment configured for RTX 2050")


# Run setup automatically on import
setup_environment()
