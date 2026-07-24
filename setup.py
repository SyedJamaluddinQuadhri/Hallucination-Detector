"""
setup.py — Install HalluDet-Lite as a Python package.

Usage:
    pip install -e .    (editable install — recommended for development)
    pip install .       (standard install)
"""

from setuptools import setup, find_packages

setup(
    name="halludet-lite",
    version="1.0.0",
    description="Lightweight LLM Hallucination Detection for Consumer GPUs",
    author="Your Name",
    packages=find_packages(exclude=["tests*", "notebooks*", "colab*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "datasets>=2.19.0",
        "accelerate>=0.29.0",
        "peft>=0.10.0",
        "trl>=0.8.6",
        "faiss-cpu>=1.8.0",
        "sentence-transformers>=2.7.0",
        "spacy>=3.7.0",
        "scikit-learn>=1.4.0",
        "evaluate>=0.4.0",
        "numpy>=1.26.0",
        "gradio>=4.26.0",
        "pyyaml>=6.0.0",
        "wandb>=0.17.0",
    ],
    extras_require={
        "dev": ["jupyter", "matplotlib", "seaborn", "ipykernel"],
        "4bit": ["bitsandbytes>=0.43.0"],
    },
)
