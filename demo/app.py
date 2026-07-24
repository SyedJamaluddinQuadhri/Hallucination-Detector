"""
app.py — HalluDet-Lite Gradio Demo Application.

Runs the full hallucination detection pipeline with a live web UI.
Features:
  - Real-time claim-by-claim analysis
  - Color-coded hallucination highlighting (red/amber/green)
  - Evidence display for each claim
  - Entity-level suspicion markers
  - Shareable public URL (share=True)

Run: python demo/app.py
     Opens at: http://localhost:7860
     Public URL: printed in terminal (valid for 72 hours)

Hardware: RTX 2050 4GB — runs full pipeline in ~3–8 seconds per response.
"""

import os
import sys
import json
import time
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import spacy

from src.utils import get_logger, free_gpu_memory, get_project_root
from src.phase1_nli.predict       import NLIPredictor
from src.phase1_nli.claim_splitter import split_claims
from src.phase2_uncertainty.ensemble   import EnsembleDisagreement
from src.phase2_uncertainty.perplexity import TinyLLaMaPerplexity
from src.phase3_rav.retriever     import LightweightRAV
from src.phase4_ensemble.meta_learner  import MetaLearnerPredictor

log = get_logger("demo")

# ── Global model instances (loaded once) ─────────────────────────────────────
# NOTE: On RTX 2050 we load models sequentially, not simultaneously.
# NLI runs on GPU. BART runs on CPU. TinyLLaMA loads on GPU when NLI is offloaded.

_models_loaded = False
_nli    = None
_ens    = None
_ppl    = None
_rav    = None
_meta   = None
_nlp_sm = None


def load_all_models():
    """Load all pipeline models. Called once on startup."""
    global _models_loaded, _nli, _ens, _ppl, _rav, _meta, _nlp_sm

    if _models_loaded:
        return

    log.info("Loading pipeline models...")
    t0 = time.time()

    # spaCy (CPU, always available)
    _nlp_sm = spacy.load("en_core_web_sm")
    log.info("spaCy loaded.")

    # Load TinyLLaMA FIRST. This prevents OS paging file exhaustion (os error 1455) 
    # because it temporarily requires larger RAM to map weights before moving to VRAM.
    _ppl = TinyLLaMaPerplexity(use_4bit=True)
    log.info("Perplexity scorer loaded.")

    # Phase 1: NLI on GPU
    _nli = NLIPredictor()
    log.info("NLI predictor loaded.")

    # Phase 2a: Ensemble (uses NLI internally + small model on GPU)
    # Disabled load_bart to save 1.6GB RAM and prevent paging file exhaustion (os error 1455)
    _ens = EnsembleDisagreement(nli_predictor=_nli, load_small=True, load_bart=False)
    log.info("Ensemble loaded.")
    # Phase 3: RAV (CPU FAISS + GPU NLI)
    _rav = LightweightRAV(nli_predictor=_nli)
    log.info("RAV loaded.")

    # Phase 4: Meta-learner (CPU)
    _meta = MetaLearnerPredictor()
    log.info("Meta-learner loaded.")

    _models_loaded = True
    log.info(f"All models loaded in {time.time()-t0:.1f}s")


# ── Core Inference ────────────────────────────────────────────────────────────

def analyze_response(question: str, response: str) -> list:
    """
    Run the full pipeline on a question + response pair.

    Returns a list of per-claim result dicts.
    """
    if not _models_loaded:
        load_all_models()

    claims = split_claims(response)
    if not claims:
        return []

    results = []
    for claim in claims:
        try:
            # Phase 1 — NLI
            nli_r = _nli.score_claim(question, claim)

            # Phase 2a — Ensemble
            ens_r = _ens.score(question, claim)

            # Phase 2b — Perplexity (TinyLLaMA on CPU for speed in demo)
            ppl_r = _ppl.score_claim(claim)

            # Phase 3 — RAV
            rav_r = _rav.verify_claim(claim)

            # Phase 4 — Meta-learner
            doc  = _nlp_sm(claim)
            feat = np.array([
                nli_r["probs"]["entailment"],
                nli_r["probs"]["contradiction"],
                ens_r["mean_entailment"],
                ens_r["disagreement"],
                min(ppl_r["log_perplexity"], 10.0),
                rav_r["max_ent"],
                max((e["retrieval_score"] for e in rav_r.get("evidence_used", [])), default=0.5),
                float(np.log1p(len(doc.ents))),
            ], dtype=np.float32)

            hal_prob = _meta.predict(feat)
            ents     = [(e.text, e.label_) for e in doc.ents]

            results.append({
                "claim":        claim,
                "hal_prob":     float(hal_prob),
                "verdict":      "HALLUCINATED" if hal_prob >= 0.5 else "SUPPORTED",
                "nli_verdict":  nli_r["verdict"],
                "nli_entail":   float(nli_r["probs"]["entailment"]),
                "disagreement": float(ens_r["disagreement"]),
                "perplexity":   float(ppl_r["perplexity"]),
                "rav_support":  rav_r["supported"],
                "evidence":     rav_r.get("best_doc", "")[:200],
                "evidence_src": rav_r.get("best_doc_title", ""),
                "entities":     ents,
            })
        except Exception as e:
            log.warning(f"Error analyzing claim '{claim[:40]}': {e}")
            results.append({
                "claim":    claim,
                "hal_prob": 0.5,
                "verdict":  "ERROR",
                "evidence": f"Processing error: {str(e)[:100]}",
                "evidence_src": "",
                "entities": [],
            })

    return results


# ── HTML Rendering ────────────────────────────────────────────────────────────

def score_to_color(score: float) -> tuple:
    """Return (background_hex, border_hex, label_color) for a hallucination score."""
    if score >= 0.7:
        return "#FEF2F2", "#EF4444", "#DC2626"   # red — high risk
    elif score >= 0.45:
        return "#FFFBEB", "#F59E0B", "#D97706"   # amber — uncertain
    else:
        return "#F0FDF4", "#22C55E", "#16A34A"   # green — supported


def build_result_html(question: str, response: str, claim_results: list) -> str:
    """Build the full HTML output for the Gradio display."""

    if not claim_results:
        return "<p style='color:#9CA3AF'>No claims detected in the response.</p>"

    overall_score = max(r["hal_prob"] for r in claim_results)

    # ── Overall score banner ──────────────────────────────────────────────────
    ov_bg, ov_brd, ov_txt = score_to_color(overall_score)
    risk_label = (
        "HIGH HALLUCINATION RISK" if overall_score >= 0.7
        else "POSSIBLE HALLUCINATION" if overall_score >= 0.45
        else "LIKELY FACTUAL"
    )
    html = f"""
    <div style="background:{ov_bg};border:2px solid {ov_brd};
                border-radius:12px;padding:16px 20px;margin-bottom:20px">
        <div style="display:flex;align-items:center;gap:12px">
            <div style="font-size:26px;font-weight:700;color:{ov_txt}">
                {overall_score*100:.0f}%
            </div>
            <div>
                <div style="font-size:15px;font-weight:600;color:{ov_txt}">
                    {risk_label}
                </div>
                <div style="font-size:12px;color:#6B7280;margin-top:2px">
                    {len(claim_results)} claim(s) analysed
                </div>
            </div>
        </div>
    </div>
    """

    # ── Per-claim results ─────────────────────────────────────────────────────
    html += "<div style='font-size:13px;font-weight:600;color:#374151;margin-bottom:10px'>Claim-by-claim analysis:</div>"

    for i, r in enumerate(claim_results, 1):
        bg, brd, txt = score_to_color(r["hal_prob"])
        verdict_icon = "✗" if r["hal_prob"] >= 0.5 else "✓"

        # Entity badges
        ent_html = ""
        if r.get("entities"):
            ent_html = "<div style='margin-top:6px;display:flex;flex-wrap:wrap;gap:4px'>"
            for ent_text, ent_type in r["entities"]:
                ent_html += (
                    f"<span style='background:#EDE9FE;color:#6D28D9;font-size:10px;"
                    f"padding:2px 7px;border-radius:10px'>{ent_text} "
                    f"<span style='opacity:0.7'>{ent_type}</span></span>"
                )
            ent_html += "</div>"

        # Signal breakdown
        signals = f"""
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-top:8px">
            <div style="background:#F3F4F6;border-radius:6px;padding:5px 8px;font-size:11px">
                <div style="color:#6B7280">NLI entailment</div>
                <div style="font-weight:600;color:#111827">{r['nli_entail']:.2f}</div>
            </div>
            <div style="background:#F3F4F6;border-radius:6px;padding:5px 8px;font-size:11px">
                <div style="color:#6B7280">Disagreement</div>
                <div style="font-weight:600;color:#111827">{r.get('disagreement',0):.3f}</div>
            </div>
            <div style="background:#F3F4F6;border-radius:6px;padding:5px 8px;font-size:11px">
                <div style="color:#6B7280">Perplexity</div>
                <div style="font-weight:600;color:#111827">{r.get('perplexity',0):.1f}</div>
            </div>
        </div>
        """

        evidence_html = ""
        if r.get("evidence"):
            evidence_html = f"""
            <div style="background:#F9FAFB;border-radius:6px;padding:8px 10px;
                        margin-top:8px;border-left:3px solid #D1D5DB">
                <div style="font-size:10px;color:#9CA3AF;margin-bottom:2px">
                    Best evidence: <b>{r.get('evidence_src','')}</b>
                </div>
                <div style="font-size:11px;color:#374151;line-height:1.5">
                    {r['evidence']}...
                </div>
            </div>
            """

        html += f"""
        <div style="background:{bg};border-left:4px solid {brd};
                    border-radius:0 8px 8px 0;padding:12px 14px;margin-bottom:10px">
            <div style="display:flex;align-items:flex-start;gap:8px">
                <span style="font-size:16px;font-weight:700;color:{txt};
                             min-width:20px">{verdict_icon}</span>
                <div style="flex:1">
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <span style="font-size:12px;font-weight:600;color:{txt}">
                            CLAIM {i}: {r['verdict']}
                        </span>
                        <span style="font-size:13px;font-weight:700;color:{txt}">
                            {r['hal_prob']*100:.0f}% risk
                        </span>
                    </div>
                    <div style="font-size:13px;color:#1F2937;margin-top:4px;
                                font-style:italic">
                        "{r['claim']}"
                    </div>
                    {ent_html}
                    {signals}
                    {evidence_html}
                </div>
            </div>
        </div>
        """

    # ── Legend ────────────────────────────────────────────────────────────────
    html += """
    <div style="display:flex;gap:16px;margin-top:16px;font-size:11px;color:#6B7280">
        <span style="display:flex;align-items:center;gap:4px">
            <span style="width:10px;height:10px;background:#22C55E;border-radius:50%;display:inline-block"></span>
            Supported (&lt;45%)
        </span>
        <span style="display:flex;align-items:center;gap:4px">
            <span style="width:10px;height:10px;background:#F59E0B;border-radius:50%;display:inline-block"></span>
            Uncertain (45–70%)
        </span>
        <span style="display:flex;align-items:center;gap:4px">
            <span style="width:10px;height:10px;background:#EF4444;border-radius:50%;display:inline-block"></span>
            Hallucinated (&gt;70%)
        </span>
    </div>
    <div style="margin-top:12px;font-size:11px;color:#9CA3AF">
        Powered by HalluDet-Lite · DeBERTa-base + MiniLM + TinyLLaMA + FAISS · RTX 2050
    </div>
    """

    return html


# ── Gradio Interface Function ─────────────────────────────────────────────────

def gradio_analyze(question: str, response: str):
    """Main Gradio callback."""
    if not question.strip():
        return "<p style='color:#EF4444'>Please enter a question.</p>"
    if not response.strip():
        return "<p style='color:#EF4444'>Please enter an LLM response to check.</p>"

    try:
        claim_results = analyze_response(question, response)
        return build_result_html(question, response, claim_results)
    except Exception as e:
        log.error(f"Pipeline error: {e}", exc_info=True)
        return f"<p style='color:#EF4444'>Error: {str(e)}</p>"


def get_json_output(question: str, response: str) -> str:
    """Return raw JSON output for the debug tab."""
    if not question.strip() or not response.strip():
        return "{}"
    try:
        results = analyze_response(question, response)
        return json.dumps(results, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


# ── Demo Examples ─────────────────────────────────────────────────────────────

EXAMPLES = [
    [
        "When did Albert Einstein win the Nobel Prize?",
        "Albert Einstein won the Nobel Prize in Physics in 1922 for his theory of relativity, which revolutionized our understanding of space and time.",
    ],
    [
        "What is the capital of Australia?",
        "The capital of Australia is Sydney, which is also the largest city and the main financial hub of the country.",
    ],
    [
        "Who wrote the play Romeo and Juliet?",
        "Romeo and Juliet was written by William Shakespeare around 1594–1596. It is one of his most famous tragedies.",
    ],
    [
        "What does DNA stand for?",
        "DNA stands for Deoxyribonucleic Acid. It carries the genetic information of living organisms.",
    ],
]


# ── Build Gradio App ──────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    """Construct the full Gradio application."""

    css = """
    .gradio-container { max-width: 900px !important; margin: 0 auto; }
    #title { text-align: center; margin-bottom: 8px; }
    #subtitle { text-align: center; color: #6B7280; margin-bottom: 24px; }
    """

    with gr.Blocks(
        title="HalluDet-Lite — LLM Hallucination Detector",
        css=css,
        theme=gr.themes.Soft(primary_hue="blue"),
    ) as demo:

        gr.HTML("""
        <div id="title">
            <h1 style="font-size:28px;font-weight:700;color:#1B3A6B;margin:0">
                🔍 HalluDet-Lite
            </h1>
            <p style="color:#6B7280;font-size:14px;margin:4px 0 0">
                Lightweight LLM Hallucination Detection · Runs on RTX 2050 4GB GPU
            </p>
        </div>
        """)

        with gr.Tabs():
            # ── Main Analysis Tab ─────────────────────────────────────────────
            with gr.Tab("Hallucination Detector"):
                with gr.Row():
                    with gr.Column(scale=1):
                        question_box = gr.Textbox(
                            label="Question / Prompt",
                            placeholder="e.g. When did Einstein win the Nobel Prize?",
                            lines=3,
                        )
                        response_box = gr.Textbox(
                            label="LLM Response to Check",
                            placeholder="Paste the LLM's response here...",
                            lines=6,
                        )
                        with gr.Row():
                            analyze_btn = gr.Button(
                                "Analyze for Hallucinations",
                                variant="primary",
                                size="lg",
                            )
                            clear_btn = gr.Button("Clear", size="lg")

                    with gr.Column(scale=1):
                        output_html = gr.HTML(
                            label="Analysis Results",
                            value="<p style='color:#9CA3AF;font-size:14px'>Results will appear here after analysis.</p>",
                        )

                gr.Examples(
                    examples=EXAMPLES,
                    inputs=[question_box, response_box],
                    label="Example Inputs (click to load)",
                )

                analyze_btn.click(
                    fn=gradio_analyze,
                    inputs=[question_box, response_box],
                    outputs=output_html,
                )
                clear_btn.click(
                    fn=lambda: ("", "", "<p style='color:#9CA3AF'>Results cleared.</p>"),
                    outputs=[question_box, response_box, output_html],
                )

            # ── Debug / JSON Tab ──────────────────────────────────────────────
            with gr.Tab("Raw JSON Output"):
                gr.Markdown("View the raw feature scores and per-claim data.")
                with gr.Row():
                    q2 = gr.Textbox(label="Question", lines=2)
                    r2 = gr.Textbox(label="Response", lines=4)
                json_btn = gr.Button("Get JSON")
                json_out = gr.Code(language="json", label="Raw pipeline output")
                json_btn.click(fn=get_json_output, inputs=[q2, r2], outputs=json_out)

            # ── About Tab ────────────────────────────────────────────────────
            with gr.Tab("About"):
                gr.Markdown("""
## HalluDet-Lite

**A consumer-hardware-first LLM hallucination detection framework.**

### How it works
1. **Phase 1 — NLI**: Fine-tuned DeBERTa-base checks if each claim is *entailed* by retrieved evidence
2. **Phase 2 — Uncertainty**: 3-model ensemble disagreement + TinyLLaMA perplexity score
3. **Phase 3 — RAV**: MiniLM + CPU FAISS retrieves Wikipedia evidence; NLI verifies each claim against it
4. **Phase 4 — Ensemble**: A calibrated 8-feature meta-learner MLP fuses all signals into a final probability

### Hardware
Runs entirely on **RTX 2050 (4GB VRAM)** — no cloud API, no A100 needed.

### Citation
If you use this work, please cite:
```
@article{halludet-lite-2025,
  title={HalluDet-Lite: Resource-Constrained Hallucination Detection on Consumer GPUs},
  author={Your Name},
  year={2025}
}
```
                """)

    return demo


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HalluDet-Lite Demo")
    parser.add_argument("--port",   type=int,  default=7860,  help="Port number")
    parser.add_argument("--share",  action="store_true",      help="Create public URL")
    parser.add_argument("--no-load", action="store_true",     help="Skip model loading (for UI testing)")
    args = parser.parse_args()

    if not args.no_load:
        log.info("Pre-loading models (this takes ~60s)...")
        load_all_models()

    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=args.share,
        show_error=True,
    )
