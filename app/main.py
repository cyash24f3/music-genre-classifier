"""
Music Genre Classifier — FastAPI + Gradio Application.

Serves both a REST API and a Spotify-inspired Gradio web UI for classifying
music audio into 10 genres using an Audio Spectrogram Transformer.

Author: Yash Chavan
GitHub: https://github.com/cyash24f3/music-genre-classifier
"""

import logging
import tempfile
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

import gradio as gr
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.model import (
    classify_audio,
    classify_audio_for_gradio,
    is_model_loaded,
    load_model,
    GENRES,
)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}


# ── FastAPI lifecycle ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load model on startup."""
    logger.info("Starting Music Genre Classifier …")
    load_model()
    logger.info("Model ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Music Genre Classifier API",
    description="Classify music audio into 10 genres using Audio Spectrogram Transformer",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REST endpoints ───────────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    """Health-check endpoint."""
    return {
        "status": "ok",
        "model": "AST-AudioSet",
        "genres": GENRES,
        "loaded": is_model_loaded(),
    }


@app.post("/api/classify")
async def classify_endpoint(file: UploadFile = File(...)):
    """
    Classify an uploaded audio file.

    Accepts: .wav, .mp3, .flac, .ogg
    Returns: { predictions: [{genre, confidence}], spectrogram: base64_png }
    """
    # Validate extension
    suffix = Path(file.filename or "audio.wav").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{suffix}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    if not is_model_loaded():
        raise HTTPException(status_code=503, detail="Model is still loading.")

    # Save to temp file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        result = classify_audio(tmp_path)

        return JSONResponse(content={
            "predictions": result["predictions"],
            "spectrogram": result["spectrogram_b64"],
        })

    except Exception as e:
        logger.exception("Classification failed")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ── Gradio UI ────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* ── Global overrides ─────────────────────────────────────────────────────── */
.gradio-container {
    max-width: 960px !important;
    margin: 0 auto !important;
    font-family: 'Segoe UI', 'Inter', system-ui, sans-serif !important;
    background: linear-gradient(165deg, #0d0b14 0%, #130f1e 40%, #1a1128 100%) !important;
}

/* ── Header ───────────────────────────────────────────────────────────────── */
#header-row {
    text-align: center;
    padding: 18px 8px 6px 8px;
}
#header-row h1 {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a855f7 0%, #ec4899 50%, #f97316 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 2px;
    letter-spacing: -0.5px;
}
#subtitle-html {
    text-align: center;
}
#subtitle-html p, #subtitle-html a {
    color: #94a3b8 !important;
    font-size: 0.95rem;
}
#subtitle-html a {
    color: #c084fc !important;
    text-decoration: none;
    font-weight: 600;
}
#subtitle-html a:hover {
    text-decoration: underline;
    color: #e879f9 !important;
}

/* ── Cards / panels ───────────────────────────────────────────────────────── */
.gr-panel, .gr-box, .gr-form {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(168,85,247,0.15) !important;
    border-radius: 14px !important;
    backdrop-filter: blur(6px);
}

/* ── Top genre label ──────────────────────────────────────────────────────── */
#top-genre-label textarea {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    text-align: center !important;
    background: transparent !important;
    border: none !important;
    color: #e2e8f0 !important;
    min-height: 60px !important;
}

/* ── Bar chart ────────────────────────────────────────────────────────────── */
#confidence-bars {
    border-radius: 12px !important;
    overflow: hidden;
}

/* ── Spectrogram image ────────────────────────────────────────────────────── */
#spectrogram-img img {
    border-radius: 10px;
    border: 1px solid rgba(168,85,247,0.2);
}

/* ── Upload box ───────────────────────────────────────────────────────────── */
#audio-upload {
    border: 2px dashed rgba(168,85,247,0.35) !important;
    border-radius: 14px !important;
    transition: border-color 0.3s ease;
}
#audio-upload:hover {
    border-color: rgba(236,72,153,0.6) !important;
}

/* ── Buttons ──────────────────────────────────────────────────────────────── */
#classify-btn {
    background: linear-gradient(135deg, #a855f7, #ec4899) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    padding: 12px 32px !important;
    border-radius: 10px !important;
    letter-spacing: 0.3px;
    transition: opacity 0.2s ease, transform 0.1s ease;
}
#classify-btn:hover {
    opacity: 0.9;
    transform: translateY(-1px);
}
#clear-btn {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: #94a3b8 !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
}

/* ── Footer ───────────────────────────────────────────────────────────────── */
#footer-html {
    text-align: center;
    padding: 14px 0 6px 0;
}
#footer-html p {
    color: #64748b !important;
    font-size: 0.82rem;
}
#footer-html a {
    color: #a78bfa !important;
    text-decoration: none;
}

/* ── Section labels ───────────────────────────────────────────────────────── */
.gr-block > label, .gr-input-label {
    color: #cbd5e1 !important;
    font-weight: 600 !important;
}
"""


def build_gradio_app() -> gr.Blocks:
    """Construct the Gradio Blocks UI."""

    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.purple,
        secondary_hue=gr.themes.colors.pink,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        body_background_fill="#0d0b14",
        body_background_fill_dark="#0d0b14",
        block_background_fill="rgba(255,255,255,0.03)",
        block_background_fill_dark="rgba(255,255,255,0.03)",
        block_border_color="rgba(168,85,247,0.15)",
        block_border_color_dark="rgba(168,85,247,0.15)",
        input_background_fill="#1e1b2e",
        input_background_fill_dark="#1e1b2e",
        button_primary_background_fill="linear-gradient(135deg, #a855f7, #ec4899)",
        button_primary_text_color="white",
    )

    with gr.Blocks(
        theme=theme,
        css=CUSTOM_CSS,
        title="Music Genre Classifier",
        analytics_enabled=False,
    ) as demo:
        # ── Header ───────────────────────────────────────────────────────
        with gr.Row(elem_id="header-row"):
            gr.Markdown("# 🎧 Music Genre Classifier")

        gr.HTML(
            '<p>Powered by <strong>Audio Spectrogram Transformer</strong> · '
            'A portfolio project by '
            '<a href="https://github.com/cyash24f3/music-genre-classifier" '
            'target="_blank">Yash Chavan</a></p>',
            elem_id="subtitle-html",
        )

        # ── Main layout ─────────────────────────────────────────────────
        with gr.Row(equal_height=False):
            # Left column — input
            with gr.Column(scale=1, min_width=340):
                audio_input = gr.Audio(
                    label="Upload or Record Audio",
                    type="filepath",
                    sources=["upload", "microphone"],
                    elem_id="audio-upload",
                )
                with gr.Row():
                    classify_btn = gr.Button(
                        "🎵  Classify Genre",
                        variant="primary",
                        elem_id="classify-btn",
                    )
                    clear_btn = gr.ClearButton(
                        value="✕  Clear",
                        elem_id="clear-btn",
                    )

            # Right column — output
            with gr.Column(scale=1, min_width=340):
                top_label = gr.Textbox(
                    label="Predicted Genre",
                    interactive=False,
                    elem_id="top-genre-label",
                    lines=1,
                )
                confidence_bars = gr.Label(
                    label="Genre Confidence Scores",
                    num_top_classes=10,
                    elem_id="confidence-bars",
                )

        # ── Spectrogram ──────────────────────────────────────────────────
        with gr.Row():
            spectrogram_img = gr.Image(
                label="Mel Spectrogram",
                type="numpy",
                interactive=False,
                elem_id="spectrogram-img",
            )

        # ── Wiring ───────────────────────────────────────────────────────
        def on_classify(audio_path: str | None):
            if audio_path is None:
                gr.Warning("Please upload or record an audio clip first.")
                return "", {}, None

            try:
                label, scores, spec_img = classify_audio_for_gradio(audio_path)
                return label, scores, spec_img
            except Exception as exc:
                gr.Error(f"Classification failed: {exc}")
                logger.exception("Gradio classification error")
                return "Error", {}, None

        classify_btn.click(
            fn=on_classify,
            inputs=[audio_input],
            outputs=[top_label, confidence_bars, spectrogram_img],
        )
        clear_btn.add([audio_input, top_label, confidence_bars, spectrogram_img])

        # ── Footer ───────────────────────────────────────────────────────
        gr.HTML(
            "<p>🎧 Portfolio Demo · "
            '<a href="https://github.com/cyash24f3/music-genre-classifier" '
            'target="_blank">GitHub</a> · '
            '<a href="https://huggingface.co/spaces/cyash1204/music-genre-classifier" '
            'target="_blank">HF Space</a> · '
            "Audio Spectrogram Transformer on AudioSet-527</p>",
            elem_id="footer-html",
        )

    return demo


# ── Mount Gradio on FastAPI ──────────────────────────────────────────────────

gradio_app = build_gradio_app()
app = gr.mount_gradio_app(app, gradio_app, path="/")
