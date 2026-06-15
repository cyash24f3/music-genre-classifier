"""
Music Genre Classification using Audio Spectrogram Transformer (AST).

Uses the pretrained MIT/ast-finetuned-audioset-10-10-0.4593 model from HuggingFace,
mapping its 527 AudioSet class predictions to 10 music genres.

Author: Yash Chavan
"""

import io
import base64
import logging
from typing import Optional

import numpy as np
import torch
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import ASTForAudioClassification, ASTFeatureExtractor

logger = logging.getLogger(__name__)

# ── Genre configuration ──────────────────────────────────────────────────────

GENRES: list[str] = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]

# Mapping from our 10 target genres → relevant AudioSet label substrings.
# Multiple AudioSet classes can map to a single genre; probabilities are summed.
AUDIOSET_GENRE_MAP: dict[str, list[str]] = {
    "blues":     ["Blues"],
    "classical": ["Classical music", "Orchestra", "Chamber music", "Opera",
                  "Symphon"],
    "country":   ["Country", "Banjo", "Bluegrass"],
    "disco":     ["Disco", "Dance music", "Electronic dance music",
                  "Techno", "House music"],
    "hiphop":    ["Hip hop music", "Rap", "Beatbox"],
    "jazz":      ["Jazz", "Swing music", "Bossa nova"],
    "metal":     ["Heavy metal", "Metal", "Thrash metal",
                  "Black metal", "Death metal", "Grindcore"],
    "pop":       ["Pop music", "Synthpop", "Electropop"],
    "reggae":    ["Reggae", "Ska", "Dub"],
    "rock":      ["Rock music", "Rock and roll", "Punk rock",
                  "Progressive rock", "Psychedelic rock", "Grunge",
                  "Alternative rock", "Indie rock"],
}

# ── Globals (lazy-loaded) ────────────────────────────────────────────────────

_model: Optional[ASTForAudioClassification] = None
_feature_extractor: Optional[ASTFeatureExtractor] = None
_genre_index_map: Optional[dict[str, list[int]]] = None
_device: Optional[torch.device] = None

TARGET_SR: int = 16_000
MAX_DURATION_SEC: float = 10.0


# ── Model lifecycle ──────────────────────────────────────────────────────────

def load_model() -> None:
    """Load the AST model and feature extractor, build genre→index mapping."""
    global _model, _feature_extractor, _genre_index_map, _device

    model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    logger.info("Loading AST model: %s", model_name)

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)
    _model = ASTForAudioClassification.from_pretrained(model_name).to(_device)
    _model.eval()

    # Build mapping: genre → list of AudioSet class indices whose label
    # contains one of the query substrings.
    id2label: dict[int, str] = _model.config.id2label  # type: ignore[assignment]
    _genre_index_map = {}
    for genre, keywords in AUDIOSET_GENRE_MAP.items():
        indices: list[int] = []
        for idx, label in id2label.items():
            if any(kw.lower() in label.lower() for kw in keywords):
                indices.append(int(idx))
        _genre_index_map[genre] = indices
        logger.info("Genre %-10s → %d AudioSet classes", genre, len(indices))

    logger.info("Model loaded on %s", _device)


def is_model_loaded() -> bool:
    """Check whether the model has been loaded."""
    return _model is not None


# ── Audio utilities ──────────────────────────────────────────────────────────

def load_audio(file_path: str) -> tuple[np.ndarray, int]:
    """
    Load an audio file, convert to mono, resample to TARGET_SR, and truncate
    to MAX_DURATION_SEC seconds.

    Returns:
        (waveform_1d, sample_rate)
    """
    waveform, sr = librosa.load(file_path, sr=TARGET_SR, mono=True,
                                duration=MAX_DURATION_SEC)
    return waveform, sr


def generate_mel_spectrogram_b64(waveform: np.ndarray, sr: int) -> str:
    """
    Generate a mel-spectrogram image from a waveform array and return it as a
    base64-encoded PNG string.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 3), dpi=120)
    fig.patch.set_facecolor("#0f0a1a")
    ax.set_facecolor("#0f0a1a")

    S = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=128,
                                       fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)

    img = librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel",
                                   ax=ax, cmap="magma", fmax=8000)
    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)
    cbar.set_label("dB", color="white", fontsize=9)

    ax.set_title("Mel Spectrogram", color="white", fontsize=11, pad=8)
    ax.tick_params(colors="white", labelsize=8)
    ax.set_xlabel("Time (s)", color="white", fontsize=9)
    ax.set_ylabel("Frequency (Hz)", color="white", fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    fig.tight_layout(pad=1.0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(),
                edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def generate_mel_spectrogram_image(waveform: np.ndarray, sr: int) -> np.ndarray:
    """
    Generate a mel-spectrogram and return it as an RGB numpy array (for Gradio).
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 3), dpi=120)
    fig.patch.set_facecolor("#0f0a1a")
    ax.set_facecolor("#0f0a1a")

    S = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=128,
                                       fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)

    img = librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel",
                                   ax=ax, cmap="magma", fmax=8000)
    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)
    cbar.set_label("dB", color="white", fontsize=9)

    ax.set_title("Mel Spectrogram", color="white", fontsize=11, pad=8)
    ax.tick_params(colors="white", labelsize=8)
    ax.set_xlabel("Time (s)", color="white", fontsize=9)
    ax.set_ylabel("Frequency (Hz)", color="white", fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    fig.tight_layout(pad=1.0)

    # Render to numpy array
    fig.canvas.draw()
    rgba = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    rgba = rgba.reshape((h, w, 4))
    # ARGB → RGB
    rgb = np.empty((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = rgba[:, :, 1]  # R
    rgb[:, :, 1] = rgba[:, :, 2]  # G
    rgb[:, :, 2] = rgba[:, :, 3]  # B
    plt.close(fig)
    return rgb


# ── Inference ────────────────────────────────────────────────────────────────

def classify_audio(file_path: str) -> dict:
    """
    Run full inference pipeline on an audio file.

    Returns:
        {
            "predictions": [{"genre": str, "confidence": float}, ...],  # sorted desc
            "spectrogram_b64": str,
        }
    """
    if _model is None or _feature_extractor is None or _genre_index_map is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    # 1. Load & preprocess audio
    waveform, sr = load_audio(file_path)

    # 2. Generate mel spectrogram for display
    spectrogram_b64 = generate_mel_spectrogram_b64(waveform, sr)

    # 3. Run AST inference
    inputs = _feature_extractor(
        waveform, sampling_rate=sr, return_tensors="pt", padding="max_length"
    )
    input_values = inputs["input_values"].to(_device)

    with torch.no_grad():
        logits = _model(input_values).logits  # (1, 527)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()  # (527,)

    # 4. Aggregate AudioSet probabilities → our 10 genres
    genre_scores: dict[str, float] = {}
    for genre in GENRES:
        indices = _genre_index_map[genre]
        if indices:
            genre_scores[genre] = float(np.sum(probs[indices]))
        else:
            genre_scores[genre] = 0.0

    # Normalize to sum to 1
    total = sum(genre_scores.values())
    if total > 0:
        genre_scores = {g: s / total for g, s in genre_scores.items()}

    # 5. Sort by confidence
    sorted_predictions = sorted(
        [{"genre": g, "confidence": round(c, 4)} for g, c in genre_scores.items()],
        key=lambda x: x["confidence"],
        reverse=True,
    )

    return {
        "predictions": sorted_predictions,
        "spectrogram_b64": spectrogram_b64,
    }


def classify_audio_for_gradio(file_path: str) -> tuple[str, dict[str, float], np.ndarray]:
    """
    Gradio-friendly wrapper. Returns:
        - top_label: str (e.g. "🎵  Rock — 87.3%")
        - genre_confidences: dict[str, float] for BarPlot
        - spectrogram_image: np.ndarray (RGB)
    """
    if _model is None or _feature_extractor is None or _genre_index_map is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    waveform, sr = load_audio(file_path)
    spectrogram_img = generate_mel_spectrogram_image(waveform, sr)

    # AST inference
    inputs = _feature_extractor(
        waveform, sampling_rate=sr, return_tensors="pt", padding="max_length"
    )
    input_values = inputs["input_values"].to(_device)

    with torch.no_grad():
        logits = _model(input_values).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    genre_scores: dict[str, float] = {}
    for genre in GENRES:
        indices = _genre_index_map[genre]
        genre_scores[genre] = float(np.sum(probs[indices])) if indices else 0.0

    total = sum(genre_scores.values())
    if total > 0:
        genre_scores = {g: s / total for g, s in genre_scores.items()}

    top_genre = max(genre_scores, key=genre_scores.get)  # type: ignore[arg-type]
    top_conf = genre_scores[top_genre]
    top_label = f"🎵  {top_genre.capitalize()} — {top_conf * 100:.1f}%"

    # Capitalize keys for display
    display_scores = {g.capitalize(): round(v, 4) for g, v in genre_scores.items()}

    return top_label, display_scores, spectrogram_img
