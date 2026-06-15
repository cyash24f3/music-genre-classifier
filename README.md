---
title: Music Genre Classifier
emoji: 🎧
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

<div align="center">

# 🎧 Music Genre Classifier

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/cyash1204/music-genre-classifier)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://github.com/cyash24f3/music-genre-classifier)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-F97316?logo=gradio&logoColor=white)](https://gradio.app)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Classify music audio into 10 genres using an Audio Spectrogram Transformer**

*A Kaggle competition portfolio project by [Yash Chavan](https://github.com/cyash24f3)*

</div>

---

## 📸 Screenshots

<div align="center">
<em>Screenshots will be added after deployment</em>

<!-- 
![Main UI](docs/screenshots/main-ui.png)
![Classification Results](docs/screenshots/results.png)
-->
</div>

---

## 🎯 Overview

This project classifies music audio clips into **10 genres** using a pretrained **Audio Spectrogram Transformer (AST)** model. It was originally built for a Kaggle competition involving noisy music mashup classification, exploring multiple approaches from MFCC+XGBoost baselines to custom CNNs to state-of-the-art transformers.

### Supported Genres

| | | | | |
|:---:|:---:|:---:|:---:|:---:|
| 🎸 Blues | 🎻 Classical | 🤠 Country | 🕺 Disco | 🎤 Hip-Hop |
| 🎷 Jazz | 🤘 Metal | 🎵 Pop | 🇯🇲 Reggae | 🎸 Rock |

---

## 🏗️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Model** | [AST (Audio Spectrogram Transformer)](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) — pretrained on AudioSet-527 |
| **Backend** | FastAPI with async endpoints |
| **Frontend** | Gradio 4.x Blocks with custom Spotify-inspired dark theme |
| **Audio Processing** | librosa, torchaudio, soundfile |
| **Visualization** | matplotlib mel spectrograms |
| **Containerization** | Docker (Python 3.11-slim) |
| **Deployment** | Hugging Face Spaces (Docker SDK) |

---

## 🧠 How It Works

```
Audio File (.wav/.mp3/.flac/.ogg)
        │
        ▼
┌──────────────────┐
│  Load & Resample │  ← librosa @ 16 kHz, mono, ≤10s
│  to 16 kHz       │
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌──────────────────┐
│  Mel   │ │  AST Feature     │
│ Spec.  │ │  Extractor       │
│ (viz)  │ │  → Model Input   │
└────────┘ └────────┬─────────┘
                    │
                    ▼
          ┌──────────────────┐
          │  AST Inference   │  → 527 AudioSet class probs
          │  (Transformer)   │
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │  Genre Mapping   │  → Aggregate to 10 genres
          │  & Normalization │
          └──────────────────┘
```

The AST model outputs probabilities for 527 AudioSet classes. We aggregate related classes (e.g., "Rock music", "Punk rock", "Indie rock" → **Rock**) and normalize to produce genre confidence scores.

---

## 🚀 Quick Start

### Local Development

```bash
# Clone the repo
git clone https://github.com/cyash24f3/music-genre-classifier.git
cd music-genre-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
# Build
docker build -t music-genre-classifier .

# Run
docker run -p 7860:7860 music-genre-classifier
```

Then open [http://localhost:7860](http://localhost:7860) in your browser.

---

## 📡 API Reference

### Health Check

```http
GET /api/health
```

```json
{
  "status": "ok",
  "model": "AST-AudioSet",
  "genres": ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"],
  "loaded": true
}
```

### Classify Audio

```http
POST /api/classify
Content-Type: multipart/form-data

file: <audio_file.wav>
```

**Response:**

```json
{
  "predictions": [
    {"genre": "rock",    "confidence": 0.4231},
    {"genre": "metal",   "confidence": 0.2118},
    {"genre": "blues",   "confidence": 0.1054},
    ...
  ],
  "spectrogram": "<base64_encoded_png>"
}
```

**Supported formats:** `.wav`, `.mp3`, `.flac`, `.ogg`

---

## 🐳 Deploy to Hugging Face Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select **Docker** as the SDK
3. Push this repository:

```bash
git remote add hf https://huggingface.co/spaces/cyash1204/music-genre-classifier
git push hf main
```

The Space will auto-build and deploy on port 7860.

---

## 📁 Project Structure

```
music-genre-classifier/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI + Gradio app
│   └── model.py         # AST model, inference, spectrogram
├── Dockerfile           # Optimised container
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── .gitignore
```

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).

---

<div align="center">

**Built with ❤️ by [Yash Chavan](https://github.com/cyash24f3)**

*Portfolio Demo · Audio Spectrogram Transformer · Kaggle Competition Project*

</div>
