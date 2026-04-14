# VADAR Visual Agent – Complete Setup Guide

This guide walks you through every step needed to get the VADAR Visual Agent
running on your machine, from prerequisites to environment verification.

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Install System Dependencies](#2-install-system-dependencies)
3. [Clone the Repository](#3-clone-the-repository)
4. [Create a Python Environment](#4-create-a-python-environment)
5. [Install Python Dependencies](#5-install-python-dependencies)
6. [API Key Configuration](#6-api-key-configuration)
7. [Model Download and Verification](#7-model-download-and-verification)
8. [Verify the Environment](#8-verify-the-environment)
9. [Troubleshooting Installation](#9-troubleshooting-installation)

---

## 1. System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Ubuntu 20.04 / macOS 12 / Windows 10 | Ubuntu 22.04 |
| Python | 3.9 | 3.10 or 3.11 |
| RAM | 16 GB | 32 GB |
| GPU VRAM | – (CPU works) | 8 GB (CUDA 11.8+) |
| Disk | 10 GB free | 30 GB free |
| Internet | Required (first run) | – |

---

## 2. Install System Dependencies

### Ubuntu / Debian

```bash
sudo apt-get update
sudo apt-get install -y \
    python3-pip python3-venv \
    libglib2.0-0 libgl1-mesa-glx \
    git curl
```

### macOS (Homebrew)

```bash
brew install python@3.11 git
```

### Windows

Install Python 3.10+ from <https://python.org>, make sure to check "Add to PATH".
Install [Git for Windows](https://git-scm.com/download/win).

---

## 3. Clone the Repository

```bash
git clone https://github.com/Sakshi2002-Sinha/vadar-visual-agent.git
cd vadar-visual-agent
```

---

## 4. Create a Python Environment

```bash
# Create a virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows PowerShell
```

You should now see `(venv)` at the start of your shell prompt.

---

## 5. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **GPU users** – install PyTorch with CUDA support *before* the rest:
>
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> pip install -r requirements.txt
> ```

### Verify the install

```bash
python -c "import torch, transformers, openai, cv2; print('All core packages OK')"
```

---

## 6. API Key Configuration

VADAR uses the **OpenAI API** for code generation.

1. Copy the example env file:

   ```bash
   cp .env.example .env
   ```

2. Open `.env` in your editor and set your key:

   ```
   OPENAI_API_KEY=sk-...your-key-here...
   ```

3. Load the variables into your shell (or configure your IDE to load `.env` automatically):

   ```bash
   export $(grep -v '^#' .env | xargs)   # Linux / macOS
   ```

> **Never commit `.env` to version control.** It is already listed in `.gitignore`.

---

## 7. Model Download and Verification

The three HuggingFace models are downloaded automatically on first use and
cached in `~/.cache/huggingface/`.

To pre-download and verify them now:

```bash
python - <<'EOF'
from transformers import pipeline

print("Downloading DETR object-detection model …")
pipeline("object-detection", model="facebook/detr-resnet-50")

print("Downloading DPT depth-estimation model …")
pipeline("depth-estimation", model="Intel/dpt-large")

print("Downloading DETR panoptic segmentation model …")
pipeline("image-segmentation", model="facebook/detr-resnet-50-panoptic")

print("All models downloaded successfully.")
EOF
```

Expected disk usage: ~2 GB total.

---

## 8. Verify the Environment

Run the bundled verification script:

```bash
python quickstart.py --verify-only
```

A passing run looks like:

```
[OK] Python 3.11.x
[OK] torch 2.1.x  (CUDA available: True)
[OK] transformers 4.38.x
[OK] openai 1.x.x
[OK] cv2 4.9.x
[OK] OPENAI_API_KEY is set
[OK] facebook/detr-resnet-50 – cached
[OK] Intel/dpt-large – cached
[OK] facebook/detr-resnet-50-panoptic – cached
Environment looks good!
```

---

## 9. Troubleshooting Installation

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed solutions to common
issues including:

- CUDA/PyTorch version mismatches
- OpenCV `libGL` errors
- Out-of-memory during model load
- `ModuleNotFoundError` on Windows
