# VADAR Repository – Clone & Setup Guide

A step-by-step guide to cloning the official VADAR repository, configuring your environment, and verifying the installation.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Cloning the Official VADAR Repository](#2-cloning-the-official-vadar-repository)
3. [Creating and Activating a Virtual Environment](#3-creating-and-activating-a-virtual-environment)
4. [Understanding `setup.sh`](#4-understanding-setupsh)
5. [Running `setup.sh`](#5-running-setupsh)
6. [Handling Different CUDA Versions](#6-handling-different-cuda-versions)
7. [Installing Additional Dependencies](#7-installing-additional-dependencies)
8. [Verifying Model Downloads](#8-verifying-model-downloads)
9. [Storing Your OpenAI API Key Securely](#9-storing-your-openai-api-key-securely)
10. [Expected Directory Structure After Setup](#10-expected-directory-structure-after-setup)
11. [Common Mistakes & Troubleshooting](#11-common-mistakes--troubleshooting)

---

## 1. Prerequisites

Before you begin, ensure the following tools are installed and meet the minimum version requirements.

### 1.1 Python 3.10+

```bash
python --version
# Expected: Python 3.10.x or higher
```

If Python is not installed or the version is too old, download it from <https://www.python.org/downloads/>.

### 1.2 Git

```bash
git --version
# Expected: git version 2.x.x
```

Install Git from <https://git-scm.com/downloads> if it is missing.

### 1.3 CUDA (for GPU acceleration)

CUDA is optional but strongly recommended for reasonable inference speed.

```bash
# Check whether an NVIDIA GPU is present and CUDA is available
nvidia-smi
# Look for "CUDA Version: XX.X" in the header line

# Alternatively, check the toolkit version
nvcc --version
```

| CUDA Version | Minimum Driver Version |
|---|---|
| 11.8 | 520.61+ |
| 12.0 | 525.60+ |
| 12.1 | 530.30+ |
| 12.2 | 535.54+ |

> **No GPU?** The project falls back to CPU inference automatically; setup still works, but model inference will be slower.

---

## 2. Cloning the Official VADAR Repository

```bash
# Clone the repo into a local folder called "vadar"
git clone https://github.com/Sakshi2002-Sinha/vadar-visual-agent.git vadar
cd vadar
```

Verify the remote is correctly set:

```bash
git remote -v
# origin  https://github.com/Sakshi2002-Sinha/vadar-visual-agent.git (fetch)
# origin  https://github.com/Sakshi2002-Sinha/vadar-visual-agent.git (push)
```

---

## 3. Creating and Activating a Virtual Environment

Using a dedicated virtual environment prevents dependency conflicts with other Python projects.

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows (Command Prompt)

```bat
python -m venv venv
venv\Scripts\activate.bat
```

### Windows (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

After activation your shell prompt will be prefixed with `(venv)`.

Upgrade pip inside the virtual environment:

```bash
pip install --upgrade pip
```

---

## 4. Understanding `setup.sh`

`setup.sh` is the one-shot installation script bundled with VADAR. It performs the following steps **in order**:

| Step | What it does |
|---|---|
| 1 | Detects your CUDA version (or falls back to CPU) |
| 2 | Installs the correct PyTorch build (CUDA-specific wheel or CPU wheel) |
| 3 | Installs `transformers`, `accelerate`, and other ML dependencies |
| 4 | Downloads the required pretrained model weights into `models/` |
| 5 | Installs the remaining Python dependencies from `requirements.txt` |
| 6 | Runs a short smoke-test to confirm the models load correctly |

Reading `setup.sh` before running it is always a good habit:

```bash
cat setup.sh
```

---

## 5. Running `setup.sh`

Make the script executable and run it:

```bash
chmod +x setup.sh
./setup.sh
```

The script will print progress messages. A successful run ends with:

```
✓ All models downloaded and verified.
✓ Setup complete. Activate your venv and run: python vadar_agent.py
```

> **Tip:** If setup fails partway through, re-running `./setup.sh` is safe — it skips steps that have already succeeded.

---

## 6. Handling Different CUDA Versions

If `setup.sh` selects the wrong PyTorch build, you can override it manually.

### Find your CUDA version

```bash
nvcc --version | grep "release"
# release 11.8, V11.8.89
```

### Install PyTorch manually

Replace `cu118` with your actual CUDA version tag (e.g., `cu121` for CUDA 12.1, `cpu` for CPU-only).

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.2
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu122

# CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Verify the installation

```python
python - <<'EOF'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF
```

---

## 7. Installing Additional Dependencies

After running `setup.sh`, install any extra packages your use-case requires:

```bash
# Core dependencies (already handled by setup.sh, listed for reference)
pip install -r requirements.txt

# Jupyter support
pip install jupyter ipywidgets

# Evaluation / visualization extras
pip install matplotlib scikit-image

# If you need the OpenAI Python SDK
pip install openai
```

---

## 8. Verifying Model Downloads

VADAR relies on several pretrained models. Confirm they were downloaded correctly:

```bash
# List the models directory
ls -lh models/

# Expected files (names may vary by version):
#   detr-resnet-50/
#   dpt-large/
#   detr-resnet-50-panoptic/
```

Run the automated verification script:

```bash
python verify_clone_setup.py
```

This script checks that every required model directory and key file exists and reports what is ready and what still needs attention.

---

## 9. Storing Your OpenAI API Key Securely

**Never hard-code API keys** in source files or commit them to version control.

### Option A – `.env` file (recommended for development)

1. Create a `.env` file in the project root:

   ```bash
   echo 'OPENAI_API_KEY="sk-..."' > .env
   ```

2. Add `.env` to `.gitignore` (it should already be there):

   ```bash
   grep '.env' .gitignore || echo '.env' >> .gitignore
   ```

3. Load the key in Python using `python-dotenv`:

   ```bash
   pip install python-dotenv
   ```

   ```python
   from dotenv import load_dotenv
   import os
   load_dotenv()
   api_key = os.getenv("OPENAI_API_KEY")
   ```

### Option B – Shell environment variable (session-only)

```bash
# Linux / macOS
export OPENAI_API_KEY="sk-..."

# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."
```

### Option C – Persistent shell profile (Linux / macOS)

```bash
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.bashrc
source ~/.bashrc
```

---

## 10. Expected Directory Structure After Setup

```
vadar/
├── agents/                  # Agent modules
├── engine/                  # Core reasoning engine
├── models/                  # Downloaded pretrained model weights
│   ├── detr-resnet-50/
│   ├── dpt-large/
│   └── detr-resnet-50-panoptic/
├── prompts/                 # LLM prompt templates
├── venv/                    # Virtual environment (not committed)
├── .env                     # API keys (not committed)
├── .gitignore
├── CLONE_AND_SETUP_GUIDE.md
├── clone_and_setup.sh
├── config.yaml
├── evaluate_benchmark.py
├── requirements.txt
├── setup.sh
├── setup_windows.ps1
├── vadar_agent.py
└── verify_clone_setup.py
```

---

## 11. Common Mistakes & Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'torch'` | Virtual environment not activated | Run `source venv/bin/activate` (Linux/macOS) or `venv\Scripts\activate` (Windows) |
| CUDA out-of-memory error | GPU VRAM insufficient | Reduce batch size or switch to CPU with `--device cpu` |
| Models not found in `models/` | `setup.sh` was interrupted | Re-run `./setup.sh` |
| `openai.error.AuthenticationError` | API key not set or incorrect | Verify `echo $OPENAI_API_KEY` returns your key |
| Permission denied on `setup.sh` | Script not executable | Run `chmod +x setup.sh` |
| Wrong PyTorch CUDA build | CUDA version mismatch | Follow [Section 6](#6-handling-different-cuda-versions) to reinstall manually |

---

> **Need help?** Open an issue at <https://github.com/Sakshi2002-Sinha/vadar-visual-agent/issues>.
