#!/usr/bin/env bash
# =============================================================================
# clone_and_setup.sh – Automated setup script for the VADAR Visual Agent
#
# Usage:
#   chmod +x clone_and_setup.sh
#   ./clone_and_setup.sh
#
# What this script does:
#   1. Checks prerequisites (Python 3.10+, Git, optional CUDA)
#   2. Prompts user for confirmation at each major step
#   3. Clones the official VADAR repository
#   4. Creates and activates a Python virtual environment
#   5. Installs PyTorch (with CUDA or CPU fallback)
#   6. Runs setup.sh from VADAR (if present)
#   7. Installs remaining dependencies from requirements.txt
#   8. Verifies model downloads
#   9. Prompts for the OpenAI API key and saves it to .env
#  10. Prints a summary and next steps
# =============================================================================

set -euo pipefail

# ── Colour helpers ──────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; }
header()  { echo -e "\n${BOLD}${CYAN}══ $* ══${RESET}\n"; }

# ── Prompt helper ────────────────────────────────────────────────────────────
confirm() {
    local prompt="${1:-Continue?}"
    while true; do
        read -r -p "$(echo -e "${YELLOW}${prompt} [y/N]: ${RESET}")" answer
        case "${answer,,}" in
            y|yes) return 0 ;;
            n|no|"") return 1 ;;
            *) echo "Please answer y or n." ;;
        esac
    done
}

# ── Constants ────────────────────────────────────────────────────────────────
REPO_URL="https://github.com/Sakshi2002-Sinha/vadar-visual-agent.git"
TARGET_DIR="vadar"
VENV_DIR="venv"
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=10

# =============================================================================
# STEP 1 – Check prerequisites
# =============================================================================
header "Step 1 – Checking prerequisites"

# -- Python -------------------------------------------------------------------
PYTHON_BIN=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || true)
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [[ "$major" -gt "$MIN_PYTHON_MAJOR" ]] || \
           [[ "$major" -eq "$MIN_PYTHON_MAJOR" && "$minor" -ge "$MIN_PYTHON_MINOR" ]]; then
            PYTHON_BIN="$cmd"
            success "Python $ver found at $(command -v "$cmd")"
            break
        else
            warn "Found Python $ver, but Python $MIN_PYTHON_MAJOR.$MIN_PYTHON_MINOR+ is required."
        fi
    fi
done

if [[ -z "$PYTHON_BIN" ]]; then
    error "Python $MIN_PYTHON_MAJOR.$MIN_PYTHON_MINOR+ is not installed or not on PATH."
    error "Install it from https://www.python.org/downloads/ and re-run this script."
    exit 1
fi

# -- Git ----------------------------------------------------------------------
if command -v git &>/dev/null; then
    success "Git $(git --version | awk '{print $3}') found."
else
    error "Git is not installed. Install it from https://git-scm.com/downloads."
    exit 1
fi

# -- CUDA (optional) ----------------------------------------------------------
CUDA_VERSION=""
TORCH_EXTRA_INDEX=""

if command -v nvcc &>/dev/null; then
    CUDA_VERSION=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+" || true)
    success "CUDA $CUDA_VERSION detected via nvcc."
elif command -v nvidia-smi &>/dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" || true)
    success "CUDA $CUDA_VERSION detected via nvidia-smi."
else
    warn "No CUDA toolkit detected. PyTorch will be installed for CPU only."
fi

if [[ -n "$CUDA_VERSION" ]]; then
    # Map CUDA version to PyTorch wheel tag
    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
    CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)

    if   [[ "$CUDA_MAJOR" -eq 11 && "$CUDA_MINOR" -ge 8 ]]; then
        TORCH_EXTRA_INDEX="https://download.pytorch.org/whl/cu118"
    elif [[ "$CUDA_MAJOR" -eq 12 && "$CUDA_MINOR" -eq 0 ]]; then
        TORCH_EXTRA_INDEX="https://download.pytorch.org/whl/cu120"
    elif [[ "$CUDA_MAJOR" -eq 12 && "$CUDA_MINOR" -eq 1 ]]; then
        TORCH_EXTRA_INDEX="https://download.pytorch.org/whl/cu121"
    elif [[ "$CUDA_MAJOR" -ge 12 ]]; then
        TORCH_EXTRA_INDEX="https://download.pytorch.org/whl/cu122"
    else
        warn "CUDA $CUDA_VERSION is older than 11.8. Falling back to CPU PyTorch."
        TORCH_EXTRA_INDEX="https://download.pytorch.org/whl/cpu"
    fi

    info "Will use PyTorch index: $TORCH_EXTRA_INDEX"
else
    TORCH_EXTRA_INDEX="https://download.pytorch.org/whl/cpu"
fi

if ! confirm "Prerequisites look good. Proceed to clone the repository?"; then
    info "Aborted by user."
    exit 0
fi

# =============================================================================
# STEP 2 – Clone the repository
# =============================================================================
header "Step 2 – Cloning the VADAR repository"

if [[ -d "$TARGET_DIR" ]]; then
    warn "Directory '$TARGET_DIR' already exists."
    if confirm "Pull latest changes instead of re-cloning?"; then
        cd "$TARGET_DIR"
        git pull
        success "Repository updated."
    else
        info "Skipping clone. Using existing directory."
        cd "$TARGET_DIR"
    fi
else
    info "Cloning $REPO_URL into '$TARGET_DIR' ..."
    git clone "$REPO_URL" "$TARGET_DIR"
    cd "$TARGET_DIR"
    success "Repository cloned successfully."
fi

info "Remote configuration:"
git remote -v

if ! confirm "Repository is ready. Create a virtual environment now?"; then
    info "Aborted by user."
    exit 0
fi

# =============================================================================
# STEP 3 – Create virtual environment
# =============================================================================
header "Step 3 – Creating virtual environment"

if [[ -d "$VENV_DIR" ]]; then
    warn "Virtual environment '$VENV_DIR' already exists."
    if ! confirm "Re-create the virtual environment? (This will delete the existing one)"; then
        info "Using existing virtual environment."
    else
        rm -rf "$VENV_DIR"
        "$PYTHON_BIN" -m venv "$VENV_DIR"
        success "Virtual environment re-created."
    fi
else
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    success "Virtual environment created in '$VENV_DIR'."
fi

# Activate venv for the remainder of this script
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
success "Virtual environment activated ($(python --version))."

# Upgrade pip
pip install --quiet --upgrade pip
success "pip upgraded to $(pip --version | awk '{print $2}')."

if ! confirm "Virtual environment is ready. Install PyTorch now?"; then
    info "Aborted by user."
    exit 0
fi

# =============================================================================
# STEP 4 – Install PyTorch
# =============================================================================
header "Step 4 – Installing PyTorch"

info "Installing torch, torchvision, torchaudio from: $TORCH_EXTRA_INDEX"
pip install torch torchvision torchaudio --index-url "$TORCH_EXTRA_INDEX"

python - <<'PYEOF'
import torch
print(f"  torch      : {torch.__version__}")
print(f"  CUDA avail : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU        : {torch.cuda.get_device_name(0)}")
PYEOF

success "PyTorch installed successfully."

# =============================================================================
# STEP 5 – Run setup.sh (if present)
# =============================================================================
header "Step 5 – Running VADAR setup.sh"

if [[ -f "setup.sh" ]]; then
    info "Found setup.sh – here is what it will do:"
    echo "---"
    cat setup.sh
    echo "---"
    if confirm "Run setup.sh now?"; then
        chmod +x setup.sh
        ./setup.sh
        success "setup.sh completed."
    else
        warn "Skipped setup.sh. You can run it manually later: ./setup.sh"
    fi
else
    warn "setup.sh not found in this repository. Skipping."
fi

# =============================================================================
# STEP 6 – Install remaining Python dependencies
# =============================================================================
header "Step 6 – Installing Python dependencies"

if [[ -f "requirements.txt" ]]; then
    info "Installing from requirements.txt ..."
    pip install -r requirements.txt
    success "All dependencies installed."
else
    warn "requirements.txt not found. Installing common VADAR dependencies ..."
    pip install \
        openai \
        transformers \
        accelerate \
        Pillow \
        numpy \
        opencv-python \
        matplotlib \
        scikit-image \
        python-dotenv
    success "Common dependencies installed."
fi

# =============================================================================
# STEP 7 – Verify model downloads
# =============================================================================
header "Step 7 – Verifying model downloads"

if [[ -f "verify_clone_setup.py" ]]; then
    python verify_clone_setup.py
elif [[ -d "models" ]]; then
    info "Contents of models/:"
    ls -lh models/ || true
    success "Models directory found."
else
    warn "models/ directory not found. Models may be downloaded on first run."
fi

# =============================================================================
# STEP 8 – OpenAI API key
# =============================================================================
header "Step 8 – OpenAI API key"

ENV_FILE=".env"

if [[ -f "$ENV_FILE" ]] && grep -q "OPENAI_API_KEY" "$ENV_FILE" 2>/dev/null; then
    success ".env already contains OPENAI_API_KEY. Skipping."
else
    if confirm "Would you like to store your OpenAI API key in .env now?"; then
        while true; do
            read -r -s -p "$(echo -e "${YELLOW}Paste your OpenAI API key (input hidden): ${RESET}")" api_key
            echo
            if [[ -z "$api_key" ]]; then
                warn "No key entered. Skipping."
                break
            fi
            if [[ "$api_key" == sk-* ]]; then
                echo "OPENAI_API_KEY=\"$api_key\"" >> "$ENV_FILE"
                success "API key saved to $ENV_FILE."
                # Ensure .env is gitignored
                if ! grep -q "^\.env" .gitignore 2>/dev/null; then
                    echo ".env" >> .gitignore
                    info "Added .env to .gitignore."
                fi
                break
            else
                warn "Key does not start with 'sk-'. Please check and try again."
            fi
        done
    else
        info "Skipped. Set the key later with: echo 'OPENAI_API_KEY=\"sk-...\"' >> .env"
    fi
fi

# =============================================================================
# STEP 9 – Summary & next steps
# =============================================================================
header "Setup Summary & Next Steps"

echo -e "${BOLD}Repository:${RESET}         $(pwd)"
echo -e "${BOLD}Python:${RESET}             $(python --version)"
echo -e "${BOLD}CUDA detected:${RESET}      ${CUDA_VERSION:-none (CPU mode)}"
echo -e "${BOLD}PyTorch index:${RESET}      $TORCH_EXTRA_INDEX"
echo ""
echo -e "${GREEN}${BOLD}✓ Setup complete!${RESET}"
echo ""
echo -e "${BOLD}Next steps:${RESET}"
echo "  1. Activate the virtual environment:"
echo -e "     ${CYAN}source $TARGET_DIR/$VENV_DIR/bin/activate${RESET}"
echo "  2. Verify the setup:"
echo -e "     ${CYAN}python verify_clone_setup.py${RESET}"
echo "  3. Run the agent:"
echo -e "     ${CYAN}python vadar_agent.py${RESET}"
echo "  4. Read the full guide:"
echo -e "     ${CYAN}cat CLONE_AND_SETUP_GUIDE.md${RESET}"
