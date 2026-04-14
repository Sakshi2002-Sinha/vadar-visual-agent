# =============================================================================
# setup_windows.ps1 – Automated VADAR setup script for Windows (PowerShell)
#
# Usage (in an elevated PowerShell window):
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\setup_windows.ps1
#
# What this script does:
#   1. Checks prerequisites (Python 3.10+, Git, optional CUDA via nvidia-smi)
#   2. Prompts user for confirmation at each major step
#   3. Clones the official VADAR repository
#   4. Creates and activates a Python virtual environment
#   5. Installs PyTorch (CUDA-specific wheel or CPU fallback)
#   6. Runs setup.sh equivalent steps
#   7. Installs remaining Python dependencies from requirements.txt
#   8. Verifies model downloads
#   9. Prompts for the OpenAI API key and saves it to .env
#  10. Prints a summary and next steps
# =============================================================================

#Requires -Version 5.1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Colour helpers ───────────────────────────────────────────────────────────
function Write-Info    ($msg) { Write-Host "[INFO]  $msg" -ForegroundColor Cyan }
function Write-Ok      ($msg) { Write-Host "[OK]    $msg" -ForegroundColor Green }
function Write-Warn    ($msg) { Write-Host "[WARN]  $msg" -ForegroundColor Yellow }
function Write-Err     ($msg) { Write-Host "[ERROR] $msg" -ForegroundColor Red }
function Write-Header  ($msg) { Write-Host "`n══ $msg ══`n" -ForegroundColor Cyan }

function Confirm-Step {
    param([string]$Prompt = "Continue?")
    $answer = Read-Host "$Prompt [y/N]"
    return ($answer -match '^[Yy](es)?$')
}

# ── Constants ────────────────────────────────────────────────────────────────
$RepoUrl   = "https://github.com/Sakshi2002-Sinha/vadar-visual-agent.git"
$TargetDir = "vadar"
$VenvDir   = "venv"

# =============================================================================
# STEP 1 – Check prerequisites
# =============================================================================
Write-Header "Step 1 – Checking prerequisites"

# -- Python -------------------------------------------------------------------
$PythonBin = $null
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $ver = & $cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($ver) {
            $parts = $ver.Split('.')
            $major = [int]$parts[0]
            $minor = [int]$parts[1]
            if ($major -gt 3 -or ($major -eq 3 -and $minor -ge 10)) {
                $PythonBin = $cmd
                Write-Ok "Python $ver found."
                break
            } else {
                Write-Warn "Found Python $ver, but Python 3.10+ is required."
            }
        }
    } catch { continue }
}

if (-not $PythonBin) {
    Write-Err "Python 3.10+ is not installed or not on PATH."
    Write-Err "Download it from https://www.python.org/downloads/"
    exit 1
}

# -- Git ----------------------------------------------------------------------
try {
    $gitVer = & git --version 2>$null
    Write-Ok "$gitVer found."
} catch {
    Write-Err "Git is not installed. Download it from https://git-scm.com/downloads."
    exit 1
}

# -- CUDA (optional) ----------------------------------------------------------
$CudaVersion     = $null
$TorchIndexUrl   = "https://download.pytorch.org/whl/cpu"

try {
    $nvidiaSmiOut = & nvidia-smi 2>$null
    if ($nvidiaSmiOut) {
        $cudaMatch = [regex]::Match($nvidiaSmiOut, "CUDA Version: (\d+\.\d+)")
        if ($cudaMatch.Success) {
            $CudaVersion = $cudaMatch.Groups[1].Value
            Write-Ok "CUDA $CudaVersion detected."
        }
    }
} catch {
    Write-Warn "nvidia-smi not found. Defaulting to CPU-only PyTorch."
}

if ($CudaVersion) {
    $cudaParts = $CudaVersion.Split('.')
    $cMajor    = [int]$cudaParts[0]
    $cMinor    = [int]$cudaParts[1]

    if     ($cMajor -eq 11 -and $cMinor -ge 8) { $TorchIndexUrl = "https://download.pytorch.org/whl/cu118" }
    elseif ($cMajor -eq 12 -and $cMinor -eq 0) { $TorchIndexUrl = "https://download.pytorch.org/whl/cu120" }
    elseif ($cMajor -eq 12 -and $cMinor -eq 1) { $TorchIndexUrl = "https://download.pytorch.org/whl/cu121" }
    elseif ($cMajor -ge 12)                     { $TorchIndexUrl = "https://download.pytorch.org/whl/cu122" }
    else {
        Write-Warn "CUDA $CudaVersion is older than 11.8. Falling back to CPU PyTorch."
    }
    Write-Info "Will use PyTorch index: $TorchIndexUrl"
} else {
    Write-Warn "No GPU detected. Using CPU-only PyTorch."
}

if (-not (Confirm-Step "Prerequisites look good. Proceed to clone the repository?")) {
    Write-Info "Aborted by user."; exit 0
}

# =============================================================================
# STEP 2 – Clone the repository
# =============================================================================
Write-Header "Step 2 – Cloning the VADAR repository"

if (Test-Path $TargetDir) {
    Write-Warn "Directory '$TargetDir' already exists."
    if (Confirm-Step "Pull latest changes instead of re-cloning?") {
        Push-Location $TargetDir
        git pull
        Write-Ok "Repository updated."
    } else {
        Write-Info "Using existing directory."
        Push-Location $TargetDir
    }
} else {
    Write-Info "Cloning $RepoUrl into '$TargetDir' ..."
    git clone $RepoUrl $TargetDir
    Push-Location $TargetDir
    Write-Ok "Repository cloned successfully."
}

Write-Info "Remote configuration:"
git remote -v

if (-not (Confirm-Step "Repository is ready. Create a virtual environment now?")) {
    Pop-Location; Write-Info "Aborted by user."; exit 0
}

# =============================================================================
# STEP 3 – Create virtual environment
# =============================================================================
Write-Header "Step 3 – Creating virtual environment"

if (Test-Path $VenvDir) {
    Write-Warn "Virtual environment '$VenvDir' already exists."
    if (Confirm-Step "Re-create the virtual environment? (This will delete the existing one)") {
        Remove-Item -Recurse -Force $VenvDir
        & $PythonBin -m venv $VenvDir
        Write-Ok "Virtual environment re-created."
    } else {
        Write-Info "Using existing virtual environment."
    }
} else {
    & $PythonBin -m venv $VenvDir
    Write-Ok "Virtual environment created in '$VenvDir'."
}

# Activate the virtual environment
$activateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
if (-not (Test-Path $activateScript)) {
    Write-Err "Could not find $activateScript. Virtual environment may be corrupted."
    exit 1
}
. $activateScript
Write-Ok "Virtual environment activated."

# Upgrade pip
python -m pip install --quiet --upgrade pip
Write-Ok "pip upgraded."

if (-not (Confirm-Step "Virtual environment is ready. Install PyTorch now?")) {
    Pop-Location; Write-Info "Aborted by user."; exit 0
}

# =============================================================================
# STEP 4 – Install PyTorch
# =============================================================================
Write-Header "Step 4 – Installing PyTorch"

Write-Info "Installing from: $TorchIndexUrl"
pip install torch torchvision torchaudio --index-url $TorchIndexUrl

python -c @"
import torch
print(f'  torch      : {torch.__version__}')
print(f'  CUDA avail : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU        : {torch.cuda.get_device_name(0)}')
"@

Write-Ok "PyTorch installed."

# =============================================================================
# STEP 5 – Install remaining Python dependencies
# =============================================================================
Write-Header "Step 5 – Installing Python dependencies"

if (Test-Path "requirements.txt") {
    Write-Info "Installing from requirements.txt ..."
    pip install -r requirements.txt
    Write-Ok "All dependencies installed."
} else {
    Write-Warn "requirements.txt not found. Installing common VADAR dependencies ..."
    pip install openai transformers accelerate Pillow numpy opencv-python matplotlib scikit-image python-dotenv
    Write-Ok "Common dependencies installed."
}

# =============================================================================
# STEP 6 – Verify model downloads
# =============================================================================
Write-Header "Step 6 – Verifying model downloads"

if (Test-Path "verify_clone_setup.py") {
    python verify_clone_setup.py
} elseif (Test-Path "models") {
    Write-Info "Contents of models/:"
    Get-ChildItem "models" | Format-Table Name, Length -AutoSize
    Write-Ok "Models directory found."
} else {
    Write-Warn "models/ directory not found. Models may be downloaded on first run."
}

# =============================================================================
# STEP 7 – OpenAI API key
# =============================================================================
Write-Header "Step 7 – OpenAI API key"

$EnvFile = ".env"
$hasKey  = (Test-Path $EnvFile) -and ((Get-Content $EnvFile -ErrorAction SilentlyContinue) -match "OPENAI_API_KEY")

if ($hasKey) {
    Write-Ok ".env already contains OPENAI_API_KEY. Skipping."
} else {
    if (Confirm-Step "Would you like to store your OpenAI API key in .env now?") {
        while ($true) {
            $apiKey = Read-Host "Paste your OpenAI API key" -AsSecureString
            $plain  = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
                          [Runtime.InteropServices.Marshal]::SecureStringToBSTR($apiKey))
            if ([string]::IsNullOrWhiteSpace($plain)) {
                Write-Warn "No key entered. Skipping."; break
            }
            if ($plain.StartsWith("sk-")) {
                Add-Content -Path $EnvFile -Value "OPENAI_API_KEY=`"$plain`""
                Write-Ok "API key saved to $EnvFile."
                # Ensure .env is gitignored
                if (-not (Test-Path ".gitignore") -or -not (Select-String -Path ".gitignore" -Pattern "^\.env" -Quiet)) {
                    Add-Content -Path ".gitignore" -Value ".env"
                    Write-Info "Added .env to .gitignore."
                }
                break
            } else {
                Write-Warn "Key does not start with 'sk-'. Please check and try again."
            }
        }
    } else {
        Write-Info "Skipped. Set the key later: Add-Content .env 'OPENAI_API_KEY=`"sk-...`"'"
    }
}

# =============================================================================
# STEP 8 – Summary & next steps
# =============================================================================
Write-Header "Setup Summary & Next Steps"

Write-Host "Repository   : $(Get-Location)"                      -ForegroundColor White
Write-Host "Python       : $(python --version)"                  -ForegroundColor White
Write-Host "CUDA         : $(if ($CudaVersion) { $CudaVersion } else { 'none (CPU mode)' })" -ForegroundColor White
Write-Host "PyTorch index: $TorchIndexUrl"                       -ForegroundColor White
Write-Host ""
Write-Host "✓ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "  1. Activate the virtual environment:"
Write-Host "       .\$TargetDir\$VenvDir\Scripts\Activate.ps1"     -ForegroundColor Cyan
Write-Host "  2. Verify the setup:"
Write-Host "       python verify_clone_setup.py"                  -ForegroundColor Cyan
Write-Host "  3. Run the agent:"
Write-Host "       python vadar_agent.py"                         -ForegroundColor Cyan
Write-Host "  4. Read the full guide:"
Write-Host "       Get-Content CLONE_AND_SETUP_GUIDE.md | more"  -ForegroundColor Cyan

Pop-Location
