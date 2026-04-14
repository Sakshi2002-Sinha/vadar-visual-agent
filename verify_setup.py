"""
verify_setup.py – Checks that the VADAR Visual Agent environment is correctly configured.

Run:
    python verify_setup.py

Exit codes:
    0 – all checks passed (warnings are acceptable)
    1 – one or more critical checks failed
"""

import sys
import os
import importlib
from pathlib import Path

PASS  = "✓"
WARN  = "⚠"
FAIL  = "✗"

critical_failures: list[str] = []
warnings: list[str] = []


def check(label: str, passed: bool, message: str = "", critical: bool = True) -> bool:
    icon = PASS if passed else (FAIL if critical else WARN)
    suffix = f"  – {message}" if message else ""
    print(f"  {icon}  {label}{suffix}")
    if not passed:
        if critical:
            critical_failures.append(label)
        else:
            warnings.append(label)
    return passed


def section(title: str) -> None:
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print("─" * 55)


# ---------------------------------------------------------------------------
# 1. Python version
# ---------------------------------------------------------------------------

section("Python version")
py = sys.version_info
check(
    f"Python {py.major}.{py.minor}.{py.micro}",
    py >= (3, 9),
    "Python 3.9+ required" if py < (3, 9) else "",
)


# ---------------------------------------------------------------------------
# 2. Required packages
# ---------------------------------------------------------------------------

section("Required packages")

REQUIRED: list[tuple[str, str, bool]] = [
    # (import_name, display_name, critical)
    ("openai",        "openai",              True),
    ("transformers",  "transformers",        True),
    ("torch",         "torch",               True),
    ("torchvision",   "torchvision",         True),
    ("PIL",           "Pillow",              True),
    ("numpy",         "numpy",               True),
    ("cv2",           "opencv-python",       True),
    ("matplotlib",    "matplotlib",          True),
    ("sklearn",       "scikit-learn",        False),
    ("skimage",       "scikit-image",        False),
    ("dotenv",        "python-dotenv",       False),
    ("jupyter",       "jupyter / notebook",  False),
]

for import_name, display_name, is_critical in REQUIRED:
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "?")
        check(f"{display_name} ({version})", True, critical=is_critical)
    except ImportError:
        check(
            display_name,
            False,
            f"install with: pip install {display_name}",
            critical=is_critical,
        )


# ---------------------------------------------------------------------------
# 3. OpenAI API key
# ---------------------------------------------------------------------------

section("Environment / credentials")

api_key = os.getenv("OPENAI_API_KEY", "")
check(
    "OPENAI_API_KEY",
    bool(api_key),
    "Set OPENAI_API_KEY in your environment or .env file",
    critical=True,
)
if api_key:
    check(
        "API key format",
        api_key.startswith("sk-"),
        "Key should start with 'sk-'" if not api_key.startswith("sk-") else "",
        critical=False,
    )


# ---------------------------------------------------------------------------
# 4. Vision model loading (lightweight smoke test)
# ---------------------------------------------------------------------------

section("Vision model smoke test")

try:
    from transformers import pipeline as hf_pipeline
    import transformers
    from packaging.version import Version
    check("transformers pipeline importable", True)
    check(
        "transformers version ≥ 4.30",
        Version(transformers.__version__) >= Version("4.30.0"),
        critical=False,
    )
except ImportError:
    # packaging may not be installed; fall back to a best-effort string comparison
    try:
        import transformers
        check("transformers pipeline importable", True)
        check("transformers version (unchecked)", True, critical=False)
    except ImportError as exc:
        check("transformers pipeline importable", False, str(exc))
except Exception as exc:
    check("transformers pipeline importable", False, str(exc))

try:
    import torch
    check("torch importable", True)
    cuda_available = torch.cuda.is_available()
    check(
        "CUDA available",
        cuda_available,
        "CPU-only mode will be slower" if not cuda_available else "",
        critical=False,
    )
except Exception as exc:
    check("torch importable", False, str(exc))


# ---------------------------------------------------------------------------
# 5. File structure
# ---------------------------------------------------------------------------

section("File structure")

EXPECTED_FILES = [
    "vadar_agent.py",
    "evaluate_benchmark.py",
    "requirements.txt",
    ".env.example",
    "demo_notebook.ipynb",
    "README.md",
]

EXPECTED_DIRS = [
    "sample_images",
]

root = Path(__file__).parent

for fname in EXPECTED_FILES:
    path = root / fname
    check(fname, path.exists(), f"not found at {path}", critical=False)

for dname in EXPECTED_DIRS:
    path = root / dname
    check(f"{dname}/", path.is_dir(), f"directory not found at {path}", critical=False)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\n{'=' * 55}")
if critical_failures:
    print(f"  {FAIL}  {len(critical_failures)} critical check(s) FAILED:")
    for item in critical_failures:
        print(f"        • {item}")
    print()
    sys.exit(1)
elif warnings:
    print(f"  {WARN}  Setup mostly OK – {len(warnings)} non-critical warning(s).")
    print("     The agent should still function; see warnings above.")
else:
    print(f"  {PASS}  All checks passed – environment looks good!")
print("=" * 55)
sys.exit(0)
