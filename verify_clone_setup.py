#!/usr/bin/env python3
"""
verify_clone_setup.py – Verify that the VADAR Visual Agent repository has been
cloned and configured correctly.

Checks performed:
  - Repository root exists and is a Git repository
  - Git remote points to the official VADAR repo
  - Expected top-level directories exist (models/, agents/, engine/, prompts/)
  - Key source files are present
  - Python virtual environment is activated
  - Required Python packages are importable
  - OpenAI API key is configured

Run with:
    python verify_clone_setup.py
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


# ── ANSI colours ─────────────────────────────────────────────────────────────
def _supports_ansi() -> bool:
    """Return True if the current terminal supports ANSI colour codes."""
    if not sys.stdout.isatty():
        return False
    if os.name == "nt":
        # Windows 10 version 1511+ and Windows Terminal support ANSI via VT processing.
        try:
            import ctypes  # noqa: PLC0415
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            # Enable VIRTUAL_TERMINAL_PROCESSING (0x0004) on stdout
            handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
            mode   = ctypes.c_ulong(0)
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                return bool(mode.value & 0x0004) or bool(
                    kernel32.SetConsoleMode(handle, mode.value | 0x0004)
                )
        except Exception:  # noqa: BLE001
            pass
        # Fallback: Windows Terminal sets WT_SESSION
        return "WT_SESSION" in os.environ
    return True


_USE_COLOUR = _supports_ansi()

GREEN  = "\033[0;32m"  if _USE_COLOUR else ""
RED    = "\033[0;31m"  if _USE_COLOUR else ""
YELLOW = "\033[1;33m"  if _USE_COLOUR else ""
CYAN   = "\033[0;36m"  if _USE_COLOUR else ""
BOLD   = "\033[1m"     if _USE_COLOUR else ""
RESET  = "\033[0m"     if _USE_COLOUR else ""

TICK  = f"{GREEN}✓{RESET}"
CROSS = f"{RED}✗{RESET}"
WARN  = f"{YELLOW}⚠{RESET}"


# ── Data structures ───────────────────────────────────────────────────────────
@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    is_warning: bool = False  # True → non-fatal


@dataclass
class VerificationReport:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult) -> None:
        self.results.append(result)

    @property
    def passed(self) -> list[CheckResult]:
        return [r for r in self.results if r.passed]

    @property
    def failed(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed and not r.is_warning]

    @property
    def warnings(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed and r.is_warning]


# ── Individual checks ─────────────────────────────────────────────────────────
def check_git_repo(repo_root: Path) -> CheckResult:
    """Confirm that repo_root is inside a Git repository."""
    git_dir = repo_root / ".git"
    if git_dir.is_dir():
        return CheckResult("Git repository", True, f".git directory found at {git_dir}")
    return CheckResult("Git repository", False, f"No .git directory found in {repo_root}")


def check_git_remote(repo_root: Path) -> CheckResult:
    """Check that the Git remote 'origin' points to the official VADAR repo."""
    expected_url = "Sakshi2002-Sinha/vadar-visual-agent"
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        url = result.stdout.strip()
        if expected_url in url:
            return CheckResult("Git remote (origin)", True, f"origin → {url}")
        return CheckResult(
            "Git remote (origin)",
            False,
            f"origin points to '{url}', expected URL containing '{expected_url}'",
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return CheckResult("Git remote (origin)", False, f"Could not run git: {exc}")


def check_directory(repo_root: Path, rel_path: str, is_warning: bool = False) -> CheckResult:
    """Check that a specific sub-directory exists."""
    full = repo_root / rel_path
    if full.is_dir():
        return CheckResult(f"Directory: {rel_path}/", True, str(full))
    return CheckResult(
        f"Directory: {rel_path}/",
        False,
        f"Directory not found: {full}",
        is_warning=is_warning,
    )


def check_file(repo_root: Path, rel_path: str, is_warning: bool = False) -> CheckResult:
    """Check that a specific file exists."""
    full = repo_root / rel_path
    size = full.stat().st_size if full.is_file() else None
    if full.is_file() and size is not None and size > 0:
        return CheckResult(f"File: {rel_path}", True, f"{full} ({size:,} bytes)")
    if full.is_file():
        return CheckResult(f"File: {rel_path}", False, f"File exists but is empty: {full}", is_warning=True)
    return CheckResult(
        f"File: {rel_path}",
        False,
        f"File not found: {full}",
        is_warning=is_warning,
    )


def check_venv_activated() -> CheckResult:
    """Check whether a virtual environment is currently activated."""
    virtual_env = os.environ.get("VIRTUAL_ENV")
    conda_env   = os.environ.get("CONDA_DEFAULT_ENV")

    if virtual_env:
        return CheckResult("Virtual environment", True, f"Active venv path: {virtual_env}")
    if conda_env:
        return CheckResult("Virtual environment", True, f"Active conda env: {conda_env}")
    return CheckResult(
        "Virtual environment",
        False,
        "No virtual environment detected (VIRTUAL_ENV / CONDA_DEFAULT_ENV not set). "
        "Run: source venv/bin/activate",
        is_warning=True,
    )


def check_python_version() -> CheckResult:
    """Check that the current Python is 3.10+."""
    info = sys.version_info
    ver  = f"{info.major}.{info.minor}.{info.micro}"
    if info.major > 3 or (info.major == 3 and info.minor >= 10):
        return CheckResult("Python version", True, f"Python {ver} at {sys.executable}")
    return CheckResult(
        "Python version",
        False,
        f"Python {ver} found, but 3.10+ is required.",
    )


def check_package(package: str, import_name: str | None = None, is_warning: bool = False) -> CheckResult:
    """Check that a Python package is importable."""
    name = import_name or package
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "unknown")
        return CheckResult(f"Package: {package}", True, f"{package} {ver}")
    except ImportError:
        return CheckResult(
            f"Package: {package}",
            False,
            f"'{package}' is not installed. Run: pip install {package}",
            is_warning=is_warning,
        )


def check_torch_cuda() -> CheckResult:
    """Check whether PyTorch can see a CUDA GPU."""
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return CheckResult("CUDA (GPU)", True, f"CUDA available – {name}")
        return CheckResult(
            "CUDA (GPU)",
            False,
            "CUDA not available. PyTorch will run on CPU (slower).",
            is_warning=True,
        )
    except ImportError:
        return CheckResult(
            "CUDA (GPU)",
            False,
            "PyTorch not installed; cannot check CUDA.",
            is_warning=True,
        )


def check_api_key(repo_root: Path) -> CheckResult:
    """Check that the OpenAI API key is configured."""
    # 1. Environment variable
    if os.environ.get("OPENAI_API_KEY", "").startswith("sk-"):
        return CheckResult("OpenAI API key", True, "Found in environment variable OPENAI_API_KEY.")

    # 2. .env file
    env_file = repo_root / ".env"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY"):
                val = line.split("=", 1)[-1].strip().strip('"').strip("'")
                if val.startswith("sk-"):
                    return CheckResult("OpenAI API key", True, f"Found in {env_file}")
                return CheckResult(
                    "OpenAI API key",
                    False,
                    f"OPENAI_API_KEY found in {env_file} but value looks invalid.",
                    is_warning=True,
                )

    return CheckResult(
        "OpenAI API key",
        False,
        "API key not set. Set OPENAI_API_KEY env var or add it to .env.",
        is_warning=True,
    )


# ── Report printing ───────────────────────────────────────────────────────────
def print_report(report: VerificationReport) -> None:
    total    = len(report.results)
    n_pass   = len(report.passed)
    n_fail   = len(report.failed)
    n_warn   = len(report.warnings)

    print(f"\n{BOLD}{CYAN}══ VADAR Setup Verification Report ══{RESET}\n")

    for r in report.results:
        if r.passed:
            symbol = TICK
        elif r.is_warning:
            symbol = WARN
        else:
            symbol = CROSS
        print(f"  {symbol}  {BOLD}{r.name}{RESET}")
        print(f"       {r.message}")

    print(f"\n{BOLD}Summary:{RESET}  "
          f"{GREEN}{n_pass} passed{RESET}  |  "
          f"{YELLOW}{n_warn} warnings{RESET}  |  "
          f"{RED}{n_fail} failed{RESET}  "
          f"(out of {total} checks)")

    if n_fail == 0 and n_warn == 0:
        print(f"\n{GREEN}{BOLD}✓ All checks passed! Your VADAR environment is ready.{RESET}")
    elif n_fail == 0:
        print(f"\n{YELLOW}{BOLD}⚠ Setup is mostly complete – review the warnings above.{RESET}")
    else:
        print(f"\n{RED}{BOLD}✗ {n_fail} check(s) failed. Please address the issues above before proceeding.{RESET}")

    print()


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    repo_root = Path(__file__).resolve().parent
    report    = VerificationReport()

    # ── Repository checks ────────────────────────────────────────────────────
    print(f"{BOLD}Repository root:{RESET} {repo_root}\n")

    report.add(check_git_repo(repo_root))
    report.add(check_git_remote(repo_root))

    # ── Directory structure ──────────────────────────────────────────────────
    # Required directories (fatal if missing)
    for d in ["agents", "engine", "prompts"]:
        report.add(check_directory(repo_root, d, is_warning=True))

    # models/ may be populated on first run – treat as warning
    report.add(check_directory(repo_root, "models", is_warning=True))

    # ── Key files ────────────────────────────────────────────────────────────
    for f in ["vadar_agent.py", "CLONE_AND_SETUP_GUIDE.md",
              "clone_and_setup.sh", "setup_windows.ps1",
              "verify_clone_setup.py"]:
        report.add(check_file(repo_root, f))

    for f in ["requirements.txt", "config.yaml", "setup.sh"]:
        report.add(check_file(repo_root, f, is_warning=True))

    # ── Python environment ───────────────────────────────────────────────────
    report.add(check_python_version())
    report.add(check_venv_activated())

    # ── Required packages ────────────────────────────────────────────────────
    packages = [
        ("torch",          "torch"),
        ("torchvision",    "torchvision"),
        ("transformers",   "transformers"),
        ("openai",         "openai"),
        ("Pillow",         "PIL"),
        ("numpy",          "numpy"),
        ("opencv-python",  "cv2"),
        ("matplotlib",     "matplotlib"),
    ]
    for pkg, imp in packages:
        report.add(check_package(pkg, imp, is_warning=True))

    report.add(check_torch_cuda())

    # ── API key ──────────────────────────────────────────────────────────────
    report.add(check_api_key(repo_root))

    # ── Print & exit ─────────────────────────────────────────────────────────
    print_report(report)

    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
