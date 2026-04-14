#!/usr/bin/env python3
"""
quickstart.py – VADAR Visual Agent quick-start script

Usage:
    # Verify all dependencies without running inference
    python quickstart.py --verify-only

    # Run a demo on a single image
    python quickstart.py --image path/to/image.jpg \
                         --question "Which object is closest to the camera?"

    # Run the built-in synthetic demo (no image file required)
    python quickstart.py --demo
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import textwrap
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_RESET = "\033[0m"


def _ok(msg: str) -> None:
    print(f"{_GREEN}[OK]{_RESET} {msg}")


def _warn(msg: str) -> None:
    print(f"{_YELLOW}[WARN]{_RESET} {msg}")


def _fail(msg: str) -> None:
    print(f"{_RED}[FAIL]{_RESET} {msg}")


# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------

def check_python_version() -> bool:
    major, minor = sys.version_info[:2]
    if major == 3 and minor >= 9:
        _ok(f"Python {major}.{minor}.{sys.version_info[2]}")
        return True
    _fail(f"Python {major}.{minor} – need 3.9+")
    return False


def check_package(import_name: str, display_name: str | None = None) -> bool:
    name = display_name or import_name
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "unknown")
        _ok(f"{name} {version}")
        return True
    except ImportError:
        _fail(f"{name} – not installed  (pip install {name})")
        return False


def check_torch_cuda() -> bool:
    try:
        import torch  # noqa: PLC0415

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            _ok(f"CUDA available – {device_name}")
        else:
            _warn("CUDA not available – will use CPU (slower)")
        return True
    except ImportError:
        _fail("torch not installed")
        return False


def check_api_key() -> bool:
    key = os.environ.get("OPENAI_API_KEY", "")
    if key and key.startswith("sk-"):
        _ok("OPENAI_API_KEY is set")
        return True
    if key:
        _warn("OPENAI_API_KEY is set but does not start with 'sk-' – may be invalid")
        return True
    _fail("OPENAI_API_KEY is not set  (export OPENAI_API_KEY=sk-...)")
    return False


def check_hf_model_cached(model_id: str) -> bool:
    """Check whether a HuggingFace model is present in the local cache."""
    try:
        from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST  # noqa: PLC0415

        # try_to_load_from_cache returns a path string if cached, or
        # _CACHED_NO_EXIST sentinel / None when not cached.
        result = try_to_load_from_cache(model_id, filename="config.json")
        if result and result is not _CACHED_NO_EXIST:
            _ok(f"{model_id} – cached")
            return True
    except Exception:  # noqa: BLE001
        pass
    _warn(f"{model_id} – not cached (will be downloaded on first use)")
    return True  # Not a blocking failure


# ---------------------------------------------------------------------------
# Full dependency verification
# ---------------------------------------------------------------------------

def verify_environment() -> bool:
    print("\n=== Verifying VADAR environment ===\n")

    checks: List[Tuple[str, bool]] = []

    checks.append(("Python version", check_python_version()))
    checks.append(("torch", check_package("torch")))
    checks.append(("CUDA", check_torch_cuda()))
    checks.append(("transformers", check_package("transformers")))
    checks.append(("openai", check_package("openai")))
    checks.append(("cv2", check_package("cv2", "opencv-python")))
    checks.append(("PIL", check_package("PIL", "Pillow")))
    checks.append(("numpy", check_package("numpy")))
    checks.append(("matplotlib", check_package("matplotlib")))
    checks.append(("OPENAI_API_KEY", check_api_key()))

    print()
    checks.append(("DETR detection", check_hf_model_cached("facebook/detr-resnet-50")))
    checks.append(("DPT depth", check_hf_model_cached("Intel/dpt-large")))
    checks.append(("DETR panoptic", check_hf_model_cached("facebook/detr-resnet-50-panoptic")))

    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)
    print(f"\n{passed}/{total} checks passed.")

    if passed == total:
        print(f"\n{_GREEN}Environment looks good!{_RESET}")
        return True

    print(f"\n{_YELLOW}Some checks failed – see messages above.{_RESET}")
    print("Consult TROUBLESHOOTING.md for help.\n")
    return False


# ---------------------------------------------------------------------------
# Single-image demo
# ---------------------------------------------------------------------------

def run_demo(image_path: str, question: str) -> None:
    """Run the full VADAR pipeline on one image and print results."""
    import json  # noqa: PLC0415

    from vadar_agent import VADARAgent  # noqa: PLC0415

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        _fail("OPENAI_API_KEY is not set – cannot run demo.")
        sys.exit(1)

    try:
        import torch  # noqa: PLC0415
        use_gpu = torch.cuda.is_available()
    except ImportError:
        use_gpu = False

    print(f"\n=== Running VADAR demo ===")
    print(f"Image:    {image_path}")
    print(f"Question: {question}")
    print(f"GPU:      {'yes' if use_gpu else 'no (CPU)'}\n")

    agent = VADARAgent(api_key=api_key, use_gpu=use_gpu)

    # Step 1 – analyse
    print("[Step 1/4] Analysing image …", end=" ", flush=True)
    scene = agent.analyze_image(image_path)
    print(f"✓  ({len(scene.objects)} objects detected)")

    for obj in scene.objects:
        print(
            f"    • {obj.label:<20} confidence={obj.confidence:.2f}  "
            f"depth={obj.depth_value:.3f}  center={obj.center}"
        )

    # Step 2 – generate code
    print("\n[Step 2/4] Generating code …", end=" ", flush=True)
    code = agent.code_generator.generate_code(question, scene)
    print("✓")

    # Step 3 – execute code
    print("[Step 3/4] Executing code …", end=" ", flush=True)
    answer, status = agent.code_generator.execute_code(code, scene)
    print(f"✓  (status: {status})")

    # Step 4 – save outputs
    output_dir = Path("outputs") / "quickstart"
    output_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime  # noqa: PLC0415

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    code_path = output_dir / f"code_{ts}.py"
    code_path.write_text(f"# Question: {question}\n# Answer: {answer}\n\n{code}")

    print(f"\n[Step 4/4] Saving outputs …  ✓")
    print(f"  Generated code → {code_path}")

    print(f"\n{_GREEN}Answer: {answer}{_RESET}\n")


# ---------------------------------------------------------------------------
# Synthetic demo (no real image required)
# ---------------------------------------------------------------------------

def run_synthetic_demo() -> None:
    """
    Demonstrate the code-generation / execution pipeline without calling
    external vision models or a real image file.
    """
    import numpy as np  # noqa: PLC0415
    from dataclasses import dataclass  # noqa: PLC0415
    from typing import Tuple as _Tuple  # noqa: PLC0415

    # Inline minimal versions of the data classes so this demo works
    # even when the heavy dependencies (torch, transformers, …) are absent.

    @dataclass
    class _SpatialObject:
        label: str
        confidence: float
        bbox: _Tuple[float, float, float, float]
        center: _Tuple[int, int]
        depth_value: float
        area: float
        image_height: int
        image_width: int

        def distance_from_camera(self) -> float:
            return self.depth_value

    class _SpatialReasoner:
        @staticmethod
        def get_object_by_label(objects, label):
            for obj in objects:
                if label.lower() in obj.label.lower():
                    return obj
            return None

        @staticmethod
        def is_farther(obj1, obj2) -> bool:
            return obj1.distance_from_camera() > obj2.distance_from_camera()

    print("\n=== Synthetic demo (no image / API key required) ===\n")

    # Build a fake scene with two objects
    chair = _SpatialObject(
        label="chair",
        confidence=0.95,
        bbox=(0.1, 0.2, 0.3, 0.8),
        center=(128, 300),
        depth_value=0.7,
        area=0.16,
        image_height=480,
        image_width=640,
    )
    table = _SpatialObject(
        label="dining table",
        confidence=0.88,
        bbox=(0.4, 0.3, 0.9, 0.9),
        center=(416, 360),
        depth_value=0.4,
        area=0.30,
        image_height=480,
        image_width=640,
    )

    objects = [chair, table]

    print("Detected objects:")
    for obj in objects:
        print(
            f"  • {obj.label:<20} depth={obj.depth_value:.2f}  "
            f"center={obj.center}  confidence={obj.confidence:.2f}"
        )

    question = "Is the chair farther from the camera than the dining table?"
    print(f"\nQuestion: {question}")

    # Manually answer using SpatialReasoner (no OpenAI call)
    c = _SpatialReasoner.get_object_by_label(objects, "chair")
    t = _SpatialReasoner.get_object_by_label(objects, "dining table")
    answer = _SpatialReasoner.is_farther(c, t)
    print(f"Answer:   {answer}  (chair depth={c.depth_value:.2f}, table depth={t.depth_value:.2f})")

    print(
        textwrap.dedent(
            f"""
            \nExample generated code that would answer this question:

              chair = SpatialReasoner.get_object_by_label(objects, "chair")
              table = SpatialReasoner.get_object_by_label(objects, "dining table")
              answer = SpatialReasoner.is_farther(chair, table)
            """
        )
    )
    print(f"{_GREEN}Synthetic demo completed.{_RESET}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="VADAR Visual Agent – quick-start helper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              python quickstart.py --verify-only
              python quickstart.py --demo
              python quickstart.py --image img.jpg --question "What is closest?"
            """
        ),
    )
    parser.add_argument("--verify-only", action="store_true", help="Check dependencies only")
    parser.add_argument("--demo", action="store_true", help="Run synthetic demo (no image/API key)")
    parser.add_argument("--image", help="Path to an image file")
    parser.add_argument(
        "--question",
        default="Which object is closest to the camera?",
        help="Spatial reasoning question (default: %(default)s)",
    )
    args = parser.parse_args()

    if args.verify_only:
        ok = verify_environment()
        sys.exit(0 if ok else 1)

    if args.demo:
        run_synthetic_demo()
        return

    if args.image:
        verify_environment()
        run_demo(args.image, args.question)
        return

    # Default: verify + synthetic demo
    verify_environment()
    run_synthetic_demo()


if __name__ == "__main__":
    main()
