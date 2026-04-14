"""
test_agent.py – Quick functional test / demonstration of the VADAR Visual Agent.

Run:
    python test_agent.py

The script will:
  1. Initialise VADARAgent
  2. Download (or reuse) a small sample image
  3. Demonstrate object detection and depth estimation
  4. Answer a simple spatial reasoning question
  5. Show how to kick off an evaluation benchmark run
"""

import os
import sys
import json
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/"
    "Cat03.jpg/320px-Cat03.jpg"
)
SAMPLE_IMAGE_PATH = Path("sample_images") / "sample.jpg"


def download_sample_image() -> Path:
    """Download a small sample image if it does not already exist."""
    SAMPLE_IMAGE_PATH.parent.mkdir(exist_ok=True)
    if not SAMPLE_IMAGE_PATH.exists():
        print(f"Downloading sample image from {SAMPLE_IMAGE_URL} …")
        try:
            urllib.request.urlretrieve(SAMPLE_IMAGE_URL, SAMPLE_IMAGE_PATH)
            # Validate that the downloaded file is a real image
            from PIL import Image as _Image
            _Image.open(SAMPLE_IMAGE_PATH).verify()
            print(f"  ✓ Saved to {SAMPLE_IMAGE_PATH}")
        except Exception as exc:
            # Remove potentially corrupted file before exiting
            if SAMPLE_IMAGE_PATH.exists():
                SAMPLE_IMAGE_PATH.unlink()
            print(f"  ✗ Download or validation failed: {exc}")
            print("  Please place a JPEG image at sample_images/sample.jpg and re-run.")
            sys.exit(1)
    else:
        print(f"  ✓ Using cached image at {SAMPLE_IMAGE_PATH}")
    return SAMPLE_IMAGE_PATH


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 1. Initialise the agent
# ---------------------------------------------------------------------------

def test_initialise_agent():
    section("Step 1 – Initialise VADARAgent")
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print(
            "  ⚠  OPENAI_API_KEY is not set.\n"
            "     Code generation will be skipped, but vision pipeline tests still run."
        )

    from vadar_agent import VADARAgent
    agent = VADARAgent(api_key)
    print("  ✓ VADARAgent initialised")
    return agent


# ---------------------------------------------------------------------------
# 2. Download sample image
# ---------------------------------------------------------------------------

def test_download_image():
    section("Step 2 – Download / locate sample image")
    return download_sample_image()


# ---------------------------------------------------------------------------
# 3. Object detection & depth estimation
# ---------------------------------------------------------------------------

def test_vision_pipeline(agent, image_path: Path):
    section("Step 3 – Object detection & depth estimation")

    print("  Running object detection …")
    try:
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        detections = agent.vision_models.detect_objects(image)
        print(f"  ✓ Detected {len(detections)} object(s):")
        for det in detections[:5]:  # show at most 5
            print(
                f"      • {det['label']:20s}  confidence={det['score']:.2f}"
                f"  box={det['box']}"
            )
    except Exception as exc:
        print(f"  ✗ Object detection failed: {exc}")

    print("  Running depth estimation …")
    try:
        import numpy as np
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        depth_map = agent.vision_models.estimate_depth(image)
        print(
            f"  ✓ Depth map shape: {depth_map.shape}  "
            f"min={depth_map.min():.3f}  max={depth_map.max():.3f}"
        )
    except Exception as exc:
        print(f"  ✗ Depth estimation failed: {exc}")


# ---------------------------------------------------------------------------
# 4. Full scene analysis
# ---------------------------------------------------------------------------

def test_scene_analysis(agent, image_path: Path):
    section("Step 4 – Full scene analysis")

    print("  Analysing image …")
    try:
        analysis = agent.analyze_image(str(image_path))
        print(f"  ✓ Scene analysis complete – {len(analysis.objects)} spatial object(s):")
        for obj in analysis.objects[:5]:
            print(
                f"      • {obj.label:20s}  depth={obj.depth_value:.3f}"
                f"  center={obj.center}"
            )
        return analysis
    except Exception as exc:
        print(f"  ✗ Scene analysis failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# 5. Spatial reasoning question
# ---------------------------------------------------------------------------

def test_spatial_reasoning(agent, image_path: Path):
    section("Step 5 – Spatial reasoning question")

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print(
            "  ⚠  Skipping code-generation step – OPENAI_API_KEY not set."
        )
        return

    question = "Which object appears to be the closest to the camera?"
    print(f"  Question: {question}")

    try:
        result = agent.answer_question(question, str(image_path))
        print(f"  ✓ Answer:  {result['answer']}")
        print(f"     Status: {result['status']}")
        print("     Generated code preview (first 5 lines):")
        for line in result["code"].splitlines()[:5]:
            print(f"       {line}")
    except Exception as exc:
        print(f"  ✗ Spatial reasoning failed: {exc}")


# ---------------------------------------------------------------------------
# 6. Evaluation benchmark (dry run)
# ---------------------------------------------------------------------------

def test_evaluation_benchmark(image_path: Path):
    section("Step 6 – Evaluation benchmark (dry run)")

    try:
        from evaluate_benchmark import BenchmarkEvaluator
    except ImportError:
        print("  ⚠  evaluate_benchmark.py not found – skipping benchmark demo.")
        return

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("  ⚠  Skipping benchmark – OPENAI_API_KEY not set.")
        return

    evaluator = BenchmarkEvaluator(api_key, output_dir="benchmark_results_test")

    test_cases = [
        {
            "sample_id": "demo_001",
            "image_path": str(image_path),
            "questions": [
                "Which object is furthest from the camera?",
                "Are there multiple objects in the scene?",
            ],
        }
    ]

    print("  Running evaluation on 1 sample …")
    try:
        evaluator.run_evaluation(test_cases)
        evaluator.generate_summary_report()
        print("  ✓ Benchmark complete.  Results saved to benchmark_results_test/")
    except Exception as exc:
        print(f"  ✗ Benchmark failed: {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("\nVADAR Visual Agent – Functional Test")
    print("=====================================")

    agent = test_initialise_agent()
    image_path = test_download_image()
    test_vision_pipeline(agent, image_path)
    test_scene_analysis(agent, image_path)
    test_spatial_reasoning(agent, image_path)
    test_evaluation_benchmark(image_path)

    print("\n✓ All steps completed.\n")


if __name__ == "__main__":
    main()
