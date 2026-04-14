#!/usr/bin/env python3
"""
evaluate_benchmark.py – Batch evaluation for the VADAR Visual Agent.

Usage:
    python evaluate_benchmark.py \
        --test-cases data/test_cases.json \
        --output-dir outputs/my_run

    # Compute accuracy from a previously saved results file
    python evaluate_benchmark.py \
        --results outputs/my_run/evaluation_results.json \
        --compute-accuracy
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from vadar_agent import SceneAnalysis, VADARAgent


# ---------------------------------------------------------------------------
# Visual trace
# ---------------------------------------------------------------------------

def create_visual_trace(
    image_path: str,
    scene: SceneAnalysis,
    question: str,
    answer: str,
    save_path: Path,
) -> None:
    """Save a 2×2 panel figure with the original image, depth map, object
    list, and the question/answer."""
    import cv2  # noqa: PLC0415

    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image)
    height, width = image_array.shape[:2]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Q: {question}\nA: {answer}",
        fontsize=11,
        fontweight="bold",
    )

    # Panel 1 – original image + bounding boxes
    ax = axes[0, 0]
    ax.imshow(image_array)
    ax.set_title("Detected Objects")
    for obj in scene.objects:
        x0, y0, x1, y1 = obj.bbox
        rect = patches.Rectangle(
            (x0 * width, y0 * height),
            (x1 - x0) * width,
            (y1 - y0) * height,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x0 * width,
            y0 * height - 5,
            f"{obj.label} ({obj.confidence:.2f})",
            color="red",
            fontsize=7,
            bbox=dict(facecolor="white", alpha=0.6),
        )
    ax.axis("off")

    # Panel 2 – depth map
    ax = axes[0, 1]
    dm = cv2.resize(scene.depth_map, (width, height))
    im = ax.imshow(dm, cmap="viridis")
    ax.set_title("Estimated Depth Map  (bright/yellow=close, dark/purple=far)")
    plt.colorbar(im, ax=ax)
    ax.axis("off")

    # Panel 3 – object list
    ax = axes[1, 0]
    ax.axis("off")
    lines = ["Detected Objects\n"]
    for i, obj in enumerate(scene.objects):
        lines.append(
            f"{i + 1}. {obj.label}\n"
            f"   conf={obj.confidence:.3f}  depth={obj.depth_value:.3f}\n"
            f"   center={obj.center}\n"
        )
    ax.text(
        0.05,
        0.95,
        "".join(lines),
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Panel 4 – question / answer
    ax = axes[1, 1]
    ax.axis("off")
    qa = f"Question:\n{question}\n\nAnswer:\n{answer}"
    ax.text(
        0.05,
        0.95,
        qa,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
    )

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class BenchmarkEvaluator:
    """Run the VADAR agent on a list of test cases and save full results."""

    def __init__(
        self,
        api_key: str,
        output_dir: str = "benchmark_results",
        use_gpu: bool = False,
        model: str = "gpt-4o",
        provider: str = "auto",
        openai_base_url: str = "",
        github_token: str = "",
        github_model: str = "gpt-4o-mini",
        github_base_url: str = "https://models.inference.ai.azure.com",
        min_detection_confidence: float = 0.35,
        max_objects: int = 40,
        enable_segmentation: bool = False,
        build_scene_graph: bool = True,
    ) -> None:
        self.agent = VADARAgent(
            api_key=api_key,
            use_gpu=use_gpu,
            model=model,
            provider=provider,
            openai_base_url=openai_base_url,
            github_token=github_token,
            github_model=github_model,
            github_base_url=github_base_url,
            min_detection_confidence=min_detection_confidence,
            max_objects=max_objects,
            enable_segmentation=enable_segmentation,
            build_scene_graph=build_scene_graph,
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate_sample(
        self,
        image_path: str,
        questions: List[str],
        sample_id: str,
        ground_truth: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Evaluate one image with one or more questions."""
        sample_start = time.perf_counter()
        sample_result: Dict[str, Any] = {
            "sample_id": sample_id,
            "image_path": image_path,
            "timestamp": datetime.now().isoformat(),
            "questions_and_answers": [],
        }

        scene = self.agent.analyze_image(image_path)
        scene_timing = getattr(scene, "processing_times", {})
        sample_result["scene_metrics"] = {
            "detection_ms": scene_timing.get("detection_ms"),
            "depth_ms": scene_timing.get("depth_ms"),
            "scene_graph_ms": scene_timing.get("scene_graph_ms"),
            "scene_total_ms": scene_timing.get("total_ms"),
            "detections_before_filter": int(scene_timing.get("detections_before_filter", 0)),
            "detections_after_filter": int(scene_timing.get("detections_after_filter", len(scene.objects))),
            "scene_graph_edges": len(scene.scene_graph.relations) if scene.scene_graph else None,
        }

        for q_idx, question in enumerate(questions):
            t_codegen = time.perf_counter()
            code = self.agent.code_generator.generate_code(question, scene)
            codegen_ms = (time.perf_counter() - t_codegen) * 1000.0

            t_exec = time.perf_counter()
            answer, status = self.agent.code_generator.execute_code(code, scene)
            exec_ms = (time.perf_counter() - t_exec) * 1000.0
            answer_str = str(answer)

            # Correctness (optional)
            correct: Optional[bool] = None
            if ground_truth and q_idx < len(ground_truth):
                correct = answer_str.strip().lower() == str(ground_truth[q_idx]).strip().lower()

            # Save code
            code_dir = self.output_dir / "code"
            code_dir.mkdir(exist_ok=True)
            code_path = code_dir / f"code_{sample_id}_q{q_idx}.py"
            code_path.write_text(
                f"# Question: {question}\n# Answer: {answer_str}\n\n{code}"
            )

            # Save trace
            trace_dir = self.output_dir / "traces"
            trace_dir.mkdir(exist_ok=True)
            trace_path = trace_dir / f"trace_{sample_id}_q{q_idx}.png"
            try:
                create_visual_trace(image_path, scene, question, answer_str, trace_path)
            except Exception as exc:  # noqa: BLE001
                print(f"  [WARN] Could not create trace: {exc}")
                trace_path = Path("N/A")

            entry: Dict[str, Any] = {
                "question_id": q_idx,
                "question": question,
                "answer": answer_str,
                "status": status,
                "code": code,
                "code_path": str(code_path),
                "trace_path": str(trace_path),
                "metrics": {
                    "codegen_ms": round(codegen_ms, 3),
                    "execution_ms": round(exec_ms, 3),
                },
            }
            if correct is not None:
                entry["correct"] = correct

            sample_result["questions_and_answers"].append(entry)
            print(
                f"    Q{q_idx}: {question[:60]!r}  →  {answer_str}  [{status}]"
                + (f"  ✓" if correct else f"  ✗" if correct is False else "")
            )

        sample_result["sample_total_ms"] = round((time.perf_counter() - sample_start) * 1000.0, 3)
        return sample_result

    def run_evaluation(self, test_cases: List[Dict[str, Any]]) -> None:
        for i, tc in enumerate(test_cases):
            sample_id = tc.get("sample_id", f"sample_{i:04d}")
            print(f"\n[{i + 1}/{len(test_cases)}] {sample_id}")
            result = self.evaluate_sample(
                image_path=tc["image_path"],
                questions=tc["questions"],
                sample_id=sample_id,
                ground_truth=tc.get("ground_truth"),
            )
            self.results.append(result)
            self._save_results()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_results(self) -> None:
        path = self.output_dir / "evaluation_results.json"
        path.write_text(json.dumps(self.results, indent=2, default=str))

    def generate_summary_report(self) -> None:
        total_q = sum(len(r["questions_and_answers"]) for r in self.results)
        successful = sum(
            1
            for r in self.results
            for q in r["questions_and_answers"]
            if q["status"] == "Success"
        )
        correct_entries = [
            q
            for r in self.results
            for q in r["questions_and_answers"]
            if "correct" in q
        ]
        accuracy: Optional[float] = None
        if correct_entries:
            accuracy = sum(1 for q in correct_entries if q["correct"]) / len(correct_entries)

        scene_total_values = [
            r.get("scene_metrics", {}).get("scene_total_ms")
            for r in self.results
            if r.get("scene_metrics", {}).get("scene_total_ms") is not None
        ]
        codegen_values = [
            q.get("metrics", {}).get("codegen_ms")
            for r in self.results
            for q in r["questions_and_answers"]
            if q.get("metrics", {}).get("codegen_ms") is not None
        ]
        exec_values = [
            q.get("metrics", {}).get("execution_ms")
            for r in self.results
            for q in r["questions_and_answers"]
            if q.get("metrics", {}).get("execution_ms") is not None
        ]
        sample_totals = [
            r.get("sample_total_ms")
            for r in self.results
            if r.get("sample_total_ms") is not None
        ]

        def _avg(values: List[float]) -> Optional[float]:
            if not values:
                return None
            return float(sum(values) / len(values))

        summary = {
            "evaluation_date": datetime.now().isoformat(),
            "total_samples": len(self.results),
            "total_questions": total_q,
            "successful_executions": successful,
            "accuracy": accuracy,
            "avg_scene_ms": _avg(scene_total_values),
            "avg_codegen_ms": _avg(codegen_values),
            "avg_execution_ms": _avg(exec_values),
            "avg_sample_total_ms": _avg(sample_totals),
            "results_directory": str(self.output_dir),
        }

        report_path = self.output_dir / "summary_report.json"
        report_path.write_text(json.dumps(summary, indent=2))

        print("\n=== Summary ===")
        print(f"  Samples   : {summary['total_samples']}")
        print(f"  Questions : {summary['total_questions']}")
        print(f"  Successful: {successful}/{total_q}")
        if accuracy is not None:
            print(f"  Accuracy  : {accuracy * 100:.1f}%")
        if summary["avg_scene_ms"] is not None:
            print(f"  Avg scene : {summary['avg_scene_ms']:.1f} ms")
        if summary["avg_codegen_ms"] is not None:
            print(f"  Avg codegen: {summary['avg_codegen_ms']:.1f} ms")
        if summary["avg_execution_ms"] is not None:
            print(f"  Avg execute: {summary['avg_execution_ms']:.1f} ms")
        print(f"  Output    : {self.output_dir}")


# ---------------------------------------------------------------------------
# Accuracy from saved results
# ---------------------------------------------------------------------------

def compute_accuracy_from_file(results_path: str) -> None:
    data = json.loads(Path(results_path).read_text())
    all_q = [q for r in data for q in r["questions_and_answers"] if "correct" in q]
    if not all_q:
        print("No ground-truth annotations found in results file.")
        return
    acc = sum(1 for q in all_q if q["correct"]) / len(all_q)
    print(f"Accuracy: {acc * 100:.1f}%  ({sum(1 for q in all_q if q['correct'])}/{len(all_q)})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Batch evaluation for VADAR Visual Agent")
    parser.add_argument("--test-cases", help="Path to test_cases.json")
    parser.add_argument("--output-dir", default="outputs/evaluation", help="Output directory")
    parser.add_argument("--results", help="Path to existing evaluation_results.json")
    parser.add_argument(
        "--compute-accuracy", action="store_true", help="Compute accuracy from --results"
    )
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model for code generation")
    parser.add_argument(
        "--provider",
        default=os.environ.get("LLM_PROVIDER", "auto"),
        choices=["auto", "github", "openai", "local"],
        help="LLM provider routing strategy",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=float(os.environ.get("VADAR_MIN_DET_CONF", "0.35")),
        help="Filter detections below this confidence before reasoning",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=int(os.environ.get("VADAR_MAX_OBJECTS", "40")),
        help="Maximum detections to keep per image (0 means unlimited)",
    )
    parser.add_argument(
        "--enable-segmentation",
        action="store_true",
        help="Preload panoptic segmentation model (off by default for speed)",
    )
    parser.add_argument(
        "--no-scene-graph",
        action="store_true",
        help="Disable scene graph construction (faster, less context for LLM)",
    )
    args = parser.parse_args()

    if args.compute_accuracy:
        if not args.results:
            print("ERROR: --results is required with --compute-accuracy")
            sys.exit(1)
        compute_accuracy_from_file(args.results)
        return

    if not args.test_cases:
        parser.print_help()
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    openai_base_url = os.environ.get("OPENAI_BASE_URL", "")
    github_token = os.environ.get("GITHUB_TOKEN", "")
    github_model = os.environ.get("GITHUB_MODEL", "gpt-4o-mini")
    github_base_url = os.environ.get("GITHUB_BASE_URL", "https://models.inference.ai.azure.com")

    if args.provider == "github" and not github_token:
        print("ERROR: provider=github requires GITHUB_TOKEN")
        sys.exit(1)
    if args.provider == "openai" and not api_key:
        print("ERROR: provider=openai requires OPENAI_API_KEY")
        sys.exit(1)

    try:
        import torch  # noqa: PLC0415
        use_gpu = torch.cuda.is_available()
    except ImportError:
        use_gpu = False

    test_cases = json.loads(Path(args.test_cases).read_text())
    evaluator = BenchmarkEvaluator(
        api_key=api_key,
        output_dir=args.output_dir,
        use_gpu=use_gpu,
        model=args.model,
        provider=args.provider,
        openai_base_url=openai_base_url,
        github_token=github_token,
        github_model=github_model,
        github_base_url=github_base_url,
        min_detection_confidence=args.min_detection_confidence,
        max_objects=args.max_objects,
        enable_segmentation=args.enable_segmentation,
        build_scene_graph=not args.no_scene_graph,
    )
    evaluator.run_evaluation(test_cases)
    evaluator.generate_summary_report()


if __name__ == "__main__":
    main()
