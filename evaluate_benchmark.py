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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from vadar_agent import SceneAnalysis, SceneGraph, SceneGraphBuilder, VADARAgent


# ---------------------------------------------------------------------------
# Visual trace
# ---------------------------------------------------------------------------

def create_visual_trace(
    image_path: str,
    scene: SceneAnalysis,
    question: str,
    answer: str,
    save_path: Path,
    scene_graph: Optional[SceneGraph] = None,
) -> None:
    """Save a 2×3 panel figure with the original image, depth map, object
    list, question/answer, scene graph edges, and depth uncertainty chart."""
    import cv2  # noqa: PLC0415

    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image)
    height, width = image_array.shape[:2]

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
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

    # Panel 3 – depth uncertainty bar chart (novel)
    ax = axes[0, 2]
    if scene.objects:
        labels = [o.label[:12] for o in scene.objects]
        uncertainties = [o.depth_uncertainty for o in scene.objects]
        depths = [o.depth_value for o in scene.objects]
        x_pos = range(len(labels))
        bars = ax.bar(x_pos, depths, color="steelblue", alpha=0.7, label="depth")
        ax.errorbar(
            x_pos, depths, yerr=uncertainties,
            fmt="none", color="black", capsize=4, linewidth=1.5, label="±uncertainty"
        )
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("Normalized depth")
        ax.set_ylim(0, 1.1)
        ax.set_title("Object Depth  (± uncertainty)")
        ax.legend(fontsize=7)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No objects", ha="center", va="center")

    # Panel 4 – object list
    ax = axes[1, 0]
    ax.axis("off")
    lines = ["Detected Objects\n"]
    for i, obj in enumerate(scene.objects):
        lines.append(
            f"{i + 1}. {obj.label}\n"
            f"   conf={obj.confidence:.3f}  depth={obj.depth_value:.3f}"
            f"  ±{obj.depth_uncertainty:.3f}\n"
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

    # Panel 5 – scene graph edges (novel)
    ax = axes[1, 1]
    ax.axis("off")
    if scene_graph and scene_graph.edges:
        rel_counts: Dict[str, int] = {}
        for e in scene_graph.edges:
            rel_counts[e.relation] = rel_counts.get(e.relation, 0) + 1
        ax.set_title("Scene Graph  (relationship counts)")
        rels = list(rel_counts.keys())
        counts = [rel_counts[r] for r in rels]
        y_pos = range(len(rels))
        ax.barh(list(y_pos), counts, color="coral", alpha=0.8)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(rels, fontsize=8)
        ax.set_xlabel("# edges")
        ax.axis("on")
    else:
        ax.text(0.5, 0.5, "Scene graph\nnot available", ha="center", va="center")

    # Panel 6 – question / answer
    ax = axes[1, 2]
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
    ) -> None:
        self.agent = VADARAgent(api_key=api_key, use_gpu=use_gpu, model=model)
        self.scene_graph_builder = SceneGraphBuilder()
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
        sample_result: Dict[str, Any] = {
            "sample_id": sample_id,
            "image_path": image_path,
            "timestamp": datetime.now().isoformat(),
            "questions_and_answers": [],
        }

        scene = self.agent.analyze_image(image_path)
        graph = self.scene_graph_builder.build(scene)

        # Save scene graph JSON once per sample
        graph_dir = self.output_dir / "scene_graphs"
        graph_dir.mkdir(exist_ok=True)
        graph_path = graph_dir / f"graph_{sample_id}.json"
        graph_path.write_text(json.dumps(graph.to_dict(), indent=2, default=str))

        for q_idx, question in enumerate(questions):
            code = self.agent.code_generator.generate_code(question, scene)
            answer, status = self.agent.code_generator.execute_code(code, scene, scene_graph=graph)
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
                create_visual_trace(image_path, scene, question, answer_str, trace_path, scene_graph=graph)
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
                "scene_graph_path": str(graph_path),
            }
            if correct is not None:
                entry["correct"] = correct

            sample_result["questions_and_answers"].append(entry)
            print(
                f"    Q{q_idx}: {question[:60]!r}  →  {answer_str}  [{status}]"
                + (f"  ✓" if correct else f"  ✗" if correct is False else "")
            )

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

        summary = {
            "evaluation_date": datetime.now().isoformat(),
            "total_samples": len(self.results),
            "total_questions": total_q,
            "successful_executions": successful,
            "accuracy": accuracy,
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
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set.")
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
    )
    evaluator.run_evaluation(test_cases)
    evaluator.generate_summary_report()


if __name__ == "__main__":
    main()
