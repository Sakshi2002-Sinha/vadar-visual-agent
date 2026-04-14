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

Novel enhancements:
  * Feature 4  – Fuzzy / semantic answer matching (boolean normalisation,
                 numeric tolerance ±5 %, case-fold)
  * Feature 3  – Uses execute_code_with_repair for automatic self-correction
  * Feature 2  – Passes image to the LLM when use_vision=True (GPT-4o vision)
  * Feature 9  – Richer summary report: per-category accuracy, execution
                 success rate, answer type distribution, confidence scatter plot
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

from vadar_agent import SceneAnalysis, VADARAgent


# ---------------------------------------------------------------------------
# Fuzzy / semantic answer matching  (Feature 4)
# ---------------------------------------------------------------------------

_BOOL_TRUE = {"true", "yes", "1", "correct", "right", "affirmative"}
_BOOL_FALSE = {"false", "no", "0", "incorrect", "wrong", "negative"}


def fuzzy_match(predicted: str, ground_truth: str) -> bool:
    """Return True when *predicted* semantically matches *ground_truth*.

    Matching strategy (applied in order):
    1. Exact match after case-fold + strip.
    2. Boolean normalisation: ``True / Yes / 1`` all match each other.
    3. Numeric tolerance: values within 5 % of each other match.
    4. Integer equality after rounding.
    """
    pred = str(predicted).strip().lower()
    gt = str(ground_truth).strip().lower()

    # 1. Exact
    if pred == gt:
        return True

    # 2. Boolean normalisation
    if pred in _BOOL_TRUE and gt in _BOOL_TRUE:
        return True
    if pred in _BOOL_FALSE and gt in _BOOL_FALSE:
        return True

    # 3. Numeric tolerance ±5 %
    try:
        p_num = float(pred)
        g_num = float(gt)
        if g_num == 0.0:
            return p_num == 0.0
        return abs(p_num - g_num) / abs(g_num) <= 0.05
    except ValueError:
        pass

    # 4. Integer equality
    try:
        return int(round(float(pred))) == int(round(float(gt)))
    except ValueError:
        pass

    return False


# ---------------------------------------------------------------------------
# Answer-type classifier  (Feature 9)
# ---------------------------------------------------------------------------

def _answer_type(answer: str) -> str:
    """Classify an answer string as 'boolean', 'numeric', or 'string'."""
    low = answer.strip().lower()
    if low in _BOOL_TRUE | _BOOL_FALSE:
        return "boolean"
    try:
        float(low)
        return "numeric"
    except ValueError:
        return "string"


# ---------------------------------------------------------------------------
# Question-category classifier  (Feature 9)
# ---------------------------------------------------------------------------

_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "depth": [
        "closest", "nearest", "farthest", "farther", "closer",
        "depth", "distance", "near", "far",
    ],
    "spatial": [
        "left", "right", "above", "below", "between", "beside",
        "next to", "front", "behind", "position",
    ],
    "counting": [
        "how many", "count", "number of", "total",
    ],
    "color": [
        "color", "colour", "red", "blue", "green", "yellow",
        "black", "white", "orange", "purple", "pink",
    ],
}


def _question_category(question: str) -> str:
    """Return the first matching category keyword group, or 'other'."""
    q = question.lower()
    for cat, kws in _CATEGORY_KEYWORDS.items():
        if any(kw in q for kw in kws):
            return cat
    return "other"


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
        label_text = f"{obj.label} ({obj.confidence:.2f})"
        if obj.color:
            label_text += f" [{obj.color}]"
        ax.text(
            x0 * width,
            y0 * height - 5,
            label_text,
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
        attr_str = ""
        if obj.color:
            attr_str = f"  [{obj.color}/{obj.material}/{obj.relative_size}]"
        lines.append(
            f"{i + 1}. {obj.label}\n"
            f"   conf={obj.confidence:.3f}  depth={obj.depth_value:.3f}\n"
            f"   center={obj.center}{attr_str}\n"
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
        use_vision: bool = True,
        use_clip: bool = False,
        detection_threshold: float = 0.7,
        nms_iou_threshold: float = 0.5,
        code_repair_retries: int = 2,
    ) -> None:
        self.agent = VADARAgent(
            api_key=api_key,
            use_gpu=use_gpu,
            model=model,
            use_vision=use_vision,
            use_clip=use_clip,
            detection_threshold=detection_threshold,
            nms_iou_threshold=nms_iou_threshold,
            code_repair_retries=code_repair_retries,
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
        sample_result: Dict[str, Any] = {
            "sample_id": sample_id,
            "image_path": image_path,
            "timestamp": datetime.now().isoformat(),
            "questions_and_answers": [],
        }

        scene = self.agent.analyze_image(image_path)
        # Store per-object info for Feature 9 confidence analysis
        from dataclasses import asdict  # noqa: PLC0415
        sample_result["objects_detected"] = [asdict(o) for o in scene.objects]

        img_ctx = image_path if self.agent.use_vision else None

        for q_idx, question in enumerate(questions):
            # Feature 2 – include image in code-generation prompt
            code = self.agent.code_generator.generate_code(
                question, scene, image_path=img_ctx
            )
            # Feature 3 – self-repair on execution failure
            answer, status = self.agent.code_generator.execute_code_with_repair(
                code, scene, question,
                image_path=img_ctx,
                max_retries=self.agent.code_repair_retries,
            )
            answer_str = str(answer)

            # Feature 4 – fuzzy matching
            correct: Optional[bool] = None
            if ground_truth and q_idx < len(ground_truth):
                correct = fuzzy_match(answer_str, str(ground_truth[q_idx]))

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
                "question_category": _question_category(question),
                "answer_type": _answer_type(answer_str),
            }
            if correct is not None:
                entry["correct"] = correct

            sample_result["questions_and_answers"].append(entry)
            print(
                f"    Q{q_idx}: {question[:60]!r}  →  {answer_str}  [{status}]"
                + ("  ✓" if correct else "  ✗" if correct is False else "")
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

    # ------------------------------------------------------------------
    # Summary report  (Feature 9 – richer metrics)
    # ------------------------------------------------------------------

    def generate_summary_report(self) -> None:
        all_qa = [
            q for r in self.results for q in r["questions_and_answers"]
        ]
        total_q = len(all_qa)
        successful = sum(1 for q in all_qa if q["status"] == "Success")
        correct_entries = [q for q in all_qa if "correct" in q]
        accuracy: Optional[float] = (
            sum(1 for q in correct_entries if q["correct"]) / len(correct_entries)
            if correct_entries
            else None
        )

        # Execution success rate
        exec_success_rate = successful / total_q if total_q else None

        # Answer type distribution
        type_dist: Dict[str, int] = {}
        for q in all_qa:
            t = q.get("answer_type", _answer_type(str(q["answer"])))
            type_dist[t] = type_dist.get(t, 0) + 1

        # Per-category accuracy
        cat_accuracy: Dict[str, float] = {}
        for cat in list(_CATEGORY_KEYWORDS.keys()) + ["other"]:
            cat_qs = [
                q for q in correct_entries
                if q.get("question_category", _question_category(q["question"])) == cat
            ]
            if cat_qs:
                cat_accuracy[cat] = (
                    sum(1 for q in cat_qs if q["correct"]) / len(cat_qs)
                )

        summary = {
            "evaluation_date": datetime.now().isoformat(),
            "total_samples": len(self.results),
            "total_questions": total_q,
            "successful_executions": successful,
            "execution_success_rate": exec_success_rate,
            "accuracy": accuracy,
            "per_category_accuracy": cat_accuracy,
            "answer_type_distribution": type_dist,
            "results_directory": str(self.output_dir),
        }

        report_path = self.output_dir / "summary_report.json"
        report_path.write_text(json.dumps(summary, indent=2))

        # Confidence vs accuracy scatter plot
        self._plot_confidence_vs_accuracy()

        # Print summary
        print("\n=== Summary ===")
        print(f"  Samples           : {summary['total_samples']}")
        print(f"  Questions         : {summary['total_questions']}")
        print(f"  Successful execs  : {successful}/{total_q}"
              + (f"  ({exec_success_rate * 100:.1f}%)" if exec_success_rate is not None else ""))
        if accuracy is not None:
            print(f"  Overall Accuracy  : {accuracy * 100:.1f}%")
        if cat_accuracy:
            print("  Per-category Accuracy:")
            for cat, acc in cat_accuracy.items():
                print(f"    {cat:<12}: {acc * 100:.1f}%")
        if type_dist:
            print(f"  Answer Types      : {type_dist}")
        print(f"  Output            : {self.output_dir}")

    def _plot_confidence_vs_accuracy(self) -> None:
        """Scatter plot: mean detection confidence vs. question correctness (Feature 9)."""
        xs: List[float] = []
        ys: List[int] = []
        for r in self.results:
            objs = r.get("objects_detected", [])
            if not objs:
                continue
            avg_conf = float(np.mean([o["confidence"] for o in objs]))
            for q in r["questions_and_answers"]:
                if "correct" not in q:
                    continue
                xs.append(avg_conf)
                ys.append(1 if q["correct"] else 0)

        if not xs:
            return

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(xs, ys, alpha=0.5, color="steelblue", edgecolors="none")
        ax.set_xlabel("Mean detection confidence")
        ax.set_ylabel("Correct (1) / Wrong (0)")
        ax.set_title("Detection Confidence vs. Answer Correctness")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Wrong", "Correct"])

        # Trend line using binned means
        bins = np.linspace(min(xs), max(xs), 6)
        bin_means = []
        bin_centers = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = [lo <= x <= hi for x in xs]
            vals = [y for x, y, m in zip(xs, ys, mask) if m]
            if vals:
                bin_means.append(np.mean(vals))
                bin_centers.append((lo + hi) / 2)
        if bin_centers:
            ax.plot(bin_centers, bin_means, "r-o", linewidth=2, label="Binned mean")
            ax.legend()

        plt.tight_layout()
        plot_path = self.output_dir / "confidence_vs_accuracy.png"
        plt.savefig(plot_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Confidence plot   : {plot_path}")


# ---------------------------------------------------------------------------
# Accuracy from saved results
# ---------------------------------------------------------------------------

def compute_accuracy_from_file(results_path: str) -> None:
    data = json.loads(Path(results_path).read_text())
    all_q = [q for r in data for q in r["questions_and_answers"] if "correct" in q]
    if not all_q:
        print("No ground-truth annotations found in results file.")
        return
    total = len(all_q)
    correct = sum(1 for q in all_q if q["correct"])
    print(f"Accuracy: {correct / total * 100:.1f}%  ({correct}/{total})")

    # Per-category breakdown
    cat_rows: Dict[str, List[bool]] = {}
    for q in all_q:
        cat = q.get("question_category", _question_category(q["question"]))
        cat_rows.setdefault(cat, []).append(q["correct"])
    if len(cat_rows) > 1:
        print("Per-category:")
        for cat, vals in sorted(cat_rows.items()):
            pct = sum(vals) / len(vals) * 100
            print(f"  {cat:<12}: {pct:.1f}%  ({sum(vals)}/{len(vals)})")


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
        "--no-vision", action="store_true",
        help="Disable GPT-4o image input (text-only mode)",
    )
    parser.add_argument(
        "--use-clip", action="store_true",
        help="Enable CLIP attribute extraction (color / material / size)",
    )
    parser.add_argument(
        "--detection-threshold", type=float, default=0.7,
        help="Minimum detection confidence to keep (default: 0.7)",
    )
    parser.add_argument(
        "--nms-iou-threshold", type=float, default=0.5,
        help="IoU threshold for NMS duplicate suppression (default: 0.5)",
    )
    parser.add_argument(
        "--code-repair-retries", type=int, default=2,
        help="Max self-repair retries on code execution failure (default: 2)",
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
        use_vision=not args.no_vision,
        use_clip=args.use_clip,
        detection_threshold=args.detection_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
        code_repair_retries=args.code_repair_retries,
    )
    evaluator.run_evaluation(test_cases)
    evaluator.generate_summary_report()


if __name__ == "__main__":
    main()
