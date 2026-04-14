"""
evaluate_benchmark.py – Evaluates the VADAR Agent on a set of benchmark images.

Usage:
    python evaluate_benchmark.py

Results are written to the directory specified by --output-dir (default: benchmark_results/).
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
from PIL import Image

from vadar_agent import VADARAgent, SceneAnalysis


class BenchmarkEvaluator:
    """Evaluates VADARAgent on a list of test cases and stores results + visual traces."""

    def __init__(self, api_key: str, output_dir: str = "benchmark_results"):
        self.agent = VADARAgent(api_key)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Visual trace
    # ------------------------------------------------------------------

    def create_visual_trace(
        self,
        image_path: str,
        scene_analysis: SceneAnalysis,
        question: str,
        answer: str,
        save_path: str,
    ) -> None:
        """Save a 2×2 figure: original + boxes, depth map, object info, Q&A."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import cv2

        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)
        height, width = scene_analysis.image_shape[:2]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"Question: {question}\nAnswer: {answer}",
            fontsize=11,
            fontweight="bold",
        )

        # — top-left: image + bounding boxes —
        ax = axes[0, 0]
        ax.imshow(image_array)
        ax.set_title("Detected Objects")
        for obj in scene_analysis.objects:
            x_min, y_min, x_max, y_max = obj.bbox
            rect = patches.Rectangle(
                (x_min * width, y_min * height),
                (x_max - x_min) * width,
                (y_max - y_min) * height,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(
                x_min * width,
                max(y_min * height - 5, 0),
                f"{obj.label} ({obj.confidence:.2f})",
                color="red",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.7),
            )
        ax.axis("off")

        # — top-right: depth map —
        ax = axes[0, 1]
        depth_display = cv2.resize(
            scene_analysis.depth_map.astype(np.float32), (width, height)
        )
        im = ax.imshow(depth_display, cmap="viridis")
        ax.set_title("Estimated Depth Map (0=close, 1=far)")
        plt.colorbar(im, ax=ax)
        ax.axis("off")

        # — bottom-left: object table —
        ax = axes[1, 0]
        ax.axis("off")
        info_text = "Detected Objects:\n\n"
        for i, obj in enumerate(scene_analysis.objects, 1):
            info_text += (
                f"{i}. {obj.label}\n"
                f"   Confidence : {obj.confidence:.3f}\n"
                f"   Depth      : {obj.depth_value:.3f}\n"
                f"   Center     : {obj.center}\n\n"
            )
        ax.text(
            0.05, 0.95, info_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # — bottom-right: Q&A —
        ax = axes[1, 1]
        ax.axis("off")
        qa_text = f"Question:\n{question}\n\nAnswer:\n{answer}"
        ax.text(
            0.05, 0.95, qa_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Single sample
    # ------------------------------------------------------------------

    def evaluate_sample(
        self,
        image_path: str,
        questions: List[str],
        sample_id: str,
    ) -> Dict[str, Any]:
        """Evaluate the agent on one image with one or more questions."""
        sample_results: Dict[str, Any] = {
            "sample_id": sample_id,
            "image_path": image_path,
            "timestamp": datetime.now().isoformat(),
            "questions_and_answers": [],
        }

        scene_analysis = self.agent.analyze_image(image_path)

        for q_idx, question in enumerate(questions):
            code = self.agent.code_generator.generate_code(question, scene_analysis)
            answer, status = self.agent.code_generator.execute_code(code, scene_analysis)

            result: Dict[str, Any] = {
                "question_id": q_idx,
                "question": question,
                "answer": str(answer),
                "status": status,
                "code": code,
            }

            # Visual trace
            trace_path = self.output_dir / f"trace_{sample_id}_q{q_idx}.png"
            self.create_visual_trace(
                image_path, scene_analysis, question, str(answer), str(trace_path)
            )
            result["trace_path"] = str(trace_path)

            # Save generated code
            code_path = self.output_dir / f"code_{sample_id}_q{q_idx}.py"
            with open(code_path, "w") as fh:
                fh.write(f"# Question: {question}\n")
                fh.write(f"# Answer: {answer}\n\n")
                fh.write(code)
            result["code_path"] = str(code_path)

            sample_results["questions_and_answers"].append(result)

        return sample_results

    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------

    def run_evaluation(self, test_cases: List[Dict[str, Any]]) -> None:
        """Evaluate all test cases in *test_cases*."""
        for i, test_case in enumerate(test_cases):
            sample_id = test_case.get("sample_id", f"sample_{i:03d}")
            print(f"\nEvaluating {sample_id} …")
            result = self.evaluate_sample(
                test_case["image_path"],
                test_case["questions"],
                sample_id,
            )
            self.results.append(result)
            self._save_results()

    def _save_results(self) -> None:
        results_file = self.output_dir / "evaluation_results.json"
        with open(results_file, "w") as fh:
            json.dump(self.results, fh, indent=2)
        print(f"  Results saved → {results_file}")

    def generate_summary_report(self) -> Dict[str, Any]:
        """Print and save a summary report; return the summary dict."""
        summary: Dict[str, Any] = {
            "evaluation_date": datetime.now().isoformat(),
            "total_samples": len(self.results),
            "total_questions": sum(
                len(r["questions_and_answers"]) for r in self.results
            ),
            "results_directory": str(self.output_dir),
            "samples": [],
        }
        for result in self.results:
            qas = result["questions_and_answers"]
            summary["samples"].append(
                {
                    "sample_id": result["sample_id"],
                    "image_path": result["image_path"],
                    "question_count": len(qas),
                    "successful_answers": sum(
                        1 for q in qas if q["status"] == "Success"
                    ),
                }
            )

        report_file = self.output_dir / "summary_report.json"
        with open(report_file, "w") as fh:
            json.dump(summary, fh, indent=2)

        print(f"\nSummary Report:")
        print(f"  Total samples   : {summary['total_samples']}")
        print(f"  Total questions : {summary['total_questions']}")
        print(f"  Results dir     : {summary['results_directory']}")
        return summary


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate VADAR Visual Agent on a benchmark dataset."
    )
    p.add_argument(
        "--output-dir",
        default=os.getenv("BENCHMARK_OUTPUT_DIR", "benchmark_results"),
        help="Directory to write results into (default: benchmark_results)",
    )
    p.add_argument(
        "--test-cases",
        default=None,
        help="Path to a JSON file containing test cases (optional).",
    )
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)

    evaluator = BenchmarkEvaluator(api_key, output_dir=args.output_dir)

    if args.test_cases:
        import json as _json
        with open(args.test_cases) as fh:
            test_cases = _json.load(fh)
    else:
        # Default demo case – requires sample_images/room_001.jpg
        test_cases = [
            {
                "sample_id": "room_001",
                "image_path": "sample_images/room_001.jpg",
                "questions": [
                    "Is the chair farther from the camera than the table?",
                    "What objects are visible in this scene?",
                    "Which object is closest to the camera?",
                ],
            }
        ]

    evaluator.run_evaluation(test_cases)
    evaluator.generate_summary_report()
