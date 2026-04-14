"""
VADAR-Inspired Visual Agent for 3D Spatial Reasoning on 2D Images
Performs agentic code generation and execution for spatial understanding.

Architecture:
  VisionModels      – wraps HuggingFace pipelines for detection / depth / segmentation
  SpatialObject     – dataclass representing one detected object with spatial attributes
  SceneAnalysis     – container holding all objects + depth map for a single image
  SpatialReasoner   – pure-function helpers for spatial comparisons
  CodeGenerator     – calls the OpenAI API to generate and exec Python for a question
  VADARAgent        – top-level orchestrator
"""

import os
import json
import base64
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime

import numpy as np
from PIL import Image
import openai


# ---------------------------------------------------------------------------
# Vision models
# ---------------------------------------------------------------------------

class VisionModels:
    """Manages pretrained vision models for object detection, segmentation, and depth estimation."""

    def __init__(self, use_gpu: bool = False):
        # Heavy imports are deferred so that the rest of the module can be
        # imported (e.g. for unit testing) without requiring torch/transformers.
        from transformers import pipeline as hf_pipeline  # noqa: PLC0415

        device = 0 if use_gpu else -1
        self.object_detector = hf_pipeline(
            "object-detection",
            model="facebook/detr-resnet-50",
            device=device,
        )
        self.depth_estimator = hf_pipeline(
            "depth-estimation",
            model="Intel/dpt-large",
            device=device,
        )
        self.segmentation = hf_pipeline(
            "image-segmentation",
            model="facebook/detr-resnet-50-panoptic",
            device=device,
        )

    def detect_objects(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect objects in the image using DETR."""
        return self.object_detector(image)

    def estimate_depth(self, image: Image.Image) -> np.ndarray:
        """Return a normalized depth map (0 = close, 1 = far) for *image*."""
        result = self.depth_estimator(image)
        depth_map = np.array(result["depth"], dtype=np.float32)
        min_val, max_val = depth_map.min(), depth_map.max()
        if max_val > min_val:
            depth_map = (depth_map - min_val) / (max_val - min_val)
        return depth_map

    def segment_objects(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Perform panoptic segmentation and return segment list."""
        return self.segmentation(image)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class SpatialObject:
    """Represents a detected object with its spatial properties."""

    label: str
    confidence: float
    # Normalised bounding box: (x_min, y_min, x_max, y_max) in [0, 1]
    bbox: Tuple[float, float, float, float]
    # Pixel coordinates of the bounding-box center
    center: Tuple[int, int]
    # Normalised depth at the object center (0 = close, 1 = far)
    depth_value: float
    # Normalised area relative to the full image
    area: float
    image_height: int
    image_width: int

    def distance_from_camera(self) -> float:
        """Return normalized distance from the camera (0 = close, 1 = far)."""
        return self.depth_value


@dataclass
class SceneAnalysis:
    """Container for scene-understanding results for a single image."""

    objects: List[SpatialObject]
    depth_map: np.ndarray
    image_shape: Tuple[int, ...]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ---------------------------------------------------------------------------
# Spatial reasoning helpers
# ---------------------------------------------------------------------------

class SpatialReasoner:
    """Pure helper functions for spatial comparisons between SpatialObjects."""

    @staticmethod
    def get_object_by_label(objects: List[SpatialObject], label: str) -> Optional[SpatialObject]:
        """Return the first object whose label contains *label* (case-insensitive)."""
        for obj in objects:
            if label.lower() in obj.label.lower():
                return obj
        return None

    @staticmethod
    def is_farther(obj1: SpatialObject, obj2: SpatialObject) -> bool:
        """Return True if *obj1* is farther from the camera than *obj2*."""
        return obj1.distance_from_camera() > obj2.distance_from_camera()

    @staticmethod
    def relative_depth_distance(obj1: SpatialObject, obj2: SpatialObject) -> float:
        """Absolute difference in normalized depth between two objects."""
        return abs(obj1.distance_from_camera() - obj2.distance_from_camera())

    @staticmethod
    def pixel_distance(obj1: SpatialObject, obj2: SpatialObject) -> float:
        """Euclidean distance (pixels) between the centers of two objects."""
        x1, y1 = obj1.center
        x2, y2 = obj2.center
        return float(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))

    @staticmethod
    def vertical_position(obj: SpatialObject) -> str:
        """Return 'upper', 'middle', or 'lower' based on vertical center position."""
        third = obj.image_height / 3
        cy = obj.center[1]
        if cy < third:
            return "upper"
        if cy > 2 * third:
            return "lower"
        return "middle"

    @staticmethod
    def horizontal_position(obj: SpatialObject) -> str:
        """Return 'left', 'center', or 'right' based on horizontal center position."""
        third = obj.image_width / 3
        cx = obj.center[0]
        if cx < third:
            return "left"
        if cx > 2 * third:
            return "right"
        return "center"


# ---------------------------------------------------------------------------
# Code generation + execution
# ---------------------------------------------------------------------------

class CodeGenerator:
    """Generates Python code via the OpenAI API and executes it with scene context."""

    _SYSTEM_PROMPT = (
        "You are an expert spatial-reasoning assistant for 2D images with depth information.\n"
        "Depth values are normalised to [0, 1]: 0 = closest to the camera, 1 = farthest.\n"
        "When asked a question about a scene you write self-contained Python code that assigns "
        "the final answer to a variable named `answer`.\n"
        "Rules:\n"
        "  • Use only the built-in `math` module, `numpy` (as `np`), and the provided scene "
        "data structures.\n"
        "  • Access scene objects via the list `objects` (List[SpatialObject]).\n"
        "  • Each SpatialObject has: label (str), confidence (float), "
        "bbox (x_min,y_min,x_max,y_max normalised), center (cx,cy in pixels), "
        "depth_value (float 0=close 1=far), area (normalised), "
        "image_height, image_width.\n"
        "  • The helper class `SpatialReasoner` is available with static methods: "
        "get_object_by_label, is_farther, relative_depth_distance, pixel_distance, "
        "vertical_position, horizontal_position.\n"
        "  • When comparing depths, consider objects with |depth_a - depth_b| < 0.05 "
        "to be at the same distance.\n"
        "  • Output ONLY valid Python – no markdown fences, no prose, no comments."
    )

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        openai.api_key = api_key
        self.model = model
        # Tracks (question, generated_code, answer_str) tuples for multi-turn context
        self.history: List[Dict[str, Any]] = []

    def _build_user_prompt(self, question: str, scene: SceneAnalysis) -> str:
        objects_desc = "\n".join(
            f"  [{i}] label={o.label!r} confidence={o.confidence:.3f} "
            f"depth={o.depth_value:.3f} center={o.center} area={o.area:.4f} "
            f"position=({SpatialReasoner.horizontal_position(o)},{SpatialReasoner.vertical_position(o)})"
            for i, o in enumerate(scene.objects)
        )
        return (
            f"Scene objects ({len(scene.objects)} detected, "
            f"image size {scene.image_shape[1]}×{scene.image_shape[0]}):\n"
            f"{objects_desc}\n\n"
            f"Question: {question}\n\n"
            "Write Python code that assigns the answer to a variable named `answer`."
        )

    def generate_code(self, question: str, scene: SceneAnalysis) -> str:
        """Call the OpenAI Chat API and return generated Python code."""
        response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user", "content": self._build_user_prompt(question, scene)},
            ],
            temperature=0.3,
            max_tokens=1500,
        )
        code = response.choices[0].message.content.strip()
        self.history.append({
            "question": question,
            "code": code,
            "answer": None,
            "timestamp": datetime.now().isoformat(),
        })
        return code

    def generate_followup(self, followup_question: str, scene: SceneAnalysis) -> str:
        """
        Generate code for a follow-up question using the full conversation history.

        The previous questions, generated code, and answers are injected as
        assistant turns so the model can reference earlier results.
        """
        if not self.history:
            return self.generate_code(followup_question, scene)

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._SYSTEM_PROMPT},
        ]
        for turn in self.history:
            messages.append({"role": "user", "content": self._build_user_prompt(turn["question"], scene)})
            assistant_content = turn["code"]
            if turn.get("answer") is not None:
                assistant_content += f"\n# answer = {turn['answer']}"
            messages.append({"role": "assistant", "content": assistant_content})

        messages.append({"role": "user", "content": self._build_user_prompt(followup_question, scene)})

        response = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=1500,
        )
        code = response.choices[0].message.content.strip()
        self.history.append({
            "question": followup_question,
            "code": code,
            "answer": None,
            "timestamp": datetime.now().isoformat(),
        })
        return code

    def reset_history(self) -> None:
        """Clear conversation history to start a new independent session."""
        self.history.clear()

    def execute_code(
        self, code: str, scene: SceneAnalysis
    ) -> Tuple[Any, str]:
        """Execute *code* with scene context injected. Returns (answer, status)."""
        exec_globals: Dict[str, Any] = {
            "np": np,
            "math": __import__("math"),
            "SpatialObject": SpatialObject,
            "SpatialReasoner": SpatialReasoner,
            "scene_analysis": scene,
            "objects": scene.objects,
        }
        try:
            exec(code, exec_globals)  # noqa: S102  # code is LLM-generated; review before production use
            answer = exec_globals.get("answer", "No answer produced")
            # Back-fill the answer into the most recent history entry that lacks it
            for entry in reversed(self.history):
                if entry.get("code") == code and entry.get("answer") is None:
                    entry["answer"] = str(answer)
                    break
            return answer, "Success"
        except Exception as exc:  # noqa: BLE001
            return None, f"Execution error: {exc}"


# ---------------------------------------------------------------------------
# Top-level agent
# ---------------------------------------------------------------------------

class VADARAgent:
    """
    VADAR Agent – combines VisionModels, SpatialReasoner, and CodeGenerator
    to answer free-form spatial questions about images.
    """

    def __init__(self, api_key: str, use_gpu: bool = False, model: str = "gpt-4o"):
        self.vision_models = VisionModels(use_gpu=use_gpu)
        self.spatial_reasoner = SpatialReasoner()
        self.code_generator = CodeGenerator(api_key, model=model)
        self._last_analysis: Optional[SceneAnalysis] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_image(self, image_path: str, use_segmentation_fallback: bool = True) -> SceneAnalysis:
        """
        Run the full vision pipeline on *image_path* and return a SceneAnalysis.

        Steps:
          1. Object detection
          2. Monocular depth estimation
          3. Build SpatialObject list from detections + depth map
          4. (Optional) For objects with heavily overlapping bounding boxes,
             use the mean depth over the segmentation mask rather than the
             single centre-pixel depth, which is more robust.
        """
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)
        height, width = image_array.shape[:2]

        detections = self.vision_models.detect_objects(image)
        depth_map = self.vision_models.estimate_depth(image)

        # Resize depth map once (reused for every object)
        if depth_map.shape != (height, width):
            import cv2  # noqa: PLC0415
            dm_resized = cv2.resize(depth_map, (width, height))
        else:
            dm_resized = depth_map

        objects: List[SpatialObject] = []
        for det in detections:
            box = det["box"]
            x_min = box["xmin"] / width
            y_min = box["ymin"] / height
            x_max = box["xmax"] / width
            y_max = box["ymax"] / height

            cx = int((box["xmin"] + box["xmax"]) / 2)
            cy = int((box["ymin"] + box["ymax"]) / 2)
            # Clamp to valid range
            cx = max(0, min(cx, width - 1))
            cy = max(0, min(cy, height - 1))

            objects.append(
                SpatialObject(
                    label=det["label"],
                    confidence=float(det["score"]),
                    bbox=(x_min, y_min, x_max, y_max),
                    center=(cx, cy),
                    depth_value=float(dm_resized[cy, cx]),
                    area=(x_max - x_min) * (y_max - y_min),
                    image_height=height,
                    image_width=width,
                )
            )

        if use_segmentation_fallback and len(objects) >= 2:
            objects = self._apply_segmentation_depth_fallback(
                image, objects, dm_resized, height, width
            )

        analysis = SceneAnalysis(
            objects=objects,
            depth_map=depth_map,
            image_shape=image_array.shape,
        )
        self._last_analysis = analysis
        return analysis

    @staticmethod
    def _bbox_iou(a: SpatialObject, b: SpatialObject) -> float:
        """Compute IoU (Intersection over Union) of two normalised bounding boxes."""
        ax1, ay1, ax2, ay2 = a.bbox
        bx1, by1, bx2, by2 = b.bbox
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        union = a.area + b.area - inter
        return inter / union if union > 0 else 0.0

    def _apply_segmentation_depth_fallback(
        self,
        image: "Image.Image",
        objects: List[SpatialObject],
        dm_resized: np.ndarray,
        height: int,
        width: int,
        iou_threshold: float = 0.15,
    ) -> List[SpatialObject]:
        """
        For each pair of objects whose bounding boxes overlap significantly
        (IoU ≥ *iou_threshold*), replace their single-pixel depth estimate
        with the mean depth over the full bounding-box region. This is more
        robust than a single centre pixel when objects partially occlude each
        other.
        """
        # Identify objects involved in any significant overlap
        overlap_indices: set = set()
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                if self._bbox_iou(objects[i], objects[j]) >= iou_threshold:
                    overlap_indices.add(i)
                    overlap_indices.add(j)

        if not overlap_indices:
            return objects

        updated: List[SpatialObject] = []
        for idx, obj in enumerate(objects):
            if idx not in overlap_indices:
                updated.append(obj)
                continue

            # Compute mean depth over the bounding-box region
            x1 = max(0, int(obj.bbox[0] * width))
            y1 = max(0, int(obj.bbox[1] * height))
            x2 = min(width, int(obj.bbox[2] * width))
            y2 = min(height, int(obj.bbox[3] * height))

            if x2 > x1 and y2 > y1:
                region_depth = float(dm_resized[y1:y2, x1:x2].mean())
            else:
                region_depth = obj.depth_value  # fallback to centre pixel

            from dataclasses import replace as _replace  # noqa: PLC0415
            updated.append(_replace(obj, depth_value=region_depth))

        return updated

    def answer_question(self, question: str, image_path: str) -> Dict[str, Any]:
        """
        End-to-end method: analyse *image_path*, generate code for *question*,
        execute it, and return a results dictionary.
        """
        scene = self.analyze_image(image_path)
        code = self.code_generator.generate_code(question, scene)
        answer, status = self.code_generator.execute_code(code, scene)

        return {
            "question": question,
            "answer": answer,
            "status": status,
            "code": code,
            "objects_detected": [asdict(o) for o in scene.objects],
            "timestamp": scene.timestamp,
        }

    def answer_followup(self, followup_question: str) -> Dict[str, Any]:
        """
        Answer a follow-up question using the most recently analysed scene and
        the full conversation history so the model can reference prior answers.

        Raises:
            RuntimeError: if no image has been analysed yet.
        """
        if self._last_analysis is None:
            raise RuntimeError(
                "No image has been analysed yet. Call answer_question() first."
            )
        scene = self._last_analysis
        code = self.code_generator.generate_followup(followup_question, scene)
        answer, status = self.code_generator.execute_code(code, scene)

        return {
            "question": followup_question,
            "answer": answer,
            "status": status,
            "code": code,
            "objects_detected": [asdict(o) for o in scene.objects],
            "timestamp": scene.timestamp,
        }

    @property
    def last_analysis(self) -> Optional[SceneAnalysis]:
        """Return the most recent SceneAnalysis, or None if no image has been analysed."""
        return self._last_analysis


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python vadar_agent.py <image_path> <question>")
        sys.exit(1)

    _api_key = os.environ.get("OPENAI_API_KEY", "")
    if not _api_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)

    _agent = VADARAgent(_api_key)
    _result = _agent.answer_question(sys.argv[2], sys.argv[1])
    print(json.dumps(_result, indent=2, default=str))