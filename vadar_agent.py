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
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime

import numpy as np
from PIL import Image
import openai
from transformers import pipeline as hf_pipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vision models
# ---------------------------------------------------------------------------

class VisionModels:
    """Manages pretrained vision models for object detection, segmentation, and depth estimation."""

    def __init__(self, use_gpu: bool = False):
        device = 0 if use_gpu else -1
        object_model = os.environ.get("OBJECT_DETECTION_MODEL", "IDEA-Research/grounding-dino-base")
        depth_model = os.environ.get("DEPTH_ESTIMATION_MODEL", "lpiccinelli/unidepth-v2-vitl14")
        segmentation_model = os.environ.get("SEGMENTATION_MODEL", "facebook/sam2-hiera-large")
        vqa_model = os.environ.get("VQA_MODEL", "allenai/Molmo-7B-D-0924")

        self.object_detector = self._build_pipeline(
            task="object-detection",
            model=object_model,
            device=device,
            fallback_model="facebook/detr-resnet-50",
        )
        self.depth_estimator = self._build_pipeline(
            task="depth-estimation",
            model=depth_model,
            device=device,
            fallback_model="Intel/dpt-large",
        )
        self.segmentation = self._build_pipeline(
            task="mask-generation",
            model=segmentation_model,
            device=device,
            fallback_model="facebook/detr-resnet-50-panoptic",
            fallback_task="image-segmentation",
        )
        self.vqa = self._build_pipeline(
            task="image-text-to-text",
            model=vqa_model,
            device=device,
            fallback_model=None,
        )

    @staticmethod
    def _build_pipeline(
        task: str,
        model: str,
        device: int,
        fallback_model: Optional[str] = None,
        fallback_task: Optional[str] = None,
    ) -> Any:
        try:
            return hf_pipeline(task, model=model, device=device)
        except Exception as primary_exc:
            if fallback_model:
                try:
                    return hf_pipeline(fallback_task or task, model=fallback_model, device=device)
                except Exception as fallback_exc:
                    logger.warning(
                        "Failed loading primary model '%s' (%s) and fallback model '%s' (%s).",
                        model,
                        primary_exc,
                        fallback_model,
                        fallback_exc,
                    )
                    return None
            logger.warning("Failed loading model '%s' for task '%s': %s", model, task, primary_exc)
            return None

    def detect_objects(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect objects in the image with the configured detector model."""
        if self.object_detector is None:
            return []
        return self.object_detector(image)

    def estimate_depth(self, image: Image.Image) -> np.ndarray:
        """Return a normalized depth map (0 = close, 1 = far) for *image*."""
        if self.depth_estimator is None:
            image_array = np.array(image)
            return np.zeros(image_array.shape[:2], dtype=np.float32)
        result = self.depth_estimator(image)
        depth_map = np.array(result["depth"], dtype=np.float32)
        min_val, max_val = depth_map.min(), depth_map.max()
        if max_val > min_val:
            depth_map = (depth_map - min_val) / (max_val - min_val)
        return depth_map

    def segment_objects(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Perform segmentation and return segment list."""
        if self.segmentation is None:
            return []
        output = self.segmentation(image)
        if isinstance(output, list):
            return output
        if isinstance(output, dict):
            if "segments_info" in output and isinstance(output["segments_info"], list):
                return output["segments_info"]
            return [output]
        return []

    def answer_vqa(self, image: Image.Image, question: str) -> Optional[str]:
        """Answer VQA question with Molmo when available."""
        if self.vqa is None:
            return None
        try:
            response = None
            for param_name in ("text", "question", "prompt"):
                try:
                    response = self.vqa(image, **{param_name: question})
                    break
                except TypeError:
                    continue

            if response is None:
                return None

            def _extract_text(payload: Dict[str, Any]) -> Optional[str]:
                for key in ("generated_text", "answer", "text"):
                    if key in payload and payload[key]:
                        return str(payload[key])
                return None

            if isinstance(response, list) and response:
                first = response[0]
                if isinstance(first, dict):
                    extracted = _extract_text(first)
                    if extracted is not None:
                        return extracted
                return str(first)
            if isinstance(response, dict):
                extracted = _extract_text(response)
                if extracted is not None:
                    return extracted
                return str(response)
            return str(response)
        except Exception as exc:
            logger.warning("Molmo VQA failed: %s", exc)
            return None


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
        "You are an expert in spatial reasoning on 2D images that include depth information. "
        "When asked a question about a scene, you write self-contained Python code that produces "
        "a variable named `answer` holding the result. "
        "Use only numpy, math, and the provided scene data structures. "
        "Output ONLY valid Python – no markdown fences, no prose."
    )

    def __init__(
        self,
        api_key: str,
        model: str = "google/gemini-2.0-flash",
        base_url: Optional[str] = None,
    ):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.history: List[Dict[str, Any]] = []

    def _build_user_prompt(self, question: str, scene: SceneAnalysis) -> str:
        objects_desc = "\n".join(
            f"  [{i}] label={o.label!r} confidence={o.confidence:.3f} "
            f"depth={o.depth_value:.3f} center={o.center} area={o.area:.4f}"
            for i, o in enumerate(scene.objects)
        )
        return (
            f"Scene objects:\n{objects_desc}\n\n"
            f"Question: {question}\n\n"
            "Write Python code that assigns the answer to a variable named `answer`. "
            "You may use numpy as `np`. "
            "The list `objects` contains SpatialObject instances with attributes: "
            "label, confidence, bbox, center, depth_value, area, image_height, image_width. "
            "The helper class `SpatialReasoner` is available."
        )

    def generate_code(self, question: str, scene: SceneAnalysis) -> str:
        """Call the OpenAI Chat API and return generated Python code."""
        response = self.client.chat.completions.create(
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
            "timestamp": datetime.now().isoformat(),
        })
        return code

    def execute_code(
        self, code: str, scene: SceneAnalysis
    ) -> Tuple[Any, str]:
        """Execute *code* with scene context injected. Returns (answer, status)."""
        exec_globals: Dict[str, Any] = {
            "np": np,
            "SpatialObject": SpatialObject,
            "SpatialReasoner": SpatialReasoner,
            "scene_analysis": scene,
            "objects": scene.objects,
        }
        try:
            exec(code, exec_globals)  # noqa: S102  # code is LLM-generated; review before production use
            return exec_globals.get("answer", "No answer produced"), "Success"
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

    def __init__(
        self,
        api_key: str,
        use_gpu: bool = False,
        model: str = "google/gemini-2.0-flash",
        base_url: Optional[str] = None,
    ):
        self.vision_models = VisionModels(use_gpu=use_gpu)
        self.spatial_reasoner = SpatialReasoner()
        self.code_generator = CodeGenerator(api_key, model=model, base_url=base_url)
        self._last_analysis: Optional[SceneAnalysis] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_image(self, image_path: str) -> SceneAnalysis:
        """
        Run the full vision pipeline on *image_path* and return a SceneAnalysis.

        Steps:
          1. Object detection
          2. Monocular depth estimation
          3. Build SpatialObject list from detections + depth map
        """
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)
        height, width = image_array.shape[:2]

        detections = self.vision_models.detect_objects(image)
        depth_map = self.vision_models.estimate_depth(image)

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

            # Resize depth map to image dimensions if needed
            if depth_map.shape != (height, width):
                import cv2  # noqa: PLC0415
                dm_resized = cv2.resize(depth_map, (width, height))
            else:
                dm_resized = depth_map

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

        analysis = SceneAnalysis(
            objects=objects,
            depth_map=depth_map,
            image_shape=image_array.shape,
        )
        self._last_analysis = analysis
        return analysis

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

    _api_key = os.environ.get("GEMINI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    if not _api_key:
        print("ERROR: GEMINI_API_KEY (or OPENAI_API_KEY) environment variable is not set.")
        sys.exit(1)

    _agent = VADARAgent(
        _api_key,
        model=os.environ.get("LLM_MODEL", "google/gemini-2.0-flash"),
        base_url=os.environ.get(
            "OPENAI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        ),
    )
    _result = _agent.answer_question(sys.argv[2], sys.argv[1])
    print(json.dumps(_result, indent=2, default=str))