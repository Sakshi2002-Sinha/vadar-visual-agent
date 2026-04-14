"""
VADAR-Inspired Visual Agent for 3D Spatial Reasoning on 2D Images
Performs agentic code generation and execution for spatial understanding.

Architecture:
  VisionModels      – wraps HuggingFace pipelines for detection / depth / segmentation
  SpatialObject     – dataclass representing one detected object with spatial attributes
  SceneAnalysis     – container holding all objects + depth map for a single image
  SpatialReasoner   – pure-function helpers for spatial comparisons
    CodeGenerator     – calls GitHub Models/OpenAI APIs (with local fallback) to generate and exec code
  VADARAgent        – top-level orchestrator
"""

import os
import json
import base64
import tempfile
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime

import numpy as np
from PIL import Image
import openai
from transformers import pipeline as hf_pipeline


# ---------------------------------------------------------------------------
# Vision models
# ---------------------------------------------------------------------------

class VisionModels:
    """Manages pretrained vision models for object detection, segmentation, and depth estimation."""

    def __init__(self, use_gpu: bool = False, enable_segmentation: bool = False):
        self.device = 0 if use_gpu else -1
        self.object_detector = hf_pipeline(
            "object-detection",
            model="facebook/detr-resnet-50",
            device=self.device,
        )
        self.depth_estimator = hf_pipeline(
            "depth-estimation",
            model="Intel/dpt-large",
            device=self.device,
        )
        self._segmentation = None
        if enable_segmentation:
            self._segmentation = hf_pipeline(
                "image-segmentation",
                model="facebook/detr-resnet-50-panoptic",
                device=self.device,
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
        if self._segmentation is None:
            self._segmentation = hf_pipeline(
                "image-segmentation",
                model="facebook/detr-resnet-50-panoptic",
                device=self.device,
            )
        return self._segmentation(image)


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
    processing_times: Dict[str, float] = field(default_factory=dict)
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
    def is_closer(obj1: SpatialObject, obj2: SpatialObject) -> bool:
        """Return True if *obj1* is closer to the camera than *obj2*."""
        return obj1.distance_from_camera() < obj2.distance_from_camera()

    @staticmethod
    def is_left_of(obj1: SpatialObject, obj2: SpatialObject) -> bool:
        """Return True if *obj1* is left of *obj2* in image coordinates."""
        return obj1.center[0] < obj2.center[0]

    @staticmethod
    def is_right_of(obj1: SpatialObject, obj2: SpatialObject) -> bool:
        """Return True if *obj1* is right of *obj2* in image coordinates."""
        return obj1.center[0] > obj2.center[0]

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
        api_key: str = "",
        model: str = "gpt-4o",
        provider: str = "auto",
        openai_base_url: str = "",
        github_token: str = "",
        github_model: str = "gpt-4o-mini",
        github_base_url: str = "https://models.inference.ai.azure.com",
    ):
        self.api_key = api_key
        self.model = model
        self.provider = provider.lower().strip() if provider else "auto"
        self.openai_base_url = openai_base_url.strip()
        self.github_token = github_token.strip()
        self.github_model = github_model.strip() or "gpt-4o-mini"
        self.github_base_url = github_base_url.strip() or "https://models.inference.ai.azure.com"
        self.history: List[Dict[str, Any]] = []

    def _provider_attempts(self) -> List[Dict[str, str]]:
        attempts: List[Dict[str, str]] = []

        if self.provider in {"auto", "github"} and self.github_token:
            attempts.append(
                {
                    "provider": "github",
                    "api_key": self.github_token,
                    "model": self.github_model,
                    "base_url": self.github_base_url,
                }
            )

        if self.provider in {"auto", "openai"} and self.api_key:
            attempts.append(
                {
                    "provider": "openai",
                    "api_key": self.api_key,
                    "model": self.model,
                    "base_url": self.openai_base_url,
                }
            )

        return attempts

    @staticmethod
    def _normalize_object_phrase(text: str) -> str:
        phrase = text.strip().lower()
        phrase = re.sub(r"\b(the|a|an)\b", "", phrase)
        phrase = re.sub(r"\s+", " ", phrase).strip(" .,?!")
        return phrase

    def _extract_required_objects(self, question: str) -> List[str]:
        """Extract object labels required by common comparative question forms."""
        q = question.strip().lower()
        patterns = [
            r"is\s+(?:the\s+)?(.+?)\s+(?:farther|further|closer|nearer)\b.*?than\s+(?:the\s+)?(.+?)[?.!]*$",
            r"is\s+(?:the\s+)?(.+?)\s+(?:left|right)\s+of\s+(?:the\s+)?(.+?)[?.!]*$",
            r"which\s+is\s+(?:farther|further|closer|nearer)\b.*?,\s*(?:the\s+)?(.+?)\s+or\s+(?:the\s+)?(.+?)[?.!]*$",
        ]

        for pattern in patterns:
            match = re.search(pattern, q)
            if match:
                return [
                    self._normalize_object_phrase(match.group(1)),
                    self._normalize_object_phrase(match.group(2)),
                ]

        return []

    def _precheck_missing_objects_code(self, question: str, scene: SceneAnalysis) -> Optional[str]:
        """Return deterministic code if required objects are missing, else None."""
        required = [obj for obj in self._extract_required_objects(question) if obj]
        if not required:
            return None

        missing: List[str] = []
        for label in required:
            if SpatialReasoner.get_object_by_label(scene.objects, label) is None:
                missing.append(label)

        if not missing:
            return None

        missing_text = ", ".join(sorted(set(missing)))
        return (
            f"answer = {f'Cannot determine: required objects not detected: {missing_text}'!r}"
        )

    def planned_provider_label(self) -> str:
        """Human-readable provider/model that will be attempted first."""
        if self.provider == "local":
            return "local/rule-based"
        attempts = self._provider_attempts()
        if attempts:
            first = attempts[0]
            return f"{first['provider']}/{first['model']}"
        return "local/rule-based (no remote credentials)"

    def _fallback_code(self, question: str, scene: SceneAnalysis) -> str:
        question_l = question.lower()
        labels = [o.label.lower() for o in scene.objects]

        def find_label_from_question(candidates: List[str], q: str) -> Optional[str]:
            # Prefer longer labels first so "dining table" matches before "table".
            for label in sorted(candidates, key=len, reverse=True):
                if re.search(rf"\\b{re.escape(label)}\\b", q):
                    return label
            return None

        obj1 = find_label_from_question(labels, question_l)
        q_without_obj1 = question_l.replace(obj1, "", 1) if obj1 else question_l
        obj2 = find_label_from_question(labels, q_without_obj1)

        if "closest" in question_l or "nearest" in question_l:
            return (
                "closest = min(objects, key=lambda o: o.depth_value)\n"
                "answer = closest.label"
            )

        if "farthest" in question_l or "furthest" in question_l:
            return (
                "farthest = max(objects, key=lambda o: o.depth_value)\n"
                "answer = farthest.label"
            )

        if obj1 and obj2 and ("farther" in question_l or "further" in question_l):
            return (
                f"obj1 = SpatialReasoner.get_object_by_label(objects, {obj1!r})\n"
                f"obj2 = SpatialReasoner.get_object_by_label(objects, {obj2!r})\n"
                "answer = SpatialReasoner.is_farther(obj1, obj2) if (obj1 and obj2) else None"
            )

        if obj1 and obj2 and "closer" in question_l:
            return (
                f"obj1 = SpatialReasoner.get_object_by_label(objects, {obj1!r})\n"
                f"obj2 = SpatialReasoner.get_object_by_label(objects, {obj2!r})\n"
                "answer = SpatialReasoner.is_closer(obj1, obj2) if (obj1 and obj2) else None"
            )

        if obj1 and obj2 and "left" in question_l:
            return (
                f"obj1 = SpatialReasoner.get_object_by_label(objects, {obj1!r})\n"
                f"obj2 = SpatialReasoner.get_object_by_label(objects, {obj2!r})\n"
                "answer = SpatialReasoner.is_left_of(obj1, obj2) if (obj1 and obj2) else None"
            )

        if obj1 and obj2 and "right" in question_l:
            return (
                f"obj1 = SpatialReasoner.get_object_by_label(objects, {obj1!r})\n"
                f"obj2 = SpatialReasoner.get_object_by_label(objects, {obj2!r})\n"
                "answer = SpatialReasoner.is_right_of(obj1, obj2) if (obj1 and obj2) else None"
            )

        return (
            "summary = [\n"
            "    {\n"
            "        'label': o.label,\n"
            "        'depth': round(float(o.depth_value), 4),\n"
            "        'center': o.center,\n"
            "        'confidence': round(float(o.confidence), 4),\n"
            "    }\n"
            "    for o in objects\n"
            "]\n"
            "answer = summary"
        )

    def fallback_code_for(self, question: str, scene: SceneAnalysis) -> str:
        """Public wrapper for deterministic local fallback code."""
        return self._fallback_code(question, scene)

    @staticmethod
    def _sanitize_generated_code(code: str) -> str:
        """Normalize model output to raw Python code."""
        cleaned = code.strip()

        if "```" in cleaned:
            parts = cleaned.split("```")
            if len(parts) >= 3:
                block = parts[1]
                if block.startswith("python"):
                    block = block[len("python"):]
                cleaned = block.strip()

        return cleaned.strip()

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
        """Call configured remote providers and fallback to local rule-based code when needed."""
        fallback_reason: Optional[str] = None

        prechecked_code = self._precheck_missing_objects_code(question, scene)
        if prechecked_code is not None:
            self.history.append({
                "question": question,
                "code": prechecked_code,
                "provider": "local",
                "model": "scene-grounded-precheck",
                "fallback": True,
                "fallback_reason": "Required objects missing in detections",
                "timestamp": datetime.now().isoformat(),
            })
            return prechecked_code

        attempts = self._provider_attempts()

        if self.provider == "local":
            fallback_reason = "LLM_PROVIDER=local"
        elif not attempts:
            fallback_reason = "No configured remote provider credentials"
        else:
            provider_errors: List[str] = []
            for attempt in attempts:
                try:
                    client = openai.OpenAI(
                        api_key=attempt["api_key"],
                        base_url=attempt["base_url"] or None,
                    )
                    response = client.chat.completions.create(
                        model=attempt["model"],
                        messages=[
                            {"role": "system", "content": self._SYSTEM_PROMPT},
                            {"role": "user", "content": self._build_user_prompt(question, scene)},
                        ],
                        temperature=0.3,
                        max_tokens=1500,
                    )
                    code = self._sanitize_generated_code(
                        response.choices[0].message.content or ""
                    )
                    self.history.append({
                        "question": question,
                        "code": code,
                        "provider": attempt["provider"],
                        "model": attempt["model"],
                        "fallback": False,
                        "fallback_reason": None,
                        "timestamp": datetime.now().isoformat(),
                    })
                    return code
                except Exception as exc:  # noqa: BLE001
                    provider_errors.append(
                        f"{attempt['provider']} unavailable ({exc.__class__.__name__})"
                    )

            fallback_reason = "; ".join(provider_errors)

        if fallback_reason:
            print(f"[CodeGenerator] Using free fallback reasoner: {fallback_reason}")
            code = self._fallback_code(question, scene)

        self.history.append({
            "question": question,
            "code": code,
            "provider": "local",
            "model": "rule-based",
            "fallback": bool(fallback_reason),
            "fallback_reason": fallback_reason,
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
        api_key: str = "",
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
    ):
        self.vision_models = VisionModels(
            use_gpu=use_gpu,
            enable_segmentation=enable_segmentation,
        )
        self.spatial_reasoner = SpatialReasoner()
        self.code_generator = CodeGenerator(
            api_key,
            model=model,
            provider=provider,
            openai_base_url=openai_base_url,
            github_token=github_token,
            github_model=github_model,
            github_base_url=github_base_url,
        )
        self.min_detection_confidence = max(0.0, min(1.0, min_detection_confidence))
        self.max_objects = max_objects
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
        t0 = time.perf_counter()
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)
        height, width = image_array.shape[:2]

        t_detect_start = time.perf_counter()
        detections = self.vision_models.detect_objects(image)
        detect_ms = (time.perf_counter() - t_detect_start) * 1000.0

        t_depth_start = time.perf_counter()
        depth_map = self.vision_models.estimate_depth(image)
        depth_ms = (time.perf_counter() - t_depth_start) * 1000.0

        filtered_detections = [
            det for det in detections if float(det.get("score", 0.0)) >= self.min_detection_confidence
        ]
        filtered_detections.sort(key=lambda d: float(d.get("score", 0.0)), reverse=True)
        if self.max_objects > 0:
            filtered_detections = filtered_detections[: self.max_objects]

        if depth_map.shape != (height, width):
            import cv2  # noqa: PLC0415
            dm_resized = cv2.resize(depth_map, (width, height))
        else:
            dm_resized = depth_map

        objects: List[SpatialObject] = []
        for det in filtered_detections:
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

        analysis = SceneAnalysis(
            objects=objects,
            depth_map=depth_map,
            image_shape=image_array.shape,
            processing_times={
                "detection_ms": detect_ms,
                "depth_ms": depth_ms,
                "total_ms": (time.perf_counter() - t0) * 1000.0,
                "detections_before_filter": float(len(detections)),
                "detections_after_filter": float(len(objects)),
            },
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

    _api_key = os.environ.get("OPENAI_API_KEY", "")
    if not _api_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)

    _agent = VADARAgent(_api_key)
    _result = _agent.answer_question(sys.argv[2], sys.argv[1])
    print(json.dumps(_result, indent=2, default=str))