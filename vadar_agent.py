"""
VADAR-Inspired Visual Agent for 3D Spatial Reasoning on 2D Images
Performs agentic code generation and execution for spatial understanding.

Architecture:
  VisionModels          – wraps HuggingFace pipelines for detection / depth
  CLIPAttributeExtractor – zero-shot color / material / size via CLIP
  SpatialObject         – dataclass representing one detected object
  SceneAnalysis         – container holding all objects + depth map
  SpatialReasoner       – pure-function helpers for spatial comparisons
  CodeGenerator         – calls the OpenAI API to generate and exec Python
  VADARAgent            – top-level orchestrator

Novel enhancements vs. baseline:
  * Feature 1  – Region-averaged (median) depth instead of single center pixel
  * Feature 2  – GPT-4o Vision multimodal context (image + text in one prompt)
  * Feature 3  – Self-repair code execution loop (up to 2 automatic retries)
  * Feature 5  – Confidence-threshold filtering + IoU-based NMS
  * Feature 6  – CLIP-based attribute extraction (color, material, size)
  * Feature 7  – Depth inversion guard heuristic
  * Feature 8  – Extended SpatialReasoner API
  * Feature 10 – Removed unused panoptic segmentation model at startup
"""

import os
import json
import base64
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
    """Manages pretrained vision models for object detection and depth estimation.

    The unused panoptic segmentation model has been removed (Feature 10).
    Detection confidence filtering and IoU-NMS are applied automatically
    (Feature 5).  The depth inversion guard is applied before returning
    the depth map (Feature 7).
    """

    def __init__(
        self,
        use_gpu: bool = False,
        detection_threshold: float = 0.7,
        nms_iou_threshold: float = 0.5,
        depth_inversion_guard: bool = True,
    ):
        device = 0 if use_gpu else -1
        self.detection_threshold = detection_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.depth_inversion_guard = depth_inversion_guard

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
        # Note: panoptic segmentation model removed (Feature 10) – saves ~2 GB
        # VRAM and ~8 s startup time.  Call segment_objects() only if you
        # explicitly load the model yourself.

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_objects(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect objects, apply confidence threshold and IoU-NMS (Feature 5)."""
        raw = self.object_detector(image)
        # Threshold filtering
        filtered = [d for d in raw if float(d["score"]) >= self.detection_threshold]
        # NMS
        return self._nms(filtered)

    @staticmethod
    def _box_iou(
        a: Dict[str, int], b: Dict[str, int]
    ) -> float:
        """Compute IoU between two raw detection boxes."""
        x0 = max(a["xmin"], b["xmin"])
        y0 = max(a["ymin"], b["ymin"])
        x1 = min(a["xmax"], b["xmax"])
        y1 = min(a["ymax"], b["ymax"])
        inter = max(0, x1 - x0) * max(0, y1 - y0)
        area_a = (a["xmax"] - a["xmin"]) * (a["ymax"] - a["ymin"])
        area_b = (b["xmax"] - b["xmin"]) * (b["ymax"] - b["ymin"])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _nms(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate detections with IoU above *nms_iou_threshold*."""
        detections = sorted(detections, key=lambda d: float(d["score"]), reverse=True)
        kept: List[Dict[str, Any]] = []
        suppressed = set()
        for i, det in enumerate(detections):
            if i in suppressed:
                continue
            kept.append(det)
            for j, other in enumerate(detections[i + 1 :], start=i + 1):
                if j in suppressed:
                    continue
                if self._box_iou(det["box"], other["box"]) >= self.nms_iou_threshold:
                    suppressed.add(j)
        return kept

    # ------------------------------------------------------------------
    # Depth
    # ------------------------------------------------------------------

    def estimate_depth(self, image: Image.Image) -> np.ndarray:
        """Return a normalized depth map (0 = close, 1 = far) for *image*.

        Feature 7 – depth inversion guard: after normalization the bottom
        third of a natural image should be on average *closer* (lower depth
        value) than the top third.  When the opposite is true the map is
        flipped automatically.
        """
        result = self.depth_estimator(image)
        depth_map = np.array(result["depth"], dtype=np.float32)
        min_val, max_val = depth_map.min(), depth_map.max()
        if max_val > min_val:
            depth_map = (depth_map - min_val) / (max_val - min_val)

        if self.depth_inversion_guard:
            h = depth_map.shape[0]
            third = h // 3
            bottom_mean = depth_map[2 * third :, :].mean()
            top_mean = depth_map[:third, :].mean()
            # Bottom of frame contains ground / closer objects.
            # If bottom is unexpectedly *farther* by more than 0.1, flip.
            if bottom_mean - top_mean > 0.10:
                depth_map = 1.0 - depth_map

        return depth_map


# ---------------------------------------------------------------------------
# CLIP attribute extractor  (Feature 6)
# ---------------------------------------------------------------------------

class CLIPAttributeExtractor:
    """Zero-shot classification of visual attributes (color, material, size)
    using ``openai/clip-vit-base-patch32`` via HuggingFace Transformers.

    Loading this model is optional.  Pass ``use_clip=True`` to VADARAgent to
    enable it; otherwise attributes are left as *None* on SpatialObject.
    """

    COLOR_LABELS = [
        "red", "orange", "yellow", "green", "blue", "purple",
        "pink", "brown", "black", "white", "gray",
    ]
    MATERIAL_LABELS = [
        "wood", "metal", "plastic", "fabric",
        "glass", "ceramic", "stone", "rubber",
    ]
    SIZE_LABELS = ["small", "medium", "large"]

    def __init__(self, device: int = -1) -> None:
        from transformers import CLIPProcessor, CLIPModel  # noqa: PLC0415
        import torch  # noqa: PLC0415

        model_name = "openai/clip-vit-base-patch32"
        self._model = CLIPModel.from_pretrained(model_name)
        self._processor = CLIPProcessor.from_pretrained(model_name)
        self._device = "cpu" if device < 0 else f"cuda:{device}"
        if device >= 0:
            self._model = self._model.to(self._device)
        self._torch = torch

    def _classify(
        self,
        crop: Image.Image,
        labels: List[str],
        template: str = "a photo of a {} object",
    ) -> str:
        """Return the most probable label for *crop* under zero-shot CLIP."""
        texts = [template.format(lbl) for lbl in labels]
        inputs = self._processor(
            text=texts, images=crop, return_tensors="pt", padding=True
        )
        if self._device != "cpu":
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with self._torch.no_grad():
            outputs = self._model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]
        return labels[int(probs.argmax().item())]

    def extract_attributes(
        self,
        image: Image.Image,
        bbox: Tuple[float, float, float, float],
    ) -> Dict[str, str]:
        """Return color, material, and relative_size for the bbox crop."""
        w, h = image.size
        x0 = max(0, int(bbox[0] * w))
        y0 = max(0, int(bbox[1] * h))
        x1 = min(w, int(bbox[2] * w))
        y1 = min(h, int(bbox[3] * h))
        crop = image.crop((x0, y0, x1, y1))
        return {
            "color": self._classify(crop, self.COLOR_LABELS),
            "material": self._classify(crop, self.MATERIAL_LABELS),
            "relative_size": self._classify(crop, self.SIZE_LABELS),
        }


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class SpatialObject:
    """Represents a detected object with its spatial properties.

    The ``depth_value`` field now holds the *median* depth of all pixels
    inside the bounding-box region (Feature 1) rather than the single
    center-pixel value.

    Optional attribute fields (Feature 6) are populated when CLIP is
    enabled via ``VADARAgent(use_clip=True)``.
    """

    label: str
    confidence: float
    # Normalised bounding box: (x_min, y_min, x_max, y_max) in [0, 1]
    bbox: Tuple[float, float, float, float]
    # Pixel coordinates of the bounding-box center
    center: Tuple[int, int]
    # Median normalised depth inside the bounding box (0 = close, 1 = far)
    depth_value: float
    # Normalised area relative to the full image
    area: float
    image_height: int
    image_width: int
    # Optional CLIP-derived attributes (Feature 6)
    color: Optional[str] = None
    material: Optional[str] = None
    relative_size: Optional[str] = None

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
# Spatial reasoning helpers  (Feature 8 – extended API)
# ---------------------------------------------------------------------------

class SpatialReasoner:
    """Pure helper functions for spatial comparisons between SpatialObjects.

    New methods added (Feature 8):
      * iou              – bounding-box overlap between two objects
      * is_occluded      – True when IoU exceeds an overlap threshold
      * is_between       – True when one object is between two others
      * count_objects_of_type – robust plural counting by label
      * closest_to       – nearest neighbour by depth + pixel distance
      * depth_rank       – sort objects from closest to furthest
    """

    # ------------------------------------------------------------------
    # Original helpers (unchanged)
    # ------------------------------------------------------------------

    @staticmethod
    def get_object_by_label(
        objects: List[SpatialObject], label: str
    ) -> Optional[SpatialObject]:
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

    # ------------------------------------------------------------------
    # New helpers (Feature 8)
    # ------------------------------------------------------------------

    @staticmethod
    def iou(obj1: SpatialObject, obj2: SpatialObject) -> float:
        """Compute Intersection-over-Union of two objects' bounding boxes."""
        x0 = max(obj1.bbox[0], obj2.bbox[0])
        y0 = max(obj1.bbox[1], obj2.bbox[1])
        x1 = min(obj1.bbox[2], obj2.bbox[2])
        y1 = min(obj1.bbox[3], obj2.bbox[3])
        inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        area1 = (obj1.bbox[2] - obj1.bbox[0]) * (obj1.bbox[3] - obj1.bbox[1])
        area2 = (obj2.bbox[2] - obj2.bbox[0]) * (obj2.bbox[3] - obj2.bbox[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def is_occluded(
        obj1: SpatialObject,
        obj2: SpatialObject,
        iou_threshold: float = 0.3,
    ) -> bool:
        """Return True if *obj1* and *obj2* overlap significantly (possible occlusion)."""
        return SpatialReasoner.iou(obj1, obj2) >= iou_threshold

    @staticmethod
    def is_between(
        obj_a: SpatialObject,
        obj_b: SpatialObject,
        obj_c: SpatialObject,
    ) -> bool:
        """Return True if *obj_b* is spatially between *obj_a* and *obj_c*.

        Checks both horizontal pixel position and depth value.
        """
        x_lo = min(obj_a.center[0], obj_c.center[0])
        x_hi = max(obj_a.center[0], obj_c.center[0])
        d_lo = min(obj_a.depth_value, obj_c.depth_value)
        d_hi = max(obj_a.depth_value, obj_c.depth_value)
        x_between = x_lo <= obj_b.center[0] <= x_hi
        d_between = d_lo <= obj_b.depth_value <= d_hi
        return x_between and d_between

    @staticmethod
    def count_objects_of_type(objects: List[SpatialObject], label: str) -> int:
        """Count objects whose label contains *label* (case-insensitive)."""
        return sum(1 for obj in objects if label.lower() in obj.label.lower())

    @staticmethod
    def closest_to(
        objects: List[SpatialObject],
        reference_obj: SpatialObject,
    ) -> Optional[SpatialObject]:
        """Return the object (excluding *reference_obj*) nearest in combined
        depth + normalised pixel distance."""
        others = [o for o in objects if o is not reference_obj]
        if not others:
            return None
        depths = [abs(o.depth_value - reference_obj.depth_value) for o in others]
        pixels = [SpatialReasoner.pixel_distance(o, reference_obj) for o in others]
        max_px = max(pixels) if max(pixels) > 0 else 1.0

        def _score(i: int) -> float:
            return 0.5 * depths[i] + 0.5 * pixels[i] / max_px

        return others[min(range(len(others)), key=_score)]

    @staticmethod
    def depth_rank(objects: List[SpatialObject]) -> List[SpatialObject]:
        """Return objects sorted from closest (lowest depth) to furthest."""
        return sorted(objects, key=lambda o: o.depth_value)


# ---------------------------------------------------------------------------
# Code generation + execution  (Features 2, 3)
# ---------------------------------------------------------------------------

class CodeGenerator:
    """Generates Python code via the OpenAI API and executes it with scene context.

    Feature 2 – when an *image_path* is supplied to generate_code(), the image
    is base64-encoded and forwarded to the model as a vision message, giving the
    LLM direct pixel-level context alongside the structured scene description.

    Feature 3 – execute_code_with_repair() retries failed executions by feeding
    the error traceback back to the model for automatic self-correction.
    """

    _SYSTEM_PROMPT = (
        "You are an expert in spatial reasoning on 2D images that include depth information. "
        "When asked a question about a scene, you write self-contained Python code that produces "
        "a variable named `answer` holding the result. "
        "Use only numpy, math, and the provided scene data structures. "
        "Output ONLY valid Python – no markdown fences, no prose."
    )

    def __init__(self, api_key: str, model: str = "gpt-4o") -> None:
        openai.api_key = api_key
        self.model = model
        self.history: List[Dict[str, Any]] = []

    def _build_user_prompt(self, question: str, scene: SceneAnalysis) -> str:
        """Build the text portion of the user prompt.

        Includes optional CLIP attributes (color, material, relative_size) when
        present on SpatialObject instances (Feature 6).
        """
        lines: List[str] = []
        has_attrs = any(o.color is not None for o in scene.objects)
        for i, o in enumerate(scene.objects):
            attr_parts: List[str] = []
            if o.color is not None:
                attr_parts.append(f"color={o.color!r}")
            if o.material is not None:
                attr_parts.append(f"material={o.material!r}")
            if o.relative_size is not None:
                attr_parts.append(f"size={o.relative_size!r}")
            attr_str = ("  " + "  ".join(attr_parts)) if attr_parts else ""
            lines.append(
                f"  [{i}] label={o.label!r} confidence={o.confidence:.3f} "
                f"depth={o.depth_value:.3f} center={o.center} area={o.area:.4f}"
                + attr_str
            )
        objects_desc = "\n".join(lines)
        attr_fields = ", color, material, relative_size" if has_attrs else ""
        return (
            f"Scene objects:\n{objects_desc}\n\n"
            f"Question: {question}\n\n"
            "Write Python code that assigns the answer to a variable named `answer`. "
            "You may use numpy as `np`. "
            "The list `objects` contains SpatialObject instances with attributes: "
            f"label, confidence, bbox, center, depth_value, area, "
            f"image_height, image_width{attr_fields}. "
            "The helper class `SpatialReasoner` is available with methods: "
            "get_object_by_label, is_farther, relative_depth_distance, pixel_distance, "
            "vertical_position, horizontal_position, iou, is_occluded, is_between, "
            "count_objects_of_type, closest_to, depth_rank."
        )

    # ------------------------------------------------------------------
    # Code generation  (Feature 2 – vision multimodal)
    # ------------------------------------------------------------------

    def generate_code(
        self,
        question: str,
        scene: SceneAnalysis,
        image_path: Optional[str] = None,
    ) -> str:
        """Call the OpenAI Chat API and return generated Python code.

        When *image_path* is provided the image is base64-encoded and sent as a
        vision message so the model can directly observe color, texture, and
        spatial layout (Feature 2).
        """
        user_text = self._build_user_prompt(question, scene)

        if image_path:
            suffix = Path(image_path).suffix.lower().lstrip(".")
            mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
            with open(image_path, "rb") as fh:
                img_b64 = base64.b64encode(fh.read()).decode("utf-8")
            user_content: Any = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{mime};base64,{img_b64}"},
                },
                {"type": "text", "text": user_text},
            ]
        else:
            user_content = user_text

        response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
            max_tokens=1500,
        )
        code = response.choices[0].message.content.strip()
        self.history.append(
            {
                "question": question,
                "code": code,
                "timestamp": datetime.now().isoformat(),
            }
        )
        return code

    # ------------------------------------------------------------------
    # Code execution
    # ------------------------------------------------------------------

    def execute_code(self, code: str, scene: SceneAnalysis) -> Tuple[Any, str]:
        """Execute *code* with scene context injected. Returns (answer, status)."""
        exec_globals: Dict[str, Any] = {
            "np": np,
            "SpatialObject": SpatialObject,
            "SpatialReasoner": SpatialReasoner,
            "scene_analysis": scene,
            "objects": scene.objects,
        }
        try:
            exec(code, exec_globals)  # noqa: S102
            return exec_globals.get("answer", "No answer produced"), "Success"
        except Exception as exc:  # noqa: BLE001
            return None, f"Execution error: {exc}"

    def execute_code_with_repair(
        self,
        code: str,
        scene: SceneAnalysis,
        question: str,
        image_path: Optional[str] = None,
        max_retries: int = 2,
    ) -> Tuple[Any, str]:
        """Execute *code* with automatic self-repair on failure (Feature 3).

        On each failed execution the error message is fed back to the model as a
        follow-up message so it can generate a corrected version.  Tries up to
        *max_retries* times before returning the last error.
        """
        current_code = code
        answer, status = self.execute_code(current_code, scene)
        if status == "Success":
            return answer, status

        base_user_content = self._build_user_prompt(question, scene)
        if image_path:
            suffix = Path(image_path).suffix.lower().lstrip(".")
            mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
            with open(image_path, "rb") as fh:
                img_b64 = base64.b64encode(fh.read()).decode("utf-8")
            base_user_content_msg: Any = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{mime};base64,{img_b64}"},
                },
                {"type": "text", "text": base_user_content},
            ]
        else:
            base_user_content_msg = base_user_content

        for _attempt in range(max_retries):
            repair_prompt = (
                f"The following Python code raised an error:\n\n"
                f"```python\n{current_code}\n```\n\n"
                f"Error: {status}\n\n"
                "Fix the code so it runs correctly and assigns the answer to `answer`. "
                "Output ONLY valid Python – no markdown fences, no prose."
            )
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._SYSTEM_PROMPT},
                    {"role": "user", "content": base_user_content_msg},
                    {"role": "assistant", "content": current_code},
                    {"role": "user", "content": repair_prompt},
                ],
                temperature=0.2,
                max_tokens=1500,
            )
            current_code = response.choices[0].message.content.strip()
            answer, status = self.execute_code(current_code, scene)
            if status == "Success":
                return answer, status

        return answer, status


# ---------------------------------------------------------------------------
# Top-level agent
# ---------------------------------------------------------------------------

class VADARAgent:
    """
    VADAR Agent – combines VisionModels, SpatialReasoner, and CodeGenerator
    to answer free-form spatial questions about images.

    Parameters
    ----------
    api_key : str
        OpenAI API key.
    use_gpu : bool
        Whether to run HuggingFace models on GPU (default False).
    model : str
        OpenAI model for code generation (default ``"gpt-4o"``).
    detection_threshold : float
        Minimum confidence score to keep a detection (default 0.7, Feature 5).
    nms_iou_threshold : float
        IoU above which duplicate detections are suppressed (default 0.5, Feature 5).
    use_vision : bool
        Pass the image directly to GPT-4o as a vision message (default True, Feature 2).
    use_clip : bool
        Extract color / material / size via CLIP (default False, Feature 6).
    depth_inversion_guard : bool
        Automatically flip inverted depth maps (default True, Feature 7).
    code_repair_retries : int
        How many times to retry failed code execution (default 2, Feature 3).
    """

    def __init__(
        self,
        api_key: str,
        use_gpu: bool = False,
        model: str = "gpt-4o",
        detection_threshold: float = 0.7,
        nms_iou_threshold: float = 0.5,
        use_vision: bool = True,
        use_clip: bool = False,
        depth_inversion_guard: bool = True,
        code_repair_retries: int = 2,
    ) -> None:
        self.vision_models = VisionModels(
            use_gpu=use_gpu,
            detection_threshold=detection_threshold,
            nms_iou_threshold=nms_iou_threshold,
            depth_inversion_guard=depth_inversion_guard,
        )
        self.spatial_reasoner = SpatialReasoner()
        self.code_generator = CodeGenerator(api_key, model=model)
        self.use_vision = use_vision
        self.code_repair_retries = code_repair_retries
        self._last_analysis: Optional[SceneAnalysis] = None

        self._clip: Optional[CLIPAttributeExtractor] = None
        if use_clip:
            device = 0 if use_gpu else -1
            self._clip = CLIPAttributeExtractor(device=device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_image(self, image_path: str) -> SceneAnalysis:
        """
        Run the full vision pipeline on *image_path* and return a SceneAnalysis.

        Steps:
          1. Object detection (with threshold + NMS, Feature 5)
          2. Monocular depth estimation (with inversion guard, Feature 7)
          3. Build SpatialObject list using region-median depth (Feature 1)
          4. Optional CLIP attribute extraction (Feature 6)
        """
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)
        height, width = image_array.shape[:2]

        detections = self.vision_models.detect_objects(image)
        depth_map = self.vision_models.estimate_depth(image)

        # Resize depth map once (if needed) rather than per-object
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
            cx = max(0, min(cx, width - 1))
            cy = max(0, min(cy, height - 1))

            # Feature 1 – region-median depth instead of single center pixel
            px0 = max(0, box["xmin"])
            py0 = max(0, box["ymin"])
            px1 = min(width, box["xmax"])
            py1 = min(height, box["ymax"])
            region = dm_resized[py0:py1, px0:px1]
            depth_value = float(np.median(region)) if region.size > 0 else float(dm_resized[cy, cx])

            obj = SpatialObject(
                label=det["label"],
                confidence=float(det["score"]),
                bbox=(x_min, y_min, x_max, y_max),
                center=(cx, cy),
                depth_value=depth_value,
                area=(x_max - x_min) * (y_max - y_min),
                image_height=height,
                image_width=width,
            )

            # Feature 6 – CLIP attribute extraction (optional)
            if self._clip is not None:
                try:
                    attrs = self._clip.extract_attributes(image, (x_min, y_min, x_max, y_max))
                    obj.color = attrs["color"]
                    obj.material = attrs["material"]
                    obj.relative_size = attrs["relative_size"]
                except Exception:  # noqa: BLE001
                    pass  # silently skip on CLIP failure

            objects.append(obj)

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
        execute it (with self-repair), and return a results dictionary.
        """
        scene = self.analyze_image(image_path)
        # Feature 2 – pass image_path for vision context when enabled
        img_ctx = image_path if self.use_vision else None
        code = self.code_generator.generate_code(question, scene, image_path=img_ctx)
        # Feature 3 – self-repair on execution failure
        answer, status = self.code_generator.execute_code_with_repair(
            code, scene, question,
            image_path=img_ctx,
            max_retries=self.code_repair_retries,
        )

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
