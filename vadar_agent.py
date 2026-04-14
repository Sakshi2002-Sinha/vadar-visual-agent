"""
VADAR-Inspired Visual Agent for 3D Spatial Reasoning on 2D Images
Performs agentic code generation and execution for spatial understanding.

Architecture:
  VisionModels      – wraps HuggingFace pipelines for detection / depth / segmentation
  SpatialObject     – dataclass representing one detected object with spatial attributes
  SceneAnalysis     – container holding all objects + depth map for a single image
  SpatialReasoner   – pure-function helpers for spatial comparisons
  SpatialRelation   – dataclass for a single directed spatial edge (subject --rel--> object)
  SceneGraph        – pairwise spatial-relation graph with multi-hop traversal
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
    # Populated by VADARAgent.analyze_image when build_scene_graph=True.
    scene_graph: Any = field(default=None)


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

    @staticmethod
    def is_above(obj1: SpatialObject, obj2: SpatialObject) -> bool:
        """Return True if *obj1* is above *obj2* in image coordinates (smaller y)."""
        return obj1.center[1] < obj2.center[1]

    @staticmethod
    def is_below(obj1: SpatialObject, obj2: SpatialObject) -> bool:
        """Return True if *obj1* is below *obj2* in image coordinates (larger y)."""
        return obj1.center[1] > obj2.center[1]

    @staticmethod
    def overlap_ratio(obj1: SpatialObject, obj2: SpatialObject) -> float:
        """Intersection-over-union of the two bounding boxes (0 = no overlap, 1 = identical)."""
        x0 = max(obj1.bbox[0], obj2.bbox[0])
        y0 = max(obj1.bbox[1], obj2.bbox[1])
        x1 = min(obj1.bbox[2], obj2.bbox[2])
        y1 = min(obj1.bbox[3], obj2.bbox[3])
        if x1 <= x0 or y1 <= y0:
            return 0.0
        inter = (x1 - x0) * (y1 - y0)
        area1 = (obj1.bbox[2] - obj1.bbox[0]) * (obj1.bbox[3] - obj1.bbox[1])
        area2 = (obj2.bbox[2] - obj2.bbox[0]) * (obj2.bbox[3] - obj2.bbox[1])
        union = area1 + area2 - inter
        return float(inter / union) if union > 0 else 0.0

    @staticmethod
    def find_nearest_neighbor(
        obj: SpatialObject, objects: List[SpatialObject]
    ) -> Optional[SpatialObject]:
        """Return the object (other than *obj*) with the smallest pixel distance to *obj*."""
        others = [o for o in objects if o is not obj]
        if not others:
            return None
        return min(others, key=lambda o: SpatialReasoner.pixel_distance(obj, o))

    @staticmethod
    def cluster_by_depth(
        objects: List[SpatialObject], n_zones: int = 3
    ) -> Dict[str, List[SpatialObject]]:
        """Partition objects into *n_zones* depth zones.

        With the default of 3 zones the keys are 'foreground', 'midground',
        and 'background' (depth 0=closest, 1=farthest).
        """
        if n_zones == 3:
            zone_labels = ["foreground", "midground", "background"]
        else:
            zone_labels = [f"zone_{i}" for i in range(n_zones)]
        result: Dict[str, List[SpatialObject]] = {lbl: [] for lbl in zone_labels}
        if not objects:
            return result
        zone_size = 1.0 / n_zones
        for obj in objects:
            idx = min(int(obj.depth_value / zone_size), n_zones - 1)
            result[zone_labels[idx]].append(obj)
        return result

    @staticmethod
    def count_by_category(objects: List[SpatialObject]) -> Dict[str, int]:
        """Return a dict mapping each label to its instance count."""
        counts: Dict[str, int] = {}
        for obj in objects:
            counts[obj.label] = counts.get(obj.label, 0) + 1
        return counts

    @staticmethod
    def scene_statistics(objects: List[SpatialObject]) -> Dict[str, Any]:
        """Return aggregate statistics (count, depth, confidence, area) for the scene."""
        if not objects:
            return {"count": 0}
        depths = [o.depth_value for o in objects]
        confs = [o.confidence for o in objects]
        areas = [o.area for o in objects]
        return {
            "count": len(objects),
            "categories": SpatialReasoner.count_by_category(objects),
            "depth": {
                "min": float(min(depths)),
                "max": float(max(depths)),
                "mean": float(sum(depths) / len(depths)),
            },
            "confidence": {
                "min": float(min(confs)),
                "max": float(max(confs)),
                "mean": float(sum(confs) / len(confs)),
            },
            "area": {
                "min": float(min(areas)),
                "max": float(max(areas)),
                "mean": float(sum(areas) / len(areas)),
            },
        }

    @staticmethod
    def reasoning_confidence(obj1: SpatialObject, obj2: SpatialObject, relation: str) -> float:
        """Estimate confidence (0–1) that *relation* holds between *obj1* and *obj2*.

        Higher confidence when the margin between the compared values is large
        relative to the measurement range.
        """
        if relation in {"closer_than", "farther_than"}:
            margin = abs(obj1.depth_value - obj2.depth_value)
            return float(min(margin / 0.5, 1.0))
        if relation in {"left_of", "right_of"}:
            w = max(obj1.image_width, obj2.image_width, 1)
            margin = abs(obj1.center[0] - obj2.center[0]) / w
            return float(min(margin / 0.5, 1.0))
        if relation in {"above", "below"}:
            h = max(obj1.image_height, obj2.image_height, 1)
            margin = abs(obj1.center[1] - obj2.center[1]) / h
            return float(min(margin / 0.5, 1.0))
        return 1.0


# ---------------------------------------------------------------------------
# Scene graph – pairwise spatial relations with multi-hop traversal
# ---------------------------------------------------------------------------

@dataclass
class SpatialRelation:
    """A directed spatial relationship between two detected objects."""

    subject: str    # label of the subject object
    relation: str   # e.g. "left_of", "closer_than", "above", "overlaps"
    object_: str    # label of the object (target)
    confidence: float = 1.0


class SceneGraph:
    """Structured graph of pairwise spatial relations between detected objects.

    Edges are directed: ``subject --relation--> object_``.

    Supported relations (auto-computed on construction):
      * ``left_of`` / ``right_of``   – horizontal order
      * ``above`` / ``below``        – vertical order
      * ``closer_than`` / ``farther_than`` – depth order
      * ``overlaps``                 – bounding-box IoU ≥ threshold

    The graph also supports **multi-hop** queries so that compound questions
    such as "What is to the left of the object closest to the camera?" can be
    answered without LLM assistance.
    """

    # Minimum normalized difference to declare a directional relation.
    THRESHOLD_DEPTH: float = 0.05
    THRESHOLD_HORIZ: float = 0.05
    THRESHOLD_VERT: float = 0.05
    OVERLAP_THRESHOLD: float = 0.10

    def __init__(self, objects: List[SpatialObject]) -> None:
        self.objects: List[SpatialObject] = objects
        self.relations: List[SpatialRelation] = []
        self._build()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _norm_center(self, obj: SpatialObject) -> Tuple[float, float]:
        """Return the object center normalized to [0, 1]."""
        return (
            obj.center[0] / max(obj.image_width, 1),
            obj.center[1] / max(obj.image_height, 1),
        )

    def _build(self) -> None:
        """Compute all pairwise spatial relations."""
        for i, a in enumerate(self.objects):
            for j, b in enumerate(self.objects):
                if i == j:
                    continue
                na = self._norm_center(a)
                nb = self._norm_center(b)

                # Horizontal
                horiz_diff = na[0] - nb[0]
                if horiz_diff < -self.THRESHOLD_HORIZ:
                    conf = SpatialReasoner.reasoning_confidence(a, b, "left_of")
                    self.relations.append(SpatialRelation(a.label, "left_of", b.label, conf))
                elif horiz_diff > self.THRESHOLD_HORIZ:
                    conf = SpatialReasoner.reasoning_confidence(a, b, "right_of")
                    self.relations.append(SpatialRelation(a.label, "right_of", b.label, conf))

                # Vertical
                vert_diff = na[1] - nb[1]
                if vert_diff < -self.THRESHOLD_VERT:
                    conf = SpatialReasoner.reasoning_confidence(a, b, "above")
                    self.relations.append(SpatialRelation(a.label, "above", b.label, conf))
                elif vert_diff > self.THRESHOLD_VERT:
                    conf = SpatialReasoner.reasoning_confidence(a, b, "below")
                    self.relations.append(SpatialRelation(a.label, "below", b.label, conf))

                # Depth
                depth_diff = a.depth_value - b.depth_value
                if depth_diff < -self.THRESHOLD_DEPTH:
                    conf = SpatialReasoner.reasoning_confidence(a, b, "closer_than")
                    self.relations.append(SpatialRelation(a.label, "closer_than", b.label, conf))
                elif depth_diff > self.THRESHOLD_DEPTH:
                    conf = SpatialReasoner.reasoning_confidence(a, b, "farther_than")
                    self.relations.append(SpatialRelation(a.label, "farther_than", b.label, conf))

                # Overlap / occlusion
                iou = SpatialReasoner.overlap_ratio(a, b)
                if iou >= self.OVERLAP_THRESHOLD:
                    self.relations.append(SpatialRelation(a.label, "overlaps", b.label, iou))

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get_relations_for(self, label: str) -> List[SpatialRelation]:
        """Return all relations whose subject matches *label* (case-insensitive)."""
        label_l = label.lower()
        return [r for r in self.relations if r.subject.lower() == label_l]

    def find_objects_by_relation(self, anchor_label: str, relation: str) -> List[str]:
        """Return labels of objects that *anchor_label* has *relation* with.

        Answers "what does *anchor_label* --relation--> ?"
        """
        anchor_l = anchor_label.lower()
        return [
            r.object_
            for r in self.relations
            if r.subject.lower() == anchor_l and r.relation == relation
        ]

    def find_subjects_by_relation(self, target_label: str, relation: str) -> List[str]:
        """Return labels of objects that have *relation* pointing to *target_label*.

        Answers "what is *relation* to *target_label*?" (i.e., ? --relation--> target).
        Example: ``find_subjects_by_relation("chair", "left_of")`` returns all objects
        that are to the left of the chair.
        """
        target_l = target_label.lower()
        return [
            r.subject
            for r in self.relations
            if r.object_.lower() == target_l and r.relation == relation
        ]

    def multi_hop(self, anchor_label: str, relations: List[str]) -> List[str]:
        """Chain *relations* from *anchor_label* through the graph.

        Example::

            graph.multi_hop("chair", ["closer_than", "left_of"])

        returns all objects Y such that chair --closer_than--> X --left_of--> Y.
        """
        current: List[str] = [anchor_label]
        for rel in relations:
            nxt: List[str] = []
            for lbl in current:
                nxt.extend(self.find_objects_by_relation(lbl, rel))
            # Deduplicate while preserving insertion order.
            seen: Dict[str, None] = {}
            for lbl in nxt:
                seen[lbl] = None
            current = list(seen)
        return current

    # ------------------------------------------------------------------
    # Serialization / display
    # ------------------------------------------------------------------

    def summary(self, max_edges: int = 20) -> str:
        """Return a compact text summary of the most salient edges."""
        if not self.relations:
            return "No spatial relations detected."
        lines = [
            f"  {r.subject} --{r.relation}--> {r.object_}  (conf={r.confidence:.2f})"
            for r in self.relations[:max_edges]
        ]
        if len(self.relations) > max_edges:
            lines.append(f"  … ({len(self.relations) - max_edges} more relations)")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "nodes": [o.label for o in self.objects],
            "edges": [
                {
                    "subject": r.subject,
                    "relation": r.relation,
                    "object": r.object_,
                    "confidence": round(r.confidence, 4),
                }
                for r in self.relations
            ],
        }


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
        "A `SceneGraph` instance named `scene_graph` is available with methods: "
        "  get_relations_for(label) → list of SpatialRelation, "
        "  find_objects_by_relation(anchor_label, relation) → list[str]  (anchor --rel--> ?), "
        "  find_subjects_by_relation(target_label, relation) → list[str]  (? --rel--> target), "
        "  multi_hop(anchor_label, [rel1, rel2, ...]) → list[str]. "
        "Relations include: left_of, right_of, above, below, closer_than, farther_than, overlaps. "
        "You may also call SpatialReasoner.cluster_by_depth(objects) for depth zones, "
        "SpatialReasoner.count_by_category(objects) for category counts, and "
        "SpatialReasoner.scene_statistics(objects) for aggregate metrics. "
        "For multi-hop questions chain scene_graph.multi_hop(). "
        "To answer 'what is to the left of X?' use scene_graph.find_subjects_by_relation(X, 'left_of'). "
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
                if re.search(rf"\b{re.escape(label)}\b", q):
                    return label
            return None

        obj1 = find_label_from_question(labels, question_l)
        q_without_obj1 = question_l.replace(obj1, "", 1) if obj1 else question_l
        obj2 = find_label_from_question(labels, q_without_obj1)

        # Multi-hop checks must come before simple "closest" / "nearest" checks
        # so that "to the left of the closest object" is handled correctly.
        if "to the left of" in question_l and ("closest" in question_l or "nearest" in question_l):
            return (
                "anchor = min(objects, key=lambda o: o.depth_value)\n"
                "answer = scene_graph.find_subjects_by_relation(anchor.label, 'left_of') "
                "if scene_graph else "
                "[o.label for o in objects if SpatialReasoner.is_left_of(o, anchor) and o is not anchor]"
            )

        if "to the right of" in question_l and ("closest" in question_l or "nearest" in question_l):
            return (
                "anchor = min(objects, key=lambda o: o.depth_value)\n"
                "answer = scene_graph.find_subjects_by_relation(anchor.label, 'right_of') "
                "if scene_graph else "
                "[o.label for o in objects if SpatialReasoner.is_right_of(o, anchor) and o is not anchor]"
            )

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

        if obj1 and obj2 and "above" in question_l:
            return (
                f"obj1 = SpatialReasoner.get_object_by_label(objects, {obj1!r})\n"
                f"obj2 = SpatialReasoner.get_object_by_label(objects, {obj2!r})\n"
                "answer = SpatialReasoner.is_above(obj1, obj2) if (obj1 and obj2) else None"
            )

        if obj1 and obj2 and "below" in question_l:
            return (
                f"obj1 = SpatialReasoner.get_object_by_label(objects, {obj1!r})\n"
                f"obj2 = SpatialReasoner.get_object_by_label(objects, {obj2!r})\n"
                "answer = SpatialReasoner.is_below(obj1, obj2) if (obj1 and obj2) else None"
            )

        if obj1 and obj2 and ("overlap" in question_l or "occlu" in question_l):
            return (
                f"obj1 = SpatialReasoner.get_object_by_label(objects, {obj1!r})\n"
                f"obj2 = SpatialReasoner.get_object_by_label(objects, {obj2!r})\n"
                "iou = SpatialReasoner.overlap_ratio(obj1, obj2) if (obj1 and obj2) else 0.0\n"
                "answer = iou"
            )

        if "how many" in question_l or "count" in question_l:
            return (
                "counts = SpatialReasoner.count_by_category(objects)\n"
                "answer = counts"
            )

        if "depth zone" in question_l or "foreground" in question_l or "background" in question_l:
            return (
                "zones = SpatialReasoner.cluster_by_depth(objects)\n"
                "answer = {z: [o.label for o in objs] for z, objs in zones.items()}"
            )

        if "statistic" in question_l or "summary" in question_l or "overview" in question_l:
            return (
                "answer = SpatialReasoner.scene_statistics(objects)"
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
        graph_section = ""
        if scene.scene_graph is not None:
            graph_section = (
                f"\nScene graph (spatial edges):\n{scene.scene_graph.summary(max_edges=30)}\n"
            )
        stats = SpatialReasoner.scene_statistics(scene.objects)
        zones = SpatialReasoner.cluster_by_depth(scene.objects)
        zone_desc = "  " + "  ".join(
            f"{z}: {[o.label for o in objs]}" for z, objs in zones.items()
        )
        return (
            f"Scene objects:\n{objects_desc}\n"
            f"{graph_section}"
            f"\nDepth zones:\n{zone_desc}\n"
            f"\nScene statistics: count={stats.get('count', 0)}, "
            f"categories={stats.get('categories', {})}\n"
            f"\nQuestion: {question}\n\n"
            "Write Python code that assigns the answer to a variable named `answer`. "
            "You may use numpy as `np`. "
            "The list `objects` contains SpatialObject instances with attributes: "
            "label, confidence, bbox, center, depth_value, area, image_height, image_width. "
            "The helper class `SpatialReasoner` is available with methods: "
            "is_closer, is_farther, is_left_of, is_right_of, is_above, is_below, "
            "overlap_ratio, find_nearest_neighbor, cluster_by_depth, count_by_category, "
            "scene_statistics, reasoning_confidence. "
            "A `scene_graph` (SceneGraph) is available with multi_hop(), "
            "get_relations_for(), find_objects_by_relation(), and find_subjects_by_relation()."
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
            # fallback_reason contains only provider names and exception class names, not credentials.
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
            "SceneGraph": SceneGraph,
            "scene_analysis": scene,
            "objects": scene.objects,
            "scene_graph": scene.scene_graph,
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
        build_scene_graph: bool = True,
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
        self.build_scene_graph = build_scene_graph
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

        if self.build_scene_graph:
            t_graph = time.perf_counter()
            analysis.scene_graph = SceneGraph(objects)
            analysis.processing_times["scene_graph_ms"] = (
                time.perf_counter() - t_graph
            ) * 1000.0
            analysis.processing_times["total_ms"] = (time.perf_counter() - t0) * 1000.0

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

        result: Dict[str, Any] = {
            "question": question,
            "answer": answer,
            "status": status,
            "code": code,
            "objects_detected": [asdict(o) for o in scene.objects],
            "timestamp": scene.timestamp,
        }
        if scene.scene_graph is not None:
            result["scene_graph"] = scene.scene_graph.to_dict()
        return result

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