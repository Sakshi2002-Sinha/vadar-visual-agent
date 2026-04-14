"""
VADAR-Inspired Visual Agent for 3D Spatial Reasoning on 2D Images
Performs agentic code generation and execution for spatial understanding.

Architecture:
  VisionModels        – wraps HuggingFace pipelines for detection / depth / segmentation
  SpatialObject       – dataclass representing one detected object with spatial attributes
  SpatialRelationship – dataclass encoding a typed directed spatial edge between two objects
  SceneAnalysis       – container holding all objects + depth map for a single image
  SceneGraph          – structured graph of nodes (objects) and edges (relationships)
  SpatialReasoner     – pure-function helpers for spatial comparisons and relationship detection
  SceneGraphBuilder   – builds a SceneGraph from a SceneAnalysis
  CodeGenerator       – calls the OpenAI API to generate and exec Python for a question
                        with optional multi-turn conversation history
  VADARAgent          – top-level orchestrator
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
from transformers import pipeline as hf_pipeline


# ---------------------------------------------------------------------------
# Vision models
# ---------------------------------------------------------------------------

class VisionModels:
    """Manages pretrained vision models for object detection, segmentation, and depth estimation."""

    def __init__(self, use_gpu: bool = False):
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
    # Normalized bounding box: (x_min, y_min, x_max, y_max) in [0, 1]
    bbox: Tuple[float, float, float, float]
    # Pixel coordinates of the bounding-box center
    center: Tuple[int, int]
    # Normalized depth at the object center (0 = close, 1 = far)
    depth_value: float
    # Normalized area relative to the full image
    area: float
    image_height: int
    image_width: int
    # Standard deviation of depth values within the bounding box (uncertainty)
    depth_uncertainty: float = 0.0

    def distance_from_camera(self) -> float:
        """Return normalized distance from the camera (0 = close, 1 = far)."""
        return self.depth_value


@dataclass
class SpatialRelationship:
    """A typed, directed spatial edge between two objects in a scene."""

    subject_label: str
    relation: str          # e.g. "left_of", "above", "closer_than", "overlaps"
    object_label: str
    confidence: float = 1.0   # geometric certainty in [0, 1]

    def __str__(self) -> str:
        return f"{self.subject_label} --[{self.relation}]--> {self.object_label}"


@dataclass
class SceneGraph:
    """
    Structured scene graph for an image.

    nodes  – detected objects (SpatialObject)
    edges  – directed spatial relationships (SpatialRelationship)
    """

    nodes: List[SpatialObject]
    edges: List[SpatialRelationship]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_node(self, label: str) -> Optional[SpatialObject]:
        """Return the first node whose label contains *label* (case-insensitive)."""
        for node in self.nodes:
            if label.lower() in node.label.lower():
                return node
        return None

    def get_edges_for(self, label: str) -> List[SpatialRelationship]:
        """Return all edges where *label* is either subject or object."""
        label_lower = label.lower()
        return [
            e for e in self.edges
            if label_lower in e.subject_label.lower()
            or label_lower in e.object_label.lower()
        ]

    def get_relationships_between(
        self, label1: str, label2: str
    ) -> List[SpatialRelationship]:
        """Return edges that connect *label1* and *label2* in either direction."""
        l1, l2 = label1.lower(), label2.lower()
        return [
            e for e in self.edges
            if (l1 in e.subject_label.lower() and l2 in e.object_label.lower())
            or (l2 in e.subject_label.lower() and l1 in e.object_label.lower())
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "nodes": [asdict(n) for n in self.nodes],
            "edges": [asdict(e) for e in self.edges],
        }


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

    # ------------------------------------------------------------------
    # Richer spatial relationship helpers (novel additions)
    # ------------------------------------------------------------------

    @staticmethod
    def is_above(obj1: SpatialObject, obj2: SpatialObject) -> bool:
        """Return True if *obj1* center is above *obj2* center (lower y value)."""
        return obj1.center[1] < obj2.center[1]

    @staticmethod
    def is_below(obj1: SpatialObject, obj2: SpatialObject) -> bool:
        """Return True if *obj1* center is below *obj2* center (higher y value)."""
        return obj1.center[1] > obj2.center[1]

    @staticmethod
    def is_left_of(obj1: SpatialObject, obj2: SpatialObject) -> bool:
        """Return True if *obj1* center is to the left of *obj2* center."""
        return obj1.center[0] < obj2.center[0]

    @staticmethod
    def is_right_of(obj1: SpatialObject, obj2: SpatialObject) -> bool:
        """Return True if *obj1* center is to the right of *obj2* center."""
        return obj1.center[0] > obj2.center[0]

    @staticmethod
    def overlap_iou(obj1: SpatialObject, obj2: SpatialObject) -> float:
        """
        Compute Intersection over Union (IoU) of the two bounding boxes.
        Bounding boxes are in normalized [0, 1] coordinates.
        """
        x0 = max(obj1.bbox[0], obj2.bbox[0])
        y0 = max(obj1.bbox[1], obj2.bbox[1])
        x1 = min(obj1.bbox[2], obj2.bbox[2])
        y1 = min(obj1.bbox[3], obj2.bbox[3])
        inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        if inter == 0.0:
            return 0.0
        union = obj1.area + obj2.area - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def is_overlapping(obj1: SpatialObject, obj2: SpatialObject, iou_threshold: float = 0.05) -> bool:
        """Return True if the bounding boxes of the two objects overlap above *iou_threshold*."""
        return SpatialReasoner.overlap_iou(obj1, obj2) >= iou_threshold

    @staticmethod
    def relative_size(obj1: SpatialObject, obj2: SpatialObject) -> float:
        """Return the area ratio obj1.area / obj2.area (>1 means obj1 is larger)."""
        if obj2.area == 0:
            return float("inf")
        return obj1.area / obj2.area

    @staticmethod
    def build_relationship_matrix(
        objects: List[SpatialObject],
        depth_threshold: float = 0.05,
        iou_threshold: float = 0.05,
    ) -> List[SpatialRelationship]:
        """
        Compute all pairwise spatial relationships between objects.

        For each ordered pair (A, B) the following directed relations are
        emitted when they hold:

          left_of      – A is to the left of B
          right_of     – A is to the right of B
          above        – A is above B
          below        – A is below B
          closer_than  – A is closer to the camera than B (by > depth_threshold)
          farther_than – A is farther from the camera than B (by > depth_threshold)
          larger_than  – A has a larger bounding-box area than B
          overlaps     – bounding boxes overlap above iou_threshold

        Returns a flat list of SpatialRelationship instances.
        """
        relationships: List[SpatialRelationship] = []
        r = SpatialReasoner

        for i, a in enumerate(objects):
            for j, b in enumerate(objects):
                if i == j:
                    continue

                if r.is_left_of(a, b):
                    relationships.append(
                        SpatialRelationship(a.label, "left_of", b.label)
                    )
                if r.is_right_of(a, b):
                    relationships.append(
                        SpatialRelationship(a.label, "right_of", b.label)
                    )
                if r.is_above(a, b):
                    relationships.append(
                        SpatialRelationship(a.label, "above", b.label)
                    )
                if r.is_below(a, b):
                    relationships.append(
                        SpatialRelationship(a.label, "below", b.label)
                    )

                depth_diff = a.depth_value - b.depth_value
                if depth_diff < -depth_threshold:
                    # A has a smaller depth value → A is closer
                    relationships.append(
                        SpatialRelationship(a.label, "closer_than", b.label)
                    )
                elif depth_diff > depth_threshold:
                    relationships.append(
                        SpatialRelationship(a.label, "farther_than", b.label)
                    )

                if a.area > b.area:
                    relationships.append(
                        SpatialRelationship(a.label, "larger_than", b.label)
                    )

                if r.is_overlapping(a, b, iou_threshold):
                    relationships.append(
                        SpatialRelationship(
                            a.label,
                            "overlaps",
                            b.label,
                            confidence=r.overlap_iou(a, b),
                        )
                    )

        return relationships



# ---------------------------------------------------------------------------
# Scene graph builder
# ---------------------------------------------------------------------------

class SceneGraphBuilder:
    """
    Constructs a SceneGraph from a SceneAnalysis by enumerating all pairwise
    spatial relationships between detected objects.

    Novel contribution: the graph provides a *symbolic* layer on top of raw
    neural detections, enabling structured queries and downstream reasoning
    (e.g. graph-based question answering, counterfactual analysis).
    """

    def __init__(
        self,
        depth_threshold: float = 0.05,
        iou_threshold: float = 0.05,
    ) -> None:
        self.depth_threshold = depth_threshold
        self.iou_threshold = iou_threshold

    def build(self, scene: SceneAnalysis) -> SceneGraph:
        """Build and return a SceneGraph for *scene*."""
        edges = SpatialReasoner.build_relationship_matrix(
            scene.objects,
            depth_threshold=self.depth_threshold,
            iou_threshold=self.iou_threshold,
        )
        return SceneGraph(nodes=list(scene.objects), edges=edges)


# ---------------------------------------------------------------------------
# Code generation + execution
# ---------------------------------------------------------------------------

class CodeGenerator:
    """Generates Python code via the OpenAI API and executes it with scene context.

    Supports optional multi-turn dialogue: if *use_conversation_history* is
    True, prior question/answer pairs for the same scene are appended as
    assistant messages so the model can refer to earlier context.
    """

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
        model: str = "gpt-4o",
        use_conversation_history: bool = True,
    ):
        openai.api_key = api_key
        self.model = model
        self.use_conversation_history = use_conversation_history
        self.history: List[Dict[str, Any]] = []
        # Per-scene conversation messages for multi-turn dialogue
        self._dialogue_messages: List[Dict[str, str]] = []
        self._dialogue_scene_timestamp: Optional[str] = None

    def reset_dialogue(self) -> None:
        """Clear the multi-turn conversation history (call when switching images)."""
        self._dialogue_messages = []
        self._dialogue_scene_timestamp = None

    def _build_user_prompt(self, question: str, scene: SceneAnalysis) -> str:
        objects_desc = "\n".join(
            f"  [{i}] label={o.label!r} confidence={o.confidence:.3f} "
            f"depth={o.depth_value:.3f} depth_uncertainty={o.depth_uncertainty:.3f} "
            f"center={o.center} area={o.area:.4f}"
            for i, o in enumerate(scene.objects)
        )
        return (
            f"Scene objects:\n{objects_desc}\n\n"
            f"Question: {question}\n\n"
            "Write Python code that assigns the answer to a variable named `answer`. "
            "You may use numpy as `np`. "
            "The list `objects` contains SpatialObject instances with attributes: "
            "label, confidence, bbox, center, depth_value, depth_uncertainty, "
            "area, image_height, image_width. "
            "The helper class `SpatialReasoner` is available with methods: "
            "get_object_by_label, is_farther, relative_depth_distance, pixel_distance, "
            "vertical_position, horizontal_position, is_above, is_below, is_left_of, "
            "is_right_of, overlap_iou, is_overlapping, relative_size, build_relationship_matrix. "
            "The `scene_graph` variable is a SceneGraph with attributes `nodes` and `edges`; "
            "each edge is a SpatialRelationship with subject_label, relation, object_label, confidence."
        )

    def generate_code(self, question: str, scene: SceneAnalysis) -> str:
        """Call the OpenAI Chat API and return generated Python code.

        When *use_conversation_history* is True, prior turns for the same scene
        (identified by timestamp) are included as context so the model can
        answer follow-up questions coherently.
        """
        # Reset dialogue history when the scene changes
        if self._dialogue_scene_timestamp != scene.timestamp:
            self.reset_dialogue()
            self._dialogue_scene_timestamp = scene.timestamp

        user_prompt = self._build_user_prompt(question, scene)

        # Build messages list with optional conversation history
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self._SYSTEM_PROMPT},
        ]
        if self.use_conversation_history:
            messages.extend(self._dialogue_messages)
        messages.append({"role": "user", "content": user_prompt})

        response = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=1500,
        )
        code = response.choices[0].message.content.strip()

        # Update dialogue history for next turn
        if self.use_conversation_history:
            self._dialogue_messages.append({"role": "user", "content": user_prompt})
            self._dialogue_messages.append({"role": "assistant", "content": code})

        self.history.append({
            "question": question,
            "code": code,
            "timestamp": datetime.now().isoformat(),
        })
        return code

    def execute_code(
        self, code: str, scene: SceneAnalysis, scene_graph: Optional["SceneGraph"] = None
    ) -> Tuple[Any, str]:
        """Execute *code* with scene context injected. Returns (answer, status).

        The execution namespace exposes:
          - ``objects``       – list of SpatialObject
          - ``scene_analysis``– SceneAnalysis
          - ``scene_graph``   – SceneGraph (or None when not provided)
          - ``SpatialObject`` / ``SpatialReasoner`` / ``SceneGraph`` / ``SpatialRelationship``
          - ``np``            – numpy
        """
        exec_globals: Dict[str, Any] = {
            "np": np,
            "SpatialObject": SpatialObject,
            "SpatialReasoner": SpatialReasoner,
            "SpatialRelationship": SpatialRelationship,
            "SceneGraph": SceneGraph,
            "scene_analysis": scene,
            "scene_graph": scene_graph,
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
    VADAR Agent – combines VisionModels, SpatialReasoner, CodeGenerator,
    and SceneGraphBuilder to answer free-form spatial questions about images.

    Novel features vs. baseline:
      • depth_uncertainty  – per-object depth variance within bounding box
      • SceneGraph         – symbolic spatial relationship graph
      • Multi-turn dialogue– conversation history kept per scene for follow-ups
      • describe_scene()   – natural language scene description from graph
    """

    def __init__(
        self,
        api_key: str,
        use_gpu: bool = False,
        model: str = "gpt-4o",
        use_conversation_history: bool = True,
    ):
        self.vision_models = VisionModels(use_gpu=use_gpu)
        self.spatial_reasoner = SpatialReasoner()
        self.code_generator = CodeGenerator(
            api_key, model=model, use_conversation_history=use_conversation_history
        )
        self.scene_graph_builder = SceneGraphBuilder()
        self._last_analysis: Optional[SceneAnalysis] = None
        self._last_scene_graph: Optional[SceneGraph] = None

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
             (includes per-object depth uncertainty from bounding-box depth variance)
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

            # Depth uncertainty: std-dev of depth values inside the bounding box
            px0 = max(0, box["xmin"])
            py0 = max(0, box["ymin"])
            px1 = min(width - 1, box["xmax"])
            py1 = min(height - 1, box["ymax"])
            bbox_depth = dm_resized[py0:py1 + 1, px0:px1 + 1]
            depth_uncertainty = float(bbox_depth.std()) if bbox_depth.size > 0 else 0.0

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
                    depth_uncertainty=depth_uncertainty,
                )
            )

        analysis = SceneAnalysis(
            objects=objects,
            depth_map=depth_map,
            image_shape=image_array.shape,
        )
        self._last_analysis = analysis
        # Reset multi-turn dialogue when the scene changes
        self.code_generator.reset_dialogue()
        return analysis

    def build_scene_graph(self, image_path: Optional[str] = None) -> SceneGraph:
        """
        Build and return a SceneGraph for the image.

        If *image_path* is provided the image is (re-)analysed first.
        Otherwise the most recent SceneAnalysis is used.
        """
        if image_path is not None:
            self.analyze_image(image_path)
        if self._last_analysis is None:
            raise RuntimeError("No scene analysis available. Call analyze_image() first.")
        graph = self.scene_graph_builder.build(self._last_analysis)
        self._last_scene_graph = graph
        return graph

    def describe_scene(self, image_path: Optional[str] = None) -> str:
        """
        Return a concise natural-language description of the scene based on
        the scene graph (no additional OpenAI call required).

        If *image_path* is provided the image is (re-)analysed first.
        """
        graph = self.build_scene_graph(image_path)
        if not graph.nodes:
            return "No objects detected in the scene."

        lines: List[str] = []
        labels = [n.label for n in graph.nodes]
        lines.append(f"The scene contains {len(labels)} object(s): {', '.join(labels)}.")

        # Depth ordering
        sorted_by_depth = sorted(graph.nodes, key=lambda o: o.depth_value)
        lines.append(
            f"From closest to farthest: "
            + ", ".join(
                f"{o.label} (depth={o.depth_value:.2f}±{o.depth_uncertainty:.2f})"
                for o in sorted_by_depth
            )
            + "."
        )

        # Notable relationships (avoid listing every pair for large scenes)
        depth_rels = [e for e in graph.edges if e.relation in ("closer_than", "farther_than")]
        if depth_rels:
            sampled = depth_rels[:5]
            lines.append(
                "Depth relationships: "
                + "; ".join(str(e) for e in sampled)
                + ("…" if len(depth_rels) > 5 else "")
                + "."
            )

        overlap_rels = [e for e in graph.edges if e.relation == "overlaps"]
        if overlap_rels:
            lines.append(
                f"{len(overlap_rels)} overlapping bounding-box pair(s) detected."
            )

        return " ".join(lines)

    def answer_question(self, question: str, image_path: str) -> Dict[str, Any]:
        """
        End-to-end method: analyse *image_path*, build scene graph, generate
        code for *question*, execute it, and return a results dictionary.
        """
        scene = self.analyze_image(image_path)
        graph = self.scene_graph_builder.build(scene)
        self._last_scene_graph = graph
        code = self.code_generator.generate_code(question, scene)
        answer, status = self.code_generator.execute_code(code, scene, scene_graph=graph)

        return {
            "question": question,
            "answer": answer,
            "status": status,
            "code": code,
            "objects_detected": [asdict(o) for o in scene.objects],
            "scene_graph": graph.to_dict(),
            "timestamp": scene.timestamp,
        }

    def answer_followup(self, question: str) -> Dict[str, Any]:
        """
        Answer a follow-up question about the *most recently analysed* image,
        using multi-turn conversation history so the model has prior context.

        Raises RuntimeError if no image has been analysed yet.
        """
        if self._last_analysis is None:
            raise RuntimeError("No scene analysis available. Call answer_question() first.")
        scene = self._last_analysis
        graph = self._last_scene_graph or self.scene_graph_builder.build(scene)
        code = self.code_generator.generate_code(question, scene)
        answer, status = self.code_generator.execute_code(code, scene, scene_graph=graph)

        return {
            "question": question,
            "answer": answer,
            "status": status,
            "code": code,
            "timestamp": datetime.now().isoformat(),
        }

    @property
    def last_analysis(self) -> Optional[SceneAnalysis]:
        """Return the most recent SceneAnalysis, or None if no image has been analysed."""
        return self._last_analysis

    @property
    def last_scene_graph(self) -> Optional[SceneGraph]:
        """Return the most recent SceneGraph, or None if no graph has been built."""
        return self._last_scene_graph


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python vadar_agent.py <image_path> <question>")
        print("       python vadar_agent.py <image_path> --describe")
        print("       python vadar_agent.py <image_path> --scene-graph")
        sys.exit(1)

    _api_key = os.environ.get("OPENAI_API_KEY", "")
    if not _api_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)

    _agent = VADARAgent(_api_key)

    if sys.argv[2] == "--describe":
        print(_agent.describe_scene(sys.argv[1]))
    elif sys.argv[2] == "--scene-graph":
        _graph = _agent.build_scene_graph(sys.argv[1])
        print(json.dumps(_graph.to_dict(), indent=2, default=str))
    else:
        _result = _agent.answer_question(sys.argv[2], sys.argv[1])
        print(json.dumps(_result, indent=2, default=str))