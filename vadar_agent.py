"""
VADAR-Inspired Visual Agent for 3D Spatial Reasoning on 2D Images
Performs agentic code generation and execution for spatial understanding.
"""

import os
import json
import base64
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

import numpy as np
from PIL import Image


def _get_openai_client():
    """Return an OpenAI client, supporting both openai>=1.0 and legacy versions."""
    try:
        import openai  # noqa: F401
        # openai >= 1.0 exposes OpenAI class
        from openai import OpenAI
        return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    except (ImportError, AttributeError):
        return None


class VisionModels:
    """Manages pretrained vision models for object detection and depth estimation."""

    def __init__(self):
        self._object_detector = None
        self._depth_estimator = None
        self._segmentation = None

    @staticmethod
    def _device() -> int:
        """Return 0 (GPU) when CUDA_AVAILABLE=1, otherwise -1 (CPU)."""
        return 0 if os.environ.get("CUDA_AVAILABLE") == "1" else -1

    def _load_object_detector(self):
        if self._object_detector is None:
            from transformers import pipeline
            self._object_detector = pipeline(
                "object-detection",
                model="facebook/detr-resnet-50",
                device=self._device(),
            )
        return self._object_detector

    def _load_depth_estimator(self):
        if self._depth_estimator is None:
            from transformers import pipeline
            self._depth_estimator = pipeline(
                "depth-estimation",
                model="Intel/dpt-large",
                device=self._device(),
            )
        return self._depth_estimator

    def _load_segmentation(self):
        if self._segmentation is None:
            from transformers import pipeline
            self._segmentation = pipeline(
                "image-segmentation",
                model="facebook/detr-resnet-50-panoptic",
                device=self._device(),
            )
        return self._segmentation

    def detect_objects(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect objects in the image using DETR."""
        detector = self._load_object_detector()
        return detector(image)

    def estimate_depth(self, image: Image.Image) -> np.ndarray:
        """Estimate a normalised (0–1) depth map from an image."""
        estimator = self._load_depth_estimator()
        result = estimator(image)
        depth_map = np.array(result["depth"], dtype=float)
        dmin, dmax = depth_map.min(), depth_map.max()
        if dmax > dmin:
            depth_map = (depth_map - dmin) / (dmax - dmin)
        return depth_map

    def segment_objects(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Perform panoptic segmentation."""
        seg = self._load_segmentation()
        return seg(image)


@dataclass
class SpatialObject:
    """Represents a detected object with spatial properties."""

    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x_min, y_min, x_max, y_max (normalised)
    center: Tuple[float, float]              # pixel coordinates
    depth_value: float                       # average depth at object location (0=close, 1=far)
    area: float                              # normalised bounding-box area
    image_height: int
    image_width: int

    def distance_from_camera(self) -> float:
        """Return normalised distance from camera (0 = close, 1 = far)."""
        return self.depth_value


@dataclass
class SceneAnalysis:
    """Container for scene understanding results."""

    objects: List[SpatialObject]
    depth_map: np.ndarray
    image_shape: Tuple[int, ...]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class SpatialReasoner:
    """Performs 3D spatial reasoning on detected objects."""

    @staticmethod
    def get_object_by_label(objects: List[SpatialObject], label: str) -> Optional[SpatialObject]:
        """Find the first object whose label contains *label* (case-insensitive)."""
        for obj in objects:
            if label.lower() in obj.label.lower():
                return obj
        return None

    @staticmethod
    def is_farther(obj1: SpatialObject, obj2: SpatialObject) -> bool:
        """Return True if *obj1* is farther from the camera than *obj2*."""
        return obj1.distance_from_camera() > obj2.distance_from_camera()

    @staticmethod
    def relative_distance(obj1: SpatialObject, obj2: SpatialObject) -> float:
        """Return the absolute depth difference between two objects."""
        return abs(obj1.distance_from_camera() - obj2.distance_from_camera())

    @staticmethod
    def horizontal_distance(obj1: SpatialObject, obj2: SpatialObject) -> float:
        """Return the 2-D pixel distance between the centres of two objects."""
        x1, y1 = obj1.center
        x2, y2 = obj2.center
        return float(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))

    @staticmethod
    def vertical_position(obj: SpatialObject) -> str:
        """Classify the vertical position of an object as 'upper', 'middle', or 'lower'."""
        _, cy = obj.center
        if cy < obj.image_height / 3:
            return "upper"
        if cy > 2 * obj.image_height / 3:
            return "lower"
        return "middle"

    @staticmethod
    def horizontal_position(obj: SpatialObject) -> str:
        """Classify the horizontal position of an object as 'left', 'center', or 'right'."""
        cx, _ = obj.center
        if cx < obj.image_width / 3:
            return "left"
        if cx > 2 * obj.image_width / 3:
            return "right"
        return "center"


class CodeGenerator:
    """Generates and executes Python code for spatial reasoning tasks using OpenAI."""

    def __init__(self, api_key: str):
        self._api_key = api_key
        self.generated_code_history: List[Dict[str, Any]] = []

    def generate_code(self, question: str, scene_analysis: SceneAnalysis) -> str:
        """Ask GPT-4 to produce Python code that answers the spatial reasoning question."""
        objects_desc = "\n".join(
            f"  - {obj.label} (confidence: {obj.confidence:.2f}, "
            f"depth: {obj.depth_value:.2f}, center: {obj.center}, area: {obj.area:.4f})"
            for obj in scene_analysis.objects
        )

        prompt = (
            "You are an expert in spatial reasoning on 2D images with depth information.\n"
            "Given a visual scene with detected objects and their depth information, "
            "generate Python code to answer a spatial reasoning question.\n\n"
            f"Scene Objects:\n{objects_desc}\n\n"
            f"Question: {question}\n\n"
            "Generate Python code that:\n"
            "1. Imports necessary modules (numpy, etc.)\n"
            "2. Defines the objects as a list of dictionaries with their properties\n"
            "3. Implements helper functions to answer the question\n"
            "4. Outputs a clear answer as the last statement: answer = <your answer>\n\n"
            "The code must be self-contained, executable, and include brief comments.\n"
            "Generate ONLY the Python code, no markdown formatting."
        )

        try:
            import openai
            from openai import OpenAI
            client = OpenAI(api_key=self._api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500,
            )
            code = response.choices[0].message.content or ""
        except Exception as exc:  # pragma: no cover
            code = f'answer = "Code generation failed: {exc}"'

        self.generated_code_history.append(
            {"question": question, "code": code, "timestamp": datetime.now().isoformat()}
        )
        return code

    def execute_code(
        self, code: str, scene_analysis: SceneAnalysis
    ) -> Tuple[Any, str]:
        """Execute generated code in a sandboxed namespace that includes the scene context.

        Security note: This executes GPT-4-generated Python code.  The agent is designed
        for trusted, local use.  Do not expose this functionality to untrusted input or
        over a network without additional sandboxing (e.g. RestrictedPython).
        """
        exec_globals: Dict[str, Any] = {
            "np": np,
            "SpatialObject": SpatialObject,
            "SpatialReasoner": SpatialReasoner,
            "scene_analysis": scene_analysis,
            "objects": scene_analysis.objects,
        }
        try:
            exec(code, exec_globals)  # noqa: S102
            answer = exec_globals.get("answer", "No answer produced")
            return answer, "Success"
        except Exception as exc:
            return None, f"Execution Error: {exc}"


class VADARAgent:
    """Main VADAR Agent combining vision models, spatial reasoning, and code generation."""

    def __init__(self, api_key: str):
        self.vision_models = VisionModels()
        self.spatial_reasoner = SpatialReasoner()
        self.code_generator = CodeGenerator(api_key)
        self.last_analysis: Optional[SceneAnalysis] = None

    def analyze_image(self, image_path: str) -> SceneAnalysis:
        """Analyse an image and return a :class:`SceneAnalysis` with detected objects."""
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)
        height, width = image_array.shape[:2]

        detections = self.vision_models.detect_objects(image)
        depth_map = self.vision_models.estimate_depth(image)

        # Resize depth map to image dimensions if necessary
        if depth_map.shape != (height, width):
            from PIL import Image as _Image
            depth_img = _Image.fromarray((depth_map * 255).astype(np.uint8)).resize(
                (width, height), _Image.BILINEAR
            )
            depth_map = np.array(depth_img, dtype=float) / 255.0

        objects: List[SpatialObject] = []
        for detection in detections:
            box = detection["box"]
            x_min = box["xmin"] / width
            y_min = box["ymin"] / height
            x_max = box["xmax"] / width
            y_max = box["ymax"] / height

            center_x = int((box["xmin"] + box["xmax"]) / 2)
            center_y = int((box["ymin"] + box["ymax"]) / 2)
            # Clamp to valid indices
            center_x = max(0, min(center_x, width - 1))
            center_y = max(0, min(center_y, height - 1))

            depth_value = float(depth_map[center_y, center_x])
            area = (x_max - x_min) * (y_max - y_min)

            objects.append(
                SpatialObject(
                    label=detection["label"],
                    confidence=float(detection["score"]),
                    bbox=(x_min, y_min, x_max, y_max),
                    center=(center_x, center_y),
                    depth_value=depth_value,
                    area=area,
                    image_height=height,
                    image_width=width,
                )
            )

        analysis = SceneAnalysis(
            objects=objects,
            depth_map=depth_map,
            image_shape=image_array.shape,
        )
        self.last_analysis = analysis
        return analysis

    def answer_question(self, question: str, image_path: str) -> Dict[str, Any]:
        """Analyse *image_path* and answer *question* using generated code."""
        scene_analysis = self.analyze_image(image_path)
        code = self.code_generator.generate_code(question, scene_analysis)
        answer, status = self.code_generator.execute_code(code, scene_analysis)

        return {
            "question": question,
            "answer": answer,
            "status": status,
            "code": code,
            "objects_detected": [asdict(obj) for obj in scene_analysis.objects],
            "depth_map_shape": list(scene_analysis.depth_map.shape),
            "timestamp": scene_analysis.timestamp,
        }


if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY", "")
    agent = VADARAgent(api_key)
    print("VADAR Agent initialised successfully!")
