"""Scene primitives and default scene data for VADAR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Object3D:
    """Represents one object in a 3D scene."""

    name: str
    object_type: str
    color: str
    position: Tuple[float, float, float]
    size: Tuple[float, float, float]

    @property
    def volume(self) -> float:
        """Return object volume computed from size dimensions."""
        return float(self.size[0] * self.size[1] * self.size[2])


@dataclass(frozen=True)
class Scene:
    """Container for a named scene and its objects."""

    scene_id: str
    objects: List[Object3D]


def default_scene() -> Scene:
    """Return the built-in default scene used for demos and evaluation."""
    return Scene(
        scene_id="default",
        objects=[
            Object3D("red chair", "chair", "red", (1.0, 0.0, 1.2), (0.6, 1.0, 0.6)),
            Object3D("blue chair", "chair", "blue", (-1.1, 0.0, 1.1), (0.6, 1.0, 0.6)),
            Object3D("main table", "table", "brown", (0.0, 0.75, 0.0), (2.2, 0.2, 1.4)),
            Object3D("green box", "box", "green", (2.2, 0.4, -0.8), (0.7, 0.7, 0.7)),
            Object3D("yellow sphere", "sphere", "yellow", (-2.0, 1.8, 0.5), (0.8, 0.8, 0.8)),
            Object3D("gray cabinet", "cabinet", "gray", (2.8, 0.0, 2.0), (1.2, 2.1, 0.8)),
            Object3D("orange lamp", "lamp", "orange", (0.1, 2.4, -1.2), (0.3, 0.9, 0.3)),
            Object3D("purple stool", "stool", "purple", (-0.4, 0.0, -2.2), (0.5, 0.5, 0.5)),
        ],
    )


def get_scene(scene_id: str = "default") -> Scene:
    """Return a scene by ID, falling back to the default scene."""
    if scene_id == "default":
        return default_scene()
    return Scene(scene_id=scene_id, objects=default_scene().objects)


def object_to_dict(obj: Object3D) -> Dict[str, object]:
    """Serialize an Object3D to a JSON-compatible dictionary."""
    return {
        "name": obj.name,
        "object_type": obj.object_type,
        "color": obj.color,
        "position": list(obj.position),
        "size": list(obj.size),
        "volume": obj.volume,
    }
