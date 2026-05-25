"""Spatial reasoning DSL functions exposed to synthesized programs."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from vadar.scene import Object3D, get_scene

BoundingBox = Dict[str, float]
Point3D = Tuple[float, float, float]


def get_all_objects(scene_id: str = "default") -> List[Object3D]:
    """Return all objects in the specified scene."""
    return list(get_scene(scene_id).objects)


def get_object_by_name(name: str, scene_id: str = "default") -> Object3D:
    """Return object whose name matches exactly or contains the provided name."""
    lowered = name.lower()
    for obj in get_all_objects(scene_id):
        if obj.name.lower() == lowered or lowered in obj.name.lower():
            return obj
    raise ValueError(f"Object '{name}' not found")


def distance(obj1: Object3D, obj2: Object3D) -> float:
    """Compute Euclidean distance between two objects."""
    return float(
        math.dist(obj1.position, obj2.position)
    )


def objects_within_radius(center: Point3D, radius: float, scene_id: str = "default") -> List[Object3D]:
    """Return objects whose center is within radius of a point."""
    return [
        obj
        for obj in get_all_objects(scene_id)
        if math.dist(obj.position, center) <= float(radius)
    ]


def filter_by_color(color: str, scene_id: str = "default") -> List[Object3D]:
    """Return objects matching color (case-insensitive)."""
    lowered = color.lower()
    return [obj for obj in get_all_objects(scene_id) if obj.color.lower() == lowered]


def filter_by_type(object_type: str, scene_id: str = "default") -> List[Object3D]:
    """Return objects matching object type (case-insensitive)."""
    lowered = object_type.lower()
    return [obj for obj in get_all_objects(scene_id) if obj.object_type.lower() == lowered]


def get_bounding_box(objects: Sequence[Object3D]) -> BoundingBox:
    """Return axis-aligned scene bounding box for provided objects."""
    if not objects:
        raise ValueError("Cannot compute bounding box of empty object list")

    xs = [obj.position[0] for obj in objects]
    ys = [obj.position[1] for obj in objects]
    zs = [obj.position[2] for obj in objects]
    return {
        "min_x": float(min(xs)),
        "min_y": float(min(ys)),
        "min_z": float(min(zs)),
        "max_x": float(max(xs)),
        "max_y": float(max(ys)),
        "max_z": float(max(zs)),
    }


def sort_by_distance(objects: Sequence[Object3D], point: Point3D) -> List[Object3D]:
    """Return objects sorted by distance to a point."""
    return sorted(objects, key=lambda obj: math.dist(obj.position, point))


def is_above(obj1: Object3D, obj2: Object3D) -> bool:
    """Return True when obj1 is vertically above obj2."""
    return bool(obj1.position[1] > obj2.position[1])


def are_overlapping(obj1: Object3D, obj2: Object3D) -> bool:
    """Return True if axis-aligned bounding boxes overlap."""
    def bounds(obj: Object3D) -> Tuple[float, float, float, float, float, float]:
        sx, sy, sz = obj.size
        px, py, pz = obj.position
        return (
            px - sx / 2.0,
            px + sx / 2.0,
            py - sy / 2.0,
            py + sy / 2.0,
            pz - sz / 2.0,
            pz + sz / 2.0,
        )

    a = bounds(obj1)
    b = bounds(obj2)
    x_overlap = a[0] <= b[1] and b[0] <= a[1]
    y_overlap = a[2] <= b[3] and b[2] <= a[3]
    z_overlap = a[4] <= b[5] and b[4] <= a[5]
    return bool(x_overlap and y_overlap and z_overlap)
