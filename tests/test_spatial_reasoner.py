"""
Unit tests for SpatialReasoner helper functions.

These tests are purely computational – no GPU, no HuggingFace models,
and no OpenAI API calls are needed.
"""

import math
import pytest

from vadar_agent import SpatialObject, SpatialReasoner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_obj(
    label: str = "cat",
    confidence: float = 0.9,
    bbox=(0.1, 0.1, 0.4, 0.5),
    center=(128, 240),
    depth_value: float = 0.5,
    area: float = 0.12,
    image_height: int = 480,
    image_width: int = 640,
) -> SpatialObject:
    return SpatialObject(
        label=label,
        confidence=confidence,
        bbox=bbox,
        center=center,
        depth_value=depth_value,
        area=area,
        image_height=image_height,
        image_width=image_width,
    )


# ---------------------------------------------------------------------------
# get_object_by_label
# ---------------------------------------------------------------------------

class TestGetObjectByLabel:
    def test_exact_match(self):
        objs = [make_obj("chair"), make_obj("table")]
        result = SpatialReasoner.get_object_by_label(objs, "chair")
        assert result is not None
        assert result.label == "chair"

    def test_partial_match(self):
        objs = [make_obj("dining table"), make_obj("chair")]
        result = SpatialReasoner.get_object_by_label(objs, "table")
        assert result is not None
        assert "table" in result.label

    def test_case_insensitive(self):
        objs = [make_obj("Chair"), make_obj("Table")]
        assert SpatialReasoner.get_object_by_label(objs, "chair") is not None
        assert SpatialReasoner.get_object_by_label(objs, "CHAIR") is not None

    def test_not_found_returns_none(self):
        objs = [make_obj("chair"), make_obj("table")]
        assert SpatialReasoner.get_object_by_label(objs, "sofa") is None

    def test_empty_list(self):
        assert SpatialReasoner.get_object_by_label([], "chair") is None

    def test_returns_first_match(self):
        objs = [make_obj("cat"), make_obj("cat")]
        objs[0] = make_obj("cat", confidence=0.95)
        objs[1] = make_obj("cat", confidence=0.50)
        result = SpatialReasoner.get_object_by_label(objs, "cat")
        assert result.confidence == 0.95


# ---------------------------------------------------------------------------
# is_farther
# ---------------------------------------------------------------------------

class TestIsFarther:
    def test_obj1_farther(self):
        far = make_obj(depth_value=0.9)
        near = make_obj(depth_value=0.2)
        assert SpatialReasoner.is_farther(far, near) is True

    def test_obj1_closer(self):
        near = make_obj(depth_value=0.1)
        far = make_obj(depth_value=0.8)
        assert SpatialReasoner.is_farther(near, far) is False

    def test_equal_depth(self):
        a = make_obj(depth_value=0.5)
        b = make_obj(depth_value=0.5)
        assert SpatialReasoner.is_farther(a, b) is False

    def test_boundary_values(self):
        close = make_obj(depth_value=0.0)
        distant = make_obj(depth_value=1.0)
        assert SpatialReasoner.is_farther(distant, close) is True
        assert SpatialReasoner.is_farther(close, distant) is False


# ---------------------------------------------------------------------------
# relative_depth_distance
# ---------------------------------------------------------------------------

class TestRelativeDepthDistance:
    def test_positive_difference(self):
        a = make_obj(depth_value=0.8)
        b = make_obj(depth_value=0.3)
        assert math.isclose(SpatialReasoner.relative_depth_distance(a, b), 0.5)

    def test_symmetric(self):
        a = make_obj(depth_value=0.2)
        b = make_obj(depth_value=0.7)
        assert math.isclose(
            SpatialReasoner.relative_depth_distance(a, b),
            SpatialReasoner.relative_depth_distance(b, a),
        )

    def test_same_depth_is_zero(self):
        a = make_obj(depth_value=0.4)
        b = make_obj(depth_value=0.4)
        assert SpatialReasoner.relative_depth_distance(a, b) == 0.0


# ---------------------------------------------------------------------------
# pixel_distance
# ---------------------------------------------------------------------------

class TestPixelDistance:
    def test_known_distance(self):
        a = make_obj(center=(0, 0))
        b = make_obj(center=(3, 4))
        assert math.isclose(SpatialReasoner.pixel_distance(a, b), 5.0)

    def test_same_point(self):
        a = make_obj(center=(100, 200))
        b = make_obj(center=(100, 200))
        assert SpatialReasoner.pixel_distance(a, b) == 0.0

    def test_symmetric(self):
        a = make_obj(center=(10, 20))
        b = make_obj(center=(30, 40))
        assert math.isclose(
            SpatialReasoner.pixel_distance(a, b),
            SpatialReasoner.pixel_distance(b, a),
        )

    def test_horizontal_only(self):
        a = make_obj(center=(0, 100))
        b = make_obj(center=(50, 100))
        assert math.isclose(SpatialReasoner.pixel_distance(a, b), 50.0)

    def test_vertical_only(self):
        a = make_obj(center=(100, 0))
        b = make_obj(center=(100, 75))
        assert math.isclose(SpatialReasoner.pixel_distance(a, b), 75.0)


# ---------------------------------------------------------------------------
# vertical_position
# ---------------------------------------------------------------------------

class TestVerticalPosition:
    def _obj_at(self, cy, image_height=300):
        return make_obj(center=(50, cy), image_height=image_height)

    def test_upper(self):
        assert SpatialReasoner.vertical_position(self._obj_at(50)) == "upper"

    def test_lower(self):
        assert SpatialReasoner.vertical_position(self._obj_at(250)) == "lower"

    def test_middle(self):
        assert SpatialReasoner.vertical_position(self._obj_at(150)) == "middle"

    def test_boundary_upper_middle(self):
        # cy == third boundary: third = 300/3 = 100, cy = 100 → "middle"
        assert SpatialReasoner.vertical_position(self._obj_at(100)) == "middle"

    def test_boundary_middle_lower(self):
        # cy == 2*third: 2*100 = 200; cy > 200 is "lower", cy = 200 is "middle"
        assert SpatialReasoner.vertical_position(self._obj_at(200)) == "middle"
        assert SpatialReasoner.vertical_position(self._obj_at(201)) == "lower"


# ---------------------------------------------------------------------------
# horizontal_position
# ---------------------------------------------------------------------------

class TestHorizontalPosition:
    def _obj_at(self, cx, image_width=300):
        return make_obj(center=(cx, 50), image_width=image_width)

    def test_left(self):
        assert SpatialReasoner.horizontal_position(self._obj_at(50)) == "left"

    def test_right(self):
        assert SpatialReasoner.horizontal_position(self._obj_at(250)) == "right"

    def test_center(self):
        assert SpatialReasoner.horizontal_position(self._obj_at(150)) == "center"

    def test_boundary_left_center(self):
        # cx == third: 300/3 = 100 → center
        assert SpatialReasoner.horizontal_position(self._obj_at(100)) == "center"

    def test_boundary_center_right(self):
        # cx > 2*third (200) → right; cx == 200 → center
        assert SpatialReasoner.horizontal_position(self._obj_at(200)) == "center"
        assert SpatialReasoner.horizontal_position(self._obj_at(201)) == "right"
