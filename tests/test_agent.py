"""Tests for VADAR agent loop failure handling and DSL output types."""

from __future__ import annotations

from vadar.agent import run_agent_loop
from vadar.dsl import (
    are_overlapping,
    filter_by_color,
    get_all_objects,
    get_bounding_box,
    get_object_by_name,
    is_above,
    objects_within_radius,
    sort_by_distance,
)


def test_agent_max_iterations_failure() -> None:
    """Agent should fail gracefully with exact message on max iterations."""
    result = run_agent_loop("force iteration failure", max_iterations=2)
    assert result.success is False
    assert result.failure_reason == "max_iterations_exceeded"
    assert result.message == "I cannot solve this query"


def test_dsl_functions_return_expected_types() -> None:
    """DSL functions should return valid expected Python types."""
    objects = get_all_objects()
    assert isinstance(objects, list)
    assert len(objects) >= 8

    target = get_object_by_name("table")
    assert isinstance(target.name, str)

    nearby = objects_within_radius((0.0, 0.0, 0.0), 2.0)
    assert isinstance(nearby, list)

    red = filter_by_color("red")
    assert isinstance(red, list)

    bbox = get_bounding_box(objects)
    assert isinstance(bbox, dict)

    ordered = sort_by_distance(objects, (0.0, 0.0, 0.0))
    assert isinstance(ordered, list)

    assert isinstance(is_above(objects[0], objects[1]), bool)
    assert isinstance(are_overlapping(objects[0], objects[1]), bool)
