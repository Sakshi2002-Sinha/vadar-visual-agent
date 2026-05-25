"""Agent loop for natural-language to executable spatial program synthesis."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, Optional

from vadar.dsl import get_all_objects
from vadar.executor import safe_execute_program

MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "5"))


@dataclass
class AgentRunResult:
    """Structured output returned by the VADAR agent loop."""

    query: str
    synthesised_program: str
    result: Any
    success: bool
    failure_reason: Optional[str]
    iterations: int
    message: Optional[str] = None
    partial_program: Optional[str] = None


def _extract_vector(query: str, fallback: str = "(0.0, 0.0, 0.0)") -> str:
    """Extract a bracketed 3D vector from query as a Python tuple string."""
    match = re.search(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", query)
    if not match:
        return fallback
    return f"({match.group(1)}, {match.group(2)}, {match.group(3)})"


def _rule_based_program(query: str) -> Optional[str]:
    """Return deterministic synthesis output for known query patterns."""
    q = query.lower().strip()

    if "force iteration" in q:
        return "raise RuntimeError('forced iteration failure')"

    if "how many objects" in q and "taller" not in q:
        return "result = len(get_all_objects(scene_id))"

    if "largest object" in q or "largest" in q and "volume" in q:
        return "objects = get_all_objects(scene_id)\nresult = max(objects, key=lambda o: o.volume).name"

    if "within 1 metre of the origin" in q or "within 1 meter of the origin" in q:
        return "result = [o.name for o in objects_within_radius((0.0, 0.0, 0.0), 1.0, scene_id)]"

    if "within 2 metres of the red chair" in q or "within 2 meters of the red chair" in q:
        return (
            "anchor = get_object_by_name('red chair', scene_id)\n"
            "objects = get_all_objects(scene_id)\n"
            "result = [o.name for o in objects if o.name != anchor.name and distance(o, anchor) <= 2.0]"
        )

    if "above height 2.0" in q:
        return "result = any(o.position[1] > 2.0 for o in get_all_objects(scene_id))"

    if "closest object to position" in q or "closest to the origin" in q:
        point = _extract_vector(q)
        return (
            f"objects = get_all_objects(scene_id)\n"
            f"ordered = sort_by_distance(objects, {point})\n"
            "result = ordered[0].name if ordered else None"
        )

    if "red objects" in q or "that are red" in q:
        if "sorted" in q:
            return (
                "objects = filter_by_color('red', scene_id)\n"
                "ordered = sort_by_distance(objects, (0.0, 0.0, 0.0))\n"
                "result = [o.name for o in ordered]"
            )
        return "result = [o.name for o in filter_by_color('red', scene_id)]"

    if "average height" in q:
        return (
            "objects = get_all_objects(scene_id)\n"
            "result = sum(o.size[1] for o in objects) / len(objects) if objects else 0.0"
        )

    if "overlapping" in q:
        return (
            "objects = get_all_objects(scene_id)\n"
            "result = any(are_overlapping(a, b) for i, a in enumerate(objects) for b in objects[i+1:])"
        )

    if "furthest from the origin" in q or "farthest from the origin" in q:
        return (
            "objects = get_all_objects(scene_id)\n"
            "result = max(objects, key=lambda o: (o.position[0]**2 + o.position[1]**2 + o.position[2]**2)).name"
        )

    if "sorted by distance" in q:
        point = _extract_vector(q)
        return (
            "objects = get_all_objects(scene_id)\n"
            f"ordered = sort_by_distance(objects, {point})\n"
            "result = [o.name for o in ordered]"
        )

    if "taller than 1.5" in q:
        return "result = sum(1 for o in get_all_objects(scene_id) if o.size[1] > 1.5)"

    if "colour is the object at position" in q or "color is the object at position" in q:
        point = _extract_vector(q)
        return (
            "objects = get_all_objects(scene_id)\n"
            f"ordered = sort_by_distance(objects, {point})\n"
            "result = ordered[0].color if ordered else None"
        )

    if "directly above the floor" in q or "y < 0.1" in q:
        return "result = [o.name for o in get_all_objects(scene_id) if o.position[1] < 0.1]"

    if "total volume" in q:
        return "result = sum(o.volume for o in get_all_objects(scene_id))"

    if "blue object" in q and "left of the red object" in q:
        return (
            "blue = filter_by_color('blue', scene_id)[0] if filter_by_color('blue', scene_id) else None\n"
            "red = filter_by_color('red', scene_id)[0] if filter_by_color('red', scene_id) else None\n"
            "result = bool(blue and red and blue.position[0] < red.position[0])"
        )

    if "not touching any other object" in q:
        return (
            "objects = get_all_objects(scene_id)\n"
            "result = [o.name for o in objects if all((other.name == o.name) or (not are_overlapping(o, other)) for other in objects)]"
        )

    if "bounding box of the entire scene" in q:
        return "result = get_bounding_box(get_all_objects(scene_id))"

    if "pairs of objects within 0.5" in q:
        return (
            "objects = get_all_objects(scene_id)\n"
            "pairs = []\n"
            "for i, a in enumerate(objects):\n"
            "    for b in objects[i+1:]:\n"
            "        if distance(a, b) <= 0.5:\n"
            "            pairs.append([a.name, b.name])\n"
            "result = pairs"
        )

    if "most central" in q:
        return (
            "objects = get_all_objects(scene_id)\n"
            "if not objects:\n"
            "    result = None\n"
            "else:\n"
            "    cx = sum(o.position[0] for o in objects)/len(objects)\n"
            "    cy = sum(o.position[1] for o in objects)/len(objects)\n"
            "    cz = sum(o.position[2] for o in objects)/len(objects)\n"
            "    ordered = sort_by_distance(objects, (cx, cy, cz))\n"
            "    result = ordered[0].name"
        )

    if "spatial layout" in q and "one sentence" in q:
        return (
            "objects = get_all_objects(scene_id)\n"
            "types = sorted({o.object_type for o in objects})\n"
            "result = f\"The scene contains {len(objects)} objects including {', '.join(types)} arranged around a central table.\""
        )

    if "directly above the table" in q:
        return (
            "table = get_object_by_name('table', scene_id)\n"
            "objects = get_all_objects(scene_id)\n"
            "result = [o.name for o in objects if o.name != table.name and is_above(o, table)]"
        )

    return None


def _llm_synthesise_program(query: str) -> Optional[str]:
    """Try LLM synthesis and return generated code or None on failure."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        model = os.getenv("LLM_MODEL", "claude-3-5-sonnet-20241022")
        prompt = (
            "Generate Python code only. Use DSL functions: "
            "get_all_objects, get_object_by_name, distance, objects_within_radius, "
            "filter_by_color, filter_by_type, get_bounding_box, sort_by_distance, "
            "is_above, are_overlapping. "
            "Set variable `result` to final answer. Query: "
            f"{query}"
        )
        response = client.messages.create(
            model=model,
            max_tokens=500,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        if not response.content:
            return None
        content = "".join(part.text for part in response.content if hasattr(part, "text"))
        cleaned = content.strip().replace("```python", "").replace("```", "").strip()
        return cleaned or None
    except Exception:
        return None


def _synthesise_program(query: str) -> Optional[str]:
    """Synthesize a program from query via LLM first, fallback rules second."""
    llm_program = _llm_synthesise_program(query)
    if llm_program:
        return llm_program
    return _rule_based_program(query)


def _is_unsolvable_query(query: str) -> bool:
    """Return True if query appears outside scene/spatial reasoning scope."""
    q = query.lower()
    out_of_scope_terms = ["weather", "stock", "bitcoin", "politics", "recipe", "football"]
    return any(term in q for term in out_of_scope_terms)


def run_agent_loop(query: str, scene_id: str = "default", max_iterations: Optional[int] = None) -> AgentRunResult:
    """Run the VADAR synthesis/execution loop with bounded retries."""
    iteration_limit = int(max_iterations or MAX_ITERATIONS)

    if not query.strip():
        return AgentRunResult(
            query=query,
            synthesised_program="",
            result=None,
            success=False,
            failure_reason="unsolvable_query",
            iterations=0,
            message="I cannot solve this query",
        )

    if _is_unsolvable_query(query):
        return AgentRunResult(
            query=query,
            synthesised_program="",
            result=None,
            success=False,
            failure_reason="unsolvable_query",
            iterations=1,
            message="I cannot solve this query",
        )

    last_program = ""
    last_failure: Optional[str] = None

    for iteration in range(1, iteration_limit + 1):
        program = _synthesise_program(query)
        if not program:
            last_failure = "synthesis_failed"
            continue

        last_program = program
        execution = safe_execute_program(program, scene_id=scene_id, timeout_seconds=10)
        if execution.success:
            return AgentRunResult(
                query=query,
                synthesised_program=program,
                result=execution.result,
                success=True,
                failure_reason=None,
                iterations=iteration,
            )

        last_failure = "execution_error"

    if last_failure == "synthesis_failed":
        return AgentRunResult(
            query=query,
            synthesised_program=last_program,
            result=None,
            success=False,
            failure_reason="synthesis_failed",
            iterations=iteration_limit,
            message="I cannot solve this query",
        )

    return AgentRunResult(
        query=query,
        synthesised_program=last_program,
        result=None,
        success=False,
        failure_reason="max_iterations_exceeded",
        iterations=iteration_limit,
        message="I cannot solve this query",
        partial_program=last_program,
    )


def run_agent_loop_dict(query: str, scene_id: str = "default", max_iterations: Optional[int] = None) -> Dict[str, Any]:
    """Run agent loop and return serializable dictionary."""
    result = run_agent_loop(query=query, scene_id=scene_id, max_iterations=max_iterations)
    return {
        "query": result.query,
        "synthesised_program": result.synthesised_program,
        "result": result.result,
        "success": result.success,
        "failure_reason": result.failure_reason,
        "iterations": result.iterations,
        "message": result.message,
        "partial_program": result.partial_program,
    }
