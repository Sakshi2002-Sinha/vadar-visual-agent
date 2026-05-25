"""Safe-ish restricted execution layer for synthesized Python programs."""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Dict, Optional

from vadar.dsl import (
    are_overlapping,
    distance,
    filter_by_color,
    filter_by_type,
    get_all_objects,
    get_bounding_box,
    get_object_by_name,
    is_above,
    objects_within_radius,
    sort_by_distance,
)
from vadar.scene import object_to_dict


@dataclass
class ExecutionResult:
    """Represents output of a synthesized program execution."""

    success: bool
    result: Any
    error: Optional[str] = None


def _execute_worker(program: str, scene_id: str, queue: mp.Queue) -> None:
    """Execute user program in a restricted namespace and publish result."""
    allowed_builtins: Dict[str, Any] = {
        "len": len,
        "min": min,
        "max": max,
        "sum": sum,
        "sorted": sorted,
        "any": any,
        "all": all,
        "round": round,
        "abs": abs,
        "float": float,
        "int": int,
        "str": str,
        "bool": bool,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        "enumerate": enumerate,
        "range": range,
    }

    globals_ctx: Dict[str, Any] = {
        "__builtins__": allowed_builtins,
        "get_all_objects": get_all_objects,
        "get_object_by_name": get_object_by_name,
        "distance": distance,
        "objects_within_radius": objects_within_radius,
        "filter_by_color": filter_by_color,
        "filter_by_type": filter_by_type,
        "get_bounding_box": get_bounding_box,
        "sort_by_distance": sort_by_distance,
        "is_above": is_above,
        "are_overlapping": are_overlapping,
        "scene_id": scene_id,
    }

    try:
        exec(program, globals_ctx, globals_ctx)  # noqa: S102
        output = globals_ctx.get("result")
        serialized: Any = output
        if isinstance(output, list):
            serialized = [object_to_dict(item) if hasattr(item, "name") else item for item in output]
        elif hasattr(output, "name"):
            serialized = object_to_dict(output)
        queue.put({"success": True, "result": serialized, "error": None})
    except Exception as exc:  # noqa: BLE001
        queue.put({"success": False, "result": None, "error": f"execution_error: {exc}"})


def safe_execute_program(program: str, scene_id: str = "default", timeout_seconds: int = 10) -> ExecutionResult:
    """Execute synthesized program with timeout and restricted builtins."""
    queue: mp.Queue = mp.Queue()
    process = mp.Process(target=_execute_worker, args=(program, scene_id, queue))
    process.start()
    process.join(timeout=timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join(timeout=1)
        return ExecutionResult(success=False, result=None, error="execution_error: execution_timeout")

    if queue.empty():
        return ExecutionResult(success=False, result=None, error="execution_error: no_result_returned")

    payload: Dict[str, Any] = queue.get()
    return ExecutionResult(success=bool(payload["success"]), result=payload["result"], error=payload["error"])
