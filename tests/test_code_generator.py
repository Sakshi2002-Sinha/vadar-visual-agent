"""
Unit tests for CodeGenerator (prompt building and code execution).

No OpenAI API calls are made – the generate_code / generate_followup
methods are tested with a unittest.mock patch.
"""

from __future__ import annotations

import math
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vadar_agent import (
    CodeGenerator,
    SceneAnalysis,
    SpatialObject,
    SpatialReasoner,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_scene(objects=None) -> SceneAnalysis:
    """Return a minimal SceneAnalysis with two objects."""
    if objects is None:
        objects = [
            SpatialObject(
                label="chair",
                confidence=0.92,
                bbox=(0.1, 0.2, 0.4, 0.7),
                center=(160, 240),
                depth_value=0.3,
                area=0.15,
                image_height=480,
                image_width=640,
            ),
            SpatialObject(
                label="table",
                confidence=0.85,
                bbox=(0.5, 0.3, 0.9, 0.8),
                center=(448, 264),
                depth_value=0.7,
                area=0.20,
                image_height=480,
                image_width=640,
            ),
        ]
    return SceneAnalysis(
        objects=objects,
        depth_map=np.zeros((480, 640), dtype=np.float32),
        image_shape=(480, 640, 3),
    )


def make_generator(api_key: str = "sk-test") -> CodeGenerator:
    return CodeGenerator(api_key=api_key, model="gpt-4o-mini")


def _mock_openai_response(content: str):
    """Build a minimal mock that mimics openai.chat.completions.create output."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# _build_user_prompt
# ---------------------------------------------------------------------------

class TestBuildUserPrompt:
    def test_contains_question(self):
        cg = make_generator()
        scene = make_scene()
        prompt = cg._build_user_prompt("Which is closer?", scene)
        assert "Which is closer?" in prompt

    def test_contains_object_labels(self):
        cg = make_generator()
        scene = make_scene()
        prompt = cg._build_user_prompt("?", scene)
        assert "chair" in prompt
        assert "table" in prompt

    def test_contains_depth_values(self):
        cg = make_generator()
        scene = make_scene()
        prompt = cg._build_user_prompt("?", scene)
        assert "0.300" in prompt or "0.3" in prompt

    def test_contains_image_dimensions(self):
        cg = make_generator()
        scene = make_scene()
        prompt = cg._build_user_prompt("?", scene)
        assert "480" in prompt or "640" in prompt

    def test_contains_object_positions(self):
        cg = make_generator()
        scene = make_scene()
        prompt = cg._build_user_prompt("?", scene)
        # Horizontal/vertical position strings should appear
        assert any(pos in prompt for pos in ("left", "center", "right", "upper", "middle", "lower"))


# ---------------------------------------------------------------------------
# execute_code
# ---------------------------------------------------------------------------

class TestExecuteCode:
    def test_simple_answer(self):
        cg = make_generator()
        scene = make_scene()
        code = "answer = 42"
        result, status = cg.execute_code(code, scene)
        assert status == "Success"
        assert result == 42

    def test_string_answer(self):
        cg = make_generator()
        scene = make_scene()
        code = "answer = 'hello'"
        result, status = cg.execute_code(code, scene)
        assert status == "Success"
        assert result == "hello"

    def test_objects_available(self):
        cg = make_generator()
        scene = make_scene()
        code = "answer = len(objects)"
        result, status = cg.execute_code(code, scene)
        assert status == "Success"
        assert result == 2

    def test_numpy_available(self):
        cg = make_generator()
        scene = make_scene()
        code = "answer = float(np.sqrt(4.0))"
        result, status = cg.execute_code(code, scene)
        assert status == "Success"
        assert math.isclose(result, 2.0)

    def test_math_module_available(self):
        cg = make_generator()
        scene = make_scene()
        code = "answer = math.floor(3.9)"
        result, status = cg.execute_code(code, scene)
        assert status == "Success"
        assert result == 3

    def test_spatial_reasoner_available(self):
        cg = make_generator()
        scene = make_scene()
        code = (
            'chair = SpatialReasoner.get_object_by_label(objects, "chair")\n'
            "answer = chair.depth_value"
        )
        result, status = cg.execute_code(code, scene)
        assert status == "Success"
        assert math.isclose(result, 0.3)

    def test_missing_answer_variable(self):
        cg = make_generator()
        scene = make_scene()
        code = "x = 1"  # no 'answer' variable
        result, status = cg.execute_code(code, scene)
        assert status == "Success"
        assert result == "No answer produced"

    def test_execution_error(self):
        cg = make_generator()
        scene = make_scene()
        code = "answer = 1 / 0"
        result, status = cg.execute_code(code, scene)
        assert result is None
        assert "Execution error" in status

    def test_syntax_error(self):
        cg = make_generator()
        scene = make_scene()
        code = "answer = ("  # incomplete expression
        result, status = cg.execute_code(code, scene)
        assert result is None
        assert "Execution error" in status

    def test_answer_recorded_in_history(self):
        cg = make_generator()
        scene = make_scene()
        # Manually push a history entry (simulates generate_code result)
        cg.history.append({"question": "q", "code": "answer = 99", "answer": None, "timestamp": ""})
        cg.execute_code("answer = 99", scene)
        assert cg.history[-1]["answer"] == "99"


# ---------------------------------------------------------------------------
# generate_code (mocked OpenAI)
# ---------------------------------------------------------------------------

class TestGenerateCodeMocked:
    def test_returns_code_string(self):
        cg = make_generator()
        scene = make_scene()
        fake_code = "answer = objects[0].label"
        with patch("openai.chat.completions.create") as mock_create:
            mock_create.return_value = _mock_openai_response(fake_code)
            code = cg.generate_code("What is the first object?", scene)
        assert code == fake_code

    def test_history_updated(self):
        cg = make_generator()
        scene = make_scene()
        with patch("openai.chat.completions.create") as mock_create:
            mock_create.return_value = _mock_openai_response("answer = 1")
            cg.generate_code("Q1", scene)
        assert len(cg.history) == 1
        assert cg.history[0]["question"] == "Q1"
        assert cg.history[0]["code"] == "answer = 1"

    def test_strips_whitespace(self):
        cg = make_generator()
        scene = make_scene()
        with patch("openai.chat.completions.create") as mock_create:
            mock_create.return_value = _mock_openai_response("  answer = 1  ")
            code = cg.generate_code("Q", scene)
        assert code == "answer = 1"


# ---------------------------------------------------------------------------
# generate_followup (mocked OpenAI)
# ---------------------------------------------------------------------------

class TestGenerateFollowupMocked:
    def test_no_history_falls_back_to_generate_code(self):
        cg = make_generator()
        scene = make_scene()
        fake_code = "answer = 'fallback'"
        with patch("openai.chat.completions.create") as mock_create:
            mock_create.return_value = _mock_openai_response(fake_code)
            code = cg.generate_followup("Followup?", scene)
        assert code == fake_code
        assert len(cg.history) == 1

    def test_with_history_sends_multi_turn(self):
        cg = make_generator()
        scene = make_scene()
        # Seed history manually
        cg.history = [
            {
                "question": "First question",
                "code": "answer = 'first'",
                "answer": "first",
                "timestamp": "2024-01-01",
            }
        ]
        with patch("openai.chat.completions.create") as mock_create:
            mock_create.return_value = _mock_openai_response("answer = 'followup'")
            cg.generate_followup("Second question", scene)
            call_kwargs = mock_create.call_args
            messages = call_kwargs[1]["messages"] if call_kwargs[1] else call_kwargs[0][1]
        # Should have: system + user(Q1) + assistant(code1) + user(Q2)
        roles = [m["role"] for m in messages]
        assert roles == ["system", "user", "assistant", "user"]

    def test_history_grows(self):
        cg = make_generator()
        scene = make_scene()
        with patch("openai.chat.completions.create") as mock_create:
            mock_create.return_value = _mock_openai_response("answer = 1")
            cg.generate_code("Q1", scene)
            cg.generate_followup("Q2", scene)
        assert len(cg.history) == 2


# ---------------------------------------------------------------------------
# reset_history
# ---------------------------------------------------------------------------

class TestResetHistory:
    def test_clears_history(self):
        cg = make_generator()
        cg.history = [{"question": "x", "code": "y", "answer": "z", "timestamp": ""}]
        cg.reset_history()
        assert cg.history == []
