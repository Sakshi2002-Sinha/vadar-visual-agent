"""
Integration tests for VADARAgent.

All heavy dependencies (VisionModels / HuggingFace pipelines and OpenAI)
are replaced with lightweight mocks so these tests run without a GPU,
model downloads, or a real API key.
"""

from __future__ import annotations

import io
import math
from dataclasses import replace as dc_replace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from vadar_agent import (
    SceneAnalysis,
    SpatialObject,
    VADARAgent,
    VisionModels,
)


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

_IMAGE_WIDTH = 640
_IMAGE_HEIGHT = 480

# A minimal in-memory RGB image (solid grey)
_FAKE_IMAGE_BYTES = io.BytesIO()
Image.new("RGB", (_IMAGE_WIDTH, _IMAGE_HEIGHT), color=(128, 128, 128)).save(
    _FAKE_IMAGE_BYTES, format="PNG"
)
_FAKE_IMAGE_BYTES.seek(0)


def _fake_image_path(tmp_path) -> str:
    p = tmp_path / "test_image.png"
    p.write_bytes(_FAKE_IMAGE_BYTES.getvalue())
    return str(p)


def _mock_vision_models() -> MagicMock:
    """Return a VisionModels mock with deterministic outputs."""
    vm = MagicMock(spec=VisionModels)
    vm.detect_objects.return_value = [
        {
            "label": "chair",
            "score": 0.92,
            "box": {"xmin": 64, "ymin": 96, "xmax": 256, "ymax": 336},
        },
        {
            "label": "table",
            "score": 0.85,
            "box": {"xmin": 320, "ymin": 144, "xmax": 576, "ymax": 384},
        },
    ]
    # Depth map: left half close (0.2), right half far (0.8)
    dm = np.full((_IMAGE_HEIGHT, _IMAGE_WIDTH), 0.2, dtype=np.float32)
    dm[:, _IMAGE_WIDTH // 2 :] = 0.8
    vm.estimate_depth.return_value = dm
    return vm


def _mock_openai_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _make_agent(tmp_path, api_key: str = "sk-test") -> tuple[VADARAgent, str]:
    """Create a VADARAgent with mocked vision models and return (agent, image_path)."""
    with patch("vadar_agent.VisionModels", return_value=_mock_vision_models()):
        agent = VADARAgent(api_key=api_key, use_gpu=False)
    agent.vision_models = _mock_vision_models()
    return agent, _fake_image_path(tmp_path)


# ---------------------------------------------------------------------------
# analyze_image
# ---------------------------------------------------------------------------

class TestAnalyzeImage:
    def test_returns_scene_analysis(self, tmp_path):
        agent, img = _make_agent(tmp_path)
        scene = agent.analyze_image(img)
        assert isinstance(scene, SceneAnalysis)

    def test_objects_count(self, tmp_path):
        agent, img = _make_agent(tmp_path)
        scene = agent.analyze_image(img)
        assert len(scene.objects) == 2

    def test_object_labels(self, tmp_path):
        agent, img = _make_agent(tmp_path)
        scene = agent.analyze_image(img)
        labels = {o.label for o in scene.objects}
        assert labels == {"chair", "table"}

    def test_object_confidences(self, tmp_path):
        agent, img = _make_agent(tmp_path)
        scene = agent.analyze_image(img)
        chair = next(o for o in scene.objects if o.label == "chair")
        assert math.isclose(chair.confidence, 0.92)

    def test_depth_values_assigned(self, tmp_path):
        agent, img = _make_agent(tmp_path)
        scene = agent.analyze_image(img)
        # Chair bbox is on the left half → depth ≈ 0.2
        chair = next(o for o in scene.objects if o.label == "chair")
        assert chair.depth_value < 0.5
        # Table bbox is on the right half → depth ≈ 0.8
        table = next(o for o in scene.objects if o.label == "table")
        assert table.depth_value > 0.5

    def test_normalised_bbox_range(self, tmp_path):
        agent, img = _make_agent(tmp_path)
        scene = agent.analyze_image(img)
        for obj in scene.objects:
            assert 0.0 <= obj.bbox[0] <= 1.0
            assert 0.0 <= obj.bbox[1] <= 1.0
            assert 0.0 <= obj.bbox[2] <= 1.0
            assert 0.0 <= obj.bbox[3] <= 1.0

    def test_center_within_image(self, tmp_path):
        agent, img = _make_agent(tmp_path)
        scene = agent.analyze_image(img)
        for obj in scene.objects:
            assert 0 <= obj.center[0] < _IMAGE_WIDTH
            assert 0 <= obj.center[1] < _IMAGE_HEIGHT

    def test_last_analysis_updated(self, tmp_path):
        agent, img = _make_agent(tmp_path)
        assert agent.last_analysis is None
        agent.analyze_image(img)
        assert agent.last_analysis is not None

    def test_image_shape_stored(self, tmp_path):
        agent, img = _make_agent(tmp_path)
        scene = agent.analyze_image(img)
        assert scene.image_shape[0] == _IMAGE_HEIGHT
        assert scene.image_shape[1] == _IMAGE_WIDTH


# ---------------------------------------------------------------------------
# Segmentation depth fallback (_bbox_iou + _apply_segmentation_depth_fallback)
# ---------------------------------------------------------------------------

class TestBboxIou:
    def test_no_overlap(self):
        a = SpatialObject("a", 1.0, (0.0, 0.0, 0.3, 0.3), (50, 50), 0.2, 0.09, 100, 100)
        b = SpatialObject("b", 1.0, (0.7, 0.7, 1.0, 1.0), (85, 85), 0.8, 0.09, 100, 100)
        assert VADARAgent._bbox_iou(a, b) == 0.0

    def test_full_overlap(self):
        a = SpatialObject("a", 1.0, (0.1, 0.1, 0.9, 0.9), (50, 50), 0.5, 0.64, 100, 100)
        b = SpatialObject("b", 1.0, (0.1, 0.1, 0.9, 0.9), (50, 50), 0.5, 0.64, 100, 100)
        assert math.isclose(VADARAgent._bbox_iou(a, b), 1.0)

    def test_partial_overlap(self):
        a = SpatialObject("a", 1.0, (0.0, 0.0, 0.5, 1.0), (25, 50), 0.3, 0.5, 100, 100)
        b = SpatialObject("b", 1.0, (0.4, 0.0, 0.9, 1.0), (65, 50), 0.7, 0.5, 100, 100)
        iou = VADARAgent._bbox_iou(a, b)
        assert 0.0 < iou < 1.0


class TestSegmentationDepthFallback:
    def _make_agent_instance(self) -> VADARAgent:
        with patch("vadar_agent.VisionModels"):
            agent = VADARAgent(api_key="sk-test")
        return agent

    def test_non_overlapping_objects_unchanged(self, tmp_path):
        agent = self._make_agent_instance()
        objs = [
            SpatialObject("a", 1.0, (0.0, 0.0, 0.3, 0.3), (50, 50), 0.2, 0.09, 100, 100),
            SpatialObject("b", 1.0, (0.7, 0.7, 1.0, 1.0), (85, 85), 0.8, 0.09, 100, 100),
        ]
        dm = np.zeros((100, 100), dtype=np.float32)
        result = agent._apply_segmentation_depth_fallback(None, objs, dm, 100, 100)
        assert result[0].depth_value == 0.2
        assert result[1].depth_value == 0.8

    def test_overlapping_objects_get_mean_depth(self, tmp_path):
        agent = self._make_agent_instance()
        # Both objects occupy roughly the same region → high IoU
        objs = [
            SpatialObject("a", 1.0, (0.2, 0.2, 0.8, 0.8), (50, 50), 0.36, 0.36, 100, 100),
            SpatialObject("b", 1.0, (0.2, 0.2, 0.8, 0.8), (50, 50), 0.36, 0.36, 100, 100),
        ]
        dm = np.full((100, 100), 0.5, dtype=np.float32)
        # Single-pixel depth was 0.0 (centre pixel of all-zeros dm), but region mean is 0.5
        objs[0] = dc_replace(objs[0], depth_value=0.0)
        objs[1] = dc_replace(objs[1], depth_value=0.0)
        result = agent._apply_segmentation_depth_fallback(None, objs, dm, 100, 100)
        assert math.isclose(result[0].depth_value, 0.5)
        assert math.isclose(result[1].depth_value, 0.5)


# ---------------------------------------------------------------------------
# answer_question
# ---------------------------------------------------------------------------

class TestAnswerQuestion:
    def test_returns_dict_with_expected_keys(self, tmp_path):
        agent, img = _make_agent(tmp_path)
        fake_code = "answer = 'chair'"
        with patch("openai.chat.completions.create") as mock_create:
            mock_create.return_value = _mock_openai_response(fake_code)
            result = agent.answer_question("Which object is closest?", img)
        for key in ("question", "answer", "status", "code", "objects_detected", "timestamp"):
            assert key in result

    def test_correct_answer_returned(self, tmp_path):
        agent, img = _make_agent(tmp_path)
        fake_code = "answer = objects[0].label"
        with patch("openai.chat.completions.create") as mock_create:
            mock_create.return_value = _mock_openai_response(fake_code)
            result = agent.answer_question("First object?", img)
        assert result["answer"] == "chair"
        assert result["status"] == "Success"

    def test_objects_detected_list(self, tmp_path):
        agent, img = _make_agent(tmp_path)
        fake_code = "answer = 'ok'"
        with patch("openai.chat.completions.create") as mock_create:
            mock_create.return_value = _mock_openai_response(fake_code)
            result = agent.answer_question("?", img)
        assert isinstance(result["objects_detected"], list)
        assert len(result["objects_detected"]) == 2

    def test_execution_error_captured(self, tmp_path):
        agent, img = _make_agent(tmp_path)
        bad_code = "answer = undefined_variable"
        with patch("openai.chat.completions.create") as mock_create:
            mock_create.return_value = _mock_openai_response(bad_code)
            result = agent.answer_question("?", img)
        assert result["answer"] is None
        assert "Execution error" in result["status"]


# ---------------------------------------------------------------------------
# answer_followup
# ---------------------------------------------------------------------------

class TestAnswerFollowup:
    def test_raises_without_prior_analysis(self):
        with patch("vadar_agent.VisionModels"):
            agent = VADARAgent(api_key="sk-test")
        with pytest.raises(RuntimeError, match="No image has been analysed yet"):
            agent.answer_followup("Follow-up?")

    def test_uses_last_analysis(self, tmp_path):
        agent, img = _make_agent(tmp_path)
        initial_code = "answer = 'first'"
        followup_code = "answer = 'followup'"
        with patch("openai.chat.completions.create") as mock_create:
            mock_create.return_value = _mock_openai_response(initial_code)
            agent.answer_question("First question?", img)
            mock_create.return_value = _mock_openai_response(followup_code)
            result = agent.answer_followup("Follow-up question?")
        assert result["answer"] == "followup"
        assert result["status"] == "Success"

    def test_history_grows_across_calls(self, tmp_path):
        agent, img = _make_agent(tmp_path)
        with patch("openai.chat.completions.create") as mock_create:
            mock_create.return_value = _mock_openai_response("answer = 1")
            agent.answer_question("Q1?", img)
            mock_create.return_value = _mock_openai_response("answer = 2")
            agent.answer_followup("Q2?")
        assert len(agent.code_generator.history) == 2
