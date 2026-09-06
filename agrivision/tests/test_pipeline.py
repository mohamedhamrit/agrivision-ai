"""Tests for the AgriVision AI OpenCV 5 pipeline and agent."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agrivision.sample_data import make_leaf_scene, make_video
from agrivision.pipeline import OpenCVPipeline, PipelineConfig
from agrivision.agent import Agent, DecisionPolicy, ActionType
from agrivision.models import Severity


@pytest.fixture(scope="module")
def pipeline():
    return OpenCVPipeline()


# ----------------------------------------------------------------------
# Segmentation primitives
# ----------------------------------------------------------------------
def test_leaf_mask_is_binary(pipeline):
    img = make_leaf_scene(0.25)
    mask = pipeline.segment_leaf_mask(img)
    assert set(np.unique(mask)) <= {0, 255}


def test_healthy_has_no_lesions(pipeline):
    img = make_leaf_scene(0.0, seed=1)
    res = pipeline.analyze_frame(img, 0)
    assert res.severity == Severity.HEALTHY.value
    assert res.lesions == 0
    assert res.lesion_area_ratio == 0.0


# ----------------------------------------------------------------------
# Severity grading across lesion loads
# ----------------------------------------------------------------------
@pytest.mark.parametrize("syn_ratio,expected", [
    (0.00, Severity.HEALTHY.value),
    (0.07, Severity.MILD.value),
    (0.35, Severity.MODERATE.value),
    (0.65, Severity.SEVERE.value),
])
def test_severity_grading(pipeline, syn_ratio, expected):
    img = make_leaf_scene(syn_ratio, seed=3)
    res = pipeline.analyze_frame(img, 0)
    assert res.severity == expected, f"{syn_ratio} -> {res.severity} (ratio={res.lesion_area_ratio})"


def test_severity_is_monotonic(pipeline):
    ratios = []
    for r in [0.0, 0.07, 0.35, 0.60]:
        res = pipeline.analyze_frame(make_leaf_scene(r, seed=3), 0)
        ratios.append(res.lesion_area_ratio)
    # Detected lesion ratio should be non-decreasing with synthetic severity.
    assert ratios == sorted(ratios)


# ----------------------------------------------------------------------
# Video-level analysis
# ----------------------------------------------------------------------
def test_analyze_video(tmp_path, pipeline):
    path = os.path.join(str(tmp_path), "mid.mp4")
    make_video(0.35, path, frames=12, seed=2)
    result = pipeline.analyze_video(path)
    assert result.frames_analyzed > 0
    assert result.worst_severity in {s.value for s in Severity}
    assert result.mean_severity_score >= 0.0
    assert result.processing_time_ms > 0.0

# ----------------------------------------------------------------------
# Agent decision mapping
# ----------------------------------------------------------------------
def _result(severity, confidence=0.9):
    from agrivision.models import VideoAnalysisResult, DetectionResult
    d = DetectionResult(severity=severity, confidence=confidence,
                        lesion_area_ratio=0.2)
    r = VideoAnalysisResult(
        frames_analyzed=1,
        detections=[d],
        worst_severity=severity,
        mean_severity_score=0.2,
        avg_lesion_area_ratio=0.2,
    )
    return r


def test_agent_no_action_healthy():
    agent = Agent()
    action = agent.run(_result(Severity.HEALTHY.value))
    assert action.action == ActionType.NO_ACTION


def test_agent_urgent_on_severe():
    agent = Agent()
    action = agent.run(_result(Severity.SEVERE.value))
    assert action.action == ActionType.URGENT_ALERT
    assert set(action.channels) == {"sns", "ses"}


def test_agent_logs_mild():
    agent = Agent()
    action = agent.run(_result(Severity.MILD.value))
    assert action.action == ActionType.LOG_ONLY


def test_agent_requests_approval_on_low_confidence_moderate():
    policy = DecisionPolicy(approval_threshold=0.85)
    agent = Agent(policy=policy)
    action = agent.run(_result(Severity.MODERATE.value, confidence=0.4))
    assert action.action == ActionType.REQUEST_APPROVAL
    assert action.requires_human_approval is True


def test_agent_emails_on_high_confidence_moderate():
    policy = DecisionPolicy(approval_threshold=0.5)
    agent = Agent(policy=policy)
    action = agent.run(_result(Severity.MODERATE.value, confidence=0.9))
    assert action.action == ActionType.NOTIFY_EMAIL
    assert action.requires_human_approval is False