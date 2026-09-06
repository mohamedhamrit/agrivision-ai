"""Autonomous agent that turns OpenCV 5 visual evidence into actions.

Implements the Agentic Vision perception-decision-action loop described in the
proposal. The OpenCV 5 detection output (severity, confidence, lesion ratio)
drives what the system does next.

Action types:
  - NO_ACTION      : healthy - nothing required, optionally log.
  - LOG_ONLY       : low threat - store for season-long tracking.
  - NOTIFY_EMAIL   : moderate - send advisory email.
  - REQUEST_APPROVAL : moderate - propose action, request human confirmation.
  - URGENT_ALERT   : severe - send SMS/urgent alert immediately.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Callable

from ..models import Severity, VideoAnalysisResult, DetectionResult

logger = logging.getLogger("agrivision.agent")


class ActionType:
    NO_ACTION = "no_action"
    LOG_ONLY = "log_only"
    NOTIFY_EMAIL = "notify_email"
    REQUEST_APPROVAL = "request_approval"
    URGENT_ALERT = "urgent_alert"
    UNKNOWN = "unknown"


@dataclass
class AgentAction:
    """An action decided by the agent."""

    action: str = ActionType.UNKNOWN
    severity: str = "unknown"
    reason: str = ""
    channels: List[str] = field(default_factory=list)
    requires_human_approval: bool = False
    message: str = ""
    decision_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DecisionPolicy:
    """Maps severity/confidence to an agent action (the decision logic)."""

    def __init__(self,
                 approval_threshold: float = 0.85,
                 low_confidence_max_action: str = ActionType.REQUEST_APPROVAL) -> None:
        self.approval_threshold = approval_threshold
        self.low_confidence_max_action = low_confidence_max_action

    def decide(self, result: VideoAnalysisResult) -> AgentAction:
        severity = result.worst_severity
        confidence = self._aggregate_confidence(result)
        avg = result.mean_severity_score

        # Base message tailored to severity.
        message = self._message_for(severity, avg)

        if severity == Severity.HEALTHY.value:
            return AgentAction(
                action=ActionType.NO_ACTION,
                severity=severity,
                reason="No disease detected; plant appears healthy.",
                channels=[],
                requires_human_approval=False,
                message=message,
                decision_metadata={"avg_severity": avg, "confidence": confidence},
            )

        if severity == Severity.MILD.value:
            return AgentAction(
                action=ActionType.LOG_ONLY,
                severity=severity,
                reason="Low lesion load; log for season-long tracking.",
                channels=["dynamodb"],
                requires_human_approval=False,
                message=message,
                decision_metadata={"avg_severity": avg, "confidence": confidence},
            )

        if severity == Severity.MODERATE.value:
            # Request human approval unless very confident, in which case email.
            if confidence >= self.approval_threshold:
                return AgentAction(
                    action=ActionType.NOTIFY_EMAIL,
                    severity=severity,
                    reason="Moderate threat with high confidence; notify via email.",
                    channels=["ses"],
                    requires_human_approval=False,
                    message=message,
                    decision_metadata={"avg_severity": avg, "confidence": confidence},
                )
            return AgentAction(
                action=ActionType.REQUEST_APPROVAL,
                severity=severity,
                reason="Moderate threat with uncertain confidence; request human approval.",
                channels=["human_in_the_loop"],
                requires_human_approval=True,
                message=message + " Human confirmation requested before action.",
                decision_metadata={"avg_severity": avg, "confidence": confidence},
            )

        # Severe
        return AgentAction(
            action=ActionType.URGENT_ALERT,
            severity=severity,
            reason="Severe disease detected; send urgent SMS/email alert.",
            channels=["sns", "ses"],
            requires_human_approval=False,
            message=message,
            decision_metadata={"avg_severity": avg, "confidence": confidence},
        )

    @staticmethod
    def _aggregate_confidence(result: VideoAnalysisResult) -> float:
        if not result.detections:
            return 0.0
        total = 0.0
        n = 0
        for d in result.detections:
            # detections are dicts after to_dict()
            if isinstance(d, dict):
                total += float(d.get("confidence", 0.0))
            else:
                total += float(d.confidence)
            n += 1
        return total / n if n else 0.0

    @staticmethod
    def _message_for(severity: str, avg: float) -> str:
        mapping = {
            Severity.HEALTHY.value: "Crop inspected - no disease detected.",
            Severity.MILD.value: (
                f"Early-stage symptoms detected (lesion ratio {avg:.2%}). "
                "Monitor and consider preventive treatment."
            ),
            Severity.MODERATE.value: (
                f"Moderate disease detected (lesion ratio {avg:.2%}). "
                "Treatment recommended within 48 hours."
            ),
            Severity.SEVERE.value: (
                f"SEVERE disease detected (lesion ratio {avg:.2%}). "
                "Immediate intervention recommended. Contact an extension officer."
            ),
        }
        return mapping.get(severity, "Analysis completed.")


class Agent:
    """The orchestration agent. In production, decision/action effects call
    AWS (SNS, SES, DynamoDB). Here effects are injected hookable functions so
    the loop is testable and portable."""

    def __init__(self,
                 policy: Optional[DecisionPolicy] = None,
                 notifier: Optional[Callable[[AgentAction], None]] = None) -> None:
        self.policy = policy or DecisionPolicy()
        self.notifier = notifier or self._default_notifier
        self.trace: List[AgentAction] = []

    def _default_notifier(self, action: AgentAction) -> None:
        if action.action in (ActionType.NOTIFY_EMAIL, ActionType.URGENT_ALERT):
            logger.info("Sending notification [%s] -> %s", action.action, action.message)

    def run(self, result: VideoAnalysisResult) -> AgentAction:
        """Perception already produced `result`; now decide and act."""
        action = self.policy.decide(result)
        action.decision_metadata["ts"] = time.time()
        # Act (this would invoke AWS resources in deployment)
        self._act(action)
        self.trace.append(action)
        return action

    def _act(self, action: AgentAction) -> None:
        if action.requires_human_approval:
            logger.info("Agent paused - awaiting human approval.")
            return
        self.notifier(action)
