"""AWS integration layer for AgriVision AI.

In a serverless deployment each of these maps to an AWS service:
  - S3            : video/result storage
  - Lambda        : OpenCV 5 pipeline execution (Graviton / Arm)
  - DynamoDB      : detection history
  - SNS / SES     : alerts
  - CloudWatch    : logs & metrics
  - API Gateway   : REST entry point

This module provides optional boto3-backed helpers and graceful degradation
when AWS credentials / SDKs are unavailable (so the pipeline runs locally).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("agrivision.aws")


def _optional_boto3():
    """Import boto3 lazily so the core pipeline runs without AWS dependencies."""
    try:
        import boto3  # type: ignore
        return boto3
    except ImportError:
        return None


class Store:
    """Stores detection results (DynamoDB) with a local fallback."""

    def __init__(self, table_name: Optional[str] = None, local_mode: bool = True) -> None:
        self.table_name = table_name or "agrivision-detections"
        self.local_mode = local_mode
        self._local_store: list = []
        self.boto3 = None if local_mode else _optional_boto3()

    def save(self, record: Dict[str, Any]) -> None:
        if self.local_mode or self.boto3 is None:
            self._local_store.append(record)
            logger.debug("Stored locally: %s", record)
            return
        try:
            table = self.boto3.resource("dynamodb").Table(self.table_name)
            table.put_item(Item=record)
        except Exception as exc:  # pragma: no cover
            logger.warning("DynamoDB write failed (falling back to local): %s", exc)
            self._local_store.append(record)

    def count(self) -> int:
        return len(self._local_store)


class Notifier:
    """Sends alerts via SNS/SES with a local/logging fallback."""

    def __init__(self, local_mode: bool = True,
                 sns_topic: Optional[str] = None,
                 email: Optional[str] = None) -> None:
        self.local_mode = local_mode
        self.sns_topic = sns_topic
        self.email = email or "farmer@example.com"
        self.sent: list = []
        self.boto3 = None if local_mode else _optional_boto3()

    def send_sms(self, message: str, phone: Optional[str] = None) -> None:
        self.sent.append({"channel": "sms", "message": message, "to": phone})
        if not (self.local_mode or self.boto3 is None):
            try:
                self.boto3.client("sns", region_name="eu-west-1").publish(
                    PhoneNumber=phone or "+213000000000", Message=message
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("SNS send failed: %s", exc)

    def send_email(self, message: str, subject: str = "AgriVision AI Alert") -> None:
        self.sent.append({"channel": "email", "subject": subject, "message": message})
        if not (self.local_mode or self.boto3 is None):
            try:
                self.boto3.client("ses", region_name="eu-west-1").send_email(
                    Source=self.email,
                    Destination={"ToAddresses": [self.email]},
                    Message={"Subject": {"Data": subject}, "Body": {"Text": {"Data": message}}},
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("SES send failed: %s", exc)


class PipelineLambda:
    """Mimics the Lambda handler that runs the OpenCV 5 pipeline.

    In real deployment this is the entry point triggered by API Gateway with
    an S3 object key. It loads the video from S3, runs OpenCVPipeline, runs the
    agent, and persists the result.
    """

    def __init__(self, pipeline, agent, store=None, notifier=None) -> None:
        self.pipeline = pipeline
        self.agent = agent
        self.store = store or Store(local_mode=True)
        self.notifier = notifier or Notifier(local_mode=True)

    def handler(self, event: Dict[str, Any], context=None) -> Dict[str, Any]:
        """Lambda-style handler. Event: {objects: [url_or_path, ...]}"""
        objects = event.get("objects") or event.get("videos") or []
        results = []
        for obj in objects:
            analysis = self.pipeline.analyze_video(obj)
            action = self.agent.run(analysis)
            record = {
                "source": obj,
                "analysis": analysis.to_dict(),
                "action": action.to_dict(),
                "store_key": "detections/" + str(len(self.store._local_store)),
            }
            self.store.save(record)
            results.append(record)
        return {"statusCode": 200, "body": json.dumps(results, default=str)}
