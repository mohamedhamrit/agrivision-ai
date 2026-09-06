"""AWS Lambda handler entry point for AgriVision AI.

Wraps the shared PipelineLambda handler so the container image is a standard
Lambda target. Event shape (from API Gateway / S3):

    { "objects": ["s3://bucket/videos/healthy.mp4", ...] }

or the standard S3 event record:

    {"Records": [{"s3": {"bucket": {"name": ...}, "object": {"key": ...}}}, ...]}

S3 objects are downloaded to /tmp first (OpenCV cannot open s3:// URLs), and
the pipeline/model are cached across invocations for fast warm starts.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, Iterable, List

from agrivision.agent import Agent
from agrivision.aws import Notifier, PipelineLambda, Store
from agrivision.pipeline import OpenCVPipeline

_SUPPORTED_EXT = (".mp4", ".mov", ".mkv", ".avi")

_pipeline = None
_s3 = None


def _get_pipeline() -> OpenCVPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = OpenCVPipeline()
    return _pipeline


def _s3_client():
    global _s3
    if _s3 is None:
        import boto3

        _s3 = boto3.client("s3")
    return _s3


def _to_local(path: str) -> str:
    """Download an s3:// object to /tmp; pass local paths through untouched."""
    if not path.startswith("s3://"):
        return path  # local file (used in offline tests/demos)
    bucket, key = path[len("s3://"):].split("/", 1)
    ext = os.path.splitext(key)[1]
    suffix = ext if ext.lower() in _SUPPORTED_EXT else ".mp4"
    fd, local = tempfile.mkstemp(prefix="agrivision_", suffix=suffix)
    os.close(fd)
    _s3_client().download_file(bucket, key, local)
    return local


def _object_list(event: Dict[str, Any]) -> List[str]:
    if "Records" in event:  # native S3 trigger
        return [
            f"s3://{r['s3']['bucket']['name']}/{r['s3']['object']['key']}"
            for r in event["Records"]
            if r.get("eventSource") == "aws:s3"
        ]
    return event.get("objects") or event.get("videos") or event.get("records") or []


def handler(event: Dict[str, Any], context=None) -> Dict[str, Any]:
    """Lambda entry point."""
    local_mode = os.getenv("LOCAL_MODE", "1") == "1"
    pipeline = _get_pipeline()
    store = Store(local_mode=local_mode,
                  table_name=os.getenv("DYNAMO_TABLE", "agrivision-detections"))
    notifier = Notifier(local_mode=local_mode)


    def notify(action) -> None:
        for ch in action.channels:
            if ch == "sns":
                notifier.send_sms(action.message)
            elif ch in ("ses", "email"):
                notifier.send_email(action.message)

    agent = Agent(notifier=notify)
    lam = PipelineLambda(pipeline, agent, store, notifier)

    local_paths = [_to_local(obj) for obj in _object_list(event)]
    return lam.handler({"objects": local_paths}, context)