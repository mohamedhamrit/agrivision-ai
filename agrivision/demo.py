"""End-to-end demo: generate synthetic clips, run OpenCV 5 pipeline,
let the agent decide actions, and save results locally.

Run:
    python -m agrivision.demo
"""

from __future__ import annotations

import json
import os

from .aws import Store, Notifier, PipelineLambda
from .agent import Agent
from .pipeline import OpenCVPipeline
from .sample_data import make_video, get_default_videos_dir

LEVELS = ["healthy", "mild", "moderate", "severe"]
RATIOS = {"healthy": 0.0, "mild": 0.07, "moderate": 0.40, "severe": 0.70}


def main() -> None:
    videos_dir = get_default_videos_dir()
    clips = {}

    print("=== Generating synthetic clips ===")
    for level in LEVELS:
        path = os.path.join(videos_dir, f"{level}.mp4")
        if not os.path.exists(path):
            make_video(RATIOS[level], path, seed=LEVELS.index(level))
        clips[level] = path
        print(f"  {level}: {path}")

    print("\n=== OpenCV 5 pipeline + Agent ===")
    pipeline = OpenCVPipeline()
    store = Store(local_mode=True)
    notifier = Notifier(local_mode=True)


    def notifier_with_log(action):
        # route agent action to AWS notifier channels
        if action.channels:
            for ch in action.channels:
                if ch == "sns":
                    notifier.send_sms(action.message)
                elif ch in ("ses", "ses_email", "email"):
                    notifier.send_email(action.message)
                elif ch == "human_in_the_loop":
                    print(f"  [HUMAN APPROVAL REQUIRED] {action.message}")

    agent = Agent(notifier=notifier_with_log)
    lambda_handler = PipelineLambda(pipeline, agent, store, notifier)

    event = {"objects": list(clips.values())}
    response = lambda_handler.handler(event)

    print("\n=== Results ===")
    for rec in response["body"] if isinstance(response["body"], list) else json.loads(response["body"]):
        src = os.path.basename(rec["source"])
        a = rec["analysis"]
        act = rec["action"]
        print(f"\n  {src}:")
        print(f"    severity = {a['worst_severity']:>10} | frames = {a['frames_analyzed']}")
        print(f"    lesion ratio = {a['avg_lesion_area_ratio']:.2%}")
        print(f"    agent action = {act['action']} | channels = {act['channels']}")
        print(f"    message      = {act['message']}")

    out_path = os.path.join(videos_dir, "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(response["body"], f, indent=2, default=str)
    print(f"\nSaved results to {out_path}")
    print(f"Local store has {store.count()} records, {len(notifier.sent)} notifications sent.")


if __name__ == "__main__":
    main()
