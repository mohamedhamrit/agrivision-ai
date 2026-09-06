"""Build a submission demo video from the built-in sample clips.

Stitches the 4 generated clips (healthy/mild/moderate/severe) into one video
with severity, measured lesion load, and agent action burned in as captions.
Requires no external resources.

Output: sample_data/videos/demo_agrivision.mp4
"""


def main() -> None:
    import json
    import os

    import cv2
    import numpy as np

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    videos = os.path.join(root, "sample_data", "videos")

    with open(os.path.join(videos, "results.json"), encoding="utf-8") as fh:
        results = json.loads(json.load(fh))

    data = {}
    for rec in results:
        key = os.path.basename(rec["source"]).replace(".mp4", "")
        a, act = rec["analysis"], rec["action"]
        data[key] = {
            "severity": a["worst_severity"],
            "ratio": a["avg_lesion_area_ratio"],
            "disease": a["dominant_disease"],
            "action": act["action"],
            "channels": act["channels"],
        }

    ORDER = ["healthy", "mild", "moderate", "severe"]
    COLOR = {
        "healthy": (60, 200, 60),
        "mild": (130, 200, 40),
        "moderate": (40, 140, 230),
        "severe": (40, 40, 230),
    }
    ACTION_LABEL = {
        "no_action": "NO ACTION",
        "log_only": "LOG ONLY",
        "notify_email": "NOTIFY EMAIL",
        "request_approval": "HUMAN APPROVAL",
        "urgent_alert": "URGENT ALERT",
    }

    SIZE, FPS = (640, 640), 10
    SEC_PER_CLIP = 4
    TITLE_SEC = 2

    def blank(bgr=(20, 20, 20)) -> np.ndarray:
        return np.full((SIZE[1], SIZE[0], 3), bgr, dtype=np.uint8)

    def put_text(img, text, y, scale=0.8, color=(255, 255, 255),
                 thickness=2, center=True):
        font = cv2.FONT_HERSHEY_DUPLEX
        (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
        x = (img.shape[1] - w) // 2 if center else 20
        cv2.putText(img, text, (x, y), font, scale, color, thickness,
                    cv2.LINE_AA)

    frames = []

    # ---- Title card ----
    for _ in range(TITLE_SEC * FPS):
        img = blank((16, 34, 46))
        put_text(img, "AgriVision AI", 250, 1.7, (140, 220, 90))
        put_text(img, "Realtime crop disease detection", 340, 0.9)
        put_text(img, "OpenCV 5  -  Agentic Vision  -  AWS", 390, 0.7,
                 (170, 170, 170))
        put_text(img, "perceive  ->  decide  ->  act", 462, 0.7,
                 (90, 190, 250))
        frames.append(img)

    # ---- Clip segments ----
    for key in ORDER:
        clip_path = os.path.join(videos, f"{key}.mp4")
        cap = cv2.VideoCapture(clip_path)
        clip = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            clip.append(frame)
        cap.release()

        info = data[key]
        sev = info["severity"].upper()
        color = COLOR[key]
        ratio_pct = info["ratio"] * 100.0
        action = ACTION_LABEL.get(info["action"], info["action"].upper())
        channels = ", ".join(info["channels"]) if info["channels"] else "none"

        for i in range(SEC_PER_CLIP * FPS):
            frame = clip[i % len(clip)]
            frame = cv2.resize(frame, SIZE, interpolation=cv2.INTER_LINEAR)

            overlay = frame.copy()
            band = frame.copy()
            cv2.rectangle(band, (0, SIZE[1] - 130), (SIZE[0], SIZE[1]),
                          (10, 10, 10), -1)
            frame = cv2.addWeighted(band, 0.55, frame, 0.45, 0)

            put_text(frame, sev, 108, 1.6, color, 4)
            put_text(frame, f"lesion load {ratio_pct:.1f}%   disease "
                            f"{info['disease']}", 178, 0.75,
                     (235, 235, 235))
            put_text(frame, f"AGENT ACTION: {action}", SIZE[1] - 90, 0.95,
                     (70, 200, 255), 3, center=False)
            put_text(frame, f"channels: {channels}", SIZE[1] - 45, 0.6,
                     (170, 170, 170), 1, center=False)
            frames.append(frame)

    # ---- Outro card ----
    for _ in range(TITLE_SEC * FPS):
        img = blank((16, 34, 46))
        put_text(img, "10 seconds of video", 240, 1.1, (235, 235, 235))
        put_text(img, "= a verdict and an action", 300, 1.1, (235, 235, 235))
        put_text(img, "human-in-the-loop for moderate cases", 400, 0.75,
                 (170, 170, 170))
        put_text(img, "AgriVision AI  -  OpenCV AI Competition 2026", 500,
                 0.7, (140, 220, 90))
        frames.append(img)

    out_path = os.path.join(videos, "demo_agrivision.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, FPS, SIZE)
    for frame in frames:
        writer.write(frame)
    writer.release()

    print(f"Wrote {out_path}  ({len(frames) / FPS:.1f}s, {SIZE[0]}x{SIZE[1]}, "
          f"{FPS} fps)")


if __name__ == "__main__":
    main()