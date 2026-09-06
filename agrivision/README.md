# AgriVision AI

Realtime crop disease detection & intelligent alert system.
OpenCV AI Competition 2026 — powered by AWS.

**Team:** AgriVision AI   **Member:** Hamrit Mohamed Lamine
**Focus path:** Agentic Vision (COOL considered)

---

## What it does

A farmer records a short clip of their crop from a smartphone. The video is
analyzed by an **OpenCV 5** pipeline that segments leaf/plant regions, detects
disease lesions, and grades severity. A transfer-learning CNN identifies the
disease (blight / rust / leaf spot). An **autonomous agent** then acts on that
visual evidence:

| Severity | Detected | Agent action | Channels |
|----------|----------|--------------|----------|
| healthy  | no lesions | `no_action` | — |
| mild     | < 30%     | `log_only`       | DynamoDB |
| moderate | 30–60%   | `notify_email` / `request_approval` | SES / human-in-loop |
| severe   | > 60%    | `urgent_alert`   | SNS + SES |

Severity is graded from **OpenCV lesion geometry** (pixel lesion load), not the
CNN label, so even a confident-but-wrong disease label never masks real damage.
The OpenCV 5 output directly changes what the system does next — satisfying the
**perceive → decide → act** Agentic Vision requirement.

---

## Project layout

```
agrivision/
├── __init__.py
├── models.py           # dataclasses & enums
├── sample_data.py      # synthetic image/video generator (no dataset needed)
├── classifier.py       # CNN disease classifier (MobileNetV2 transfer)
├── train_model.py      # transfer-learning training script
├── evaluate.py         # evaluation report (accuracy/F1/severity/latency)
├── app.py              # FastAPI web endpoint
├── demo.py             # end-to-end demo
├── pipeline/           # OpenCV 5 vision pipeline
│   └── __init__.py     #   frame sampling, HSV segmentation, severity scoring
├── agent/              # decision layer
│   └── __init__.py     #   DecisionPolicy + Agent (perceive->decide->act)
├── aws/                # AWS integration (SNS/SES/S3/DynamoDB/Lambda)
│   └── __init__.py
├── models/
│   └── leaf_cnn.h5     # trained transfer-learning classifier
├── tests/
│   └── test_pipeline.py
├── requirements.txt    # pinned base runtime (incl. tensorflow)
└── requirements-web.txt# + fastapi/uvicorn for the web endpoint

sample_data/
├── dataset/            # generated training images (128x128, per class)
└── videos/             # generated demo clips + results.json

deploy/                 # Lambda container (Dockerfile, handler, README)
```

---

## Install & run

```bash
pip install -r agrivision/requirements.txt
pip install -r agrivision/requirements-web.txt   # only for the web endpoint

# End-to-end demo (generates synthetic clips, runs pipeline + agent)
python -m agrivision.demo

# Train the classifier (first run downloads ImageNet weights once)
python -m agrivision.train_model --epochs 12 --images-per-class 1200

# Evaluation report -> sample_data/evaluation_report.json
python -m agrivision.evaluate

# Web endpoint (working web endpoint for the submission)
uvicorn agrivision.app:app --host 0.0.0.0 --port 8000
#   POST /analyze      -> raw video bytes
#   POST /analyze_url  -> {"video_url": "https://..."}
#   GET  /examples     -> run the 4 built-in clips

# Run tests
python -m pytest agrivision/tests -q
```

Run on Python 3.12 with OpenCV 5 (`opencv-python==5.0.0`) and
TensorFlow 2.21 (CPU-only on Windows). boto3 is only needed for real AWS
deployment; the code degrades gracefully to local mode when absent.

---

## OpenCV 5 pipeline

`agrivision/pipeline/__init__.py` implements:
1. **Video frame sampling** — uniform sampling capped at `max_frames`.
2. **HSV color transformation** — robust leaf/plant separation.
3. **Morphological operations** — open/close to clean masks.
4. **Lesion segmentation** — hue-distinct (orange/brown/dark) regions inside
   the dominant plant cluster.
5. **Severity scoring** — lesion area ratio vs. leaf area mapped to
   healthy/mild/moderate/severe (thresholds 0.05 / 0.30 / 0.60).

Disease classification is a frozen-MobileNetV2 transfer-learning head
(128x128, ImageNet backbone) trained on the synthetic generator. All stages are
tunable via `PipelineConfig`.

---

## Measured results (see sample_data/evaluation_report.json)

| Metric | Value |
|--------|-------|
| CNN disease accuracy (4 classes) | 100% |
| CNN macro-F1 | 100% |
| Predictions > 0.9 confidence | 99.9% |
| Severity-grade agreement vs. known load | 100% |
| Lesion ratio MAE (0.07/0.40/0.70 targets) | 0.4–1.2% |
| Pipeline latency (steady state) | ~115 ms/frame |

Severity grading agrees perfectly with the planted lesion load, and disease
labels separate cleanly — while the false-failure path (label disagreement →
`request_approval`) keeps extreme cases answerable by a human.

---

## AWS deployment

The `deploy/` folder contains the serverless container plan (see
`deploy/README.md`). Architecture:

```
Farmer app
   └─> API Gateway ─> S3 (video)
                └─> Lambda (OpenCV 5 + COOL, Graviton/Arm) ─> Agent
                         ├─> SNS/SES (alerts)
                         ├─> DynamoDB (history)
                         └─> CloudWatch (monitoring)
```

The Lambda handler (`deploy/lambda_handler.py`) downloads S3 objects to `/tmp`
before analysis and caches the model across invocations. Run everything locally
with zero AWS dependencies first (`python -m agrivision.demo`),
then build the Graviton container:

```bash
docker build --platform linux/arm64 -f deploy/Dockerfile -t agrivision:latest .
```

---

## Responsible use

Advisory only; human-in-the-loop for moderate/critical cases; farmer data kept
private on AWS; alerts direct users to extension officers rather than
prescribing treatment.