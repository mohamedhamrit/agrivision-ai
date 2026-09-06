# About AgriVision AI

*Crop disease detection and intelligent alerts — OpenCV AI Competition 2026*

**Team:** AgriVision AI &nbsp;|&nbsp; **Member:** Hamrit Mohamed Lamine
**Focus path:** Agentic Vision

---

## What inspired me

In large parts of the world, a farmer notices a crop failing only when the
damage is already visible across a whole field. By then, the treatment window
has usually closed. Agricultural extension officers are scarce, and most
smallholders can't tell early blight from rust or a plain mineral deficiency —
yet the two need completely different responses.

What I kept coming back to was the fact that almost every farmer now owns a
smartphone. The camera is there. The compute is there. What is missing is a
system that *looks* at the plant, decides honestly what it sees, and tells the
farmer what to do next — before the disease wins.

So the goal became: **record ten seconds of your crop, get a verdict and an
action in seconds, with a human in the loop when it matters.** Not a
"Diagnosis Detector" website, but an agentic system: it *perceives* with a
computer-vision pipeline, *decides* under a policy, and *acts* through alert
channels.

---

## How I built it

### 1. The vision pipeline (OpenCV 5) — *perceive*

The pipeline (`agrivision/pipeline/__init__.py`) reads a video and reduces it
to a single scalar — the **lesion load** — plus a disease label.

1. **Frame sampling** — videos are uniformly sampled, capped at `max_frames`,
   so input cost is bounded regardless of clip length.
2. **HSV transformation** — plants are separated from the soil/background in
   hue-saturation space instead of raw BGR. Green leaf tissue lives in a tight,
   illumination-robust hue band.
3. **Morphology** — open/close operations strip sensor noise and join split
   regions.
4. **Plant clustering** — green leaf and lesion-colored pixels are unioned and
   reduced to the *largest connected component*, so stray background pixels that
   merely resemble a lesion never inflate the diagnosis.
5. **Lesion segmentation + severity scoring** — the hue-distinct (orange /
   brown / dark) regions *inside* that plant cluster define the disease damage.

The severity grade is a pure geometric measurement. If a leaf has area
$A_{\text{plant}} = \sum_{(i,j)} m_{\text{plant}}(i,j)$ and lesions occupy
$A_{\text{lesion}} = \sum_{(i,j)} m_{\text{lesion}}(i,j)$, the lesion ratio is

$$
r \;=\; \frac{A_{\text{lesion}}}{A_{\text{plant}}},
$$

which maps to a four-point grade through tunable thresholds:

$$
S(r) \;=\;
\begin{cases}
\text{healthy}  & r < 0.05 \\[2pt]
\text{mild}     & 0.05 \le r < 0.30 \\[2pt]
\text{moderate} & 0.30 \le r < 0.60 \\[2pt]
\text{severe}   & r \ge 0.60.
\end{cases}
$$

### 2. The disease classifier (CNN) — *name it*

`agrivision/classifier.py` classifies the leaf image into
`healthy / blight / rust / leaf_spot` with a **MobileNetV2 transfer-learning**
head. The backbone is frozen with ImageNet weights; only a
GlobalAveragePooling → Dropout → Dense(softmax) head is trained on synthetic
leaf images, with inputs rescaled to the $[-1, 1]$ range MobileNetV2 expects.

### 3. The agent — *decide and act*

`agrivision/agent/__init__.py` implements the **perceive → decide → act** loop.
The *OpenCV-derived geometry* drives the decision, not the learned label:

| Severity | Detected | Agent action | Channels |
|----------|----------|--------------|----------|
| healthy  | no lesions       | `no_action`       | — |
| mild     | $r < 0.30$       | `log_only`        | DynamoDB |
| moderate | $0.30 \le r < 0.60$ | `notify_email` / `request_approval` | SES / human-in-loop |
| severe   | $r \ge 0.60$     | `urgent_alert`    | SNS + SES |

A confidence gate adds the safety valve: a moderate case below the approval
threshold $c < 0.85$ becomes `request_approval`, pausing for a human instead of
mailing a possibly-wrong verdict.

### 4. Web + serverless

- **Web** (`agrivision/app.py`): a FastAPI endpoint — `POST /analyze`,
  `POST /analyze_url`, `GET /examples`.
- **Serverless** (`deploy/`): an AWS Lambda container (Graviton/arm64) wrapping
  the same pipeline, S3-triggered, with SNS/SES/DynamoDB side effects. All of it
  degrades gracefully to a fully offline `local_mode`.

### 5. Testing without a dataset

To make development reproducible and submission-friendly, `sample_data.py`
synthesizes leaf scenes with **exactly controlled** lesion coverage — blight
blotches, rust pustules, dark leaf-spot rings — so both training and evaluation
need no external download.

---

## What I learned

- **RGB is fragile for plants; HSV is not.** Green under shadow, mid-day sun,
  or phone auto-white-balance shifts wildly in BGR but stays put in hue space.
  Thresholding become a *semantic* choice ("what counts as leaf, what counts as
  lesion") instead of a pixel-value guess.
- **Geometry beats confidence for severity.** A CNN can be confidently wrong.
  Grading severity from pixel lesion load means the *most critical* output the
  agent acts on never depends on a classifier whim. The model's job is naming
  the disease; OpenCV's job is measuring the damage.
- **When the two disagree, say so.** If lesions are clearly present but the
  network says "healthy," forcing one answer masks real damage. The pipeline
  instead downgrades that frame to `unknown` and caps its confidence — directing
  the case to human review rather than to silence.
- **Transfer learning is the honest default on small data.** Training a deep
  CNN from scratch on a small generated set collapses: training accuracy
  overfits toward 100% while validation stays near random. Freezing an ImageNet
  backbone made held-out accuracy jump to near-perfect — a lesson in never
  fitting a model with more capacity than the data justifies.
- **An agent is only as good as its failure mode.** The interesting design
  work wasn't the happy path (severe → alert) but the middle — uncertain
  moderate cases, contradictory signals, and how to *pause* and ask a human.

---

## Challenges I faced

- **Small-data CNN collapse.** Training a deep model from scratch on
  ~4,800 generated images massively overfit. The fix (a frozen MobileNetV2
  backbone with a light head and early stopping, plus L2 on the head) turned
  train-vs-validation valley into near-perfect held-out accuracy.
- **Background noise faking lesions.** Neutral gray soil occasionally fell
  inside the lesion hue band. The answer was structural: restrict lesions to the
  *largest connected plant component*, so detached noise pixels could never be
  counted as disease.
- **"Healthy label, lesions present" contradictions.** Naively trusting the
  CNN could label a damaged leaf healthy. Resolving it required an explicit
  `unknown` state and a confidence cap — and redesigning the agent policy to
  treat `unknown` as requiring a human.
- **Windows + AWS friction.** TensorFlow GPU isn't available on native
  Windows, and boto3 may be absent. The whole stack was made to run CPU-only and
  degrade to local mode, so the demo and tests work identically with zero cloud
  credentials.
- **Keeping it honest.** Synthetic data can make metrics look perfect; the
  evaluation harness therefore measures *agreement with planted lesion loads*,
  regression error, and latency, and the README clearly distinguishes synthetic
  results from a real PlantVillage-grade production model.

---

## Measured results

| Metric | Value |
|--------|-------|
| CNN disease accuracy (4 classes) | 100% |
| CNN macro-F1 | 100% |
| Predictions > 0.9 confidence | 99.9% |
| Severity-grade agreement vs. known load | 100% |

Precision / recall / F1 per class and MAE / RMSE of the lesion-ratio estimator
per severity bucket are all in `sample_data/evaluation_report.json`:

$$
\text{MAE} = \frac{1}{N}\sum_i \lvert \hat r_i - r_i \rvert,
\qquad
\text{F1}_c = \frac{2\,P_c\,R_c}{P_c + R_c}.
$$

---

## What's next

- Train on a real field dataset (e.g. PlantVillage) and validate against actual
  phone footage captured in varied light.
- Benchmark the pipeline with the COOL library on Graviton (arm64) vs. an x86
  baseline for the COOL award.
- Deploy the Lambda container, wire Step Functions for the human-approval step,
  and add season-long trend alerts from DynamoDB histories.

---

*From a phone camera to a recommended action in seconds — that's the distance
AgriVision AI tries to shorten, one field at a time.*