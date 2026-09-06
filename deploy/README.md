# AWS Deployment — AgriVision AI

Serverless architecture for the OpenCV 5 + Agentic Vision pipeline.

## Architecture

```
[ Farmer Mobile App ]
        |  HTTPS upload
        v
[ Amazon API Gateway ]  (REST endpoint, auth via Cognito)
        |  PUT object
        v
[ Amazon S3 ]  (agrivision-uploads bucket, videos + results)
        |
        |  put-object event
        v
[ AWS Lambda ]  container image, runs OpenCV 5 (+ COOL), Graviton (Arm)
   OCV5 pipeline: frame sampling -> HSV -> morphology -> lesions -> severity
        |
        v
[ Agent Layer ]  (in Lambda; LangChain or custom)
   decide: no_action | log_only | notify_email | request_approval | urgent_alert
        |
        +--[ moderate/low-confidence ]--> human approval (Step Functions)
        |
        v
[ Amazon SNS ] -> SMS            [ Amazon SES ] -> email
[ Amazon DynamoDB ] detection history
[ Amazon CloudWatch ] logs + metrics + cost
```

## Services & why

| Service | Role |
|---------|------|
| API Gateway | Secure serverless entry point |
| S3 | Video/result object storage |
| Lambda (Graviton) | Runs the OpenCV 5 pipeline; **COOL** evaluated on Arm |
| Step Functions (optional) | Human-in-the-loop approval workflow |
| SNS | SMS/urgent alerts |
| SES | Email alerts |
| DynamoDB | Detection history & trend analysis |
| CloudWatch | Observability, metrics, cost |

## Steps

1. **Create an S3 bucket** for uploads and set lifecycle rules.
2. **Build the Lambda container** targeting `arm64` (Graviton):

   ```bash
   docker build --platform linux/arm64 -f deploy/Dockerfile -t agrivision:latest .
   aws ecr create-repository --repository-name agrivision --region eu-west-1
   aws ecr get-login-password | docker login --username AWS --password-stdin <acct>.dkr.ecr.eu-west-1.amazonaws.com
   docker tag agrivision:latest <acct>.dkr.ecr.eu-west-1.amazonaws.com/agrivision:latest
   docker push <acct>.dkr.ecr.eu-west-1.amazonaws.com/agrivision:latest
   ```

   Dependencies are the pinned `agrivision/requirements.txt`; `tensorflow==2.21.0`
   ships official aarch64 (Graviton) CPU wheels on PyPI.

3. **Create the Lambda function** (`Architecture=arm64`, image from ECR). Grant
   it S3 read, DynamoDB read/write, SNS publish, SES send.
4. **Wire the S3 event** → Lambda trigger (filter on `videos/` prefix).
5. **Create DynamoDB table** `agrivision-detections` (key: `source`).
6. **Set up SNS topic + phone/email subscriptions**; configure SES verified
   domain.
7. **Step Functions approval flow** for moderate-low-confidence decisions:
   an `ActionTaken` activity awaits a human reviewer.
8. **CloudWatch alarms** on Lambda errors, duration, and invocation counts.

## Lambda handler

The container entry point is `deploy/lambda_handler.py` (wraps
`agrivision.aws.PipelineLambda.handler`). It:
- downloads each `s3://` object to `/tmp` (OpenCV cannot open `s3://` URLs),
- runs `OpenCVPipeline.analyze_video`,
- runs `Agent.run` (which routes to SNS/SES/DynamoDB via `Store`/`Notifier`),
- persists a detection record.
- caches the model/pipeline across warm invocations.

Pass `local_mode=True` (env `LOCAL_MODE=1`) to stay fully offline during
development.

## Notes

- Run the code **locally first** (`python -m agrivision.demo`) — it needs no AWS
  credentials.
- For COOL: bundle the COOL library in the image and call its Arm-optimized
  ops from the OpenCV 5 pipeline on Graviton; report latency/throughput vs. an
  x86 baseline for the COOL award.
- Keep dependencies reproducible via `Dockerfile` + pinned `requirements.txt`.