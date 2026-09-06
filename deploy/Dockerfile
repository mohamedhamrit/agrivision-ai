# Lambda container image for AgriVision AI (Arm / Graviton).
# Build:  docker build --platform linux/arm64 -t agrivision:latest .
FROM python:3.12-slim AS runtime

# System libs needed by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY agrivision/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY agrivision/ ./agrivision/

# AWSLAMBDA runtime (only sets the entrypoint wrapper)
FROM public.ecr.aws/lambda/python:3.12

COPY --from=runtime /app /var/task
COPY deploy/lambda_handler.py /var/task/lambda_handler.py

CMD [ "lambda_handler.handler" ]