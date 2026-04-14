# ---------------------------------------------------------------------------
# VADAR Visual Agent – Docker image
#
# CPU-only build (suitable for inference without a GPU).
# For CUDA support, replace the base image with the appropriate
# pytorch/pytorch:<version>-cuda<XX.X>-cudnn<Y>-runtime image.
#
# Build:
#   docker build -t vadar-visual-agent .
#
# Run (verify environment):
#   docker run --rm -e OPENAI_API_KEY=sk-... vadar-visual-agent --verify-only
#
# Run (synthetic demo – no API key required):
#   docker run --rm vadar-visual-agent --demo
#
# Run on a local image:
#   docker run --rm \
#     -e OPENAI_API_KEY=sk-... \
#     -v /path/to/images:/data \
#     vadar-visual-agent --image /data/photo.jpg \
#                        --question "Which object is closest?"
# ---------------------------------------------------------------------------

FROM python:3.11-slim

# System packages needed by OpenCV and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (CPU torch wheel – much smaller than CUDA)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY vadar_agent.py evaluate_benchmark.py quickstart.py config.yaml ./

# Pre-download HuggingFace models into the image cache so the container
# starts quickly (remove this block to keep the image smaller and download
# models on first run instead).
# RUN python -c "\
# from transformers import pipeline; \
# pipeline('object-detection', model='facebook/detr-resnet-50'); \
# pipeline('depth-estimation', model='Intel/dpt-large'); \
# pipeline('image-segmentation', model='facebook/detr-resnet-50-panoptic')"

# Create output directory
RUN mkdir -p /app/outputs

ENTRYPOINT ["python", "quickstart.py"]
CMD ["--demo"]
