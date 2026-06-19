FROM --platform=linux/arm64 nvcr.io/nvidia/pytorch:25.04-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_CACHE=/root/.cache/huggingface/hub \
    TORCH_HOME=/root/.cache/torch \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /workspace/RobustVGGT

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Keep NVIDIA's preinstalled CUDA/PyTorch stack from the base image and
# install project packages into an isolated venv layered on top of it.
RUN mkdir -p /opt/venv "$HF_HUB_CACHE" "$TORCH_HOME" "$MPLCONFIGDIR" && \
    python -m venv /opt/venv --system-site-packages
ENV PATH=/opt/venv/bin:$PATH

COPY requirements.txt ./requirements.txt

RUN /opt/venv/bin/pip install --upgrade pip setuptools wheel && \
    /opt/venv/bin/pip install matplotlib && \
    /opt/venv/bin/pip install -r requirements.txt

COPY robust_vggt.py ./robust_vggt.py
COPY visualize_demo_result.py ./visualize_demo_result.py
COPY vggt ./vggt
COPY examples ./examples
COPY assets ./assets
COPY README.md ./README.md
COPY README_docker.MD ./README_docker.MD

# Pre-download the only Hugging Face model used by robust_vggt.py into the
# standard cache location so container startup does not need to fetch weights.
RUN python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="facebook/VGGT-1B",
    cache_dir=os.environ["HF_HUB_CACHE"],
    allow_patterns=["*.json", "*.safetensors", "*.txt"],
)
PY

# Build-time smoke test so failures happen during docker build, not runtime.
# `local_files_only=True` guarantees the model is loaded from the image cache.
RUN python - <<'PY'
import matplotlib
import torch
import torchvision
import robust_vggt
from vggt.models.vggt import VGGT

model = VGGT.from_pretrained("facebook/VGGT-1B", local_files_only=True)
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("model_loaded", model.__class__.__name__)
PY

# User args replace CMD only, so `docker run ... --image-dir foo` still runs
# `python robust_vggt.py --image-dir foo` (not `python --image-dir foo`).
ENTRYPOINT ["python", "robust_vggt.py"]
CMD ["--image-dir", "examples/trevi"]
