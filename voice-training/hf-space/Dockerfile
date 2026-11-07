FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/voice-data/hf-cache \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-dev git git-lfs ffmpeg wget \
    ca-certificates libsndfile1 libgomp1 build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch 2.4.0 and torchaudio 2.4.0 are a matched CUDA 12.1 set.
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install \
      --index-url https://download.pytorch.org/whl/cu121 \
      torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install -r /app/requirements.txt

# Bake the three official codebases into the Space container once.
RUN git clone --depth 1 https://github.com/tbenitz/IRL-Voice-Changer.git /app/irl-voice && \
    git clone --depth 1 https://github.com/Plachtaa/seed-vc.git /app/seed-vc && \
    git clone --depth 1 https://github.com/KoeAI/LLVC.git /app/llvc-upstream

COPY app.py /app/app.py

RUN useradd -m -u 1000 user && \
    mkdir -p /home/user/voice-data/voice-jobs /home/user/voice-data/voice-cache /home/user/voice-data/hf-cache /home/user/voice-data/tmp && \
    chown -R user:user /home/user/voice-data /app/app.py /app/irl-voice /app/seed-vc /app/llvc-upstream

ENV HOME=/home/user TMPDIR=/home/user/voice-data/tmp

USER user
EXPOSE 7860
CMD ["python3", "/app/app.py"]
