FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive

RUN sed -i 's/main/main contrib non-free/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && \
    apt-get install -y \
    curl \
    git \
    build-essential \
    unrar \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

# ===== CÀI PYTORCH CPU VERSION =====
# PyTorch 2.6 CPU-only version
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cpu

RUN python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

WORKDIR /app

COPY . /app

RUN if [ -f "requirements.txt" ]; then pip install --no-cache-dir -r requirements.txt; fi

RUN mkdir -p /app/uploads && \
    chmod 777 /app/uploads

ENV HF_HOME=/app/model_cache
RUN mkdir -p /app/model_cache && chmod 777 /app/model_cache

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8011"]
