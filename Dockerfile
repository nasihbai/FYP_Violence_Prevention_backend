# ─── Violence Detection Backend ───────────────────────────────────────────────
# Python 3.11 required: TF 2.17 + MediaPipe don't have wheels for 3.12/3.13.
# GPU detection is not available inside this container (no CUDA runtime).
# For live detection with GPU, run the backend natively and use Docker only
# for the database (postgres) and frontend.
FROM python:3.11-slim

WORKDIR /app

# System libs needed by OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch before the rest of requirements.txt.
# The main requirements.txt doesn't pin a torch source so pip would
# try to pull the CUDA wheel (multi-GB) — we override it here.
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose Flask port
EXPOSE 5000

# Run the web dashboard only (no live camera feed in container).
# Scene classifier is disabled because the 345 MB model.safetensors
# is not tracked in git — mount models/ as a volume if you need it.
CMD ["python", "run_detection.py", \
     "--web", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--no-scene-classifier"]
