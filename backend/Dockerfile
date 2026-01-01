# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for OpenCV and wget for downloads
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install CPU-only PyTorch first (saves ~1.5GB vs GPU version)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create models directory and download ML models from HuggingFace
RUN mkdir -p models && \
    wget -q --show-progress -O models/model_84_acc_10_frames_final_data.pt "https://huggingface.co/Devanshu2025/Deepfake-video-detection/resolve/main/model_84_acc_10_frames_final_data.pt" && \
    wget -q --show-progress -O models/model_90_acc_20_frames_FF_data.pt "https://huggingface.co/Devanshu2025/Deepfake-video-detection/resolve/main/model_90_acc_20_frames_FF_data.pt" && \
    wget -q --show-progress -O models/model_95_acc_40_frames_FF_data.pt "https://huggingface.co/Devanshu2025/Deepfake-video-detection/resolve/main/model_95_acc_40_frames_FF_data.pt" && \
    wget -q --show-progress -O models/model_97_acc_60_frames_FF_data.pt "https://huggingface.co/Devanshu2025/Deepfake-video-detection/resolve/main/model_97_acc_60_frames_FF_data.pt" && \
    wget -q --show-progress -O models/model_97_acc_80_frames_FF_data.pt "https://huggingface.co/Devanshu2025/Deepfake-video-detection/resolve/main/model_97_acc_80_frames_FF_data.pt" && \
    wget -q --show-progress -O models/model_97_acc_100_frames_FF_data.pt "https://huggingface.co/Devanshu2025/Deepfake-video-detection/resolve/main/model_97_acc_100_frames_FF_data.pt"

# Expose the port Render will use
EXPOSE 8000

# Run the application with the PORT environment variable from Render
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
