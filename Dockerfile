# Use official PyTorch runtime with CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (cache friendly)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p \
    data/raw \
    data/processed/train/images data/processed/train/labels \
    data/processed/val/images data/processed/val/labels \
    data/processed/test/images data/processed/test/labels \
    models/original models/enhanced \
    results/metrics results/visualizations

# Make pipeline executable
RUN chmod +x /app/script/pipeline.sh

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV OPENCV_IO_ENABLE_OPENEXR=1

# Auto-run pipeline
CMD ["/app/script/pipeline.sh"]
