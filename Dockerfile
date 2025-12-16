# Python base image (stable)
FROM python:3.10-slim

# Prevent tzdata from asking questions
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Phnom_Penh

# Set working directory
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    tzdata \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements
COPY requirements.txt .

# Install Python dependencies (CPU PyTorch)
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create folders
RUN mkdir -p \
    data/raw \
    data/processed/train/images data/processed/train/labels \
    data/processed/val/images data/processed/val/labels \
    data/processed/test/images data/processed/test/labels \
    models/original models/enhanced \
    results/metrics results/visualizations

# Make pipeline executable
RUN chmod +x /app/script/pipeline.sh

# Environment
ENV PYTHONUNBUFFERED=1
ENV OPENCV_IO_ENABLE_OPENEXR=1

# Run pipeline
CMD ["/app/script/pipeline.sh"]
