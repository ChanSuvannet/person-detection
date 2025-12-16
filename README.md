# Person Detection with Open Images Dataset

## Project Overview

This project implements single-class object detection for the "Person" class using the Open Images Dataset. The project compares model performance between original and quality-enhanced datasets.

## Project Objectives

1. Download and preprocess Person class images from Open Images Dataset
2. Implement data quality enhancement techniques
3. Train YOLOv8 models on both original and enhanced datasets
4. Compare and evaluate model performance
5. Provide comprehensive analysis and recommendations

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- NVIDIA GPU with CUDA support (recommended)
- At least 20GB free disk space

### Step 1: Clone and Setup

```bash
# Clone or extract the project
cd person-detection

# Build Docker container
docker-compose build
```

### Step 2: Run the Complete Pipeline

```bash
# Option A: Run everything sequentially (recommended for first time)
docker-compose run app bash run_pipeline.sh

# Option B: Run individual steps
docker-compose run app python src/data_collection.py
docker-compose run app python src/data_preprocessing.py
docker-compose run app python src/data_quality.py
docker-compose run app python src/train.py --data original
docker-compose run app python src/train.py --data enhanced
docker-compose run app python src/evaluate.py
```

### Data Collection & Preprocessing

**Tasks:**

1. Download Person class images from Open Images Dataset
2. Remove duplicate images using perceptual hashing
3. Remove unlabeled or invalid images
4. Validate bounding box coordinates
5. Split into train/val/test sets (70%/15%/15%)
6. Convert annotations to YOLO format

**Commands:**

```bash
docker-compose run app python src/data_collection.py
docker-compose run app python src/data_preprocessing.py
```

**Expected Output:**

- ~1500 images downloaded and organized
- Train: ~1050 images, Val: ~225 images, Test: ~225 images
- YOLO format labels created

---

### Quality Enhancement & Initial Training

**Tasks:**

1. Analyze image quality (brightness, contrast, sharpness)
2. Apply enhancement techniques:
   - Brightness adjustment (gamma correction)
   - Contrast enhancement (CLAHE)
   - Image sharpening (unsharp masking)
3. Start training on original dataset
4. Start training on enhanced dataset

**Commands:**

```bash
# Enhance data quality
docker-compose run app python src/data_quality.py

# Train on original data (will run for ~2-4 hours)
docker-compose run app python src/train.py --data original

# Train on enhanced data (parallel or sequential)
docker-compose run app python src/train.py --data enhanced
```

**Expected Output:**

- Enhanced dataset created
- Training progress visible with metrics
- Model checkpoints saved every 10 epochs

---

### Evaluation & Analysis**

**Tasks:**

1. Evaluate both models on test set
2. Generate performance metrics (mAP, Precision, Recall)
3. Create detection visualizations
4. Analyze errors (false positives/negatives)
5. Compare original vs enhanced models

**Commands:**

```bash
# Run complete evaluation
docker-compose run app python src/evaluate.py

# Generate additional visualizations if needed
docker-compose run app python src/utils.py
```

**Expected Output:**

- Performance metrics for both models
- Comparison plots and tables
- Detection visualizations
- Error analysis reports

## Key Features

### Data Quality Enhancements

1. **Brightness Adjustment**
   - Uses gamma correction to normalize lighting
   - Targets average brightness of 128 (0-255 scale)

2. **Contrast Enhancement**
   - CLAHE (Contrast Limited Adaptive Histogram Equalization)
   - Improves local contrast without over-amplifying noise

3. **Sharpening**
   - Unsharp masking technique
   - Enhances edges and fine details

### Model Training

- **Architecture:** YOLOv8 (nano, small, or medium variants)
- **Optimizer:** Adam with adaptive learning rate
- **Data Augmentation:** HSV, translation, scaling, flipping, mosaic
- **Early Stopping:** Patience of 20 epochs
- **Validation:** Continuous monitoring on validation set

### Evaluation Metrics

- **mAP@0.5:** Mean Average Precision at 0.5 IoU threshold
- **mAP@0.5:0.95:** Mean Average Precision averaged over IoU 0.5-0.95
- **Precision:** True Positives / (True Positives + False Positives)
- **Recall:** True Positives / (True Positives + False Negatives)
- **F1-Score:** Harmonic mean of Precision and Recall

## Configuration

Edit `config/config.yaml` to customize:

```yaml
# Dataset settings
dataset:
  total_images: 1500
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15

# Quality thresholds
quality:
  min_brightness: 30
  max_brightness: 225
  min_contrast: 20
  blur_threshold: 100

# Training parameters
training:
  model: "yolov8n.pt"  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt
  epochs: 100
  batch_size: 16
  img_size: 640
  patience: 20
```

## Expected Results

### Training Time (approximate)

- **YOLOv8 Nano:** 2-3 hours on GPU, 8-10 hours on CPU
- **YOLOv8 Small:** 3-4 hours on GPU, 12-15 hours on CPU
- **YOLOv8 Medium:** 4-6 hours on GPU, 20-25 hours on CPU

### Performance Expectations

**Original Dataset:**

- mAP@0.5: 0.60-0.70
- Precision: 0.65-0.75
- Recall: 0.60-0.70

**Enhanced Dataset (Expected Improvement):**

- mAP@0.5: 0.65-0.75 (+5-10%)
- Precision: 0.70-0.80 (+5-10%)
- Recall: 0.65-0.75 (+5-10%)

## Troubleshooting

### Common Issues

**1. Out of Memory Error**

```bash
# Reduce batch size in config/config.yaml
training:
  batch_size: 8  # Instead of 16
```

**2. CUDA Not Available**

```bash
# Check GPU access
docker-compose run app python -c "import torch; print(torch.cuda.is_available())"

# If False, training will use CPU (slower but works)
```

**3. Download Failures**

```bash
# Re-run data collection
docker-compose run app python src/data_collection.py

# Downloads will resume from where they stopped
```

**4. Docker Build Errors**

```bash
# Clear Docker cache and rebuild
docker-compose down
docker system prune -a
docker-compose build --no-cache
```


## Learning Outcomes

After completing this project, you will understand:

- Open Images Dataset structure and usage
- Image preprocessing and quality enhancement techniques
- YOLO object detection architecture
- Model training, validation, and evaluation
- Performance metrics interpretation
- Comparative analysis methodology

## References

- [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [CLAHE Algorithm](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html)
- [Object Detection Metrics](https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173)