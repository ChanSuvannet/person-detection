"""
Utility Functions
Helper functions used across different modules
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml


def load_config(config_path="config/config.yaml"):
    """
    Load configuration from YAML file

    Args:
        config_path: Path to configuration file

    Returns:
        Dictionary with configuration
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_image(image_path):
    """
    Load image from file path

    Args:
        image_path: Path to image file

    Returns:
        numpy array: Loaded image in BGR format
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return image


def save_image(image, output_path):
    """
    Save image to file

    Args:
        image: Image array to save
        output_path: Destination path

    Returns:
        bool: True if successful
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(output_path), image)


def visualize_bounding_boxes(image, boxes, labels=None, colors=None):
    """
    Draw bounding boxes on image

    Args:
        image: Input image
        boxes: List of bounding boxes in format [x1, y1, x2, y2]
        labels: List of labels for each box
        colors: List of colors for each box

    Returns:
        Image with drawn bounding boxes
    """
    image_copy = image.copy()

    if colors is None:
        colors = [(0, 255, 0)] * len(boxes)  # Green by default

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        color = colors[idx] if idx < len(colors) else (0, 255, 0)

        # Draw rectangle
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)

        # Draw label if provided
        if labels and idx < len(labels):
            label = labels[idx]
            # Get text size for background rectangle
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            # Draw background rectangle
            cv2.rectangle(
                image_copy,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color,
                -1,
            )
            # Draw text
            cv2.putText(
                image_copy,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

    return image_copy


def calculate_image_stats(image):
    """
    Calculate statistics for image quality assessment

    Args:
        image: Input image (BGR format)

    Returns:
        Dictionary with image statistics
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    stats = {
        "mean_brightness": np.mean(gray),
        "std_brightness": np.std(gray),
        "min_brightness": np.min(gray),
        "max_brightness": np.max(gray),
        "contrast": np.std(gray),
        "sharpness": cv2.Laplacian(gray, cv2.CV_64F).var(),
        "height": image.shape[0],
        "width": image.shape[1],
        "channels": image.shape[2] if len(image.shape) == 3 else 1,
    }

    return stats


def plot_training_curves(history_csv, output_path):
    """
    Plot training curves from results CSV

    Args:
        history_csv: Path to results CSV file
        output_path: Path to save plot
    """
    import pandas as pd

    # Load training history
    df = pd.read_csv(history_csv)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Training Curves", fontsize=16, fontweight="bold")

    # Plot mAP
    axes[0, 0].plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP50")
    axes[0, 0].plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP50-95")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("mAP")
    axes[0, 0].set_title("Mean Average Precision")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot Loss
    axes[0, 1].plot(df["epoch"], df["train/box_loss"], label="Box Loss")
    axes[0, 1].plot(df["epoch"], df["train/cls_loss"], label="Class Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title("Training Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot Precision and Recall
    axes[1, 0].plot(df["epoch"], df["metrics/precision(B)"], label="Precision")
    axes[1, 0].plot(df["epoch"], df["metrics/recall(B)"], label="Recall")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].set_title("Precision and Recall")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot Learning Rate
    axes[1, 1].plot(df["epoch"], df["lr/pg0"], label="Learning Rate")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Learning Rate")
    axes[1, 1].set_title("Learning Rate Schedule")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_dataset_statistics(data_dir, output_path):
    """
    Create statistics visualization for dataset

    Args:
        data_dir: Path to data directory
        output_path: Path to save visualization
    """
    data_dir = Path(data_dir)

    stats = {}
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split

        if not split_dir.exists():
            continue

        # Count images
        images = list((split_dir / "images").glob("*.jpg"))
        labels = list((split_dir / "labels").glob("*.txt"))

        # Count annotations
        total_annotations = 0
        for label_file in labels:
            with open(label_file, "r") as f:
                total_annotations += len(f.readlines())

        stats[split] = {
            "images": len(images),
            "annotations": total_annotations,
            "avg_annotations": total_annotations / len(images) if images else 0,
        }

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot image counts
    splits = list(stats.keys())
    image_counts = [stats[s]["images"] for s in splits]
    axes[0].bar(splits, image_counts, color=["#3498db", "#2ecc71", "#e74c3c"])
    axes[0].set_title("Images per Split")
    axes[0].set_ylabel("Number of Images")
    axes[0].grid(axis="y", alpha=0.3)

    # Add count labels
    for i, count in enumerate(image_counts):
        axes[0].text(i, count, str(count), ha="center", va="bottom", fontweight="bold")

    # Plot annotation counts
    annotation_counts = [stats[s]["annotations"] for s in splits]
    axes[1].bar(splits, annotation_counts, color=["#3498db", "#2ecc71", "#e74c3c"])
    axes[1].set_title("Annotations per Split")
    axes[1].set_ylabel("Number of Annotations")
    axes[1].grid(axis="y", alpha=0.3)

    # Add count labels
    for i, count in enumerate(annotation_counts):
        axes[1].text(i, count, str(count), ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def convert_yolo_to_xyxy(yolo_box, img_width, img_height):
    """
    Convert YOLO format box to xyxy format

    Args:
        yolo_box: Box in YOLO format [x_center, y_center, width, height] (normalized)
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        Box in [x1, y1, x2, y2] format (pixel coordinates)
    """
    x_center, y_center, width, height = yolo_box

    x1 = int((x_center - width / 2) * img_width)
    y1 = int((y_center - height / 2) * img_height)
    x2 = int((x_center + width / 2) * img_width)
    y2 = int((y_center + height / 2) * img_height)

    return [x1, y1, x2, y2]


def convert_xyxy_to_yolo(xyxy_box, img_width, img_height):
    """
    Convert xyxy format box to YOLO format

    Args:
        xyxy_box: Box in [x1, y1, x2, y2] format (pixel coordinates)
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        Box in YOLO format [x_center, y_center, width, height] (normalized)
    """
    x1, y1, x2, y2 = xyxy_box

    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height

    return [x_center, y_center, width, height]


def print_progress_bar(
    iteration, total, prefix="Progress:", suffix="Complete", length=50, fill="█"
):
    """
    Print progress bar in console

    Args:
        iteration: Current iteration
        total: Total iterations
        prefix: Prefix string
        suffix: Suffix string
        length: Character length of bar
        fill: Bar fill character
    """
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + "-" * (length - filled_length)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end="\r")

    # Print new line on completion
    if iteration == total:
        print()


if __name__ == "__main__":
    # Test utility functions
    print("Utility functions module - no standalone execution")
    print("Import this module to use helper functions")
