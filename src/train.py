"""
Model Training
This script trains YOLOv8 models on original and enhanced datasets
"""

import yaml
import argparse
from pathlib import Path
from ultralytics import YOLO
import torch


class PersonDetectionTrainer:
    """
    Handles training of YOLO object detection models
    """

    def __init__(self, config_path="config/config.yaml", dataset_type="original"):
        """
        Initialize trainer with configuration

        Args:
            config_path: Path to configuration file
            dataset_type: 'original' or 'enhanced'
        """
        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.dataset_type = dataset_type

        # Set up paths
        if dataset_type == "original":
            self.data_yaml = "data/dataset.yaml"
            self.output_dir = Path(self.config["paths"]["models"]) / "original"
        else:
            self.data_yaml = "data/dataset_enhanced.yaml"
            self.output_dir = Path(self.config["paths"]["models"]) / "enhanced"

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Check if CUDA is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(
                f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n"
            )

    def train(self):
        """
        Train YOLO model with configured parameters

        This function:
        1. Loads pre-trained YOLO weights
        2. Trains on Person detection dataset
        3. Saves best model weights
        4. Generates training metrics and plots
        """
        print("\n" + "=" * 70)
        print(f"TRAINING MODEL ON {self.dataset_type.upper()} DATASET")
        print("=" * 70 + "\n")

        # Load pre-trained YOLO model
        # YOLOv8n = nano (fastest), YOLOv8s = small, YOLOv8m = medium
        model_name = self.config["training"]["model"]
        print(f"Loading model: {model_name}")
        model = YOLO(model_name)

        # Get training parameters from config
        epochs = self.config["training"]["epochs"]
        batch_size = self.config["training"]["batch_size"]
        img_size = self.config["training"]["img_size"]
        patience = self.config["training"]["patience"]

        print(f"\nTraining Configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Image size: {img_size}")
        print(f"  Early stopping patience: {patience}")
        print(f"  Device: {self.device}\n")

        # Start training
        print("Starting training...\n")

        results = model.train(
            # Data configuration
            data=self.data_yaml,
            # Training duration
            epochs=epochs,
            patience=patience,  # Early stopping after N epochs without improvement
            # Batch and image settings
            batch=batch_size,
            imgsz=img_size,
            # Device settings
            device=self.device,
            # Optimizer settings
            optimizer="Adam",  # Adam optimizer (adaptive learning rate)
            lr0=self.config["training"]["learning_rate"],
            momentum=self.config["training"]["momentum"],
            weight_decay=self.config["training"]["weight_decay"],
            # Data augmentation settings
            augment=self.config["training"]["augment"],
            hsv_h=self.config["training"]["hsv_h"],
            hsv_s=self.config["training"]["hsv_s"],
            hsv_v=self.config["training"]["hsv_v"],
            degrees=self.config["training"]["degrees"],
            translate=self.config["training"]["translate"],
            scale=self.config["training"]["scale"],
            flipud=self.config["training"]["flipud"],
            fliplr=self.config["training"]["fliplr"],
            mosaic=self.config["training"]["mosaic"],
            # Output settings
            project=str(self.output_dir),
            name="train",
            exist_ok=True,
            # Visualization
            plots=True,  # Save training plots
            save=True,  # Save checkpoints
            save_period=10,  # Save checkpoint every N epochs
            # Performance
            workers=8,  # Number of data loading workers
            verbose=True,  # Print training progress
        )

        print("\nTraining complete!")
        print(f"Results saved to: {self.output_dir / 'train'}")

        # Save best model to standard location
        best_model_path = self.output_dir / "train" / "weights" / "best.pt"
        if best_model_path.exists():
            # Copy to easy-to-find location
            import shutil

            dest_path = self.output_dir / "best.pt"
            shutil.copy2(best_model_path, dest_path)
            print(f"Best model saved to: {dest_path}")

        return results

    def validate(self):
        """
        Validate trained model on validation set

        Returns validation metrics
        """
        print("\n🔍 Validating model on validation set...")

        # Load best model
        best_model = self.output_dir / "best.pt"
        if not best_model.exists():
            best_model = self.output_dir / "train" / "weights" / "best.pt"

        model = YOLO(str(best_model))

        # Run validation
        results = model.val(
            data=self.data_yaml,
            batch=self.config["training"]["batch_size"],
            imgsz=self.config["training"]["img_size"],
            device=self.device,
            plots=True,
            save_json=True,  # Save results in COCO format
        )

        print("\nValidation Results:")
        print(f"  mAP50: {results.box.map50:.4f}")
        print(f"  mAP50-95: {results.box.map:.4f}")
        print(f"  Precision: {results.box.mp:.4f}")
        print(f"  Recall: {results.box.mr:.4f}")

        return results

    def create_training_summary(self, results):
        """
        Create a summary of training results

        Args:
            results: Training results object
        """
        summary_path = self.output_dir / "training_summary.txt"

        with open(summary_path, "w") as f:
            f.write(f"TRAINING SUMMARY - {self.dataset_type.upper()} DATASET\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Model: {self.config['training']['model']}\n")
            f.write(f"Epochs: {self.config['training']['epochs']}\n")
            f.write(f"Batch size: {self.config['training']['batch_size']}\n")
            f.write(f"Image size: {self.config['training']['img_size']}\n")
            f.write(f"Device: {self.device}\n\n")

            f.write("Final Metrics:\n")
            if hasattr(results, "box"):
                f.write(f"  mAP50: {results.box.map50:.4f}\n")
                f.write(f"  mAP50-95: {results.box.map:.4f}\n")
                f.write(f"  Precision: {results.box.mp:.4f}\n")
                f.write(f"  Recall: {results.box.mr:.4f}\n")

            f.write(f"\nModel saved to: {self.output_dir / 'best.pt'}\n")

        print(f"\nTraining summary saved to: {summary_path}")


def main():
    """
    Main training function with command-line arguments
    """
    parser = argparse.ArgumentParser(description="Train Person Detection Model")
    parser.add_argument(
        "--data",
        type=str,
        default="original",
        choices=["original", "enhanced"],
        help="Dataset type to train on",
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config file"
    )

    args = parser.parse_args()

    # Initialize trainer
    trainer = PersonDetectionTrainer(config_path=args.config, dataset_type=args.data)

    # Train model
    results = trainer.train()

    # Validate model
    val_results = trainer.validate()

    # Create summary
    trainer.create_training_summary(val_results)

    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nNext step: Run evaluation with:")
    print(f"   python src/evaluate.py --data {args.data}\n")


if __name__ == "__main__":
    main()
