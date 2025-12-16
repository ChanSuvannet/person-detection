"""
Data Quality Enhancement
This script improves image quality through brightness, contrast, and sharpness adjustments
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm
import shutil


class DataQualityEnhancer:
    """
    Handles image quality enhancement: brightness, contrast, and sharpening
    """

    def __init__(self, config_path="config/config.yaml"):
        """
        Initialize quality enhancer with configuration

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Set up paths
        self.processed_dir = Path(self.config["paths"]["processed_data"])

        # Create enhanced data directory (copy of processed)
        self.enhanced_dir = self.processed_dir.parent / "enhanced"

        # Statistics for reporting
        self.stats = {
            "brightness_adjusted": 0,
            "contrast_adjusted": 0,
            "sharpened": 0,
            "total_processed": 0,
        }

    def check_brightness(self, image):
        """
        Check if image brightness is within acceptable range

        Args:
            image: Input image (BGR format)

        Returns:
            tuple: (is_acceptable, avg_brightness)
        """
        # Convert to grayscale for brightness calculation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)

        # Get thresholds from config
        min_bright = self.config["quality"]["min_brightness"]
        max_bright = self.config["quality"]["max_brightness"]

        is_acceptable = min_bright <= avg_brightness <= max_bright

        return is_acceptable, avg_brightness

    def adjust_brightness(self, image, target_brightness=128):
        """
        Adjust image brightness to target value using gamma correction

        Args:
            image: Input image (BGR format)
            target_brightness: Desired average brightness (0-255)

        Returns:
            Brightness-adjusted image
        """
        # Calculate current brightness
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        current_brightness = np.mean(gray)

        # Calculate gamma for adjustment
        # gamma < 1 brightens, gamma > 1 darkens
        gamma = np.log(target_brightness / 255.0) / np.log(current_brightness / 255.0)

        # Clamp gamma to reasonable range
        gamma = np.clip(gamma, 0.5, 2.0)

        # Build lookup table for gamma correction
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(
            "uint8"
        )

        # Apply gamma correction
        adjusted = cv2.LUT(image, table)

        return adjusted

    def check_contrast(self, image):
        """
        Check if image has sufficient contrast
        Uses standard deviation of pixel intensities

        Args:
            image: Input image (BGR format)

        Returns:
            tuple: (is_acceptable, contrast_value)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate contrast as standard deviation
        contrast = np.std(gray)

        # Get minimum contrast threshold
        min_contrast = self.config["quality"]["min_contrast"]

        is_acceptable = contrast >= min_contrast

        return is_acceptable, contrast

    def enhance_contrast(self, image):
        """
        Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        CLAHE improves local contrast without over-amplifying noise

        Args:
            image: Input image (BGR format)

        Returns:
            Contrast-enhanced image
        """
        # Convert to LAB color space
        # L channel = lightness, A and B = color channels
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel only (preserves colors)
        clahe = cv2.createCLAHE(
            clipLimit=self.config["enhancement"]["clahe_clip_limit"],
            tileGridSize=tuple(self.config["enhancement"]["clahe_grid_size"]),
        )
        l_clahe = clahe.apply(l)

        # Merge channels back
        lab_clahe = cv2.merge([l_clahe, a, b])

        # Convert back to BGR
        enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

        return enhanced

    def check_sharpness(self, image):
        """
        Check image sharpness using Laplacian variance
        Higher variance = sharper image

        Args:
            image: Input image (BGR format)

        Returns:
            tuple: (is_acceptable, sharpness_value)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()

        # Get blur threshold
        blur_threshold = self.config["quality"]["blur_threshold"]

        is_acceptable = sharpness >= blur_threshold

        return is_acceptable, sharpness

    def sharpen_image(self, image):
        """
        Sharpen image using unsharp masking technique
        Enhances edges without creating artifacts

        Args:
            image: Input image (BGR format)

        Returns:
            Sharpened image
        """
        # Create Gaussian blur
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)

        # Unsharp mask: original + (original - blur) * amount
        sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)

        return sharpened

    def enhance_image(self, image_path, output_path):
        """
        Apply all quality enhancements to a single image

        Args:
            image_path: Path to input image
            output_path: Path to save enhanced image

        Returns:
            bool: True if enhancements were applied, False otherwise
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            return False

        enhanced = image.copy()
        modified = False

        # Check and adjust brightness
        if self.config["enhancement"]["brightness_adjustment"]:
            is_ok, brightness = self.check_brightness(enhanced)
            if not is_ok:
                enhanced = self.adjust_brightness(enhanced)
                self.stats["brightness_adjusted"] += 1
                modified = True

        # Check and enhance contrast
        if self.config["enhancement"]["contrast_adjustment"]:
            is_ok, contrast = self.check_contrast(enhanced)
            if not is_ok:
                enhanced = self.enhance_contrast(enhanced)
                self.stats["contrast_adjusted"] += 1
                modified = True

        # Check and sharpen
        if self.config["enhancement"]["sharpening"]:
            is_ok, sharpness = self.check_sharpness(enhanced)
            if not is_ok:
                enhanced = self.sharpen_image(enhanced)
                self.stats["sharpened"] += 1
                modified = True

        # Save enhanced image
        cv2.imwrite(str(output_path), enhanced)
        self.stats["total_processed"] += 1

        return modified

    def process_split(self, split_name):
        """
        Process all images in a dataset split

        Args:
            split_name: Name of split ('train', 'val', or 'test')
        """
        print(f"Enhancing {split_name} set...")

        # Source and destination directories
        src_img_dir = self.processed_dir / split_name / "images"
        dst_img_dir = self.enhanced_dir / split_name / "images"
        dst_img_dir.mkdir(parents=True, exist_ok=True)

        # Copy labels (they don't need enhancement)
        src_label_dir = self.processed_dir / split_name / "labels"
        dst_label_dir = self.enhanced_dir / split_name / "labels"
        if dst_label_dir.exists():
            shutil.rmtree(dst_label_dir)
        shutil.copytree(src_label_dir, dst_label_dir)

        # Process each image
        image_files = list(src_img_dir.glob("*.jpg"))

        for img_path in tqdm(image_files, desc=f"Processing {split_name}"):
            output_path = dst_img_dir / img_path.name
            self.enhance_image(img_path, output_path)

    def create_enhanced_yaml(self):
        """
        Create dataset.yaml for enhanced dataset
        """
        # Create YAML content
        dataset_config = {
            "path": str(self.enhanced_dir.absolute()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": 1,
            "names": ["Person"],
        }

        # Write YAML file
        yaml_path = Path("data/dataset_enhanced.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(dataset_config, f, default_flow_style=False)

        print(f"Created {yaml_path}")

    def create_summary(self):
        """
        Create quality enhancement summary report
        """
        print("\n" + "=" * 60)
        print("DATA QUALITY ENHANCEMENT SUMMARY")
        print("=" * 60)
        print(f"Total images processed: {self.stats['total_processed']}")
        print(f"Brightness adjusted: {self.stats['brightness_adjusted']}")
        print(f"Contrast enhanced: {self.stats['contrast_adjusted']}")
        print(f"Sharpened: {self.stats['sharpened']}")

        # Calculate percentages
        if self.stats["total_processed"] > 0:
            bright_pct = (
                self.stats["brightness_adjusted"] / self.stats["total_processed"] * 100
            )
            contrast_pct = (
                self.stats["contrast_adjusted"] / self.stats["total_processed"] * 100
            )
            sharp_pct = self.stats["sharpened"] / self.stats["total_processed"] * 100

            print(f"\nPercentage enhanced:")
            print(f"  Brightness: {bright_pct:.1f}%")
            print(f"  Contrast: {contrast_pct:.1f}%")
            print(f"  Sharpness: {sharp_pct:.1f}%")

        print("=" * 60 + "\n")

        # Save summary
        summary_path = self.enhanced_dir / "enhancement_summary.txt"
        with open(summary_path, "w") as f:
            for key, value in self.stats.items():
                f.write(f"{key}: {value}\n")

    def run(self):
        """
        Main execution function - runs all enhancement steps
        """
        print("\n" + "=" * 70)
        print("DATA QUALITY ENHANCEMENT")
        print("=" * 70 + "\n")

        # Process each split
        for split in ["train", "val", "test"]:
            self.process_split(split)

        # Create enhanced dataset YAML
        self.create_enhanced_yaml()

        # Create summary
        self.create_summary()

        print("Data quality enhancement complete! Ready for training.\n")


if __name__ == "__main__":
    # Initialize and run enhancer
    enhancer = DataQualityEnhancer()
    enhancer.run()
