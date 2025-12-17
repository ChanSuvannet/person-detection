"""
Data Collection from Open Images Dataset
This script downloads Person class images and annotations from Open Images Dataset V7
"""

import os
import requests
import pandas as pd
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm
import yaml
import shutil
from pathlib import Path


class OpenImagesDownloader:
    """
    Handles downloading images and annotations from Open Images Dataset
    """

    def __init__(self, config_path="config/config.yaml"):
        """
        Initialize downloader with configuration

        Args:
            config_path: Path to configuration file
        """
        # Load configuration from YAML file
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Set up AWS S3 client for Open Images (public bucket, no credentials needed)
        self.s3_client = boto3.client(
            "s3", config=Config(signature_version=UNSIGNED), region_name="us-east-1"
        )

        # Open Images bucket name
        self.bucket_name = "open-images-dataset"

        # Person class ID in Open Images Dataset
        self.person_class_id = "/m/01g317"  # Person class ID

        # Create raw data directory
        self.raw_dir = Path(self.config["paths"]["raw_data"])
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def download_metadata(self):
        """
        Step 1: Download metadata files (class descriptions, annotations)
        These files tell us which images contain persons and where they are
        """
        print("Downloading metadata files...")

        # URLs for Open Images V7 metadata
        base_url = "https://storage.googleapis.com/openimages/v7/"

        metadata_files = {
            "class_descriptions": "oidv7-class-descriptions.csv",
            "train_annotations": "oidv7-train-annotations-bbox.csv",
            "validation_annotations": "oidv7-validation-annotations-bbox.csv",
            "test_annotations": "oidv7-test-annotations-bbox.csv",
        }

        for name, filename in metadata_files.items():
            filepath = self.raw_dir / filename

            # Skip if already downloaded
            if filepath.exists():
                print(f"✓ {filename} already exists")
                continue

            print(f"⬇ Downloading {filename}...")
            url = base_url + filename

            response = requests.get(url, stream=True)
            response.raise_for_status()

            # Write file with progress bar
            total_size = int(response.headers.get("content-length", 0))
            with open(filepath, "wb") as f, tqdm(
                desc=filename,
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)

        print("Metadata download complete!\n")

    def filter_person_annotations(self):
        """
        Step 2: Filter annotations to get only Person class images
        This reduces the dataset to only images containing persons

        Returns:
            DataFrame with person annotations
        """
        print("Filtering Person class annotations...")

        # Load validation annotations (this file still works)
        val_ann = pd.read_csv(self.raw_dir / "oidv7-validation-annotations-bbox.csv")

        # Filter for Person class only
        person_annotations = val_ann[val_ann["LabelName"] == self.person_class_id]

        print(f"✓ Found {len(person_annotations)} person annotations in validation set")

        # Get unique image IDs and limit to target number
        target_images = self.config["dataset"]["total_images"]
        unique_images = person_annotations["ImageID"].unique()[:target_images]

        # Filter annotations for selected images
        person_annotations = person_annotations[
            person_annotations["ImageID"].isin(unique_images)
        ]

        # Save filtered annotations
        output_path = self.raw_dir / "person_annotations.csv"
        person_annotations.to_csv(output_path, index=False)

        print(f"✓ Filtered {len(unique_images)} images with person annotations")
        print(f"✓ Saved to {output_path}\n")

        return person_annotations

    def download_images(self, annotations):
        """
        Step 3: Download actual images from S3

        Args:
            annotations: DataFrame with person annotations
        """
        print("Downloading images from Open Images S3 bucket...")

        # Get unique image IDs
        image_ids = annotations["ImageID"].unique()

        # Create images directory
        images_dir = self.raw_dir / "images"
        images_dir.mkdir(exist_ok=True)

        # Download images with progress bar
        downloaded = 0
        failed = []

        for image_id in tqdm(image_ids, desc="Downloading images"):
            # Check if already downloaded
            image_path = images_dir / f"{image_id}.jpg"
            if image_path.exists():
                downloaded += 1
                continue

            # Try to download from train, validation, or test set
            for subset in ["train", "validation", "test"]:
                try:
                    # S3 path in Open Images bucket
                    s3_key = f"{subset}/{image_id}.jpg"

                    # Download from S3
                    self.s3_client.download_file(
                        self.bucket_name, s3_key, str(image_path)
                    )

                    downloaded += 1
                    break  # Success, move to next image

                except Exception:
                    continue  # Try next subset
            else:
                # Failed to download from all subsets
                failed.append(image_id)

        print(f"\nDownloaded {downloaded} images")
        if failed:
            print(f"Failed to download {len(failed)} images")

        return downloaded, failed

    def create_summary(self, annotations):
        """
        Step 4: Create a summary report of downloaded data

        Args:
            annotations: DataFrame with person annotations
        """
        print("\nCreating download summary...")

        images_dir = self.raw_dir / "images"
        downloaded_images = len(list(images_dir.glob("*.jpg")))

        summary = {
            "Total Annotations": len(annotations),
            "Unique Images": annotations["ImageID"].nunique(),
            "Downloaded Images": downloaded_images,
            "Avg Annotations per Image": len(annotations)
            / annotations["ImageID"].nunique(),
        }

        # Print summary
        print("\n" + "=" * 50)
        print("DATA COLLECTION SUMMARY")
        print("=" * 50)
        for key, value in summary.items():
            print(
                f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}"
            )
        print("=" * 50 + "\n")

        # Save summary
        summary_path = self.raw_dir / "download_summary.txt"
        with open(summary_path, "w") as f:
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")

    def run(self):
        """
        Main execution function - runs all download steps
        """
        print("\n" + "=" * 70)
        print("OPEN IMAGES PERSON DETECTION - DATA COLLECTION (DAY 1)")
        print("=" * 70 + "\n")

        # Step 1: Download metadata
        # self.download_metadata()

        # Step 2: Filter person annotations
        annotations = self.filter_person_annotations()

        # Step 3: Download images
        self.download_images(annotations)

        # Step 4: Create summary
        self.create_summary(annotations)

        print("Data collection complete! Ready for preprocessing.\n")


if __name__ == "__main__":
    # Initialize and run downloader
    downloader = OpenImagesDownloader()
    downloader.run()
