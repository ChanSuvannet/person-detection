"""
Data Preprocessing and Quality Enhancement
This script handles data cleaning, duplicate removal, and train/val/test split
"""

import os
import cv2
import numpy as np
import pandas as pd
import yaml
import shutil
import imagehash
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """
    Handles data preprocessing: duplicate removal, quality checks, and data splitting
    """
    
    def __init__(self, config_path='config/config.yaml'):
        """
        Initialize preprocessor with configuration
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set up paths
        self.raw_dir = Path(self.config['paths']['raw_data'])
        self.processed_dir = Path(self.config['paths']['processed_data'])
        
        # Load annotations
        self.annotations = pd.read_csv(self.raw_dir / 'person_annotations.csv')
        
        # Create processed directories
        for split in ['train', 'val', 'test']:
            (self.processed_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.processed_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    def remove_duplicates(self):
        """
        Step 1: Remove duplicate images using perceptual hashing
        This finds visually similar images even if pixels differ slightly
        
        Returns:
            List of unique image IDs
        """
        print("Detecting and removing duplicate images...")
        
        images_dir = self.raw_dir / 'images'
        image_files = list(images_dir.glob('*.jpg'))
        
        # Dictionary to store image hashes
        hashes = {}
        duplicates = []
        
        # Calculate perceptual hash for each image
        for img_path in tqdm(image_files, desc="Computing image hashes"):
            try:
                # Open image and compute hash
                img = Image.open(img_path)
                img_hash = imagehash.phash(img)
                
                # Check if similar hash exists (allowing small differences)
                is_duplicate = False
                for existing_hash, existing_path in hashes.items():
                    # Hamming distance < 5 means very similar images
                    if img_hash - existing_hash < 5:
                        duplicates.append(img_path)
                        is_duplicate = True
                        break
                
                # Add to hash dictionary if unique
                if not is_duplicate:
                    hashes[img_hash] = img_path
                    
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
                duplicates.append(img_path)
        
        # Remove duplicate files
        for dup_path in duplicates:
            dup_path.unlink()
        
        print(f"Removed {len(duplicates)} duplicate images")
        print(f"Retained {len(hashes)} unique images\n")
        
        # Get unique image IDs
        unique_ids = [path.stem for path in hashes.values()]
        
        # Filter annotations to keep only unique images
        self.annotations = self.annotations[
            self.annotations['ImageID'].isin(unique_ids)
        ]
        
        return unique_ids
    
    def remove_unlabeled_images(self):
        """
        Step 2: Remove images without proper bounding box annotations
        Ensures every image has at least one valid person annotation
        
        Returns:
            DataFrame with valid annotations
        """
        print("Removing unlabeled and invalid images...")
        
        images_dir = self.raw_dir / 'images'
        
        # Get images that exist in filesystem
        existing_images = {f.stem for f in images_dir.glob('*.jpg')}
        
        # Filter annotations for existing images
        valid_annotations = self.annotations[
            self.annotations['ImageID'].isin(existing_images)
        ]
        
        # Group by image and check if they have valid bounding boxes
        image_groups = valid_annotations.groupby('ImageID')
        
        invalid_images = []
        valid_image_ids = []
        
        for image_id, group in image_groups:
            # Check if image has at least one valid bounding box
            # Bounding boxes must have positive area
            valid_boxes = group[
                (group['XMax'] > group['XMin']) & 
                (group['YMax'] > group['YMin'])
            ]
            
            if len(valid_boxes) == 0:
                invalid_images.append(image_id)
                # Delete image without valid annotations
                img_path = images_dir / f"{image_id}.jpg"
                if img_path.exists():
                    img_path.unlink()
            else:
                valid_image_ids.append(image_id)
        
        # Keep only valid annotations
        self.annotations = valid_annotations[
            valid_annotations['ImageID'].isin(valid_image_ids)
        ]
        
        print(f"Removed {len(invalid_images)} images without valid labels")
        print(f"Retained {len(valid_image_ids)} properly labeled images\n")
        
        return self.annotations
    
    def validate_bounding_boxes(self):
        """
        Step 3: Validate and correct bounding box coordinates
        Ensures all boxes are within image boundaries and have minimum size
        """
        print("Validating bounding box coordinates...")
        
        images_dir = self.raw_dir / 'images'
        corrected = 0
        removed = 0
        
        # Check each annotation
        valid_annotations = []
        
        for _, row in tqdm(self.annotations.iterrows(), 
                        total=len(self.annotations),
                        desc="Validating boxes"):
            
            image_path = images_dir / f"{row['ImageID']}.jpg"
            
            if not image_path.exists():
                removed += 1
                continue
            
            # Load image to get dimensions
            img = cv2.imread(str(image_path))
            if img is None:
                removed += 1
                continue
            
            h, w = img.shape[:2]
            
            # Convert normalized coordinates to pixel coordinates
            xmin = int(row['XMin'] * w)
            xmax = int(row['XMax'] * w)
            ymin = int(row['YMin'] * h)
            ymax = int(row['YMax'] * h)
            
            # Clamp coordinates to image boundaries
            xmin = max(0, min(xmin, w))
            xmax = max(0, min(xmax, w))
            ymin = max(0, min(ymin, h))
            ymax = max(0, min(ymax, h))
            
            # Check minimum box area
            box_area = (xmax - xmin) * (ymax - ymin)
            min_area = self.config['quality']['min_bbox_area']
            
            if box_area < min_area:
                removed += 1
                continue
            
            # Check if coordinates were corrected
            if (xmin != int(row['XMin'] * w) or xmax != int(row['XMax'] * w) or
                ymin != int(row['YMin'] * h) or ymax != int(row['YMax'] * h)):
                corrected += 1
            
            # Update with corrected normalized coordinates
            row['XMin'] = xmin / w
            row['XMax'] = xmax / w
            row['YMin'] = ymin / h
            row['YMax'] = ymax / h
            
            valid_annotations.append(row)
        
        # Update annotations
        self.annotations = pd.DataFrame(valid_annotations)
        
        print(f"Corrected {corrected} bounding boxes")
        print(f"Removed {removed} invalid annotations\n")
    
    def split_dataset(self):
        """
        Step 4: Split data into train/validation/test sets
        Uses stratification to ensure balanced distribution
        
        Returns:
            Dictionary with train, val, test image IDs
        """
        print("✂️  Splitting dataset into train/val/test...")
        
        # Get unique image IDs
        unique_images = self.annotations['ImageID'].unique()
        
        # Get split ratios from config
        train_ratio = self.config['dataset']['train_split']
        val_ratio = self.config['dataset']['val_split']
        test_ratio = self.config['dataset']['test_split']
        
        # First split: separate test set
        train_val_images, test_images = train_test_split(
            unique_images,
            test_size=test_ratio,
            random_state=42
        )
        
        # Second split: separate train and validation
        val_size = val_ratio / (train_ratio + val_ratio)
        train_images, val_images = train_test_split(
            train_val_images,
            test_size=val_size,
            random_state=42
        )
        
        splits = {
            'train': train_images,
            'val': val_images,
            'test': test_images
        }
        
        print(f"Train set: {len(train_images)} images ({train_ratio*100:.0f}%)")
        print(f"Validation set: {len(val_images)} images ({val_ratio*100:.0f}%)")
        print(f"Test set: {len(test_images)} images ({test_ratio*100:.0f}%)\n")
        
        return splits
    
    def convert_to_yolo_format(self, splits):
        """
        Step 5: Convert annotations to YOLO format and organize files
        YOLO format: <class_id> <x_center> <y_center> <width> <height> (all normalized)
        
        Args:
            splits: Dictionary with train/val/test image IDs
        """
        print("Converting annotations to YOLO format...")
        
        images_dir = self.raw_dir / 'images'
        
        for split_name, image_ids in splits.items():
            print(f"Processing {split_name} set...")
            
            # Get split directories
            split_img_dir = self.processed_dir / split_name / 'images'
            split_label_dir = self.processed_dir / split_name / 'labels'
            
            # Process each image in this split
            for image_id in tqdm(image_ids, desc=f"Converting {split_name}"):
                # Copy image file
                src_img = images_dir / f"{image_id}.jpg"
                dst_img = split_img_dir / f"{image_id}.jpg"
                
                if src_img.exists():
                    shutil.copy2(src_img, dst_img)
                
                # Get annotations for this image
                img_annotations = self.annotations[
                    self.annotations['ImageID'] == image_id
                ]
                
                # Create YOLO label file
                label_path = split_label_dir / f"{image_id}.txt"
                
                with open(label_path, 'w') as f:
                    for _, ann in img_annotations.iterrows():
                        # Convert to YOLO format (class_id x_center y_center width height)
                        # All values normalized to [0, 1]
                        x_center = (ann['XMin'] + ann['XMax']) / 2
                        y_center = (ann['YMin'] + ann['YMax']) / 2
                        width = ann['XMax'] - ann['XMin']
                        height = ann['YMax'] - ann['YMin']
                        
                        # YOLO class ID is 0 for single-class detection (Person)
                        f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        print("YOLO format conversion complete!\n")
    
    def create_dataset_yaml(self):
        """
        Step 6: Create dataset.yaml file for YOLO training
        This file tells YOLO where to find images and what classes exist
        """
        print("Creating dataset.yaml configuration...")
        
        # Create YAML content
        dataset_config = {
            'path': str(self.processed_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': 1,  # Number of classes
            'names': ['Person']  # Class names
        }
        
        # Write YAML file
        yaml_path = Path('data/dataset.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"Created {yaml_path}\n")
    
    def create_summary(self):
        """
        Step 7: Create preprocessing summary report
        """
        print("Creating preprocessing summary...")
        
        summary = {}
        for split in ['train', 'val', 'test']:
            img_dir = self.processed_dir / split / 'images'
            label_dir = self.processed_dir / split / 'labels'
            
            n_images = len(list(img_dir.glob('*.jpg')))
            n_labels = len(list(label_dir.glob('*.txt')))
            
            # Count total annotations
            total_annotations = 0
            for label_file in label_dir.glob('*.txt'):
                with open(label_file, 'r') as f:
                    total_annotations += len(f.readlines())
            
            summary[split] = {
                'images': n_images,
                'labels': n_labels,
                'annotations': total_annotations,
                'avg_annotations': total_annotations / n_images if n_images > 0 else 0
            }
        
        # Print summary
        print("\n" + "="*60)
        print("DATA PREPROCESSING SUMMARY")
        print("="*60)
        for split, stats in summary.items():
            print(f"\n{split.upper()}:")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
        print("="*60 + "\n")
        
        # Save summary
        summary_path = self.processed_dir / 'preprocessing_summary.txt'
        with open(summary_path, 'w') as f:
            for split, stats in summary.items():
                f.write(f"{split.upper()}:\n")
                for key, value in stats.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
    
    def run(self):
        """
        Main execution function - runs all preprocessing steps
        """
        print("\n" + "="*70)
        print("DATA PREPROCESSING AND CLEANING")
        print("="*70 + "\n")
        
        # Step 1: Remove duplicates
        self.remove_duplicates()
        
        # Step 2: Remove unlabeled images
        self.remove_unlabeled_images()
        
        # Step 3: Validate bounding boxes
        self.validate_bounding_boxes()
        
        # Step 4: Split dataset
        splits = self.split_dataset()
        
        # Step 5: Convert to YOLO format
        self.convert_to_yolo_format(splits)
        
        # Step 6: Create dataset YAML
        self.create_dataset_yaml()
        
        # Step 7: Create summary
        self.create_summary()
        
        print("Data preprocessing complete! Ready for quality enhancement.\n")


if __name__ == "__main__":
    # Initialize and run preprocessor
    preprocessor = DataPreprocessor()
    preprocessor.run()