"""
DAY 3: Model Evaluation and Comparison
This script evaluates trained models and compares original vs enhanced datasets
"""

import yaml
import argparse
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import json


class ModelEvaluator:
    """
    Handles model evaluation and performance comparison
    """
    
    def __init__(self, config_path='config/config.yaml'):
        """
        Initialize evaluator with configuration
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set up paths
        self.results_dir = Path(self.config['paths']['results'])
        self.models_dir = Path(self.config['paths']['models'])
        
        # Create results directories
        (self.results_dir / 'metrics').mkdir(parents=True, exist_ok=True)
        (self.results_dir / 'visualizations').mkdir(parents=True, exist_ok=True)
        (self.results_dir / 'comparison').mkdir(parents=True, exist_ok=True)
    
    def load_model(self, dataset_type):
        """
        Load trained model for evaluation
        
        Args:
            dataset_type: 'original' or 'enhanced'
            
        Returns:
            YOLO model object
        """
        model_path = self.models_dir / dataset_type / 'best.pt'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"Loading {dataset_type} model from {model_path}")
        model = YOLO(str(model_path))
        
        return model
    
    def evaluate_on_test_set(self, model, dataset_type):
        """
        Evaluate model on test set
        
        Args:
            model: YOLO model object
            dataset_type: 'original' or 'enhanced'
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\n🧪 Evaluating {dataset_type} model on test set...")
        
        # Get data YAML path
        data_yaml = f'data/dataset{"_enhanced" if dataset_type == "enhanced" else ""}.yaml'
        
        # Run evaluation on test set
        results = model.val(
            data=data_yaml,
            split='test',  # Evaluate on test split
            batch=16,
            imgsz=self.config['training']['img_size'],
            conf=self.config['evaluation']['conf_threshold'],
            iou=self.config['evaluation']['iou_threshold'],
            plots=True,
            save_json=True,
            project=str(self.results_dir / 'metrics'),
            name=f'test_{dataset_type}'
        )
        
        # Extract key metrics
        metrics = {
            'dataset': dataset_type,
            'mAP50': float(results.box.map50),
            'mAP50-95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
            'f1_score': 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr) 
                       if (results.box.mp + results.box.mr) > 0 else 0
        }
        
        print(f"\nTest Set Metrics ({dataset_type}):")
        for key, value in metrics.items():
            if key != 'dataset':
                print(f"  {key}: {value:.4f}")
        
        return metrics
    
    def visualize_detections(self, model, dataset_type, num_samples=10):
        """
        Create visualization of model predictions on test images
        
        Args:
            model: YOLO model object
            dataset_type: 'original' or 'enhanced'
            num_samples: Number of sample images to visualize
        """
        print(f"\nCreating detection visualizations for {dataset_type}...")
        
        # Get test images
        if dataset_type == 'original':
            test_dir = Path('data/processed/test/images')
        else:
            test_dir = Path('data/enhanced/test/images')
        
        test_images = list(test_dir.glob('*.jpg'))[:num_samples]
        
        # Create visualization directory
        vis_dir = self.results_dir / 'visualizations' / dataset_type
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each image
        for img_path in tqdm(test_images, desc="Visualizing"):
            # Run inference
            results = model.predict(
                source=str(img_path),
                conf=self.config['evaluation']['conf_threshold'],
                iou=self.config['evaluation']['iou_threshold'],
                verbose=False
            )[0]
            
            # Get annotated image
            annotated = results.plot()
            
            # Save visualization
            output_path = vis_dir / img_path.name
            cv2.imwrite(str(output_path), annotated)
        
        print(f"Saved {len(test_images)} visualizations to {vis_dir}")
    
    def analyze_errors(self, model, dataset_type, num_samples=100):
        """
        Analyze false positives and false negatives
        
        Args:
            model: YOLO model object
            dataset_type: 'original' or 'enhanced'
            num_samples: Number of images to analyze
            
        Returns:
            Dictionary with error analysis
        """
        print(f"\nAnalyzing errors for {dataset_type} model...")
        
        # Get test images and labels
        if dataset_type == 'original':
            img_dir = Path('data/processed/test/images')
            label_dir = Path('data/processed/test/labels')
        else:
            img_dir = Path('data/enhanced/test/images')
            label_dir = Path('data/enhanced/test/labels')
        
        test_images = list(img_dir.glob('*.jpg'))[:num_samples]
        
        # Error counters
        false_positives = 0
        false_negatives = 0
        true_positives = 0
        
        iou_threshold = 0.5  # IoU threshold for matching
        
        for img_path in tqdm(test_images, desc="Analyzing errors"):
            # Run inference
            results = model.predict(
                source=str(img_path),
                conf=self.config['evaluation']['conf_threshold'],
                verbose=False
            )[0]
            
            # Get predictions
            pred_boxes = results.boxes.xyxyn.cpu().numpy() if len(results.boxes) > 0 else np.array([])
            
            # Load ground truth labels
            label_path = label_dir / f"{img_path.stem}.txt"
            gt_boxes = []
            
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        # Parse YOLO format: class x_center y_center width height
                        parts = line.strip().split()
                        if len(parts) == 5:
                            _, x_c, y_c, w, h = map(float, parts)
                            # Convert to xyxy format
                            x1 = x_c - w/2
                            y1 = y_c - h/2
                            x2 = x_c + w/2
                            y2 = y_c + h/2
                            gt_boxes.append([x1, y1, x2, y2])
            
            gt_boxes = np.array(gt_boxes)
            
            # Match predictions to ground truth
            matched_preds = set()
            matched_gts = set()
            
            # Calculate IoU for all pairs
            for i, pred in enumerate(pred_boxes):
                best_iou = 0
                best_gt = -1
                
                for j, gt in enumerate(gt_boxes):
                    iou = self.calculate_iou(pred, gt)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = j
                
                if best_iou >= iou_threshold:
                    matched_preds.add(i)
                    matched_gts.add(best_gt)
                    true_positives += 1
            
            # False positives: predictions without matching GT
            false_positives += len(pred_boxes) - len(matched_preds)
            
            # False negatives: GT without matching predictions
            false_negatives += len(gt_boxes) - len(matched_gts)
        
        # Calculate error rates
        total_predictions = true_positives + false_positives
        total_ground_truth = true_positives + false_negatives
        
        error_analysis = {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': true_positives / total_predictions if total_predictions > 0 else 0,
            'recall': true_positives / total_ground_truth if total_ground_truth > 0 else 0
        }
        
        print(f"\nError Analysis ({dataset_type}):")
        print(f"  True Positives: {error_analysis['true_positives']}")
        print(f"  False Positives: {error_analysis['false_positives']}")
        print(f"  False Negatives: {error_analysis['false_negatives']}")
        print(f"  Precision: {error_analysis['precision']:.4f}")
        print(f"  Recall: {error_analysis['recall']:.4f}")
        
        return error_analysis
    
    @staticmethod
    def calculate_iou(box1, box2):
        """
        Calculate Intersection over Union (IoU) between two boxes
        
        Args:
            box1, box2: Boxes in [x1, y1, x2, y2] format
            
        Returns:
            IoU value
        """
        # Calculate intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Calculate union
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0
    
    def compare_models(self, metrics_original, metrics_enhanced):
        """
        Create comparison plots between original and enhanced models
        
        Args:
            metrics_original: Metrics dictionary for original model
            metrics_enhanced: Metrics dictionary for enhanced model
        """
        print("\nCreating comparison visualizations...")
        
        # Prepare data for plotting
        metrics_df = pd.DataFrame([metrics_original, metrics_enhanced])
        metrics_df.set_index('dataset', inplace=True)
        
        # Create comparison bar plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison: Original vs Enhanced', 
                    fontsize=16, fontweight='bold')
        
        metrics_to_plot = ['mAP50', 'mAP50-95', 'precision', 'recall', 'f1_score']
        
        for idx, metric in enumerate(metrics_to_plot):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            # Create bar plot
            values = metrics_df[metric].values
            datasets = metrics_df.index.values
            colors = ['#3498db', '#2ecc71']
            
            bars = ax.bar(datasets, values, color=colors, alpha=0.8, edgecolor='black')
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}',
                       ha='center', va='bottom', fontweight='bold')
            
            # Calculate improvement
            improvement = ((values[1] - values[0]) / values[0] * 100) if values[0] > 0 else 0
            ax.set_title(f'{metric.upper()}\n({"+" if improvement >= 0 else ""}{improvement:.2f}% change)')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1.1)
            ax.grid(axis='y', alpha=0.3)
        
        # Remove extra subplot
        fig.delaxes(axes[1, 2])
        
        plt.tight_layout()
        
        # Save comparison plot
        comparison_path = self.results_dir / 'comparison' / 'metrics_comparison.png'
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {comparison_path}")
        plt.close()
        
        # Create improvement summary table
        summary_path = self.results_dir / 'comparison' / 'improvement_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("MODEL PERFORMANCE COMPARISON\n")
            f.write("="*70 + "\n\n")
            f.write(f"{'Metric':<15} {'Original':<12} {'Enhanced':<12} {'Improvement':<15}\n")
            f.write("-"*70 + "\n")
            
            for metric in metrics_to_plot:
                orig_val = metrics_df.loc['original', metric]
                enh_val = metrics_df.loc['enhanced', metric]
                improvement = ((enh_val - orig_val) / orig_val * 100) if orig_val > 0 else 0
                
                f.write(f"{metric:<15} {orig_val:<12.4f} {enh_val:<12.4f} "
                       f"{"+" if improvement >= 0 else ""}{improvement:<14.2f}%\n")
            
            f.write("="*70 + "\n")
        
        print(f"Saved improvement summary to {summary_path}")
        
        # Save metrics as JSON
        metrics_json = {
            'original': metrics_original,
            'enhanced': metrics_enhanced
        }
        
        json_path = self.results_dir / 'comparison' / 'metrics.json'
        with open(json_path, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        
        print(f"Saved metrics JSON to {json_path}")
    
    def run_full_evaluation(self):
        """
        Run complete evaluation pipeline for both models
        """
        print("\n" + "="*70)
        print("FULL MODEL EVALUATION AND COMPARISON (DAY 3)")
        print("="*70 + "\n")
        
        # Load models
        model_original = self.load_model('original')
        model_enhanced = self.load_model('enhanced')
        
        # Evaluate on test set
        metrics_original = self.evaluate_on_test_set(model_original, 'original')
        metrics_enhanced = self.evaluate_on_test_set(model_enhanced, 'enhanced')
        
        # Create visualizations
        self.visualize_detections(model_original, 'original')
        self.visualize_detections(model_enhanced, 'enhanced')
        
        # Analyze errors
        error_original = self.analyze_errors(model_original, 'original')
        error_enhanced = self.analyze_errors(model_enhanced, 'enhanced')
        
        # Compare models
        self.compare_models(metrics_original, metrics_enhanced)
        
        print("\n" + "="*70)
        print("✅ EVALUATION COMPLETE!")
        print("="*70)
        print(f"\n📁 Results saved to: {self.results_dir}")
        print(f"\n📊 Key Findings:")
        
        # Print summary of improvements
        for metric in ['mAP50', 'mAP50-95', 'precision', 'recall']:
            orig = metrics_original[metric]
            enh = metrics_enhanced[metric]
            improvement = ((enh - orig) / orig * 100) if orig > 0 else 0
            symbol = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
            print(f"  {symbol} {metric}: {orig:.4f} → {enh:.4f} "
                  f"({"+" if improvement >= 0 else ""}{improvement:.2f}%)")


def main():
    """
    Main evaluation function
    """
    parser = argparse.ArgumentParser(description='Evaluate Person Detection Models')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(config_path=args.config)
    
    # Run full evaluation
    evaluator.run_full_evaluation()


if __name__ == "__main__":
    main()