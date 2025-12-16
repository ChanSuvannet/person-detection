
"""
Model Training
This script trains YOLOv8 models on original and enhanced datasets
"""
class PersonDetectionTrainer:
    """
    Handles training of YOLO object detection models
    """

    def __init__(self, config_path="config/config.yaml", dataset_type="original"):
        print("Hello World - Trainer initialized")
        self.config_path = config_path
        self.dataset_type = dataset_type


if __name__ == "__main__":
    trainer = PersonDetectionTrainer()
