import torch
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
import os
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from torchvision import transforms
import time

class ObjectDetectionModel:
    
    def __init__(self, model_name='yolov5s', confidence_threshold=0.5, device='auto'):
        self.confidence_threshold = confidence_threshold
        self.device = self._get_device(device)
        self.model = self._load_model(model_name)
        self.class_names = self.model.names
        self.results_history = []
        
        print(f"Model loaded: {model_name}")
        print(f"Device: {self.device}")
        print(f"Classes: {len(self.class_names)}")
        
    def _get_device(self, device):
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _load_model(self, model_name):
        try:
            # Use local YOLOv5 repo to avoid GitHub rate limits
            model = torch.hub.load(
                './yolov5',  # local path to cloned repo
                model_name,
                source='local',
                pretrained=True
            )
            model.to(self.device)
            return model
        except Exception as e:
            print(f"Error loading model locally: {e}")
            print("Please ensure YOLOv5 is cloned locally and requirements are installed.")
            return None

    def detect_objects(self, image_path, save_results=True):
        try:
            if image_path.startswith('http'):
                response = requests.get(image_path)
                image = Image.open(BytesIO(response.content))
            else:
                image = Image.open(image_path)
            
            results = self.model(image)
            
            detections = results.pandas().xyxy[0]
            
            detections = detections[detections['confidence'] >= self.confidence_threshold]
            
            detection_info = {
                'image_path': image_path,
                'detections': detections,
                'num_objects': len(detections),
                'classes_detected': detections['name'].unique().tolist() if len(detections) > 0 else [],
                'image_size': image.size,
                'inference_time': None
            }
            
            if save_results:
                self.results_history.append(detection_info)
            
            return results, detection_info
            
        except Exception as e:
            print(f"Error detecting objects in {image_path}: {e}")
            return None, None
    
    def detect_batch(self, image_paths, batch_size=4):
        all_results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_results = []
            
            for path in batch_paths:
                results, detection_info = self.detect_objects(path)
                if results is not None:
                    batch_results.append((results, detection_info))
            
            all_results.extend(batch_results)
            print(f"Processed batch {i//batch_size + 1}/{(len(image_paths)-1)//batch_size + 1}")
        
        return all_results
    
    def visualize_detections(self, image_path, figsize=(12, 8), save_path=None):
        results, detection_info = self.detect_objects(image_path)
        
        if results is None:
            print("No results to visualize")
            return
        
        rendered_image = results.render()[0]
        
        # Ensure rendered_image is a numpy array and convert to RGB for matplotlib
        if isinstance(rendered_image, np.ndarray):
            rendered_image = cv2.cvtColor(rendered_image, cv2.COLOR_BGR2RGB)
        else:
            rendered_image = np.array(rendered_image)
        
        plt.figure(figsize=figsize)
        plt.imshow(rendered_image)
        plt.axis('off')
        plt.title(f'Object Detection Results\n'
                 f'Objects detected: {detection_info["num_objects"]}\n'
                 f'Classes: {", ".join(detection_info["classes_detected"])}')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        else:
            plt.savefig(f"{os.path.basename(image_path)}_detection.png")
        plt.close()
        
        return rendered_image, detection_info

    def plot_class_distribution(self, figsize=(12, 6)):
        if not self.results_history:
            print("No detection history available")
            return
        
        all_classes = []
        for result in self.results_history:
            all_classes.extend(result['classes_detected'])
        
        if not all_classes:
            print("No classes detected in history")
            return
        
        class_counts = pd.Series(all_classes).value_counts()
        
        plt.figure(figsize=figsize)
        sns.barplot(y=class_counts.index, x=class_counts.values, orient='h')
        plt.title('Distribution of Detected Object Classes')
        plt.xlabel('Number of Detections')
        plt.ylabel('Object Class')
        plt.tight_layout()
        plt.show()
        
        plt.savefig("class_distribution.png")
        plt.close()
        
        return class_counts

    def _load_model(self, model_name):
        try:
            # Use local YOLOv5 repo to avoid GitHub rate limits
            model = torch.hub.load(
                './yolov5',  # local path to cloned repo
                model_name,
                source='local',
                pretrained=True
            )
            model.to(self.device)
            return model
        except Exception as e:
            print(f"Error loading model locally: {e}")
            print("Please ensure YOLOv5 is cloned locally and requirements are installed.")
            return None

    def detect_objects(self, image_path, save_results=True):
        try:
            if image_path.startswith('http'):
                response = requests.get(image_path)
                image = Image.open(BytesIO(response.content))
            else:
                image = Image.open(image_path)
            
            results = self.model(image)
            
            detections = results.pandas().xyxy[0]
            
            detections = detections[detections['confidence'] >= self.confidence_threshold]
            
            detection_info = {
                'image_path': image_path,
                'detections': detections,
                'num_objects': len(detections),
                'classes_detected': detections['name'].unique().tolist() if len(detections) > 0 else [],
                'image_size': image.size,
                'inference_time': None
            }
            
            if save_results:
                self.results_history.append(detection_info)
            
            return results, detection_info
            
        except Exception as e:
            print(f"Error detecting objects in {image_path}: {e}")
            return None, None
    
    def detect_batch(self, image_paths, batch_size=4):
        all_results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_results = []
            
            for path in batch_paths:
                results, detection_info = self.detect_objects(path)
                if results is not None:
                    batch_results.append((results, detection_info))
            
            all_results.extend(batch_results)
            print(f"Processed batch {i//batch_size + 1}/{(len(image_paths)-1)//batch_size + 1}")
        
        return all_results
    
    def visualize_detections(self, image_path, figsize=(12, 8), save_path=None):
        results, detection_info = self.detect_objects(image_path)
        
        if results is None:
            print("No results to visualize")
            return
        
        rendered_image = results.render()[0]
        
        # Ensure rendered_image is a numpy array and convert to RGB for matplotlib
        if isinstance(rendered_image, np.ndarray):
            rendered_image = cv2.cvtColor(rendered_image, cv2.COLOR_BGR2RGB)
        else:
            rendered_image = np.array(rendered_image)
        
        plt.figure(figsize=figsize)
        plt.imshow(rendered_image)
        plt.axis('off')
        plt.title(f'Object Detection Results\n'
                 f'Objects detected: {detection_info["num_objects"]}\n'
                 f'Classes: {", ".join(detection_info["classes_detected"])}')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        else:
            plt.savefig(f"{os.path.basename(image_path)}_detection.png")
        plt.close()
        
        return rendered_image, detection_info

    def evaluate_model(self, test_images, ground_truth=None):
        print("Evaluating model performance...")
        
        inference_times = []
        detection_counts = []
        class_distributions = {}
        
        for image_path in test_images:
            start_time = time.time()
            results, detection_info = self.detect_objects(image_path)
            inference_time = time.time() - start_time
            
            if detection_info:
                inference_times.append(inference_time)
                detection_counts.append(detection_info['num_objects'])
                
                for class_name in detection_info['classes_detected']:
                    class_distributions[class_name] = class_distributions.get(class_name, 0) + 1
        
        evaluation_results = {
            'total_images': len(test_images),
            'avg_inference_time': np.mean(inference_times) if inference_times else 0,
            'avg_detections_per_image': np.mean(detection_counts) if detection_counts else 0,
            'class_distribution': class_distributions,
            'detection_rate': len([d for d in detection_counts if d > 0]) / len(test_images) if test_images else 0
        }
        
        return evaluation_results
    
    def generate_report(self, evaluation_results=None):
        print("="*60)
        print("OBJECT DETECTION MODEL REPORT")
        print("="*60)
        
        print(f"\nModel Configuration:")
        print(f"- Model: YOLOv5")
        print(f"- Device: {self.device}")
        print(f"- Confidence Threshold: {self.confidence_threshold}")
        print(f"- Total Classes: {len(self.class_names)}")
        
        if evaluation_results:
            print(f"\nPerformance Metrics:")
            print(f"- Total Images Processed: {evaluation_results['total_images']}")
            print(f"- Average Inference Time: {evaluation_results['avg_inference_time']:.4f} seconds")
            print(f"- Average Detections per Image: {evaluation_results['avg_detections_per_image']:.2f}")
            print(f"- Detection Rate: {evaluation_results['detection_rate']:.2%}")
            
            print(f"\nClass Distribution:")
            for class_name, count in sorted(evaluation_results['class_distribution'].items()):
                print(f"- {class_name}: {count}")
        
        if self.results_history:
            print(f"\nOverall Statistics:")
            total_detections = sum([r['num_objects'] for r in self.results_history])
            print(f"- Total Images Processed: {len(self.results_history)}")
            print(f"- Total Objects Detected: {total_detections}")
            
            all_classes = []
            for result in self.results_history:
                all_classes.extend(result['classes_detected'])
            unique_classes = list(set(all_classes))
            print(f"- Unique Classes Detected: {len(unique_classes)}")
            print(f"- Classes: {', '.join(sorted(unique_classes))}")
        
        print("="*60)
    
    def plot_class_distribution(self, figsize=(12, 6)):
        if not self.results_history:
            print("No detection history available")
            return
        
        all_classes = []
        for result in self.results_history:
            all_classes.extend(result['classes_detected'])
        
        if not all_classes:
            print("No classes detected in history")
            return
        
        class_counts = pd.Series(all_classes).value_counts()
        
        plt.figure(figsize=figsize)
        sns.barplot(y=class_counts.index, x=class_counts.values, orient='h')
        plt.title('Distribution of Detected Object Classes')
        plt.xlabel('Number of Detections')
        plt.ylabel('Object Class')
        plt.tight_layout()
        plt.show()
        
        plt.savefig("class_distribution.png")
        plt.close()
        
        return class_counts

def demo_with_sample_images():
    print("Initializing Object Detection Model...")
    
    detector = ObjectDetectionModel(
        model_name='yolov5s',
        confidence_threshold=0.5,
        device='auto'
    )
    
    # Use your provided image paths
    sample_images = [
        r"c:\Users\Shubham\Downloads\hazel123.jpg",
        r"c:\Users\Shubham\Downloads\tatum.jpg",
    ]
    
    print(f"\nTesting with {len(sample_images)} sample images...")
    
    for i, image_path in enumerate(sample_images, 1):
        print(f"\nProcessing image {i}/{len(sample_images)}")
        
        try:
            rendered_image, detection_info = detector.visualize_detections(
                image_path, 
                figsize=(10, 6)
            )
            
            if detection_info:
                print(f"- Objects detected: {detection_info['num_objects']}")
                print(f"- Classes: {', '.join(detection_info['classes_detected'])}")
            
        except Exception as e:
            print(f"Error processing image {i}: {e}")
    
    evaluation_results = detector.evaluate_model(sample_images)
    
    detector.generate_report(evaluation_results)
    
    detector.plot_class_distribution()
    
    return detector, evaluation_results

def custom_dataset_training_example():
    print("\nCustom Dataset Training Example:")
    print("="*50)
    
    print("""
    To train on a custom dataset, you would typically:
    
    1. Prepare Dataset:
       - Collect images
       - Create annotations in YOLO format
       - Split into train/val/test sets
    
    2. Dataset Structure:
       dataset/
       ├── images/
       │   ├── train/
       │   ├── val/
       │   └── test/
       └── labels/
           ├── train/
           ├── val/
           └── test/
    
    3. Training Configuration:
       - Modify data.yaml file
       - Set number of classes
       - Configure training parameters
    
    4. Training Command:
       python train.py --data custom_data.yaml --cfg yolov5s.yaml --weights yolov5s.pt --epochs 100
    
    5. Evaluation:
       python val.py --data custom_data.yaml --weights runs/train/exp/weights/best.pt
    """)

if __name__ == "__main__":
    print("Object Detection and Recognition Model")
    print("Using Pre-trained YOLOv5 Model")
    print("="*60)
    
    detector, results = demo_with_sample_images()
    
    custom_dataset_training_example()
    
    print("\nDemo completed successfully!")
    print("The model has been tested and evaluated on sample images.")
    print("Refer to the generated report for detailed performance metrics.")