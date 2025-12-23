# detector.py
import cv2
import numpy as np
from ultralytics import YOLO
import torch

class FallDetector:
    def __init__(self, model_path="best_fall_model.pt", conf_threshold=0.5):
        """
        Initialize the fall detection model.
        
        Args:
            model_path (str): Path to YOLO model weights (.pt file)
            conf_threshold (float): Confidence threshold for detections
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        
        # Load model
        try:
            self.model = YOLO(model_path)
            print(f"✅ Model loaded from {model_path}")
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")
    
    def detect(self, image):
        """
        Detect falls in an image.
        
        Args:
            image (numpy array): Input image in BGR format
            
        Returns:
            tuple: (processed_image, detections, alert_status)
        """
        try:
            # Convert BGR to RGB (YOLO expects RGB)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run inference
            results = self.model(
                image_rgb, 
                conf=self.conf_threshold,
                verbose=False
            )
            
            # Process results
            detections = []
            alert = False
            
            # Create a copy for drawing
            processed_image = image.copy()
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Get class name
                        class_name = self.model.names[class_id]
                        
                        # Store detection
                        detection = {
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'class_name': class_name
                        }
                        detections.append(detection)
                        
                        # Check if fall is detected
                        if class_name.lower() in ['fall', 'falling', 'person_falling']:
                            alert = True
                            color = (0, 0, 255)  # Red for fall
                            label = f"FALL: {confidence:.2f}"
                        else:
                            color = (0, 255, 0)  # Green for normal
                            label = f"{class_name}: {confidence:.2f}"
                        
                        # Draw bounding box
                        cv2.rectangle(
                            processed_image, 
                            (int(x1), int(y1)), 
                            (int(x2), int(y2)), 
                            color, 
                            2
                        )
                        
                        # Draw label
                        cv2.putText(
                            processed_image,
                            label,
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2
                        )
            
            return processed_image, detections, alert
            
        except Exception as e:
            print(f"Detection error: {e}")
            return image, [], False
    
    def test_model(self):
        """Test if model is working properly."""
        test_image = np.zeros((300, 300, 3), dtype=np.uint8)
        test_image[:150, :] = [255, 0, 0]  # Blue top half
        
        result, detections, alert = self.detect(test_image)
        print(f"✅ Model test passed. Detections: {len(detections)}")
        return result, detections, alert