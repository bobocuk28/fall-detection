import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import time

class FallDetector:
    """Fall Detection System - Streamlit Cloud Compatible"""
    
    def __init__(self, model_path='best_fall_model.pt', conf_threshold=0.5):
        """
        Initialize Fall Detector
        
        Args:
            model_path: Path to YOLOv8 model (.pt file)
            conf_threshold: Confidence threshold for detections
        """
        try:
            # Load YOLO model
            self.model = YOLO(model_path)
            self.model.conf = conf_threshold
            
            # Class names (adjust based on your model)
            # Class 0 = falling, Class 1 = normal
            self.classes = ['falling', 'normal']
            
            # Colors for visualization
            self.colors = {
                'falling': (0, 0, 255),      # RED for falling
                'normal': (0, 255, 0)        # GREEN for normal
            }
            
            # Alert system
            self.fall_buffer = deque(maxlen=10)
            self.alert_threshold = 5
            self.alert_active = False
            self.alert_start_time = None
            
            # Statistics
            self.total_frames = 0
            self.fall_detections = 0
            self.normal_detections = 0
            self.fps = 0
            self.last_fps_time = time.time()
            self.fps_frames = 0
            
            print(f"✅ FallDetector initialized")
            print(f"   Model: {model_path}")
            print(f"   Classes: {self.classes}")
            print(f"   Confidence threshold: {conf_threshold}")
            
        except Exception as e:
            print(f"❌ Error initializing detector: {e}")
            raise
    
    def detect(self, frame):
        """
        Detect falls in a frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            processed_frame: Frame with bounding boxes and labels
            detections: List of detections
            alert_status: True if fall alert is active
        """
        self.total_frames += 1
        self.fps_frames += 1
        
        # Calculate FPS
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.fps_frames / (current_time - self.last_fps_time)
            self.last_fps_time = current_time
            self.fps_frames = 0
        
        # Run inference
        results = self.model(frame, verbose=False)
        
        detections = []
        current_fall = False
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.cpu().numpy()
                
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].astype(int)
                    conf = float(boxes.conf[i])
                    cls_id = int(boxes.cls[i])
                    
                    if cls_id < len(self.classes):
                        class_name = self.classes[cls_id]
                        
                        # Update statistics
                        if class_name == 'falling':
                            self.fall_detections += 1
                            current_fall = True
                        else:
                            self.normal_detections += 1
                        
                        # Store detection info
                        detections.append({
                            'bbox': bbox.tolist(),
                            'confidence': conf,
                            'class_name': class_name,
                            'class_id': cls_id
                        })
                        
                        # Draw bounding box
                        color = self.colors.get(class_name, (255, 255, 255))
                        x1, y1, x2, y2 = bbox
                        
                        # Draw rectangle
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label
                        label = f"{class_name.upper()} {conf:.1%}"
                        label_size, _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        
                        # Label background
                        cv2.rectangle(frame, 
                                    (x1, y1 - label_size[1] - 10),
                                    (x1 + label_size[0], y1),
                                    color, -1)
                        
                        # Label text
                        cv2.putText(frame, label, 
                                (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                (255, 255, 255), 2)
        
        # Update alert status
        self.fall_buffer.append(current_fall)
        
        if sum(self.fall_buffer) >= self.alert_threshold:
            if not self.alert_active:
                self.alert_active = True
                self.alert_start_time = time.time()
        else:
            self.alert_active = False
        
        # Draw alert indicator
        if self.alert_active:
            # Blinking red border
            if int(time.time() * 2) % 2 == 0:
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 255), 10)
            
            # Alert text
            cv2.putText(frame, "FALL DETECTED!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        # Draw FPS
        fps_text = f"FPS: {self.fps:.1f}"
        h = frame.shape[0]
        cv2.putText(frame, fps_text, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return frame, detections, self.alert_active
    
    def get_statistics(self):
        """Get current statistics"""
        fall_ratio = self.fall_detections / max(self.total_frames, 1)
        
        return {
            'total_frames': self.total_frames,
            'fall_detections': self.fall_detections,
            'normal_detections': self.normal_detections,
            'fall_ratio': fall_ratio,
            'alert_active': self.alert_active,
            'alert_duration': time.time() - self.alert_start_time if self.alert_start_time else 0,
            'fps': self.fps
        }
    
    def reset_statistics(self):
        """Reset all statistics"""
        self.total_frames = 0
        self.fall_detections = 0
        self.normal_detections = 0
        self.alert_active = False
        self.alert_start_time = None
        self.fall_buffer.clear()
        self.fps = 0
        self.fps_frames = 0
        self.last_fps_time = time.time()
        print("✅ Statistics reset")