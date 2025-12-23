import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import time
import threading
import queue

class DroidCamStream:
    """Class untuk menangani koneksi DroidCam"""
    
    def __init__(self, ip_address="192.168.1.5", port=4747):
        self.ip = ip_address
        self.port = port
        self.url = f"http://{ip_address}:{port}/video"
        self.cap = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.thread = None
        self.connection_status = "disconnected"
        
    def start(self):
        """Start DroidCam stream"""
        try:
            self.cap = cv2.VideoCapture(self.url)
            
            if not self.cap.isOpened():
                self.connection_status = "error"
                raise ConnectionError(f"Cannot connect to DroidCam at {self.url}")
            
            # Set properties for smoother streaming
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 15)
            
            self.running = True
            self.thread = threading.Thread(target=self._stream_loop, daemon=True)
            self.thread.start()
            
            # Wait for first frame
            timeout = time.time() + 5
            while self.frame_queue.empty() and time.time() < timeout:
                time.sleep(0.1)
            
            if self.frame_queue.empty():
                self.connection_status = "timeout"
                raise TimeoutError("DroidCam stream timeout")
            
            self.connection_status = "connected"
            return True
            
        except Exception as e:
            self.connection_status = f"error: {str(e)}"
            raise
    
    def _stream_loop(self):
        """Background thread untuk streaming"""
        retry_count = 0
        max_retries = 3
        
        while self.running and retry_count < max_retries:
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    print(f"DroidCam: Frame read failed (attempt {retry_count + 1}/{max_retries})")
                    retry_count += 1
                    time.sleep(1)
                    continue
                
                retry_count = 0  # Reset on success
                
                # Put frame in queue
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.frame_queue.put(frame)
                
            except Exception as e:
                print(f"DroidCam stream error: {e}")
                retry_count += 1
                time.sleep(1)
    
    def get_frame(self, timeout=1.0):
        """Get latest frame dari DroidCam"""
        try:
            frame = self.frame_queue.get(timeout=timeout)
            return True, frame
        except queue.Empty:
            return False, None
    
    def stop(self):
        """Stop DroidCam stream"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()
        self.connection_status = "disconnected"
    
    def get_status(self):
        """Get connection status"""
        return self.connection_status

class FallDetector:
    """Class utama untuk deteksi jatuh"""
    
    def __init__(self, model_path='best_fall_model.pt', conf_threshold=0.5):
        """
        Initialize Fall Detector
        
        Args:
            model_path: Path ke model YOLOv8
            conf_threshold: Threshold confidence
        """
        # Load model
        self.model = YOLO(model_path)
        self.model.conf = conf_threshold
        
        # ============================================
        # PERBAIKAN DI SINI: 
        # Sesuai dengan model: Class 0 = 'falling', Class 1 = 'normal'
        # ============================================
        self.classes = ['falling', 'normal']
        
        # Colors untuk visualisasi
        self.colors = {
            'falling': (0, 0, 255),      # MERAH untuk falling
            'normal': (0, 255, 0)        # HIJAU untuk normal
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
        
        # DroidCam stream
        self.droidcam = None
        
        # Print debug info
        print(f"✓ FallDetector initialized")
        print(f"  Model: {model_path}")
        print(f"  Class mapping: {self.classes}")
        print(f"  Colors: falling=RED, normal=GREEN")
    
    def detect(self, frame):
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
            
            if result.boxes is not None:
                boxes = result.boxes.cpu().numpy()
                
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].astype(int)
                    conf = boxes.conf[i]
                    cls_id = int(boxes.cls[i])
                    
                    # ============================================
                    # VERIFIKASI: Print class mapping
                    # ============================================
                    print(f"DEBUG: Model predicted class_id={cls_id}")
                    
                    if cls_id < len(self.classes):
                        class_name = self.classes[cls_id]
                        print(f"DEBUG: Mapped to '{class_name}'")
                        
                        # Update statistics
                        if class_name == 'falling':
                            self.fall_detections += 1
                            current_fall = True
                            print(f"DEBUG: FALLING detected! Confidence: {conf:.2%}")
                        else:
                            self.normal_detections += 1
                        
                        # Store detection info
                        detections.append({
                            'bbox': bbox.tolist(),
                            'confidence': float(conf),
                            'class_name': class_name,
                            'class_id': cls_id
                        })
                        
                        # Draw bounding box dengan warna yang sesuai
                        color = self.colors.get(class_name, (255, 255, 255))
                        x1, y1, x2, y2 = bbox
                        
                        # Draw box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label background
                        label = f"{class_name.upper()} {conf:.1%}"
                        (label_width, label_height), baseline = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        
                        cv2.rectangle(frame, 
                                    (x1, y1 - label_height - 10),
                                    (x1 + label_width, y1),
                                    color, -1)
                        
                        # Draw label text
                        cv2.putText(frame, label, 
                                (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Check alert status
        self.fall_buffer.append(current_fall)
        
        if sum(self.fall_buffer) >= self.alert_threshold:
            if not self.alert_active:
                self.alert_active = True
                self.alert_start_time = time.time()
        else:
            self.alert_active = False
        
        # Draw alert if active
        if self.alert_active:
            # Blinking red border
            if int(time.time() * 2) % 2 == 0:
                cv2.rectangle(frame, (0, 0), 
                            (frame.shape[1]-1, frame.shape[0]-1), 
                            (0, 0, 255), 10)
            
            # Alert text
            cv2.putText(frame, "🚨 FALL DETECTED!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        # Draw FPS
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(frame, fps_text, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return frame, detections, self.alert_active
    
    def connect_droidcam(self, ip_address="192.168.1.5", port=4747):
        """Connect to DroidCam"""
        if self.droidcam:
            self.droidcam.stop()
        
        self.droidcam = DroidCamStream(ip_address, port)
        return self.droidcam.start()
    
    def get_droidcam_frame(self):
        """Get frame from DroidCam"""
        if self.droidcam and self.droidcam.connection_status == "connected":
            success, frame = self.droidcam.get_frame(timeout=0.5)
            return success, frame
        return False, None
    
    def disconnect_droidcam(self):
        """Disconnect DroidCam"""
        if self.droidcam:
            self.droidcam.stop()
            self.droidcam = None
    
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
            'fps': self.fps,
            'droidcam_status': self.droidcam.get_status() if self.droidcam else "not_connected"
        }
    
    def reset_statistics(self):
        """Reset semua statistics"""
        self.total_frames = 0
        self.fall_detections = 0
        self.normal_detections = 0
        self.alert_active = False
        self.fall_buffer.clear()
        self.fps = 0
        self.fps_frames = 0
        self.last_fps_time = time.time()