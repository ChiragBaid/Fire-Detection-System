import os
import cv2
import numpy as np
import torch
import logging
import json
import queue
import datetime
import threading
from ultralytics import YOLO
import tkinter as tk
from tkinter import messagebox
from datetime import datetime, timedelta
from pathlib import Path
from skimage.feature import graycomatrix, graycoprops

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fire_detection_yolov8.log'),
        logging.StreamHandler()
    ]
)

class TextureFeatureExtractor:
    def __init__(self):
        self.distances = [1, 2, 3]
        self.angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        self.properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        
    def extract_features(self, img):
        try:
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
                
            # Normalize to 8-bit
            if gray.dtype != np.uint8:
                gray = (gray * 255).astype(np.uint8)
                
            # Calculate GLCM
            glcm = graycomatrix(gray, distances=self.distances, angles=self.angles,
                            symmetric=True, normed=True)
            
            # Extract features
            features = []
            for prop in self.properties:
                feature = graycoprops(glcm, prop).ravel()
                features.extend(feature)
                
            return np.array(features)
        except Exception as e:
            logging.error(f"Error extracting texture features: {e}")
            return np.zeros(len(self.distances) * len(self.angles) * len(self.properties))

    def analyze_texture(self, features):
        try:
            # Define typical fire texture characteristics
            fire_characteristics = {
                'contrast': (4.0, 15.0),      # Fire tends to have high contrast
                'dissimilarity': (1.5, 5.0),  # Fire regions show significant dissimilarity
                'homogeneity': (0.3, 0.7),    # Fire texture is moderately homogeneous
                'energy': (0.1, 0.4),         # Fire patterns have moderate energy
                'correlation': (0.5, 0.9)      # Fire regions show high correlation
            }
            
            # Calculate average values for each property
            avg_values = {}
            feature_idx = 0
            for prop in self.properties:
                prop_values = features[feature_idx:feature_idx + len(self.distances) * len(self.angles)]
                avg_values[prop] = np.mean(prop_values)
                feature_idx += len(self.distances) * len(self.angles)
            
            # Check if values fall within fire characteristics ranges
            matches = 0
            for prop, (min_val, max_val) in fire_characteristics.items():
                if min_val <= avg_values[prop] <= max_val:
                    matches += 1
            
            # Calculate confidence based on number of matching characteristics
            confidence = matches / len(fire_characteristics)
            return confidence
            
        except Exception as e:
            logging.error(f"Error analyzing texture: {e}")
            return 0.0

class FireDetectionSystem:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.texture_extractor = TextureFeatureExtractor()
        self.load_config()
        self.frame_buffer = []
        self.alert_queue = queue.Queue()
        self.last_alert_time = datetime.now() - timedelta(minutes=5)
        self.fire_detected_count = 0
        self.consecutive_static_frames = 0
        self.first_detection_time = None
        self.continuous_detection = False
        self.screenshot_dir = "fire_detections"
        os.makedirs(self.screenshot_dir, exist_ok=True)
        
    def load_model(self):
        try:
            # Load YOLOv8 model
            model_path = "runs/train/fire_detection_yolov8_optimized/weights/best.pt"
            if not os.path.exists(model_path):
                logging.error(f"Model not found at {model_path}")
                raise FileNotFoundError(f"Model not found at {model_path}")
            
            model = YOLO(model_path)
            model.to(self.device)
            logging.info("YOLOv8 model loaded successfully")
            return model
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

    def load_config(self):
        config_path = "fire_detection_config.json"
        default_config = {
            "min_confidence": 0.4,        # Increased threshold for more reliable detections
            "frame_buffer_size": 5,       # Reduced for faster response
            "alert_cooldown_minutes": 1,  # Reduced for more frequent alerts
            "consecutive_detections_threshold": 2,  # Reduced for faster detection
            "constant_detection_seconds": 2,       # Reduced for faster response
            "detection_area_threshold": 50,        # Reduced minimum area for detecting smaller fires
            "max_detections_per_frame": 5,        # Reduced to focus on clearer detections
            "nms_threshold": 0.4,                 # Adjusted for better overlapping detection handling
            "frame_skip": 1,                      # Process every frame for better accuracy
            "color_thresholds": {
                "red_lower": [0, 120, 100],      # HSV lower bound for fire colors
                "red_upper": [10, 255, 255],     # HSV upper bound for fire colors
                "yellow_lower": [20, 100, 100],  # HSV lower bound for yellow flames
                "yellow_upper": [35, 255, 255]   # HSV upper bound for yellow flames
            },
            "motion_detection": {
                "enabled": True,
                "history": 20,
                "var_threshold": 16,
                "detect_shadows": False
            }
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    self.config = default_config.copy()
                    self.config.update(loaded_config)
            else:
                self.config = default_config
                with open(config_path, 'w') as f:
                    json.dump(default_config, f, indent=4)
            logging.info("Configuration loaded successfully")
        except Exception as e:
            logging.warning(f"Error loading config: {e}. Using default values.")
            self.config = default_config

    def color_filter(self, frame):
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Define fire color ranges in LAB
            # L: Brightness (0-100)
            # a: Red-Green (negative = green, positive = red)
            # b: Blue-Yellow (negative = blue, positive = yellow)
            
            # Create masks for different fire characteristics
            # Bright regions
            l_mask = cv2.inRange(l, 50, 100)  # Medium to high brightness
            
            # Red-Yellow regions (high a and b values)
            a_mask = cv2.inRange(a, 128 + 20, 255)  # Strong red component
            b_mask = cv2.inRange(b, 128 + 10, 255)  # Yellow component
            
            # Combine masks
            fire_mask = cv2.bitwise_and(l_mask, cv2.bitwise_and(a_mask, b_mask))
            
            # Apply morphological operations
            kernel = np.ones((3,3), np.uint8)
            fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
            fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)
            
            # Additional filtering for better accuracy
            contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            refined_mask = np.zeros_like(fire_mask)
            
            min_area = frame.shape[0] * frame.shape[1] * 0.001  # Minimum area threshold
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    # Calculate aspect ratio and roundness
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h if h > 0 else 0
                    perimeter = cv2.arcLength(contour, True)
                    roundness = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                    
                    # Fire tends to have irregular shapes
                    if aspect_ratio > 0.2 and roundness < 0.8:
                        cv2.drawContours(refined_mask, [contour], -1, 255, -1)
            
            return refined_mask
            
        except Exception as e:
            logging.error(f"Error in color filtering: {e}")
            return None

    def enhance_frame(self, frame):
        try:
            # Convert to LAB for better color enhancement
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Enhance luminance
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            
            # Enhance color channels (a and b)
            # Boost red and yellow components
            a = cv2.add(a, 5)  # Boost red
            b = cv2.add(b, 3)  # Slightly boost yellow
            
            # Merge channels
            enhanced_lab = cv2.merge([cl, a, b])
            enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            # Apply subtle sharpening
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]]) / 9.0
            enhanced_bgr = cv2.filter2D(enhanced_bgr, -1, kernel)
            
            return enhanced_bgr
            
        except Exception as e:
            logging.error(f"Error enhancing frame: {e}")
            return frame

    def process_frame(self, frame):
        try:
            # Enhance frame
            enhanced_frame = self.enhance_frame(frame)
            
            # Apply color filtering
            fire_mask = self.color_filter(enhanced_frame)
            if fire_mask is None:
                return [], enhanced_frame
            
            # Run YOLOv8 detection
            results = self.model(enhanced_frame, conf=self.config["min_confidence"])
            
            # Process detections
            detections = []
            if len(results) > 0:
                result = results[0]  # Get first result
                boxes = result.boxes
                
                for box in boxes:
                    confidence = float(box.conf[0])
                    if confidence >= self.config["min_confidence"]:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Extract ROI for texture analysis
                        roi = enhanced_frame[y1:y2, x1:x2]
                        if roi.size > 0:  # Check if ROI is valid
                            # Get texture features
                            texture_features = self.texture_extractor.extract_features(roi)
                            texture_confidence = self.texture_extractor.analyze_texture(texture_features)
                            
                            # Check if detection overlaps with fire-colored regions
                            roi_mask = fire_mask[y1:y2, x1:x2]
                            if roi_mask.size > 0:
                                fire_pixel_ratio = np.count_nonzero(roi_mask) / roi_mask.size
                                if fire_pixel_ratio > 0.3:  # At least 30% fire-colored pixels
                                    # Combine confidences from YOLOv8, color, and texture
                                    combined_confidence = (
                                        confidence * 0.4 +          # YOLOv8 detection weight
                                        fire_pixel_ratio * 0.3 +    # Color detection weight
                                        texture_confidence * 0.3    # Texture analysis weight
                                    )
                                    
                                    detections.append({
                                        'bbox': (x1, y1, x2, y2),
                                        'confidence': combined_confidence,
                                        'yolo_conf': confidence,
                                        'color_conf': fire_pixel_ratio,
                                        'texture_conf': texture_confidence
                                    })
            
            # Sort detections by confidence and apply limit
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            detections = detections[:self.config["max_detections_per_frame"]]
            
            return detections, enhanced_frame
            
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            return [], frame

    def draw_detections(self, frame, detections):
        try:
            overlay = frame.copy()
            alpha = 0.3  # Transparency factor
            
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                conf = det['confidence']
                yolo_conf = det.get('yolo_conf', 0)
                color_conf = det.get('color_conf', 0)
                texture_conf = det.get('texture_conf', 0)
                
                # Draw filled rectangle with transparency
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
                
                # Add the transparent overlay
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                
                # Draw border
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Draw confidence scores with background
                labels = [
                    f'Combined: {conf:.2f}',
                    f'YOLO: {yolo_conf:.2f}',
                    f'Color: {color_conf:.2f}',
                    f'Texture: {texture_conf:.2f}'
                ]
                
                y_offset = y1 - 10
                for label in labels:
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(frame, (x1, y_offset-h-5), (x1+w+10, y_offset+5), (0, 0, 255), -1)
                    cv2.putText(frame, label, (x1+5, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    y_offset -= h + 10
                
            return frame
        except Exception as e:
            logging.error(f"Error drawing detections: {e}")
            return frame

    def save_detection_data(self, frame, detections, timestamp):
        try:
            # Create filename with timestamp
            filename = f"fire_detection_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            
            # Save frame with detections
            frame_path = os.path.join(self.screenshot_dir, f"{filename}.jpg")
            cv2.imwrite(frame_path, frame)
            
            # Save detection data
            data = {
                'timestamp': timestamp.isoformat(),
                'detections': [
                    {
                        'bbox': list(det['bbox']),
                        'confidence': float(det['confidence']),
                        'yolo_conf': float(det.get('yolo_conf', 0)),
                        'color_conf': float(det.get('color_conf', 0)),
                        'texture_conf': float(det.get('texture_conf', 0))
                    }
                    for det in detections
                ]
            }
            
            json_path = os.path.join(self.screenshot_dir, f"{filename}.json")
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=4)
                
            logging.info(f"Detection data saved to {json_path}")
            
        except Exception as e:
            logging.error(f"Error saving detection data: {e}")

    def send_alert(self, detections):
        try:
            current_time = datetime.now()
            if (current_time - self.last_alert_time).total_seconds() > self.config["alert_cooldown_minutes"] * 60:
                max_conf = max(det['confidence'] for det in detections)
                alert_msg = f"FIRE DETECTED!\nConfidence: {max_conf:.2f}\nNumber of detections: {len(detections)}"
                
                # Add alert sound
                print('\a')  # System beep
                
                # Send to alert queue
                self.alert_queue.put({
                    'message': alert_msg,
                    'timestamp': current_time,
                    'confidence': max_conf,
                    'num_detections': len(detections)
                })
                
                self.last_alert_time = current_time
                logging.info(f"Alert sent: {alert_msg}")
                
        except Exception as e:
            logging.error(f"Error sending alert: {e}")

    def process_alerts(self):
        try:
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            
            while True:
                if not self.alert_queue.empty():
                    alert = self.alert_queue.get()
                    messagebox.showwarning("Fire Alert", alert['message'])
                threading.Event().wait(1.0)  # Check every second
        except Exception as e:
            logging.error(f"Error in alert processing: {e}")

    def run_detection(self):
        try:
            # Start alert processing thread
            alert_thread = threading.Thread(target=self.process_alerts, daemon=True)
            alert_thread.start()
            
            # Open video capture
            cap = cv2.VideoCapture(0)  # Use 0 for webcam
            if not cap.isOpened():
                raise RuntimeError("Error opening video capture")
            
            # Set camera properties for better quality
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames based on config
                frame_count += 1
                if frame_count % self.config["frame_skip"] != 0:
                    continue
                
                # Process frame
                detections, enhanced_frame = self.process_frame(frame)
                
                # Draw detections
                output_frame = self.draw_detections(enhanced_frame.copy(), detections)
                
                # Add FPS counter
                fps = cap.get(cv2.CAP_PROP_FPS)
                cv2.putText(output_frame, f'FPS: {int(fps)}', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Handle detections
                if detections:
                    self.fire_detected_count += 1
                    if self.fire_detected_count >= self.config["consecutive_detections_threshold"]:
                        current_time = datetime.now()
                        
                        # Save detection data
                        self.save_detection_data(output_frame, detections, current_time)
                        
                        # Send alert
                        self.send_alert(detections)
                else:
                    self.fire_detected_count = max(0, self.fire_detected_count - 1)
                
                # Display frame
                cv2.imshow('Fire Detection', output_frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            logging.error(f"Error in detection loop: {e}")
            raise

if __name__ == "__main__":
    try:
        detector = FireDetectionSystem()
        detector.run_detection()
    except Exception as e:
        logging.error(f"Error in main: {e}")
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
