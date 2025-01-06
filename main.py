import os
import torch
import cv2
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from PIL import Image
import tkinter as tk
from tkinter import messagebox
import time
from datetime import datetime, timedelta
import logging
import threading
import queue
import json
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fire_detection.log'),
        logging.StreamHandler()
    ]
)

class FireDetectionSystem:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_model()
        self.setup_transforms()
        self.load_config()
        self.frame_buffer = []
        self.alert_queue = queue.Queue()
        self.last_alert_time = datetime.now() - timedelta(minutes=5)
        self.fire_detected_count = 0
        self.consecutive_static_frames = 0
        # Add detection time tracking
        self.first_detection_time = None
        self.continuous_detection = False
        
    def setup_model(self):
        try:
            self.model = models.resnet18(weights="IMAGENET1K_V1")
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, 2)
            model_path = "best_fire_detection_model.pth"
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file {model_path} not found!")
                
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model = self.model.to(self.device)
            self.model.eval()
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

    def setup_transforms(self):
        self.data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def load_config(self):
        config_path = "fire_detection_config.json"
        default_config = {
            "min_confidence": 0.7,      # Increased for better accuracy
            "frame_buffer_size": 10,
            "alert_cooldown_minutes": 5,
            "min_fire_size_ratio": 0.005,
            "motion_threshold": 0.98,
            "consecutive_detections_threshold": 3,
            "constant_detection_seconds": 3,
            "color_detection": {
                "min_saturation": 85,    # Adjusted for better color discrimination
                "min_value": 120,
                "red_hue_ranges": [[0, 12], [165, 180]],  # Tightened ranges
                "yellow_hue_range": [12, 30],
                "min_intensity_ratio": 0.4,
                "intensity_threshold": 160,
                "max_roundness": 0.85,    # Maximum roundness ratio for fire regions
                "min_aspect_ratio": 1.2,  # Minimum aspect ratio for fire regions
                "edge_threshold": 50      # Threshold for edge detection
            }
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Ensure all default config keys exist in loaded config
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

    def enhance_frame(self, frame):
        try:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            enhanced_lab = cv2.merge((cl, a, b))
            enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            # Additional enhancement for better fire detection
            enhanced_bgr = cv2.detailEnhance(enhanced_bgr, sigma_s=10, sigma_r=0.15)
            return enhanced_bgr
        except Exception as e:
            logging.error(f"Error in frame enhancement: {e}")
            return frame

    def analyze_light_characteristics(self, frame, roi, mask):
        """Analyze if the region has characteristics of artificial light or reflection."""
        try:
            # Convert ROI to grayscale if it's not already
            if len(roi.shape) > 2:
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                roi_gray = roi

            # Edge detection
            edges = cv2.Canny(roi_gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size

            # Check for uniform intensity (characteristic of artificial lights)
            intensity_std = np.std(roi_gray)
            mean_intensity = np.mean(roi_gray)
            intensity_variation_ratio = intensity_std / (mean_intensity + 1e-6)

            # Check for sudden intensity changes (characteristic of reflections)
            gradient_x = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            gradient_mean = np.mean(gradient_magnitude)

            # Calculate texture features
            glcm = np.zeros((256, 256), dtype=np.uint8)
            for i in range(roi_gray.shape[0]-1):
                for j in range(roi_gray.shape[1]-1):
                    i_intensity = roi_gray[i, j]
                    j_intensity = roi_gray[i, j+1]
                    glcm[i_intensity, j_intensity] += 1

            glcm_norm = glcm / (glcm.sum() + 1e-6)
            texture_uniformity = np.sum(glcm_norm**2)

            # Characteristics of artificial light/reflection:
            is_artificial = (
                edge_density < 0.05 or           # Very few edges
                intensity_variation_ratio < 0.1 or  # Too uniform
                texture_uniformity > 0.8 or      # Too uniform texture
                gradient_mean < 5                # Too smooth gradients
            )

            return not is_artificial

        except Exception as e:
            logging.error(f"Error in light characteristics analysis: {e}")
            return True  # Default to true to avoid false negatives

    def analyze_color_patterns(self, frame):
        try:
            enhanced_frame = self.enhance_frame(frame)
            hsv = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)
            
            # Get color detection parameters from config
            color_config = self.config.get("color_detection", {})
            min_s = color_config.get("min_saturation", 85)
            min_v = color_config.get("min_value", 120)
            red_ranges = color_config.get("red_hue_ranges", [[0, 12], [165, 180]])
            yellow_range = color_config.get("yellow_hue_range", [12, 30])
            min_intensity_ratio = color_config.get("min_intensity_ratio", 0.4)
            intensity_threshold = color_config.get("intensity_threshold", 160)
            max_roundness = color_config.get("max_roundness", 0.85)
            min_aspect_ratio = color_config.get("min_aspect_ratio", 1.2)
            edge_threshold = color_config.get("edge_threshold", 50)

            # Create masks for different color ranges
            masks = []
            
            # Red masks (both ranges)
            for red_range in red_ranges:
                lower_red = np.array([red_range[0], min_s, min_v])
                upper_red = np.array([red_range[1], 255, 255])
                red_mask = cv2.inRange(hsv, lower_red, upper_red)
                masks.append(red_mask)
            
            # Yellow mask
            lower_yellow = np.array([yellow_range[0], min_s, min_v])
            upper_yellow = np.array([yellow_range[1], 255, 255])
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            masks.append(yellow_mask)
            
            # Combine all color masks
            combined_mask = np.zeros_like(masks[0])
            for mask in masks:
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            # Apply morphological operations
            kernel = np.ones((5, 5), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find and filter contours
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_contour_area = frame.shape[0] * frame.shape[1] * self.config["min_fire_size_ratio"]
            valid_contours = []
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > min_contour_area:
                    # Get the region of the contour
                    x, y, w, h = cv2.boundingRect(cnt)
                    
                    # Check aspect ratio
                    aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
                    if aspect_ratio < min_aspect_ratio:
                        continue
                    
                    # Check roundness
                    perimeter = cv2.arcLength(cnt, True)
                    roundness = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
                    if roundness > max_roundness:
                        continue
                    
                    # Get ROI for further analysis
                    roi = frame[y:y+h, x:x+w]
                    roi_mask = combined_mask[y:y+h, x:x+w]
                    
                    # Check if region has fire-like characteristics
                    if self.analyze_light_characteristics(frame, roi, roi_mask):
                        valid_contours.append(cnt)
            
            # Convert valid contours to regions
            fire_regions = [cv2.boundingRect(cnt) for cnt in valid_contours]
            
            # Final decision
            is_fire = len(valid_contours) > 0
            
            return is_fire, combined_mask, fire_regions

        except Exception as e:
            logging.error(f"Error in color pattern analysis: {e}")
            return False, None, []

    def analyze_temporal_patterns(self):
        if len(self.frame_buffer) < 2:
            return False, 0

        try:
            gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in self.frame_buffer]
            diffs = [np.mean(cv2.absdiff(gray_frames[i], gray_frames[i+1])) 
                    for i in range(len(gray_frames)-1)]

            mean_diff = np.mean(diffs)
            std_diff = np.std(diffs)
            
            # Safer frequency analysis
            frequency_analysis = np.fft.fft(diffs)
            frequency_magnitudes = np.abs(frequency_analysis)
            # Only analyze if we have enough data points
            if len(frequency_magnitudes) > 1:
                dominant_frequency = np.argmax(frequency_magnitudes[1:]) + 1
                is_fluctuating = (std_diff > 2.0 and mean_diff > 1.0 and 
                                dominant_frequency > 1 and dominant_frequency < len(diffs)//2)
            else:
                is_fluctuating = std_diff > 2.0 and mean_diff > 1.0
            
            return is_fluctuating, mean_diff
        except Exception as e:
            logging.error(f"Error in temporal pattern analysis: {e}")
            return False, 0

    def detect_motion(self, prev_frame, curr_frame):
        if prev_frame is None:
            return False, None, 0

        try:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
            # Multi-scale motion detection
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mean_magnitude = np.mean(magnitude)
            
            # Advanced motion analysis
            motion_mask = magnitude > mean_magnitude
            motion_ratio = np.sum(motion_mask) / motion_mask.size
            
            return motion_ratio > 0.01, motion_mask, mean_magnitude
        except Exception as e:
            logging.error(f"Error in motion detection: {e}")
            return False, None, 0

    def save_detection_data(self, frame, confidence, fire_regions):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            detection_dir = Path("fire_detections")
            detection_dir.mkdir(exist_ok=True)
            
            # Save annotated frame
            annotated_frame = frame.copy()
            for x, y, w, h in fire_regions:
                cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(annotated_frame, f"Fire: {confidence:.2f}", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            cv2.imwrite(str(detection_dir / f"fire_detection_{timestamp}.jpg"), annotated_frame)
            
            # Save detection metadata
            metadata = {
                "timestamp": timestamp,
                "confidence": float(confidence),
                "fire_regions": fire_regions,
                "device": str(self.device)
            }
            with open(detection_dir / f"detection_{timestamp}.json", 'w') as f:
                json.dump(metadata, f, indent=4)
                
        except Exception as e:
            logging.error(f"Error saving detection data: {e}")

    def send_alert(self, confidence, fire_regions):
        current_time = datetime.now()
        if (current_time - self.last_alert_time).total_seconds() < self.config["alert_cooldown_minutes"] * 60:
            return

        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            
            alert_message = (
                f"Fire detected with {confidence:.2f}% confidence!\n"
                f"Number of fire regions: {len(fire_regions)}\n"
                f"Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            messagebox.showwarning("FIRE ALERT!", alert_message)
            root.destroy()
            
            self.last_alert_time = current_time
            logging.warning(f"Fire alert sent: {alert_message}")
            
        except Exception as e:
            logging.error(f"Error sending alert: {e}")

    def process_frame(self, frame):
        try:
            # Convert frame for model input
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            input_tensor = self.data_transforms(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence = probabilities[0][1].item()  # Probability of fire class
            
            # Multiple analysis methods
            color_detected, color_mask, fire_regions = self.analyze_color_patterns(frame)
            temporal_detected, _ = self.analyze_temporal_patterns()
            motion_detected, _, _ = self.detect_motion(
                self.frame_buffer[-1] if self.frame_buffer else None, 
                frame
            )
            
            # Less strict combination of detection methods
            current_detection = (
                confidence > self.config["min_confidence"] and 
                color_detected and 
                (temporal_detected or motion_detected)
            )
            
            current_time = datetime.now()
            
            # Handle continuous detection logic
            constant_detection_seconds = self.config.get("constant_detection_seconds", 3)  # Default to 3 if not found
            
            if current_detection:
                if self.first_detection_time is None:
                    self.first_detection_time = current_time
                    self.continuous_detection = False
                elif (current_time - self.first_detection_time).total_seconds() >= constant_detection_seconds:
                    self.continuous_detection = True
            else:
                self.first_detection_time = None
                self.continuous_detection = False
            
            # Update frame buffer
            self.frame_buffer.append(frame)
            if len(self.frame_buffer) > self.config["frame_buffer_size"]:
                self.frame_buffer.pop(0)
            
            if self.continuous_detection:
                self.fire_detected_count += 1
                if self.fire_detected_count >= self.config["consecutive_detections_threshold"]:
                    self.save_detection_data(frame, confidence * 100, fire_regions)
                    self.send_alert(confidence * 100, fire_regions)
            else:
                self.fire_detected_count = max(0, self.fire_detected_count - 1)
            
            return self.continuous_detection, confidence, color_mask if color_detected else None, fire_regions, current_detection
            
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            return False, 0.0, None, [], False

    def run_detection(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logging.error("Error: Could not open webcam.")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logging.error("Error: Could not read frame.")
                    break

                is_fire, confidence, mask, fire_regions, current_detection = self.process_frame(frame)
                
                # Display results
                display_frame = frame.copy()
                
                # Add detection status visualization
                status_text = "Detecting..." if current_detection else "No Fire"
                if current_detection and not self.continuous_detection:
                    constant_detection_seconds = self.config.get("constant_detection_seconds", 3)
                    remaining_time = max(0, constant_detection_seconds - 
                                      (datetime.now() - self.first_detection_time).total_seconds())
                    status_text = f"Validating... {remaining_time:.1f}s"
                elif self.continuous_detection:
                    status_text = "FIRE DETECTED!"

                cv2.putText(display_frame, status_text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                if is_fire and mask is not None:
                    # Overlay fire detection visualization
                    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    mask_colored[mask > 0] = [0, 0, 255]  # Red color for fire regions
                    display_frame = cv2.addWeighted(display_frame, 0.7, mask_colored, 0.3, 0)
                    
                    for x, y, w, h in fire_regions:
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(display_frame, f"Fire: {confidence:.2f}", 
                                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                cv2.imshow('Fire Detection', display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            logging.error(f"Error in main detection loop: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        detector = FireDetectionSystem()
        detector.run_detection()
    except Exception as e:
        logging.critical(f"Critical error in main program: {e}")
