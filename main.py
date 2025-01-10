import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import logging
import json
import queue
import datetime
import threading
from skimage.feature import graycomatrix, graycoprops
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

class TextureFeatureExtractor:
    def __init__(self):
        self.distances = [1, 2, 3]
        self.angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        self.properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        
    def extract_features(self, img):
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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

class FireTextureNet(nn.Module):
    def __init__(self, num_texture_features):
        super(FireTextureNet, self).__init__()
        
        # Load pre-trained ResNet
        self.resnet = models.resnet18(weights="IMAGENET1K_V1")
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove final FC layer
        
        # Texture feature processing
        self.texture_fc = nn.Sequential(
            nn.Linear(num_texture_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Combined features processing
        self.combined_fc = nn.Sequential(
            nn.Linear(num_features + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
        
    def forward(self, x):
        img, texture_features = x
        
        # Process image through ResNet
        img_features = self.resnet(img)
        
        # Process texture features
        texture_processed = self.texture_fc(texture_features)
        
        # Combine features
        combined = torch.cat((img_features, texture_processed), dim=1)
        
        # Final classification
        return self.combined_fc(combined)

class FireDetectionSystem:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.texture_extractor = TextureFeatureExtractor()
        self.model = self.load_model("best_fire_detection_model.pth")
        self.load_config()
        self.frame_buffer = []
        self.alert_queue = queue.Queue()
        self.last_alert_time = datetime.now() - timedelta(minutes=5)
        self.fire_detected_count = 0
        self.consecutive_static_frames = 0
        # Add detection time tracking
        self.first_detection_time = None
        self.continuous_detection = False
        # Add screenshot directory
        self.screenshot_dir = "fire_detections"
        os.makedirs(self.screenshot_dir, exist_ok=True)
        
    def load_model(self, model_path):
        try:
            # Get number of texture features
            sample_image = np.zeros((224, 224, 3), dtype=np.uint8)  # Create a dummy image
            num_texture_features = len(self.texture_extractor.extract_features(sample_image))
            
            # Create model with correct architecture
            model = FireTextureNet(num_texture_features)
            
            # Load state dict
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model = model.to(self.device)
            model.eval()
            logging.info("Model loaded successfully")
            return model
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

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
        """Analyze if the region has characteristics of artificial light or reflection,
        with specific handling for low-light backgrounds and reflections."""
        try:
            # Convert ROI to different color spaces for better analysis
            if len(roi.shape) > 2:
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            else:
                roi_gray = roi
                roi_hsv = cv2.cvtColor(cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)
                roi_lab = cv2.cvtColor(cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2LAB)

            # Low-light background analysis
            mean_intensity = np.mean(roi_gray)
            is_low_light = mean_intensity < 50

            # Enhanced color variation analysis
            hsv_channels = cv2.split(roi_hsv)
            lab_channels = cv2.split(roi_lab)
            
            # Analyze temporal color variation (if region is static, likely a reflection)
            color_std = np.std(hsv_channels[0])  # Hue standard deviation
            sat_std = np.std(hsv_channels[1])
            val_std = np.std(hsv_channels[2])
            
            # Check for color consistency in LAB space
            a_std = np.std(lab_channels[1])  # a channel variation (green-red)
            b_std = np.std(lab_channels[2])  # b channel variation (blue-yellow)
            
            # Calculate color variation ratios
            sat_variation = sat_std / (np.mean(hsv_channels[1]) + 1e-6)
            val_variation = val_std / (np.mean(hsv_channels[2]) + 1e-6)
            
            # Analyze edge characteristics
            edges = cv2.Canny(roi_gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Analyze gradient patterns
            sobelx = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            gradient_direction = np.arctan2(sobely, sobelx)
            
            # Calculate gradient statistics
            gradient_mean = np.mean(gradient_magnitude)
            gradient_std = np.std(gradient_magnitude)
            direction_std = np.std(gradient_direction)
            
            # Enhanced texture analysis
            glcm = self.calculate_glcm(roi_gray)
            contrast = self.calculate_glcm_feature(glcm, 'contrast')
            homogeneity = self.calculate_glcm_feature(glcm, 'homogeneity')
            energy = self.calculate_glcm_feature(glcm, 'energy')
            correlation = self.calculate_glcm_feature(glcm, 'correlation')

            # Specific checks for orange/yellow reflections
            is_reflection = (
                (sat_variation < 0.15 or sat_variation > 0.8) or  # Either too uniform or too varied saturation
                (val_variation < 0.1) or  # Too uniform brightness
                (gradient_std / (gradient_mean + 1e-6) < 0.2) or  # Too uniform gradient
                (direction_std < 0.3) or  # Too uniform gradient direction
                (energy > 0.6) or  # Too uniform texture
                (homogeneity > 0.9) or  # Too smooth texture
                (contrast < 0.2)  # Too little contrast
            )

            # Additional checks for artificial light patterns
            is_artificial = (
                is_reflection or
                (edge_density < 0.03) or  # Too few edges
                (correlation > 0.95) or  # Too regular pattern
                (a_std < 5 and b_std < 5)  # Too consistent color in LAB space
            )

            return not is_artificial

        except Exception as e:
            logging.error(f"Error in light characteristics analysis: {e}")
            return True

    def calculate_glcm(self, image, distance=1, angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
        """Calculate Gray-Level Co-occurrence Matrix."""
        glcm = np.zeros((256, 256))
        for angle in angles:
            dx = int(distance * np.cos(angle))
            dy = int(distance * np.sin(angle))
            for i in range(image.shape[0]-dx):
                for j in range(image.shape[1]-dy):
                    i_intensity = image[i, j]
                    j_intensity = image[i+dx, j+dy]
                    glcm[i_intensity, j_intensity] += 1
        return glcm / (glcm.sum() + 1e-6)

    def calculate_glcm_feature(self, glcm, feature):
        """Calculate specific GLCM feature."""
        if feature == 'contrast':
            i, j = np.ogrid[0:256, 0:256]
            return np.sum(glcm * ((i-j)**2))
        elif feature == 'homogeneity':
            i, j = np.ogrid[0:256, 0:256]
            return np.sum(glcm / (1 + (i-j)**2))
        elif feature == 'energy':
            return np.sum(glcm**2)
        elif feature == 'correlation':
            i, j = np.ogrid[0:256, 0:256]
            mu_i = np.sum(i * np.sum(glcm, axis=1))
            mu_j = np.sum(j * np.sum(glcm, axis=0))
            sigma_i = np.sqrt(np.sum((i - mu_i)**2 * np.sum(glcm, axis=1)))
            sigma_j = np.sqrt(np.sum((j - mu_j)**2 * np.sum(glcm, axis=0)))
            if sigma_i * sigma_j == 0:
                return 0
            return np.sum(glcm * (i - mu_i) * (j - mu_j)) / (sigma_i * sigma_j)

    def analyze_color_patterns(self, frame):
        try:
            enhanced_frame = self.enhance_frame(frame)
            hsv = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2LAB)
            
            # Get color detection parameters from config
            color_config = self.config.get("color_detection", {})
            min_s = color_config.get("min_saturation", 120)  # Increased saturation threshold
            min_v = color_config.get("min_value", 150)
            red_ranges = color_config.get("red_hue_ranges", [[0, 10], [170, 180]])  # Narrowed red range
            orange_range = color_config.get("orange_range", [10, 20])  # Specific orange range
            yellow_range = color_config.get("yellow_range", [20, 25])  # Narrowed yellow range
            
            masks = []
            
            # Red detection with stricter criteria
            for red_range in red_ranges:
                lower_red = np.array([red_range[0], min_s, min_v])
                upper_red = np.array([red_range[1], 255, 255])
                red_mask = cv2.inRange(hsv, lower_red, upper_red)
                masks.append(red_mask)
            
            # Orange and yellow detection with additional validation
            for color_range in [orange_range, yellow_range]:
                lower_color = np.array([color_range[0], min_s + 30, min_v + 20])  # Stricter thresholds
                upper_color = np.array([color_range[1], 255, 255])
                color_mask = cv2.inRange(hsv, lower_color, upper_color)
                
                # Additional validation for orange/yellow regions
                contours = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                validated_mask = np.zeros_like(color_mask)
                
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    roi = enhanced_frame[y:y+h, x:x+w]
                    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
                    
                    # Advanced color analysis
                    sat_std = np.std(roi_hsv[:,:,1])
                    val_std = np.std(roi_hsv[:,:,2])
                    a_std = np.std(roi_lab[:,:,1])
                    b_std = np.std(roi_lab[:,:,2])
                    
                    # Check for fire-like characteristics
                    if (sat_std > 40 and val_std > 50 and  # High variation in saturation and value
                        a_std > 10 and b_std > 10 and  # Significant color variation in LAB space
                        w/h < 2 and h/w < 2):  # Reasonable aspect ratio
                        cv2.drawContours(validated_mask, [cnt], -1, 255, -1)
            
            masks.append(validated_mask)
            
            # Combine masks and process
            combined_mask = np.zeros_like(masks[0])
            for mask in masks:
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            kernel = np.ones((5,5), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_contour_area = frame.shape[0] * frame.shape[1] * self.config["min_fire_size_ratio"]
            valid_contours = []
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > min_contour_area:
                    x, y, w, h = cv2.boundingRect(cnt)
                    roi = frame[y:y+h, x:x+w]
                    roi_mask = combined_mask[y:y+h, x:x+w]
                    
                    if self.analyze_light_characteristics(frame, roi, roi_mask):
                        valid_contours.append(cnt)
            
            fire_regions = [cv2.boundingRect(cnt) for cnt in valid_contours]
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
        """Save detection data including screenshot when fire is detected."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save the screenshot with detection visualization
            screenshot = frame.copy()
            for x, y, w, h in fire_regions:
                cv2.rectangle(screenshot, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(screenshot, f"Fire: {confidence:.2f}", 
                          (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Create filename with timestamp
            filename = os.path.join(self.screenshot_dir, f"fire_detection_{timestamp}.jpg")
            cv2.imwrite(filename, screenshot)
            
            logging.info(f"Fire detection screenshot saved: {filename}")
            
            # Save additional detection data if needed
            detection_data = {
                "timestamp": timestamp,
                "confidence": confidence,
                "regions": fire_regions
            }
            
            # Save detection data to a JSON file
            json_filename = os.path.join(self.screenshot_dir, f"detection_data_{timestamp}.json")
            with open(json_filename, 'w') as f:
                json.dump(detection_data, f, indent=4)
                
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
            # Convert frame to RGB for model input
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Prepare image for model
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            # Transform image
            img_tensor = transform(rgb_frame).unsqueeze(0)
            
            # Extract texture features
            texture_features = self.texture_extractor.extract_features(rgb_frame)
            texture_tensor = torch.FloatTensor(texture_features).unsqueeze(0)
            
            # Move to device
            device = next(self.model.parameters()).device
            img_tensor = img_tensor.to(device)
            texture_tensor = texture_tensor.to(device)
            
            # Get model prediction
            with torch.no_grad():
                output = self.model((img_tensor, texture_tensor))
                probabilities = torch.softmax(output, dim=1)
                confidence, prediction = torch.max(probabilities, 1)
                
            # Multiple analysis methods
            color_detected, color_mask, fire_regions = self.analyze_color_patterns(frame)
            temporal_detected, _ = self.analyze_temporal_patterns()
            motion_detected, _, _ = self.detect_motion(
                self.frame_buffer[-1] if self.frame_buffer else None, 
                frame
            )
            
            # Less strict combination of detection methods
            current_detection = (
                confidence.item() > self.config["min_confidence"] and 
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
                    self.save_detection_data(frame, confidence.item(), fire_regions)
                    self.send_alert(confidence.item(), fire_regions)
            else:
                self.fire_detected_count = max(0, self.fire_detected_count - 1)
            
            return self.continuous_detection, confidence.item(), color_mask if color_detected else None, fire_regions, current_detection
            
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            return False, 0.0, None, [], False

    def run_detection(self):
        """Run the fire detection system on video input."""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                logging.error("Error: Could not open webcam.")
                return

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
                    constant_detection_seconds = self.config.get("constant_detection_seconds", 1)
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
