import cv2
import torch
import numpy as np
from pathlib import Path
import time
import logging
import json
import os
import sys
from datetime import datetime, timedelta
import queue
import threading
from skimage.feature import graycomatrix, graycoprops

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

class FireDetectorYOLOv5:
    def __init__(self, weights_path='runs/train/fire_detection_yolov5/weights/best.pt', conf_threshold=0.25):
        """
        Initialize the YOLOv5 fire detector with additional features
        """
        self.conf_threshold = conf_threshold
        self.setup_logging()
        self.texture_extractor = TextureFeatureExtractor()
        self.frame_buffer = []
        self.alert_queue = queue.Queue()
        self.last_alert_time = datetime.now() - timedelta(minutes=5)
        self.fire_detected_count = 0
        self.consecutive_static_frames = 0
        self.first_detection_time = None
        self.continuous_detection = False
        self.screenshot_dir = "fire_detections"
        os.makedirs(self.screenshot_dir, exist_ok=True)
        
        # Load configuration
        self.load_config()
        
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
            self.model.conf = conf_threshold
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

    def setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('fire_detection_yolov5.log'),
                logging.StreamHandler()
            ]
        )

    def load_config(self):
        """Load configuration from JSON file"""
        config_path = "fire_detection_config.json"
        default_config = {
            "min_confidence": 0.7,
            "frame_buffer_size": 10,
            "alert_cooldown_minutes": 5,
            "min_fire_size_ratio": 0.005,
            "motion_threshold": 0.98,
            "consecutive_detections_threshold": 3,
            "constant_detection_seconds": 3,
            "color_detection": {
                "min_saturation": 85,
                "min_value": 120,
                "red_hue_ranges": [[0, 12], [165, 180]],
                "yellow_hue_range": [12, 30],
                "min_intensity_ratio": 0.4,
                "intensity_threshold": 160,
                "max_roundness": 0.85,
                "min_aspect_ratio": 1.2,
                "edge_threshold": 50
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

    def analyze_light_characteristics(self, frame, roi, mask=None):
        """
        Analyze if the region has characteristics of artificial light or reflection
        """
        try:
            # Convert ROI to different color spaces for analysis
            if len(roi.shape) == 3:
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
            
            # Analyze temporal color variation
            color_std = np.std(hsv_channels[0])
            sat_std = np.std(hsv_channels[1])
            val_std = np.std(hsv_channels[2])
            
            # Check for color consistency in LAB space
            a_std = np.std(lab_channels[1])
            b_std = np.std(lab_channels[2])
            
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

            # Check for reflections and artificial light
            is_reflection = (
                (sat_variation < 0.15 or sat_variation > 0.8) or
                (val_variation < 0.1) or
                (gradient_std / (gradient_mean + 1e-6) < 0.2) or
                (direction_std < 0.3) or
                (energy > 0.6) or
                (homogeneity > 0.9) or
                (contrast < 0.2)
            )

            is_artificial = (
                is_reflection or
                (edge_density < 0.03) or
                (correlation > 0.95) or
                (a_std < 5 and b_std < 5)
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
            return np.sum((i - mu_i) * (j - mu_j) * glcm) / (sigma_i * sigma_j + 1e-6)

    def process_frame(self, frame):
        """
        Process a single frame and detect fire with enhanced features
        """
        try:
            # Store frame in buffer for temporal analysis
            self.frame_buffer.append(frame.copy())
            if len(self.frame_buffer) > 5:
                self.frame_buffer.pop(0)

            # Enhance frame
            enhanced_frame = self.enhance_frame(frame)
            
            # Run YOLOv5 detection
            results = self.model(enhanced_frame)
            
            detections = []
            highest_confidence = 0
            
            # Process each detection
            for det in results.xyxy[0]:
                x1, y1, x2, y2, conf, cls = det.cpu().numpy()
                if conf > self.conf_threshold:
                    roi = frame[int(y1):int(y2), int(x1):int(x2)]
                    if roi.size > 0:
                        # Apply enhanced analysis
                        if self.analyze_light_characteristics(frame, roi):
                            detections.append((int(x1), int(y1), int(x2), int(y2), conf))
                            highest_confidence = max(highest_confidence, conf)

            # Update detection tracking
            if detections:
                if self.first_detection_time is None:
                    self.first_detection_time = datetime.now()
                    self.continuous_detection = True
                elif (datetime.now() - self.first_detection_time).total_seconds() > 3:
                    self.save_detection_data(frame, highest_confidence, detections)
                    self.send_alert(highest_confidence, detections)
            else:
                self.first_detection_time = None
                self.continuous_detection = False

            return frame, detections, highest_confidence

        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            return frame, [], 0.0

    def enhance_frame(self, frame):
        """Enhance frame for better fire detection"""
        try:
            # Convert to float32
            frame_float = frame.astype(np.float32) / 255.0
            
            # Apply adaptive contrast enhancement
            lab = cv2.cvtColor(frame_float, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply((l * 255).astype(np.uint8)) / 255.0
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Apply sharpening
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            # Convert back to uint8
            enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
            
            return enhanced
            
        except Exception as e:
            logging.error(f"Error enhancing frame: {e}")
            return frame

    def save_detection_data(self, frame, confidence, boxes):
        """Save detection data including screenshot"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save annotated frame
            screenshot_path = os.path.join(self.screenshot_dir, f"fire_detection_{timestamp}.jpg")
            cv2.imwrite(screenshot_path, frame)
            
            # Save detection metadata
            metadata = {
                "timestamp": timestamp,
                "confidence": float(confidence),
                "num_detections": len(boxes),
                "boxes": boxes
            }
            
            metadata_path = os.path.join(self.screenshot_dir, f"detection_{timestamp}.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
                
            logging.info(f"Detection data saved: {screenshot_path}")
            
        except Exception as e:
            logging.error(f"Error saving detection data: {e}")

    def send_alert(self, confidence, boxes):
        """Send alert notification"""
        try:
            # Simulate sending alert notification
            logging.info(f"Sending alert notification with confidence {confidence:.2f} and {len(boxes)} detections")
            
        except Exception as e:
            logging.error(f"Error sending alert notification: {e}")

    def process_video(self, video_path, output_path=None, display=True):
        """
        Process video file for fire detection with enhanced features
        """
        try:
            logging.info(f"Starting fire detection on video: {video_path}")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Error opening video file: {video_path}")

            # Get video properties
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = 0
            
            # Initialize video writer if output path is specified
            out = None
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

            start_time = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                processed_frame, detections, confidence = self.process_frame(frame)
                frame_count += 1

                # Write frame if output path is specified
                if out:
                    out.write(processed_frame)

                # Display frame if requested
                if display:
                    cv2.imshow('Fire Detection', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # Log detections if any
                if len(detections) > 0:
                    logging.info(f"Frame {frame_count}: Detected {len(detections)} fire instances with confidence {confidence:.2f}")

            # Calculate and log processing statistics
            end_time = time.time()
            processing_time = end_time - start_time
            fps = frame_count / processing_time
            logging.info(f"Processed {frame_count} frames in {processing_time:.2f} seconds ({fps:.2f} FPS)")

            # Clean up
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()

        except Exception as e:
            logging.error(f"Error processing video: {str(e)}")
            if 'cap' in locals():
                cap.release()
            if 'out' in locals() and out:
                out.release()
            cv2.destroyAllWindows()
            raise

def main():
    try:
        # Initialize the detector
        detector = FireDetectorYOLOv5()
        
        # Use webcam by default
        video_path = 0
        output_path = None
        
        # Process the video
        detector.process_video(video_path, output_path)
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
