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

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights="IMAGENET1K_V1")
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model.load_state_dict(torch.load("best_fire_detection_model.pth"))
model = model.to(device)
model.eval()

def send_alert(title, message):
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        messagebox.showinfo(title, message)
        root.destroy()
        print("Pop-up alert displayed!")
    except Exception as e:
        print(f"Error displaying pop-up alert: {e}")

def enhance_frame(frame):
    # Convert to LAB color space for better color separation
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Merge the CLAHE enhanced L-channel back with A and B channels
    enhanced_lab = cv2.merge((cl,a,b))
    
    # Convert back to BGR color space
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced_bgr

def is_fire_colored(frame):
    # Enhance the frame first
    enhanced_frame = enhance_frame(frame)
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2HSV)
    
    # More permissive color ranges for fire detection
    # Red-Orange range
    lower_red1 = np.array([0, 70, 70])
    upper_red1 = np.array([15, 255, 255])
    # Upper red range
    lower_red2 = np.array([165, 70, 70])
    upper_red2 = np.array([180, 255, 255])
    # Yellow range for bright flames
    lower_yellow = np.array([15, 70, 70])
    upper_yellow = np.array([35, 255, 255])
    
    # Create masks for each color range
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask3 = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Combine masks
    mask = cv2.bitwise_or(cv2.bitwise_or(mask1, mask2), mask3)
    
    # Apply morphological operations to reduce noise
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours to analyze blob size
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter small contours
    min_contour_area = frame.shape[0] * frame.shape[1] * 0.001
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
    # If no valid contours found, return False
    if not valid_contours:
        return False, mask
    
    # Calculate the ratio of fire-colored pixels
    fire_pixel_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
    
    # More permissive threshold for fire pixel ratio
    return fire_pixel_ratio > 0.005, mask

def analyze_temporal_patterns(frame_buffer):
    if len(frame_buffer) < 2:
        return False, 0
    
    # Convert frames to grayscale
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frame_buffer]
    
    # Calculate frame differences
    diffs = []
    for i in range(len(gray_frames)-1):
        diff = cv2.absdiff(gray_frames[i], gray_frames[i+1])
        diffs.append(np.mean(diff))
    
    # Analyze the pattern of changes
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    
    # Real fire typically shows fluctuating patterns
    is_fluctuating = std_diff > 2.0 and mean_diff > 1.0
    
    return is_fluctuating, mean_diff

def detect_motion(prev_frame, curr_frame):
    if prev_frame is None:
        return False, None, 0, None
    
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    prev_gray = cv2.GaussianBlur(prev_gray, (15, 15), 0)
    curr_gray = cv2.GaussianBlur(curr_gray, (15, 15), 0)
    
    # Calculate frame difference
    frame_diff = cv2.absdiff(prev_gray, curr_gray)
    
    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 
                                      pyr_scale=0.5, levels=3, winsize=15, 
                                      iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    
    # Calculate flow magnitude and angle
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Analyze flow patterns
    mean_magnitude = np.mean(magnitude)
    flow_mask = magnitude > mean_magnitude
    
    # Apply thresholding to frame difference
    thresh = cv2.threshold(frame_diff, 20, 255, cv2.THRESH_BINARY)[1]
    
    # Apply morphological operations
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Calculate motion metrics
    motion_ratio = cv2.countNonZero(thresh) / (thresh.shape[0] * thresh.shape[1])
    
    # Create a combined motion score using both difference and flow
    flow_score = np.mean(magnitude[flow_mask]) if np.any(flow_mask) else 0
    combined_score = (motion_ratio + flow_score) / 2
    
    return combined_score > 0.01, thresh, combined_score, magnitude

def is_video_static(prev_frame, curr_frame, threshold=0.98):
    """
    Check if the video feed is static by comparing consecutive frames.
    Returns True if frames are nearly identical (static), False otherwise.
    """
    if prev_frame is None or curr_frame is None:
        return False
        
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate structural similarity index
    score = cv2.matchTemplate(prev_gray, curr_gray, cv2.TM_CCOEFF_NORMED)[0][0]
    
    return score > threshold

def save_screenshot(frame, confidence):
    """
    Save a screenshot of the frame when fire is detected
    """
    # Create screenshots directory if it doesn't exist
    screenshots_dir = "fire_screenshots"
    if not os.path.exists(screenshots_dir):
        os.makedirs(screenshots_dir)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{screenshots_dir}/fire_detected_{timestamp}_conf_{confidence:.2f}.jpg"
    
    # Save the image
    cv2.imwrite(filename, frame)
    print(f"Screenshot saved: {filename}")

def check_fire_in_live_video():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    fire_start_time = None
    fire_detected = False
    alert_triggered = False
    screenshot_taken = False
    prev_frame = None
    consecutive_detections = 0
    system_pause_time = None
    window_name = "Fire Detection"
    cv2.namedWindow(window_name)
    
    # Initialize variables
    avg_motion = 0
    mean_diff = 0
    is_fluctuating = False
    
    # Buffers for temporal analysis
    frame_buffer = []
    detection_buffer = []
    motion_buffer = []
    flow_buffer = []
    BUFFER_SIZE = 10
    FRAME_BUFFER_SIZE = 5
    DETECTION_THRESHOLD = 0.5
    MOTION_THRESHOLD = 0.2
    
    # Initialize frame counter and timing variables
    frame_counter = 0
    fps_start_time = time.time()
    fps = 0
    last_check_time = time.time()
    check_interval = 2.0  
    is_live_video = True  
    static_count = 0
    max_static_count = 30  # Number of consecutive static frames before warning
    
    try:
        while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) > 0:
            if system_pause_time is not None:
                current_time = datetime.now()
                if current_time - system_pause_time >= timedelta(seconds=15):
                    print("Restarting detection after 15-second pause...")
                    system_pause_time = None
                    break
                else:
                    time.sleep(1)
                    continue

            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam.")
                break

            current_time = time.time()

            # Update frame buffer
            frame_buffer.append(frame.copy())
            if len(frame_buffer) > FRAME_BUFFER_SIZE:
                frame_buffer.pop(0)

            # Check live/static status every 2 seconds
            if current_time - last_check_time >= check_interval and len(frame_buffer) >= 2:
                # Analyze temporal patterns
                is_fluctuating, mean_diff = analyze_temporal_patterns(frame_buffer)
                
                # Check for motion
                has_motion, motion_mask, motion_score, flow_magnitude = detect_motion(prev_frame, frame)
                
                # Update motion buffer
                if motion_score is not None:  
                    motion_buffer.append(motion_score)
                    if len(motion_buffer) > BUFFER_SIZE:
                        motion_buffer.pop(0)
                
                # Calculate average motion
                if motion_buffer:  
                    avg_motion = sum(motion_buffer) / len(motion_buffer)
                
                # Update live video status
                is_live_video = (avg_motion > 0.005 and is_fluctuating and mean_diff > 1.0)
                last_check_time = current_time

            # Calculate FPS
            frame_counter += 1
            if frame_counter >= 30:
                fps = frame_counter / (current_time - fps_start_time)
                frame_counter = 0
                fps_start_time = current_time

            # Enhance the frame
            enhanced_frame = enhance_frame(frame)
            
            if not is_live_video:
                cv2.putText(enhanced_frame, "STATIC IMAGE DETECTED - MONITORING PAUSED", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow(window_name, enhanced_frame)
                cv2.waitKey(1)
                prev_frame = frame.copy()
                continue
            
            # Check if video is static
            if prev_frame is not None:
                if is_video_static(prev_frame, frame):
                    static_count += 1
                    if static_count >= max_static_count:
                        cv2.putText(enhanced_frame, "Warning: Static Video Feed Detected!", 
                                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                                  (0, 0, 255), 2)
                else:
                    static_count = 0
                
            # Check for fire colors
            is_fire, fire_mask = is_fire_colored(enhanced_frame)
            
            # Convert frame for model input
            frame_rgb = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            input_tensor = data_transforms(pil_image).unsqueeze(0).to(device)

            # Model prediction
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = nn.Softmax(dim=1)(outputs)
                confidence, predicted = torch.max(probabilities, 1)

            is_model_confident = predicted.item() == 1 and confidence.item() > 0.85

            prev_frame = frame.copy()

            # Update detection buffer
            current_detection = is_model_confident and is_fire
            detection_buffer.append(current_detection)
            if len(detection_buffer) > BUFFER_SIZE:
                detection_buffer.pop(0)

            # Calculate detection ratio
            detection_ratio = sum(detection_buffer) / len(detection_buffer)
            
            # Only proceed if we have enough consistent detections
            if detection_ratio >= DETECTION_THRESHOLD:
                consecutive_detections += 1
                if consecutive_detections >= 3:
                    if fire_start_time is None:
                        fire_start_time = time.time()
                    elif time.time() - fire_start_time >= 4.0: 
                        if not alert_triggered:
                            send_alert("Fire Alert!", "Live fire detected in the video feed!")
                            alert_triggered = True
                        
                        if not screenshot_taken:
                            save_screenshot(enhanced_frame, confidence.item())
                            screenshot_taken = True
                        
                        fire_detected = True
                        cv2.putText(enhanced_frame, "LIVE FIRE DETECTED!", (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                consecutive_detections = max(0, consecutive_detections - 1)
                if consecutive_detections == 0:
                    fire_start_time = None
                    fire_detected = False
                    alert_triggered = False
                    screenshot_taken = False

            # Display status information
            cv2.putText(enhanced_frame, f"FPS: {fps:.1f}", (10, enhanced_frame.shape[0] - 120), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(enhanced_frame, f"Motion: {avg_motion:.3f}", (10, enhanced_frame.shape[0] - 90), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(enhanced_frame, f"Temporal Change: {mean_diff:.3f}", (10, enhanced_frame.shape[0] - 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(enhanced_frame, f"Detection Ratio: {detection_ratio:.2f}", (10, enhanced_frame.shape[0] - 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            status_color = (0, 255, 0) if not fire_detected else (0, 0, 255)
            status_msg = "MONITORING LIVE VIDEO"
            if fire_start_time is not None and not fire_detected:
                remaining_time = max(0, 4.0 - (time.time() - fire_start_time))
                status_msg = f"Potential Fire - Confirming ({remaining_time:.1f}s)"
            elif fire_detected:
                status_msg = "LIVE FIRE ALERT"
            
            cv2.putText(enhanced_frame, f"Status: {status_msg}",
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

            cv2.imshow(window_name, enhanced_frame)
            cv2.waitKey(1)

    except cv2.error as e:
        print(f"OpenCV Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    check_fire_in_live_video()
