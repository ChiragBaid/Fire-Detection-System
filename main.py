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
model.load_state_dict(torch.load("MOdels/best_fire_detection_model.pth"))
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

def is_fire_colored(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 150, 150])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 150, 150])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    fire_pixel_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
    return fire_pixel_ratio > 0.01

def detect_motion(prev_frame, curr_frame):
    if prev_frame is None:
        return False
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    frame_diff = cv2.absdiff(prev_gray, curr_gray)
    _, threshold = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
    motion_ratio = np.sum(threshold > 0) / (threshold.shape[0] * threshold.shape[1])
    return motion_ratio > 0.01

def check_fire_in_live_video():
    while True:
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

        print("Press 'q' to quit the live video feed.")

        try:
            while True:
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

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                input_tensor = data_transforms(pil_image).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = nn.Softmax(dim=1)(outputs)
                    confidence, predicted = torch.max(probabilities, 1)

                is_model_confident = predicted.item() == 1 and confidence.item() > 0.9
                has_fire_colors = is_fire_colored(frame)
                has_motion = detect_motion(prev_frame, frame)

                prev_frame = frame.copy()

                if is_model_confident and has_fire_colors and has_motion:
                    consecutive_detections += 1
                    if consecutive_detections >= 3:
                        if fire_start_time is None:
                            fire_start_time = time.time()
                        elif time.time() - fire_start_time >= 3:
                            fire_detected = True

                            if not alert_triggered:
                                send_alert("Fire Alert!", "Fire detected in your surroundings. Please take action immediately!")
                                alert_triggered = True
                                system_pause_time = datetime.now()
                                print("Fire detected! System paused for 15 seconds.")

                            height, width, _ = frame.shape
                            cv2.rectangle(frame, (10, 10), (width - 10, height - 10), (0, 0, 255), 10)

                            if not screenshot_taken:
                                output_image_path = "screenshot_fire.jpg"
                                cv2.imwrite(output_image_path, frame)
                                screenshot_taken = True
                                print(f"Screenshot saved at {output_image_path}")
                else:
                    consecutive_detections = 0
                    if time.time() - fire_start_time > 3 if fire_start_time else True:
                        fire_start_time = None
                        fire_detected = False
                        alert_triggered = False

                cv2.imshow("Fire Detection", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Exiting live video feed...")
                    return

        finally:
            cap.release()
            cv2.destroyAllWindows()

        if 'key' in locals() and key != ord('q'):
            print("Restarting video capture...")
            time.sleep(1)

if __name__ == "__main__":
    check_fire_in_live_video()
