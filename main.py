import os
import torch
import cv2
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import tkinter as tk
from tkinter import messagebox

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights="IMAGENET1K_V1")
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # Output layer for 2 classes (fire, no fire)
model.load_state_dict(torch.load("final_fire_detection_model.pth"))
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

# Function to process video, check for fire, and take a screenshot
def check_fire_in_video(video_path, output_image_path):
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fire_detected = False
    alert_triggered = False  
    screenshot_taken = False  

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Convert the frame to PIL image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Apply transformations
        input_tensor = data_transforms(pil_image).unsqueeze(0).to(device)

        # Classify the frame
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)

        # If fire is detected, process the frame
        if predicted.item() == 1:
            fire_detected = True
            if not alert_triggered:
                send_alert("Fire Alert!", "Fire detected in your surroundings. Please take action immediately!")
                alert_triggered = True  # Ensure the alert is only triggered once

            # Draw a red border around the frame
            height, width, _ = frame.shape
            cv2.rectangle(frame, (10, 10), (width - 10, height - 10), (0, 0, 255), 10)

            # Save a screenshot of the first fire-detected frame
            if not screenshot_taken:
                cv2.imwrite(output_image_path, frame)
                screenshot_taken = True
                print(f"Screenshot saved at {output_image_path}")

    cap.release()

    if not fire_detected:
        print("No fire detected in the video.")
    else:
        print("Fire detected. Screenshot saved.")

# Example usage
if __name__ == "__main__":
    video_path = r"C:\Users\Admin\Desktop\Fire Detection From a Video\Videos\video10.mp4"
    output_image_path = r"C:\Users\Admin\Desktop\Fire Detection From a Video\Videos\screenshot_fire.jpg"
    check_fire_in_video(video_path, output_image_path)
