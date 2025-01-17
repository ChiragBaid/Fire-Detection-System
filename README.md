# Fire Detection System

This is a real-time fire detection system using YOLOv8 with enhanced detection through color and texture analysis.

## Setup Instructions

1. Install Python 3.9+ if not already installed

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Files included:
- `main_improved_yolov8.py`: Main detection script
- `fire_detection_config.json`: Configuration file
- `yolov8n.pt`: YOLOv8 model
- `requirements.txt`: Required Python packages

4. Run the detection:
```bash
python main_improved_yolov8.py
```

## Features
- Real-time fire detection using YOLOv8
- Enhanced detection using color analysis in LAB color space
- Texture analysis using GLCM features
- Automatic alerts when fire is detected
- Detection confidence visualization

## Controls
- Press 'q' to quit the application
- Detections are automatically saved in the `fire_detections` folder

## System Requirements
- Python 3.9+
- CUDA-capable GPU recommended but not required
- Webcam or video input device

## Configuration
You can modify detection parameters in `fire_detection_config.json`:
- Confidence thresholds
- Alert settings
- Detection parameters
