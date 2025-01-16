import os
import subprocess
import sys

def setup_yolov5():
    # Clone YOLOv5 if not already present
    if not os.path.exists('yolov5/models'):
        print("Cloning YOLOv5 repository...")
        subprocess.run(['git', 'clone', 'https://github.com/ultralytics/yolov5.git'], check=True)
    
    # Install YOLOv5 requirements
    print("Installing YOLOv5 requirements...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'yolov5/requirements.txt'], check=True)

def start_training():
    # Training hyperparameters
    batch_size = 32   # Reduced batch size for 4GB VRAM
    epochs = 150     # Increased epochs for better convergence
    img_size = 640   # Reduced image size for memory efficiency
    model_type = 'yolov5s'  # Using small model for faster training
    
    print("Starting YOLOv5 training...")
    training_command = [
        sys.executable,
        'yolov5/train.py',
        '--img', str(img_size),
        '--batch', str(batch_size),
        '--epochs', str(epochs),
        '--data', 'data.yaml',  
        '--weights', f'{model_type}.pt',
        '--cache',
        '--project', 'runs/train',
        '--name', 'fire_detection_yolov5',
        '--patience', '30',     
        '--save-period', '10',  
        '--workers', '8',       
        '--exist-ok',          
        '--label-smoothing', '0.1',  
        '--multi-scale',       
        '--optimizer', 'Adam'  
    ]
    
    subprocess.run(training_command, check=True)

def main():
    try:
        # Create necessary directories
        os.makedirs('runs/train', exist_ok=True)
        
        # Setup and start training
        setup_yolov5()
        start_training()
        
        print("Training completed successfully!")
        print("The trained model is saved in: runs/train/fire_detection_yolov5/weights/best.pt")
        
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during training: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
