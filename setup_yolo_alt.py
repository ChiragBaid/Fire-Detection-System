import subprocess
import sys

def setup_yolov5():
    print("Installing YOLOv5...")
    # Install YOLOv5 directly via pip
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'yolov5'])
    
    # Install additional requirements for our project
    additional_requirements = [
        'pytorch-lightning',  # for better training organization
        'wandb',             # for experiment tracking (optional)
        'albumentations'     # for advanced data augmentation
    ]
    
    print("Installing additional requirements...")
    for req in additional_requirements:
        subprocess.run([sys.executable, '-m', 'pip', 'install', req])
    
    print("\nYOLOv5 setup completed successfully!")
    print("\nNext steps:")
    print("1. Prepare your dataset in YOLO format")
    print("2. Create a dataset.yaml file")
    print("3. Start training using YOLOv5")

if __name__ == "__main__":
    setup_yolov5()
