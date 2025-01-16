import os
import shutil
from pathlib import Path
import random
import cv2
import numpy as np

def create_yolo_structure(base_path):
    # Create main dataset directory
    dataset_path = Path(base_path) / 'yolo_dataset_v5'
    
    # Create train, val, test directories with images and labels subdirectories
    splits = ['train', 'val', 'test']
    for split in splits:
        for subdir in ['images', 'labels']:
            os.makedirs(dataset_path / split / subdir, exist_ok=True)
    
    return dataset_path

def convert_to_yolo_format(bbox, img_width, img_height):
    # Convert bbox from [x1, y1, x2, y2] to YOLO format [x_center, y_center, width, height]
    x1, y1, x2, y2 = bbox
    
    # Normalize coordinates
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    # Class index for fire is 0
    return f"0 {x_center} {y_center} {width} {height}"

def process_dataset(source_path, dest_path, split_ratios=(0.7, 0.2, 0.1)):
    fire_images = list(Path(source_path).glob('fire/*.jpg'))
    no_fire_images = list(Path(source_path).glob('no_fire/*.jpg'))
    
    # Shuffle images
    random.shuffle(fire_images)
    random.shuffle(no_fire_images)
    
    # Calculate split indices
    fire_train_end = int(len(fire_images) * split_ratios[0])
    fire_val_end = fire_train_end + int(len(fire_images) * split_ratios[1])
    
    no_fire_train_end = int(len(no_fire_images) * split_ratios[0])
    no_fire_val_end = no_fire_train_end + int(len(no_fire_images) * split_ratios[1])
    
    # Split datasets
    splits = {
        'train': (fire_images[:fire_train_end], no_fire_images[:no_fire_train_end]),
        'val': (fire_images[fire_train_end:fire_val_end], no_fire_images[no_fire_train_end:no_fire_val_end]),
        'test': (fire_images[fire_val_end:], no_fire_images[no_fire_val_end:])
    }
    
    for split_name, (split_fire, split_no_fire) in splits.items():
        print(f"Processing {split_name} split...")
        
        # Process fire images
        for img_path in split_fire:
            img = cv2.imread(str(img_path))
            height, width = img.shape[:2]
            
            # Create label for fire images (assuming full image bounding box)
            label = convert_to_yolo_format([0, 0, width, height], width, height)
            
            # Copy image
            shutil.copy2(img_path, dest_path / split_name / 'images' / img_path.name)
            
            # Save label
            label_path = dest_path / split_name / 'labels' / (img_path.stem + '.txt')
            with open(label_path, 'w') as f:
                f.write(label)
        
        # Process no_fire images (no labels needed as they're negative samples)
        for img_path in split_no_fire:
            shutil.copy2(img_path, dest_path / split_name / 'images' / img_path.name)
            # Create empty label file
            label_path = dest_path / split_name / 'labels' / (img_path.stem + '.txt')
            open(label_path, 'w').close()

def create_data_yaml(base_path):
    yaml_content = f"""
# Train/val/test sets
path: {base_path}/yolo_dataset_v5
train: train/images
val: val/images
test: test/images

# Classes
nc: 1  # number of classes
names: ['fire']  # class names
"""
    
    with open(os.path.join(base_path, 'data_yolov5.yaml'), 'w') as f:
        f.write(yaml_content.strip())

def main():
    base_path = 'c:/Users/Admin/Desktop/Fire Detection From a Video'
    source_path = os.path.join(base_path, 'Datasets')
    
    # Create YOLOv5 directory structure
    dataset_path = create_yolo_structure(base_path)
    
    # Process and convert dataset
    process_dataset(source_path, dataset_path)
    
    # Create data.yaml file
    create_data_yaml(base_path)
    
    print("Dataset conversion completed successfully!")

if __name__ == "__main__":
    main()
