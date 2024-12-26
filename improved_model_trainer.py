import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
from tqdm import tqdm

class FireDataset(Dataset):
    def __init__(self, fire_dir, nofire_dir, transform=None, generate_crops=True):
        self.transform = transform
        self.generate_crops = generate_crops
        self.images = []
        self.labels = []
        
        # Load fire images (label 1)
        for img_name in os.listdir(fire_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(fire_dir, img_name)
                self.images.append((img_path, True))  # True indicates fire image
                self.labels.append(1)
        
        # Load no-fire images (label 0)
        for img_name in os.listdir(nofire_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(nofire_dir, img_name)
                self.images.append((img_path, False))  # False indicates no-fire image
                self.labels.append(0)
    
    def __len__(self):
        return len(self.images)
    
    def create_crop(self, image, size):
        """Create a random crop of the image"""
        width, height = image.size
        crop_size = min(width, height) // size
        left = torch.randint(0, width - crop_size, (1,)).item()
        top = torch.randint(0, height - crop_size, (1,)).item()
        crop = image.crop((left, top, left + crop_size, top + crop_size))
        return crop
    
    def __getitem__(self, idx):
        # Load image and get type
        img_path, is_fire = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if is_fire and self.generate_crops:
            # For fire images, create additional crops
            crops = []
            
            # Original image
            if self.transform:
                crops.append(self.transform(image))
            
            # Create smaller crops
            for size in [2, 3, 4]:  # 1/2, 1/3, 1/4 of original size
                crop = self.create_crop(image, size)
                if self.transform:
                    crops.append(self.transform(crop))
            
            # Return all crops
            return torch.stack(crops), torch.tensor([label] * len(crops))
        else:
            # For no-fire images or when crops not needed
            if self.transform:
                image = self.transform(image)
            return image.unsqueeze(0), torch.tensor([label])

def collate_fn(batch):
    """Custom collate function to handle different sized batches"""
    images = []
    labels = []
    
    for img, lbl in batch:
        images.append(img)
        labels.append(lbl)
    
    # Concatenate all images and labels
    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    
    return images, labels

# Training transforms with extensive augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Validation transforms
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    device = torch.device("cpu")
    model = model.to(device)
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        for inputs, labels in tqdm(train_loader, desc='Training'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validation'):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)
            
            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples
            print(f'Validation Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Save best model
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'Models/best_fire_detection_model.pth')
                print(f'New best model saved with accuracy: {epoch_acc:.4f}')
    
    return model

if __name__ == "__main__":
    # Dataset paths
    fire_dir = r"C:\Users\Admin\Desktop\Fire Detection From a Video\Datasets\fire"
    nofire_dir = r"C:\Users\Admin\Desktop\Fire Detection From a Video\Datasets\no_fire"
    
    # Create datasets with appropriate transforms
    train_dataset = FireDataset(fire_dir, nofire_dir, transform=train_transform, generate_crops=True)
    val_dataset = FireDataset(fire_dir, nofire_dir, transform=val_transform, generate_crops=False)
    
    # Split datasets
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(train_dataset)))
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset, 
        batch_size=8,  # Smaller batch size since we're generating multiple crops
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Initialize model
    model = models.resnet18(weights="IMAGENET1K_V1")
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    print("Starting training...")
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer)
    
    # Save final model
    torch.save(trained_model.state_dict(), "Models/final_fire_detection_model.pth")
    print("Training completed!")
