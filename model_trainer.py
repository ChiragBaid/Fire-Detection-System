import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2
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

class FireTextureDataset(Dataset):
    def __init__(self, fire_dir, nofire_dir, transform=None, generate_crops=True):
        self.transform = transform
        self.generate_crops = generate_crops
        self.images = []
        self.labels = []
        self.texture_extractor = TextureFeatureExtractor()
        
        # Load fire images (label 1)
        for img_name in os.listdir(fire_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(fire_dir, img_name)
                self.images.append((img_path, True))
                self.labels.append(1)
        
        # Load no-fire images (label 0)
        for img_name in os.listdir(nofire_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(nofire_dir, img_name)
                self.images.append((img_path, False))
                self.labels.append(0)
    
    def extract_texture_features(self, image):
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        return self.texture_extractor.extract_features(img_array)
    
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
            texture_features = []
            
            # Original image
            if self.transform:
                crops.append(self.transform(image))
                texture_features.append(torch.FloatTensor(self.extract_texture_features(image)))
            
            # Create smaller crops
            for size in [2, 3, 4]:
                crop = self.create_crop(image, size)
                if self.transform:
                    crops.append(self.transform(crop))
                    texture_features.append(torch.FloatTensor(self.extract_texture_features(crop)))
            
            return (torch.stack(crops), 
                   torch.stack(texture_features)), torch.tensor([label] * len(crops))
        else:
            if self.transform:
                transformed_image = self.transform(image)
                texture_features = torch.FloatTensor(self.extract_texture_features(image))
            return (transformed_image.unsqueeze(0), 
                   texture_features.unsqueeze(0)), torch.tensor([label])

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

def collate_fn(batch):
    """Custom collate function to handle different sized batches with texture features"""
    images = []
    texture_features = []
    labels = []
    
    for (img, tex), lbl in batch:
        images.append(img)
        texture_features.append(tex)
        labels.append(lbl)
    
    # Concatenate all images, texture features, and labels
    images = torch.cat(images, dim=0)
    texture_features = torch.cat(texture_features, dim=0)
    labels = torch.cat(labels, dim=0)
    
    return (images, texture_features), labels

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        for (inputs, texture_features), labels in tqdm(train_loader):
            inputs = inputs.to(device)
            texture_features = texture_features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model((inputs, texture_features))
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)
            total_samples += labels.size(0)
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        with torch.no_grad():
            for (inputs, texture_features), labels in tqdm(val_loader):
                inputs = inputs.to(device)
                texture_features = texture_features.to(device)
                labels = labels.to(device)
                
                outputs = model((inputs, texture_features))
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)
                total_samples += labels.size(0)
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        print(f'Validation Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # Save best model
        if epoch_acc > best_val_acc:
            best_val_acc = epoch_acc
            torch.save(model.state_dict(), 'best_fire_detection_model.pth')
            print(f'New best model saved with accuracy: {best_val_acc:.4f}')
        
        print()

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

if __name__ == "__main__":
    # Dataset paths
    fire_dir = r"C:\Users\Admin\Desktop\Fire Detection From a Video\Datasets\fire"
    nofire_dir = r"C:\Users\Admin\Desktop\Fire Detection From a Video\Datasets\no_fire"
    
    # Create datasets with texture analysis
    texture_extractor = TextureFeatureExtractor()
    full_dataset = FireTextureDataset(fire_dir, nofire_dir, transform=train_transform)
    
    # Calculate number of features
    sample_image = Image.open(full_dataset.images[0][0]).convert('RGB')
    num_texture_features = len(texture_extractor.extract_features(np.array(sample_image)))
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                            collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                          collate_fn=collate_fn, num_workers=4)
    
    # Create model with texture features
    model = FireTextureNet(num_texture_features)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25)
