import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
from tqdm import tqdm  

class CustomImageDataset(Dataset):
    def __init__(self, fire_dir, nofire_dir, transform=None):
        self.fire_images = [
            os.path.join(fire_dir, f) for f in os.listdir(fire_dir) if f.endswith(".jpg")
        ]
        self.nofire_images = [
            os.path.join(nofire_dir, f) for f in os.listdir(nofire_dir) if f.endswith(".jpg")
        ]
        self.image_paths = self.fire_images + self.nofire_images
        self.labels = [1] * len(self.fire_images) + [0] * len(self.nofire_images)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Data transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Dataset directories
fire_dir = r"C:\Users\Admin\Desktop\Fire Detection From a Video\Datasets\fire"
nofire_dir = r"C:\Users\Admin\Desktop\Fire Detection From a Video\Datasets\no_fire"

# Create dataset
dataset = CustomImageDataset(fire_dir=fire_dir, nofire_dir=nofire_dir, transform=data_transforms)

# Split into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load the model
model = models.resnet18(weights="IMAGENET1K_V1")
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # Two output classes: fire and no fire

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=15):
    best_val_loss = float("inf")
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        print(f"Train Loss: {train_loss:.4f}")

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss = running_val_loss / len(val_loader)
        val_acc = correct / total
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_fire_detection_model.pth")
            print("Saved best model!")

    return model

# Train the model
trained_model = train_model(model, train_loader, val_loader, criterion, optimizer)

# Save the final model
torch.save(trained_model.state_dict(), "final_fire_detection_model.pth")
