import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import timm  # used for many models/backbones

# Custom dataset class


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                path = os.path.join(cls_dir, fname)
                self.images.append((path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# Define transforms
img_size = 224
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset directory and creation
dataset_root = "dataset"  # change to your path
dataset = CustomDataset(root_dir=dataset_root, transform=transform)

# Split into train and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8,
                          shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset,   batch_size=8,
                        shuffle=False, num_workers=4)

# Define model using MambaVision backbone
num_classes = len(dataset.classes)


class MambaVisionClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        # Use timm to load a MambaVision model – example model name:
        self.backbone = timm.create_model(
            'mambavision_s_1k', pretrained=pretrained, num_classes=num_classes)
        # If backbone has a head you might replace it:
        self.backbone.head = nn.Linear(
            self.backbone.head.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MambaVisionClassifier(num_classes=num_classes).to(device)

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 10
train_losses = []
val_losses = []
all_preds = []
all_targets = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_train_loss = running_loss / train_size
    train_losses.append(epoch_train_loss)

    model.eval()
    val_running_loss = 0.0
    correct = 0
    total = 0
    preds = []
    targets = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            preds.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    epoch_val_loss = val_running_loss / val_size
    val_losses.append(epoch_val_loss)
    val_acc = correct / total

    print(f"Epoch {epoch+1}/{num_epochs} – Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {val_acc:.4f}")

    all_preds = preds
    all_targets = targets

# Plot losses
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), val_losses,   label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.show()

# Confusion matrix
cm = confusion_matrix(all_targets, all_preds)
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, None] * 100

plt.figure(figsize=(8, 8))
plt.imshow(cm_percent, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, dataset.classes, rotation=45)
plt.yticks(tick_marks, dataset.classes)
thresh = cm_percent.max() / 2.
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, f'{cm[i, j]} ({cm_percent[i, j]:.1f}%)',
                 horizontalalignment="center",
                 color="white" if cm_percent[i, j] > thresh else "black")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix (Percent)')
plt.tight_layout()
plt.show()

# Save model checkpoint
torch.save(model.state_dict(), 'mambavision_model.pth')
