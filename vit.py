import os
import torch
import timm
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from torchviz import make_dot
from torchsummary import summary
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms


# ================================================================
# Dataset Definition
# ================================================================
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = [(os.path.join(root_dir, cls, img), self.class_to_idx[cls])
                       for cls in self.classes for img in os.listdir(os.path.join(root_dir, cls))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# ================================================================
# Model Definition
# ================================================================
class ViTForClassification(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vit = timm.create_model(
            'vit_base_patch16_224', pretrained=True, num_classes=num_classes)
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

    def forward(self, x):
        return self.vit(x)


# ================================================================
# Training and Evaluation Functions
# ================================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    preds, labels_list = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return total_loss / len(loader.dataset), accuracy, labels_list, preds


# ================================================================
# Main Execution
# ================================================================
def main():
    # ----- Configuration -----
    dataset_root = 'dataset'
    batch_size = 8
    num_epochs = 10
    lr = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----- Data -----
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = CustomDataset(root_dir=dataset_root, transform=transform)
    num_classes = len(dataset.classes)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # ----- Model, Loss, Optimizer -----
    model = ViTForClassification(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ----- Training -----
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, true_labels, preds = evaluate(
            model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # ----- Save Model -----
    torch.save(model.state_dict(), 'vit_classification_model.pth')

    # ----- Loss Plot -----
    plt.plot(range(1, num_epochs + 1), train_losses,
             label='Train', color='blue')
    plt.plot(range(1, num_epochs + 1), val_losses,
             label='Validation', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.show()

    # ----- Confusion Matrix -----
    cm = confusion_matrix(true_labels, preds)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(10, 8))
    plt.imshow(cm_percent, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (%)')
    plt.colorbar()
    ticks = np.arange(num_classes)
    plt.xticks(ticks, dataset.classes, rotation=45)
    plt.yticks(ticks, dataset.classes)

    thresh = cm_percent.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, f'{cm[i, j]} ({cm_percent[i, j]:.1f}%)',
                     ha="center", color="white" if cm_percent[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # ----- Model Summary & Graph -----
    summary(model, input_size=(3, 224, 224))
    x = torch.zeros((1, 3, 224, 224)).to(device)
    vis_graph = make_dot(model(x), params=dict(model.named_parameters()))
    vis_graph.render('vit_model_architecture', format='png')


if __name__ == "__main__":
    main()
