import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = 'dataset'
    img_height, img_width = 180, 180
    batch_size = 32
    num_epochs = 10
    num_classes = 3

    data_transforms = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' not found.")
        return

    full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
    class_names = full_dataset.classes
    print(f"Classes found: {class_names}")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(123)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("Initializing ResNet50...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Linear(512, num_classes)
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct_train / total_train
        train_acc_history.append(epoch_acc)

        # Validation Loop
        model.eval()
        correct_val = 0
        total_val = 0
        val_preds = []
        val_true = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)

                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                val_preds.extend(predicted.cpu().numpy())
                val_true.extend(labels.cpu().numpy())

        val_acc = correct_val / total_val
        val_acc_history.append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Loss: {epoch_loss:.4f} - "
              f"Train Acc: {epoch_acc:.4f} - "
              f"Val Acc: {val_acc:.4f}")

    # 9. Results
    plt.figure(figsize=(8, 8))
    plt.plot(range(num_epochs), train_acc_history, label='Training Accuracy')
    plt.plot(range(num_epochs), val_acc_history, label='Validation Accuracy')
    plt.ylim([0.4, 1.0])
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

    # 10. Confusion Matrix
    cm = confusion_matrix(val_true, val_preds)

    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Matrix count
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.show()

    #  Prediction
    img_path = 'test.png'
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = transforms.ToPILImage()(img)

        input_tensor = data_transforms(img_pil).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            _, pred_idx = torch.max(probs, 1)

        predicted_class = class_names[pred_idx.item()]
        print(f"The predicted class is: {predicted_class}")
    else:
        print(f"Test image '{img_path}' not found.")

    print("\nModel Architecture:")
    print(model.fc)

    # weights save
    torch.save(model.state_dict(), 'resnet50_model.pth')


if __name__ == '__main__':
    main()
