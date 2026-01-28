import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import os
import matplotlib.pyplot as plt
import pandas as pd
import itertools


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, device, save_dir):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        self.best_acc = 0.0
        # Auto-detect classes from the custom wrapper or standard dataset
        if hasattr(train_loader.dataset, 'classes'):
            self.class_names = train_loader.dataset.classes
        else:
            self.class_names = [str(i) for i in range(10)]

        os.makedirs(save_dir, exist_ok=True)

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(self.train_loader, desc="Training", leave=False):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc="Validation", leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(self.val_loader.dataset)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def fit(self, num_epochs):
        metrics = []
        print(f"Starting training on {self.device}...")

        for epoch in range(num_epochs):
            t_loss, t_acc = self.train_epoch()
            v_loss, v_acc = self.validate()

            if self.scheduler:
                self.scheduler.step()

            print(
                f"Epoch {epoch+1}/{num_epochs} | Train Loss: {t_loss:.4f} Acc: {t_acc:.4f} | Val Loss: {v_loss:.4f} Acc: {v_acc:.4f}")

            metrics.append({
                "epoch": epoch + 1,
                "train_loss": t_loss,
                "train_accuracy": t_acc,
                "val_loss": v_loss,
                "val_accuracy": v_acc
            })

            # Save best model
            if v_acc > self.best_acc:
                self.best_acc = v_acc
                torch.save(self.model.state_dict(), os.path.join(
                    self.save_dir, 'best_model.pth'))
                print("  --> Saved new best model.")

        # 1. Save CSV logs
        pd.DataFrame(metrics).to_csv(os.path.join(
            self.save_dir, 'training_metrics.csv'), index=False)
        print(
            f"Saved logs to {os.path.join(self.save_dir, 'training_metrics.csv')}")

        # 2. Plot Curves
        self.plot_curves(metrics)

        # 3. Generate Confusion Matrix (using best model)
        print("Generating Confusion Matrix on Best Model...")
        self.save_confusion_matrix()

        print("Training Workflow Complete.")

    def plot_curves(self, metrics):
        df = pd.DataFrame(metrics)
        epochs = df['epoch']

        plt.figure(figsize=(12, 5))

        # Loss Plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, df['train_loss'], label='Train Loss')
        plt.plot(epochs, df['val_loss'], label='Val Loss')
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Accuracy Plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, df['train_accuracy'], label='Train Acc')
        plt.plot(epochs, df['val_accuracy'], label='Val Acc')
        plt.title('Accuracy Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        save_path = os.path.join(self.save_dir, 'training_curves.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved curves to {save_path}")

    def save_confusion_matrix(self):
        # Load best weights to ensure we plot the best version of the model
        best_path = os.path.join(self.save_dir, 'best_model.pth')
        self.model.load_state_dict(torch.load(
            best_path, map_location=self.device))
        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)

        # Plot
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix (Best Acc: {self.best_acc*100:.2f}%)')
        plt.colorbar()

        tick_marks = np.arange(len(self.class_names))
        plt.xticks(tick_marks, self.class_names, rotation=45)
        plt.yticks(tick_marks, self.class_names)

        # Add text annotations
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

        save_path = os.path.join(self.save_dir, 'confusion_matrix.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved confusion matrix to {save_path}")
