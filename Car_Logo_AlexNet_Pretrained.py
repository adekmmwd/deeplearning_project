import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time


DATA_DIR = "Car_Logo_Dataset"

import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader

# Only preprocessing (NO augmentation)
train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load dataset WITHOUT transform first
full_dataset = ImageFolder(DATA_DIR)

total = len(full_dataset)
val_size = int(0.2 * total)
train_size = total - val_size

train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)

# Apply transforms correctly (IMPORTANT FIX)
train_dataset.dataset.transform = train_transforms
val_dataset.dataset.transform = val_transforms

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

class_names = full_dataset.classes
NUM_CLASSES = len(class_names)

print(f"Training samples: {train_size}")
print(f"Validation samples: {val_size}")
print(f"Number of classes: {NUM_CLASSES}")

# Load pretrained AlexNet
model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)

# Freeze all feature layers
for param in model.features.parameters():
    param.requires_grad = False

# Replace the classifier head for 35 classes
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(9216, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(inplace=True),
    nn.Linear(4096, NUM_CLASSES),
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(model)
print(f"\nDevice: {device}")
print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")


def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, patience=5, model_name="model"):
    history = {
        "accuracy": [], "val_accuracy": [], 
        "loss": [], "val_loss": [],
        "val_precision": [], "val_recall": [], "val_f1": []
    }
    
    best_f1 = 0.0
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # --- Validation ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= val_total
        val_acc = val_correct / val_total
        
        # Calculate P, R, F1 (macro average)
        v_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        v_rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        v_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        history["accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)
        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_precision"].append(v_prec)
        history["val_recall"].append(v_rec)
        history["val_f1"].append(v_f1)

        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"loss: {train_loss:.4f} - acc: {train_acc:.4f} - "
            f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f} - "
            f"val_f1: {v_f1:.4f}"
        )

        # Early Stopping
        if v_f1 > best_f1:
            best_f1 = v_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"best_{model_name}_weights.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                early_stop = True
                break
        
    if os.path.exists(f"best_{model_name}_weights.pth"):
        model.load_state_dict(torch.load(f"best_{model_name}_weights.pth"))

    return history


start_time = time.time()

criterion = nn.CrossEntropyLoss()

# Phase 1: Train only the classifier head
optimizer = optim.Adam(model.classifier.parameters(), lr=1e-4)
history = train_model(model, train_loader, val_loader, optimizer, criterion, epochs=10, model_name="alexnet_phase1")

# Phase 2: Unfreeze all layers for fine-tuning
for param in model.features.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=1e-5)
history_finetune = train_model(
    model, train_loader, val_loader, optimizer, criterion, epochs=20, model_name="alexnet_phase2"
)

# Merge histories
for key in history:
    history[key] = history[key] + history_finetune[key]

end_time = time.time()
training_time = end_time - start_time
params = sum(p.numel() for p in model.parameters())
best_val_acc = max(history["val_accuracy"])

print("Total Parameters:", params)
print("Training Time:", round(training_time, 2), "seconds")
print("Best Validation Accuracy:", round(best_val_acc * 100, 2), "%")

# Plotting and Auto-saving
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(history["accuracy"], label="Train Acc")
plt.plot(history["val_accuracy"], label="Val Acc")
plt.title("Accuracy")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(history["loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.title("Loss")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(history["val_precision"], label="Val Precision")
plt.plot(history["val_recall"], label="Val Recall")
plt.title("Precision & Recall")
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(history["val_f1"], label="Val F1 Score")
plt.title("F1 Score")
plt.legend()

plt.tight_layout()
plt.savefig("alexnet_metrics.png")
print("Metrics graph saved as alexnet_metrics.png")
plt.show()

model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(labels.numpy())

print("\nAlexNet Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("AlexNet Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("alexnet_confusion_matrix.png")
print("Confusion matrix saved as alexnet_confusion_matrix.png")
plt.show()
