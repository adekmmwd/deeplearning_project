# =========================
# 1. IMPORTS
# =========================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import vgg16, VGG16_Weights
import numpy as np
import time
import os
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 2. DATASET PREPARATION
# =========================
DATA_DIR = "Car_Logo_Dataset"

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
full_dataset = datasets.ImageFolder(DATA_DIR)

total = len(full_dataset)
val_size = int(0.2 * total)
train_size = total - val_size

train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)

# Apply transforms correctly
train_dataset.dataset.transform = train_transforms
val_dataset.dataset.transform = val_transforms

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

class_names = full_dataset.classes
NUM_CLASSES = len(class_names)

print(f"Training samples: {train_size}")
print(f"Validation samples: {val_size}")
print(f"Number of classes: {NUM_CLASSES}")

# =========================
# 3. MODEL SETUP
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

# Freeze feature layers
for param in model.features.parameters():
    param.requires_grad = False

# Replace the classifier head for NUM_CLASSES
model.classifier[6] = nn.Linear(model.classifier[6].in_features, NUM_CLASSES)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

print(model)
print(f"\nDevice: {device}")
print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")

# =========================
# 4. TRAINING FUNCTIONS
# =========================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    v_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    v_rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    v_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return total_loss / len(loader), correct / total, v_prec, v_rec, v_f1

# =========================
# 5. TRAINING LOOP
# =========================
start_time = time.time()
epochs = 20
patience = 5
best_f1 = 0.0
epochs_no_improve = 0

history = {
    "train_loss": [], "train_acc": [],
    "val_loss": [], "val_acc": [],
    "val_precision": [], "val_recall": [], "val_f1": []
}

for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc, v_prec, v_rec, v_f1 = evaluate(model, val_loader, criterion)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["val_precision"].append(v_prec)
    history["val_recall"].append(v_rec)
    history["val_f1"].append(v_f1)

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val F1: {v_f1:.4f}")

    # Early Stopping based on F1
    if v_f1 > best_f1:
        best_f1 = v_f1
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_vgg16_weights.pth")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

if os.path.exists("best_vgg16_weights.pth"):
    model.load_state_dict(torch.load("best_vgg16_weights.pth"))

# =========================
# 6. EVALUATION & METRICS
# =========================
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(labels.numpy())

# Plotting and Auto-saving
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(history["train_acc"], label="Train Acc")
plt.plot(history["val_acc"], label="Val Acc")
plt.title("VGG16 Accuracy")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.title("VGG16 Loss")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(history["val_precision"], label="Val Precision")
plt.plot(history["val_recall"], label="Val Recall")
plt.title("VGG16 Precision & Recall")
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(history["val_f1"], label="Val F1 Score")
plt.title("VGG16 F1 Score")
plt.legend()

plt.tight_layout()
plt.savefig("vgg16_metrics.png")
print("Metrics graph saved as vgg16_metrics.png")
plt.show()

print("\nVGG16 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("VGG16 Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("vgg16_confusion_matrix.png")
print("Confusion matrix saved as vgg16_confusion_matrix.png")
plt.show()

end_time = time.time()
print("Training Time:", end_time - start_time, "seconds")
