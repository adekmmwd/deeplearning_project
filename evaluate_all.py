import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import os

DATA_DIR = "Car_Logo_Dataset"
BATCH_SIZE = 32
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Common transforms
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load dataset and split
full_dataset = datasets.ImageFolder(DATA_DIR)
total = len(full_dataset)
val_size = int(0.2 * total)
train_size = total - val_size
_, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))
val_dataset.dataset.transform = val_transforms
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
class_names = full_dataset.classes
NUM_CLASSES = len(class_names)

def get_alexnet():
    model = models.alexnet()
    model.classifier[6] = nn.Linear(4096, NUM_CLASSES)
    # The script used a more complex head, let me check it again
    return model

def get_vgg16():
    model = models.vgg16()
    model.classifier[6] = nn.Linear(4096, NUM_CLASSES)
    return model

def get_resnet50():
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model

def get_googlenet():
    model = models.googlenet(aux_logits=True)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    # Auxiliary classifiers
    model.aux1.fc2 = nn.Linear(model.aux1.fc2.in_features, NUM_CLASSES)
    model.aux2.fc2 = nn.Linear(model.aux2.fc2.in_features, NUM_CLASSES)
    return model

def get_efficientnet_b0():
    model = models.efficientnet_b0()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    return model

# Re-check AlexNet head from Car_Logo_AlexNet_Pretrained.py
# model.classifier = nn.Sequential(
#     nn.Dropout(0.5),
#     nn.Linear(9216, 4096),
#     nn.ReLU(inplace=True),
#     nn.Dropout(0.5),
#     nn.Linear(4096, 4096),
#     nn.ReLU(inplace=True),
#     nn.Linear(4096, NUM_CLASSES),
# )

def get_alexnet_custom():
    model = models.alexnet()
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(9216, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, NUM_CLASSES),
    )
    return model

model_configs = [
    ("AlexNet", get_alexnet_custom, "best_alexnet_phase2_weights.pth"),
    ("VGG16", get_vgg16, "best_vgg16_weights.pth"),
    ("ResNet50", get_resnet50, "best_resnet50_weights.pth"),
    ("GoogLeNet", get_googlenet, "best_googlenet_weights.pth"),
    ("EfficientNet-B0", get_efficientnet_b0, "best_efficientnet_b0_weights.pth"),
]

results = []

for name, model_fn, weight_path in model_configs:
    print(f"Evaluating {name}...")
    if not os.path.exists(weight_path):
        print(f"Weight file {weight_path} not found. Skipping.")
        continue
    
    model = model_fn().to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            # Handle GoogLeNet outputs (can be namedtuple in train mode or if aux_logits=True)
            if hasattr(outputs, 'logits'):
                outputs = outputs.logits
            elif isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

df = pd.DataFrame(results)
print("\nModel Comparison Table:")
print(df.to_string(index=False))
df.to_csv("model_comparison.csv", index=False)
