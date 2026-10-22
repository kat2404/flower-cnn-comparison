
from pathlib import Path
import time

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import random_split, DataLoader

# 1. Project dataset location
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

CLASSES = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

print("Dataset location:", DATASET_DIR)

# 2. Check flower image counts
for flower in CLASSES:
    folder = DATASET_DIR / flower
    if folder.exists():
        images = [
            file for file in folder.iterdir()
            if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]
        print(f"{flower}: {len(images)} images")
    else:
        print(f"{flower}: Folder not found!")

# 3. Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 4. Load dataset
dataset = datasets.ImageFolder(
    root=str(DATASET_DIR),
    transform=transform
)

print("\nClasses:", dataset.classes)

# 5. Split dataset reproducibly
train_size = int(0.70 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_set, val_set, test_set = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

print("Total images:", len(dataset))
print("Training images:", len(train_set))
print("Validation images:", len(val_set))
print("Testing images:", len(test_set))

# 6. Create DataLoaders
train_loader = DataLoader(
    train_set, batch_size=16, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_set, batch_size=16, shuffle=False, num_workers=0
)
test_loader = DataLoader(
    test_set, batch_size=16, shuffle=False, num_workers=0
)

print("\nDataLoaders created successfully!")

# 7. Select CPU or GPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Device:", device)

# 8. Load pretrained MobileNetV2
print("\nLoading pretrained MobileNetV2...")

mobilenet = models.mobilenet_v2(
    weights=models.MobileNet_V2_Weights.DEFAULT
)

# Freeze pretrained feature extractor
for param in mobilenet.features.parameters():
    param.requires_grad = False

# Replace final layer for five flower classes
in_features = mobilenet.classifier[1].in_features
mobilenet.classifier[1] = nn.Linear(
    in_features, len(dataset.classes)
)

mobilenet = mobilenet.to(device)

print("MobileNetV2 model loaded successfully!")

# 9. Configure training
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    mobilenet.classifier.parameters(),
    lr=0.001
)

EPOCHS = 5
train_losses = []
val_accuracies = []

# 10. Train MobileNetV2
print("\nStarting MobileNetV2 training...")
start_time = time.perf_counter()

for epoch in range(EPOCHS):
    mobilenet.train()
    mobilenet.features.eval()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = mobilenet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_set)
    train_losses.append(epoch_loss)

    # Validation
    mobilenet.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = mobilenet(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    val_accuracy = 100 * correct / total
    val_accuracies.append(val_accuracy)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {epoch_loss:.4f} | "
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )

training_time = time.perf_counter() - start_time
print(f"\nTraining time: {training_time:.2f} seconds")

# 11. Save trained model
MODEL_DIR.mkdir(parents=True, exist_ok=True)
torch.save(
    mobilenet.state_dict(),
    MODEL_DIR / "mobilenetv2_flowers.pth"
)
print("Model saved successfully!")

# 12. Save training graphs
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), train_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("MobileNetV2 Training Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "mobilenetv2_loss.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), val_accuracies, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy (%)")
plt.title("MobileNetV2 Validation Accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "mobilenetv2_accuracy.png")
plt.close()

print("Training graphs saved in results folder.")

# 13. Evaluate on unseen test images
print("\nEvaluating MobileNetV2 on test dataset...")
mobilenet.eval()
correct = 0
total = 0
all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = mobilenet(images)
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

test_accuracy = 100 * correct / total

print(f"\nMobileNetV2 Test Accuracy: {test_accuracy:.2f}%")
print(f"Correct predictions: {correct}/{total}")

# 14. Save test results
with open(RESULTS_DIR / "mobilenetv2_test_results.txt", "w") as file:
    file.write("MobileNetV2 Test Results\n")
    file.write("========================\n")
    file.write(f"Test Accuracy: {test_accuracy:.2f}%\n")
    file.write(f"Correct Predictions: {correct}/{total}\n")
    file.write(f"Training Time: {training_time:.2f} seconds\n")

print("Test results saved in results folder.")
print("\nMobileNetV2 training and evaluation completed!")
