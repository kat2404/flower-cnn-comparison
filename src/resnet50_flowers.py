
from pathlib import Path
import time

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import random_split, DataLoader

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = datasets.ImageFolder(
    root=str(DATASET_DIR),
    transform=transform
)

train_size = int(0.70 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_set, val_set, test_set = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_set, batch_size=16, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_set, batch_size=16, shuffle=False, num_workers=0
)
test_loader = DataLoader(
    test_set, batch_size=16, shuffle=False, num_workers=0
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Device:", device)
print("Classes:", dataset.classes)

print("\nLoading pretrained ResNet50...")
resnet = models.resnet50(
    weights=models.ResNet50_Weights.DEFAULT
)

# Freeze pretrained feature layers
for param in resnet.parameters():
    param.requires_grad = False

# Replace final classifier for 5 flower classes
in_features = resnet.fc.in_features
resnet.fc = nn.Linear(in_features, len(dataset.classes))

resnet = resnet.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet.fc.parameters(), lr=0.001)

EPOCHS = 5
train_losses = []
val_accuracies = []

print("\nStarting ResNet50 training...")
start_time = time.perf_counter()

for epoch in range(EPOCHS):
    resnet.train()
    # Keep frozen layers in evaluation mode
    resnet.eval()
    resnet.fc.train()

    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = resnet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_set)
    train_losses.append(epoch_loss)

    resnet.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            predictions = resnet(images).argmax(dim=1)
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

MODEL_DIR.mkdir(parents=True, exist_ok=True)
torch.save(resnet.state_dict(), MODEL_DIR / "resnet50_flowers.pth")
print("ResNet50 model saved!")

# Test evaluation
resnet.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        predictions = resnet(images).argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

test_accuracy = 100 * correct / total
print(f"\nResNet50 Test Accuracy: {test_accuracy:.2f}%")
print(f"Correct predictions: {correct}/{total}")

# Save graphs
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), train_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("ResNet50 Training Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "resnet50_loss.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), val_accuracies, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy (%)")
plt.title("ResNet50 Validation Accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "resnet50_accuracy.png")
plt.close()

with open(RESULTS_DIR / "resnet50_test_results.txt", "w") as file:
    file.write(f"Test Accuracy: {test_accuracy:.2f}%\n")
    file.write(f"Correct Predictions: {correct}/{total}\n")
    file.write(f"Training Time: {training_time:.2f} seconds\n")

print("Graphs and test results saved!")
