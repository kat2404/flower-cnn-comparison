
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "mobilenetv2_flowers.pth"

CLASSES = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(CLASSES)
)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=True)
)
model = model.to(device)
model.eval()

image_path = input("Enter flower image path: ").strip().strip('"')

try:
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_index = probabilities.argmax().item()

    print("\nFlower Prediction:", CLASSES[predicted_index])
    print(f"Confidence: {probabilities[predicted_index].item() * 100:.2f}%")

except FileNotFoundError:
    print("Image file not found. Check the image path.")
except Exception as error:
    print("Prediction error:", error)
