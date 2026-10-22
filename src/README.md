# Flower Classification Using CNN Architectures

## 1. Project Overview
This project compares two CNN architectures, MobileNetV2 and ResNet50, for classifying flower images into five categories: Daisy, Dandelion, Rose, Sunflower, and Tulip.

## 2. Objective
To compare the classification accuracy of MobileNetV2 and ResNet50 on a flower image dataset.

## 3. Technologies Used
- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Visual Studio Code

## 4. Dataset
The dataset contains 4,317 images across five flower classes.

- Daisy: 764 images
- Dandelion: 1,052 images
- Rose: 784 images
- Sunflower: 733 images
- Tulip: 984 images

## 5. Methodology
1. Load and preprocess flower images.
2. Resize images to 224 × 224 pixels.
3. Normalize images.
4. Train MobileNetV2 and ResNet50 for five-class classification.
5. Evaluate both models using the test dataset.
6. Compare model accuracy.
7. Predict the class of an individual flower image.

## 6. Results

| Model | Test Accuracy | Correct Predictions |
|---|---:|---:|
| MobileNetV2 | 90.14% | 585/649 |
| ResNet50 | 90.45% | 587/649 |

ResNet50 achieved 0.31 percentage points higher test accuracy than MobileNetV2.

## 7. Sample Prediction
- Input: Daisy flower image
- Predicted class: Daisy
- Model: MobileNetV2
- Confidence: 92.93%

## 8. Conclusion
Both models achieved approximately 90% test accuracy. ResNet50 performed slightly better in accuracy, while MobileNetV2 correctly classified the sample daisy image.

## 9. Project Structure
- `dataset/` — Flower images
- `models/` — Saved model weights
- `src/flower_classification.py` — MobileNetV2 training
- `src/resnet50_flowers.py` — ResNet50 training
- `src/test_mobilenet.py` — MobileNetV2 evaluation
- `src/compare_models.py` — Model comparison
- `src/predict_flower.py` — Single-image prediction
- `results/` — Graphs and evaluation results

## 10. Future Enhancements
- Create a web interface for uploading flower images.
- Compare precision, recall, and F1-score.
- Generate confusion matrices.
- Compare model inference speed.
