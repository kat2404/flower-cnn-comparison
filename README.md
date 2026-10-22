# Flower Classification Using CNN Architectures

## Project Overview
This project compares MobileNetV2 and ResNet50 CNN architectures for classifying flower images into five categories: Daisy, Dandelion, Rose, Sunflower, and Tulip.

## Objective
To evaluate and compare the test accuracy of two CNN architectures on a flower image dataset.

## Technologies Used
- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Visual Studio Code

## Dataset
The dataset contains 4,317 images across five classes:
- Daisy: 764
- Dandelion: 1,052
- Rose: 784
- Sunflower: 733
- Tulip: 984

The dataset is stored locally and excluded from this Git repository.

## Methodology
1. Load and preprocess flower images.
2. Resize images to 224 x 224 pixels.
3. Normalize the images.
4. Train MobileNetV2 and ResNet50 for five-class classification.
5. Evaluate both models using the test dataset.
6. Compare test accuracy.
7. Predict the class of an individual flower image.

## Results

| Model | Test Accuracy | Correct Predictions |
|---|---:|---:|
| MobileNetV2 | 90.14% | 585/649 |
| ResNet50 | 90.45% | 587/649 |

ResNet50 achieved 0.31 percentage points higher test accuracy.

## Sample Prediction
- Predicted class: Daisy
- Model: MobileNetV2
- Confidence: 92.93%

## Project Structure
- `src/` - Training, evaluation, comparison, and prediction scripts
- `results/` - Accuracy graphs, loss graphs, and comparison results
- `models/` - Locally saved model weights, excluded from Git
- `dataset/` - Flower images, excluded from Git
- `requirements.txt` - Python dependencies

## Conclusion
Both models achieved approximately 90% test accuracy. ResNet50 performed slightly better in test accuracy, while MobileNetV2 successfully classified the sample daisy image.

## Future Enhancements
- Develop a web interface for image uploads.
- Evaluate precision, recall, and F1-score.
- Generate confusion matrices.
- Compare inference speed and computational requirements.
