
import matplotlib.pyplot as plt
import os

# Verified test results
models = ["MobileNetV2", "ResNet50"]
accuracies = [90.14, 90.45]
correct = [585, 587]
incorrect = [64, 62]

# Create results folder if needed
os.makedirs("results", exist_ok=True)

# Accuracy comparison graph
plt.figure(figsize=(8, 5))
bars = plt.bar(models, accuracies)

plt.title("Flower Classification: CNN Model Comparison")
plt.ylabel("Test Accuracy (%)")
plt.ylim(0, 100)

for bar, accuracy in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{accuracy:.2f}%",
        ha="center"
    )

plt.tight_layout()
plt.savefig("results/model_accuracy_comparison.png", dpi=300)
plt.show()

# Save final comparison report
with open("results/final_comparison.txt", "w") as file:
    file.write("FLOWER CNN COMPARISON\n")
    file.write("=====================\n")
    file.write("MobileNetV2: 90.14% (585/649 correct)\n")
    file.write("ResNet50: 90.45% (587/649 correct)\n")
    file.write("Accuracy difference: 0.31 percentage points\n")
    file.write("ResNet50 has slightly higher test accuracy.\n")

print("Comparison graph saved to results/model_accuracy_comparison.png")
print("Final report saved to results/final_comparison.txt")
