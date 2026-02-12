import torch
import numpy as np
import os
from feature_extraction import extract_features
from train_model import MotionCNN

classes = ["walking", "running", "standing"]


def main():
    # Ask user for video path
    video_path = input("Enter video path (example: dataset/walking/test1.mp4): ")

    # Check if file exists
    if not os.path.exists(video_path):
        print("Error: File does not exist.")
        return

    print("Extracting features...")

    features = extract_features(video_path)

    if features is None:
        print("Could not extract features from video.")
        return

    features = torch.tensor(features, dtype=torch.float32)

    # Load model
    model = MotionCNN(features.shape[0])
    model.load_state_dict(torch.load("motion_model.pth"))
    model.eval()

    # Predict
    with torch.no_grad():
        output = model(features)
        probabilities = torch.softmax(output, dim=0)
        _, prediction = torch.max(output, 0)

    print("\nPrediction:", classes[prediction.item()])
    print("\nConfidence Scores:")
    for i, class_name in enumerate(classes):
        print(f"{class_name}: {probabilities[i].item() * 100:.2f}%")


if __name__ == "__main__":
    main()
