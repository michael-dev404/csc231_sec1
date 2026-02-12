import os
import numpy as np
from feature_extraction import extract_features

dataset_path = "D:\Stamford (Term 2)\System Programming\Final Project\Project_videos"

labels_dict = {
    "walking": 0,
    "running": 1,
    "standing": 2
}

X = []
y = []

for category in labels_dict:
    folder_path = os.path.join(dataset_path, category)

    for video_file in os.listdir(folder_path):
        video_path = os.path.join(folder_path, video_file)

        print("Processing:", video_path)

        features = extract_features(video_path)

        if features is not None:
            X.append(features)
            y.append(labels_dict[category])

X = np.array(X)
y = np.array(y)

np.save("X.npy", X)
np.save("y.npy", y)

print("Dataset prepared successfully.")
print("Total samples:", len(X))
