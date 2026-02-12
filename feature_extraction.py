import cv2
import numpy as np
from skimage.feature import hog


def extract_features(video_path):
    cap = cv2.VideoCapture(video_path)
    features = []

    ret, prev_frame = cap.read()
    if not ret:
        return None

    prev_frame = cv2.resize(prev_frame, (128, 128))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (128, 128))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Gaussian Blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # HOG
        hog_features = hog(
            blurred,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False
        )

        # Optical Flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_feature = np.mean(magnitude)

        # Harris Corners
        corners = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
        corner_feature = np.sum(corners > 0.01 * corners.max())

        combined_features = np.hstack((hog_features, motion_feature, corner_feature))
        features.append(combined_features)

        prev_gray = gray

    cap.release()

    if len(features) == 0:
        return None

    return np.mean(features, axis=0)
