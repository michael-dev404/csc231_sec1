import numpy as np
import torch
from train_model import MotionCNN
from sklearn.model_selection import train_test_split

# Load data
X = np.load("X.npy")
y = np.load("y.npy")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

model = MotionCNN(X_test.shape[1])
model.load_state_dict(torch.load("motion_model.pth"))
model.eval()

with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test).float().mean()

print("Test Accuracy:", accuracy.item())
