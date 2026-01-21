import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import(
    Conv2D, MaxPooling2D,
    Dense, Flatten, Dropout
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# ---------- PATHS ----------
DATASET_PATH = r"D:\Stamford (Term 2)\System Programming\Image processing coding stuffs\dataset\train"
MODEL_SAVE_PATH = r"D:\Stamford (Term 2)\System Programming\Image processing coding stuffs\dataset\fruit_cnn_deep_tf215.keras"

# ---------- PARAMETERS ----------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

# ---------- DATA GENERATORS ----------
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_data = train_gen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = train_gen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

NUM_CLASSES = train_data.num_classes

# ---------- DEEPER CNN MODEL ----------
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu"),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------- TRAIN ----------
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data
)

# ---------- ACCURACY GRAPH ----------
plt.figure()
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Deeper CNN - Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Training Accuracy", "Validation Accuracy"])
plt.show()

# ---------- SAVE MODEL ----------
model.save(MODEL_SAVE_PATH)

# ---------- COMPARISON INFO ----------
print("\n📊 Deeper CNN Summary")
print("Architecture Type: Deeper CNN")
print("Epochs:", EPOCHS)
print("Total Parameters:", model.count_params())
print("Final Training Accuracy:", history.history["accuracy"][-1])
print("Final Validation Accuracy:", history.history["val_accuracy"][-1])

print("✅ Deeper CNN model saved successfully")
