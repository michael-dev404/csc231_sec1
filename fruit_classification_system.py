import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------- PATHS ----------
DATASET_PATH = r"D:\Stamford (Term 2)\System Programming\Image processing coding stuffs\dataset\train"
MODEL_PATH = "fruit_cnn_model_tf215.keras"

# ---------- PARAMETERS ----------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# ---------- DATA GENERATOR ----------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    zoom_range=0.3,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

num_classes = train_data.num_classes

# ---------- CNN MODEL ----------
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),

    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ---------- TRAIN ----------
print("\n🚀 Training CNN model...\n")
model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

# ---------- SAVE MODEL ----------
model.save(MODEL_PATH)
print(f"\n✅ Model saved as {MODEL_PATH}")
print("📂 Class mapping:", train_data.class_indices)
