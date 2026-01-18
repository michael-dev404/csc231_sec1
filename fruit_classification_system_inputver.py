import tensorflow as tf
import numpy as np
import cv2
import os
from tkinter import Tk, filedialog

# ---------- PATHS ----------
MODEL_PATH = r"D:\Stamford (Term 2)\System Programming\Image processing coding stuffs\dataset\fruit_cnn_model_tf215.keras"
DATASET_PATH = r"D:\Stamford (Term 2)\System Programming\Image processing coding stuffs\dataset\train"

IMG_SIZE = (224, 224)

# ---------- LOAD MODEL ----------
model = tf.keras.models.load_model(MODEL_PATH)

# ---------- LOAD CLASS NAMES (MATCH TRAINING ORDER) ----------
# This matches ImageDataGenerator.flow_from_directory behavior
class_names = sorted(
    entry.name for entry in os.scandir(DATASET_PATH) if entry.is_dir()
)

print("✅ Loaded Classes:", class_names)

# ---------- IMAGE FILE DIALOG ----------
def select_image():
    root = Tk()
    root.withdraw()
    return filedialog.askopenfilename(
        title="Select Fruit Image",
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )

# ---------- PREPROCESS (MATCH TRAINING) ----------
def preprocess_image(img_path):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None, None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    img_norm = img_resized / 255.0
    img_input = np.expand_dims(img_norm, axis=0)

    return img_bgr, img_input

# ---------- PREDICT ----------
def predict_fruit():
    img_path = select_image()
    if not img_path:
        print("❌ No image selected")
        return

    original_img, img_input = preprocess_image(img_path)
    if original_img is None:
        print("❌ Failed to load image")
        return

    predictions = model.predict(img_input, verbose=0)[0]

    class_id = int(np.argmax(predictions))
    fruit = class_names[class_id]
    confidence = predictions[class_id] * 100

    # ---------- DEBUG OUTPUT ----------
    print("\n📊 All Class Probabilities:")
    for i, prob in enumerate(predictions):
        print(f"{class_names[i]:15s}: {prob*100:.2f}%")

    # ---------- DRAW BOX ----------
    h, w, _ = original_img.shape
    cv2.rectangle(original_img, (20, 20), (w - 20, h - 20), (0, 255, 0), 3)
    cv2.putText(
        original_img,
        f"{fruit} ({confidence:.2f}%)",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Fruit Classification Result", original_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n🎯 FINAL RESULT")
    print(f"🍎 Fruit Type : {fruit}")
    print(f"📊 Confidence : {confidence:.2f}%")

# ---------- RUN ----------
if __name__ == "__main__":
    predict_fruit()
