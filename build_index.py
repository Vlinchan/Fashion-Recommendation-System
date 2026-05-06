import numpy as np
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# ── Point this to your unzipped Kaggle images folder ──
SOURCE_DIR = "fashion-small/images"
SAVE_DIR   = "static/dataset_images"
MAX_IMAGES = 2000  # increase if your PC can handle it

os.makedirs(SAVE_DIR, exist_ok=True)

# Collect image paths
all_images = [
    f for f in os.listdir(SOURCE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
][:MAX_IMAGES]

image_paths = []
print(f"Copying {len(all_images)} images...")

for i, fname in enumerate(all_images):
    src = os.path.join(SOURCE_DIR, fname)
    dst = os.path.join(SAVE_DIR, f"{i}.png")
    img = Image.open(src).convert("RGB").resize((96, 96), Image.LANCZOS)
    img.save(dst)
    image_paths.append(dst)
    if i % 200 == 0:
        print(f"  {i}/{len(all_images)}")

# Build feature extractor
model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(96, 96, 3)
)

def extract_features(img_path):
    img = Image.open(img_path).convert("RGB").resize((96, 96))
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return model.predict(arr, verbose=0).flatten()

print("Extracting features...")
features = []
for i, path in enumerate(image_paths):
    if i % 100 == 0:
        print(f"  {i}/{len(image_paths)}")
    features.append(extract_features(path))

features = np.array(features)
np.save("features.npy", features)
np.save("image_paths.npy", np.array(image_paths))
print("✅ Done!")