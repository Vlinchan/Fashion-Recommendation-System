import numpy as np
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Load Fashion MNIST
(x_train, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()

# Use first 2000 images for speed (increase if you want)
x_train = x_train[:2000]

# Save images as PNGs
os.makedirs("static/dataset_images", exist_ok=True)
image_paths = []

print("Saving dataset images...")
for i, img_array in enumerate(x_train):
    img = Image.fromarray(img_array).convert("RGB").resize((96, 96))
    path = f"static/dataset_images/{i}.png"
    img.save(path)
    image_paths.append(path)

# Build feature extractor
model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(96, 96, 3))

def extract_features(img_path):
    img = Image.open(img_path).convert("RGB").resize((96, 96))
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return model.predict(arr, verbose=0).flatten()

print("Extracting features (this takes a few minutes)...")
features = []
for i, path in enumerate(image_paths):
    if i % 100 == 0:
        print(f"  {i}/{len(image_paths)}")
    features.append(extract_features(path))

features = np.array(features)
np.save("features.npy", features)
np.save("image_paths.npy", np.array(image_paths))
print("✅ Done! features.npy and image_paths.npy saved.")