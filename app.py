from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import io, base64, os

app = Flask(__name__)

# Load precomputed index
features = np.load("features.npy")
image_paths = np.load("image_paths.npy")

# Load model
model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(96, 96, 3))

def extract_features(pil_image):
    img = pil_image.convert("RGB").resize((96, 96))
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return model.predict(arr, verbose=0).flatten()

def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    img = Image.open(io.BytesIO(file.read()))
    query_feat = extract_features(img).reshape(1, -1)

    # Cosine similarity
    sims = cosine_similarity(query_feat, features)[0]
    top_indices = np.argsort(sims)[::-1][:5]  # Top 8 results

    results = []
    for idx in top_indices:
        results.append({
            "image": image_to_base64(image_paths[idx]),
            "score": round(float(sims[idx]) * 100, 1),
            "path": image_paths[idx]
        })

    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(debug=True)