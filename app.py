from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import io, base64, os
from gradio_client import Client, handle_file

app = Flask(__name__)

# ── Load precomputed index ──
features    = np.load("features.npy")
image_paths = np.load("image_paths.npy")

# ── Feature extractor ──
model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(96, 96, 3)
)

# ── Free HuggingFace try-on client ──
print("Connecting to IDM-VTON on HuggingFace (free)...")
tryon_client = Client("yisol/IDM-VTON")
print("✅ Connected!")

BASE_MODEL = "static/base_model.jpg"


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

    sims = cosine_similarity(query_feat, features)[0]
    top_indices = np.argsort(sims)[::-1][:5]

    results = []
    for idx in top_indices:
        results.append({
            "image": image_to_base64(image_paths[idx]),
            "score": round(float(sims[idx]) * 100, 1),
            "path": str(image_paths[idx])   # send path to frontend for try-on
        })

    return jsonify({"results": results})


@app.route("/tryon", methods=["POST"])
def tryon():
    cloth_path = request.form.get("cloth_path")

    # Validate paths
    if not cloth_path or not os.path.exists(cloth_path):
        return jsonify({"error": "Clothing image not found on server"}), 400

    if not os.path.exists(BASE_MODEL):
        return jsonify({
            "error": "Base model image missing. Add static/base_model.jpg"
        }), 400

    try:
        print(f"Running try-on for: {cloth_path}")

        result = tryon_client.predict(
            dict={
                "background": handle_file(BASE_MODEL),
                "layers":    [],
                "composite": None
            },
            garm_img=handle_file(cloth_path),
            garment_des="clothing item",
            is_checked=True,
            is_checked_crop=False,
            denoise_steps=30,
            seed=42,
            api_name="/tryon"
        )

        # result[0] is the output image path
        result_image_path = result[0]
        with open(result_image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        print("✅ Try-on complete!")
        return jsonify({"image": b64})

    except Exception as e:
        print(f"❌ Try-on error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)