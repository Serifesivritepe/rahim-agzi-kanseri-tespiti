import io
import json
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os


app = Flask(
    __name__,
    static_folder="frontend",  
    static_url_path=""         
)
CORS(app)


MODEL_PATH = "vgg16_sipakmed.h5"   
model = load_model(MODEL_PATH)

with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
INV_MAP = {int(v): k for k, v in class_indices.items()}


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)


@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        return jsonify({"error": "file alanı bulunamadı"}), 400

    img_bytes = request.files["file"].read()
    print("⏺ Received bytes:", len(img_bytes))

 
    x = preprocess_image(img_bytes)
    print(f"⏺ Input tensor shape: {x.shape}")
    print(f"⏺ Input tensor min: {x.min():.4f}, max: {x.max():.4f}")

  
    preds = model.predict(x)[0]
    print("⏺ Prediction vector:", preds)

  
    idx = int(np.argmax(preds))
    label = INV_MAP[idx]
    confidence = float(preds[idx])

  
    return jsonify({
        "prediction": label,
        "confidence": round(confidence, 4),
        "scores": {INV_MAP[i]: float(preds[i]) for i in range(len(preds))}
    })


@app.route("/", methods=["GET"])
def serve_index():
    return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
