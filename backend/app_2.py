import os, io, json
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# --- Dizinler ---
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))        # ...\cervical-kanser-tespiti\backend
ROOT_DIR  = os.path.abspath(os.path.join(BASE_DIR, ".."))     # ...\cervical-kanser-tespiti
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")             # ...\cervical-kanser-tespiti\frontend

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)

# --- Dosya yolları ---
MODEL_PATH = os.path.join(BASE_DIR, "vgg16_sipakmed.h5")      # backend içindeki .h5
CLASS_JSON = os.path.join(ROOT_DIR, "class_indices.json")     # kökteki json (gerekirse BASE_DIR'e düşeriz)

if not os.path.exists(CLASS_JSON):
    CLASS_JSON = os.path.join(BASE_DIR, "class_indices.json")

# --- Model & sınıf isimleri ---
model = load_model(MODEL_PATH)
with open(CLASS_JSON, "r") as f:
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
    x = preprocess_image(request.files["file"].read())
    preds = model.predict(x)[0]
    idx = int(np.argmax(preds))
    return jsonify({
        "prediction": INV_MAP[idx],
        "confidence": round(float(preds[idx]), 4),
        "scores": {INV_MAP[i]: float(preds[i]) for i in range(len(preds))}
    })

@app.route("/", methods=["GET"])
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
