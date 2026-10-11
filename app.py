"""
CropSense — AI Crop Disease Detector
Uses a pre-trained MobileNetV2 (fine-tuned on PlantVillage) via a Keras model.
If you don't have a trained model yet, the app will use Gemini Vision as fallback.
"""

from flask import Flask, request, render_template, jsonify # type: ignore
import os, base64, io, json
from PIL import Image
import numpy as np
import google.generativeai as genai # type: ignore

app = Flask(__name__)

# ── CONFIG ──────────────────────────────────────────────────────────────────
# Get FREE Gemini key → https://aistudio.google.com/app/apikey
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
genai.configure(api_key=GEMINI_KEY)
gemini_vision = genai.GenerativeModel("gemini-1.5-flash")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB max

ALLOWED = {"png", "jpg", "jpeg", "webp"}

# ── Try to load Keras model (optional — falls back to Gemini Vision) ──────
keras_model   = None
CLASS_NAMES   = []
try:
    import tensorflow as tf # type: ignore
    MODEL_PATH  = "model/cropsense_model.h5"
    LABELS_PATH = "model/class_names.json"
    if os.path.exists(MODEL_PATH):
        keras_model = tf.keras.models.load_model(MODEL_PATH)
        if os.path.exists(LABELS_PATH):
            with open(LABELS_PATH) as f:
                CLASS_NAMES = json.load(f)
        print("✅ Keras model loaded.")
    else:
        print("ℹ️  No local model found — using Gemini Vision.")
except ImportError:
    print("ℹ️  TensorFlow not installed — using Gemini Vision.")

# ── HELPERS ───────────────────────────────────────────────────────────────
def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED

def preprocess(image: Image.Image, size=(224, 224)):
    img = image.convert("RGB").resize(size)
    arr = np.array(img, dtype="float32") / 255.0
    return np.expand_dims(arr, 0)

def diagnose_with_keras(image: Image.Image):
    arr  = preprocess(image)
    preds = keras_model.predict(arr, verbose=0)[0]
    idx  = int(np.argmax(preds))
    conf = float(preds[idx])
    label = CLASS_NAMES[idx] if CLASS_NAMES else f"Class {idx}"
    # PlantVillage labels look like "Tomato___Late_blight"
    parts = label.replace("___", " — ").replace("_", " ")
    return parts, round(conf * 100, 1)

def diagnose_with_gemini(image: Image.Image, filename: str):
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    prompt = """You are an expert plant pathologist AI.
Analyse this leaf image and respond ONLY with a JSON object (no markdown) like:
{
  "crop": "Tomato",
  "disease": "Late Blight",
  "confidence": 87,
  "healthy": false,
  "symptoms": "Dark brown water-soaked lesions on leaves with white mouldy growth.",
  "treatment": "Apply copper-based fungicide. Remove and destroy infected plant parts. Avoid overhead watering.",
  "prevention": "Use resistant varieties, ensure good air circulation, and rotate crops annually."
}
If the leaf looks healthy, set "disease" to "Healthy" and "healthy" to true."""

    response = gemini_vision.generate_content([
        prompt,
        {"mime_type": "image/jpeg", "data": base64.b64encode(img_bytes).decode()}
    ])
    raw = response.text.strip().replace("```json", "").replace("```", "")
    return json.loads(raw)

# ── ROUTES ────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/diagnose", methods=["POST"])
def diagnose():
    file = request.files.get("leaf")
    if not file or not allowed(file.filename):
        return jsonify({"error": "Upload a JPG, PNG, or WEBP leaf image."}), 400

    image = Image.open(file.stream)

    if keras_model is not None:
        label, conf = diagnose_with_keras(image)
        result = {
            "crop":       label.split(" — ")[0] if " — " in label else "Unknown",
            "disease":    label.split(" — ")[1] if " — " in label else label,
            "confidence": conf,
            "healthy":    "healthy" in label.lower(),
            "symptoms":   "Detected via local CNN model.",
            "treatment":  "Consult a local agriculture expert for treatment advice.",
            "prevention": "Practice crop rotation and use disease-resistant varieties."
        }
    else:
        try:
            result = diagnose_with_gemini(image, file.filename)
        except Exception as e:
            return jsonify({"error": f"Diagnosis failed: {e}"}), 500

    return jsonify(result)

if __name__ == "__main__":
    print("🌱 CropSense running → http://127.0.0.1:5001")
    app.run(debug=True, port=5001)
