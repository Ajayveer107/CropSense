# 🌿 CropSense — AI Crop Disease Detector

Upload a leaf photo → get instant disease diagnosis, symptoms, treatment, and prevention advice.  
Works with **Gemini Vision** out of the box (no GPU needed). Optionally plug in your trained CNN model.

---

## ⚡ Run in VS Code (Step-by-Step)

### 1. Open in VS Code
```
File → Open Folder → select the CropSense folder
```

### 2. Open Terminal  (`Ctrl + \``)

### 3. Create & activate virtual environment
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Get FREE Gemini API key
→ https://aistudio.google.com/app/apikey

### 6. Set API key
```bash
# Windows:
set GEMINI_API_KEY=your_key_here

# Mac/Linux:
export GEMINI_API_KEY=your_key_here
```

### 7. Run
```bash
python app.py
```

### 8. Open browser
```
http://127.0.0.1:5001
```

---

## 🔧 Optional: Use Your Trained CNN Model
If you have your trained `.h5` model:
1. Create a `model/` folder inside CropSense
2. Place `cropsense_model.h5` inside it
3. Create `model/class_names.json` — array of class names matching your model output
4. Uncomment `tensorflow` in `requirements.txt` and run `pip install tensorflow`

The app auto-detects the model file and uses it instead of Gemini Vision.

---

## 🛠 Tech Stack
| Component | Technology |
|-----------|-----------|
| Backend | Flask (Python) |
| Image Processing | Pillow + OpenCV (optional) |
| AI Diagnosis | Google Gemini 1.5 Flash Vision |
| Optional CNN | TensorFlow / Keras (MobileNetV2) |
| Dataset Reference | PlantVillage (54K images, 38 classes) |
