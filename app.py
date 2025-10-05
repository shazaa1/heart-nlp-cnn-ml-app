
# Gradio app for "AI-based Heart Disease Prediction & Detection"
# --------------------------------------------------------------
# Usage:
# 1) Put your model files (if you have them) in a folder named `models/` next to this file:
#    - ML model (joblib/pickle): models/ml_model.pkl
#    - NLP model (joblib/pickle): models/nlp_model.pkl
#    - CNN image model (tensorflow .h5) : models/cnn_model.h5
# 2) Install requirements: pip install -r requirements.txt
# 3) Run locally: python gradio_heart_app.py
# 4) To deploy on Hugging Face Spaces: create a new Space (Gradio), push this file + requirements.txt + models

# Note: If you don't have real model files, this app includes simple fallback heuristics so it runs
# and demonstrates the full integration (ML tabular, NLP symptoms, CNN image). Replace the
# fallback functions with real model loading when you upload model files.

import os
import joblib
import numpy as np
import gradio as gr
from PIL import Image
import random
import json

# -------------------- configuration --------------------
ML_MODEL_PATH = "models/ml_model.pkl"
NLP_MODEL_PATH = "models/nlp_model.pkl"
CNN_MODEL_PATH = "models/cnn_model.h5"

# -------------------- helpers / fallbacks --------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Fallback ML predictor: simple logistic-like heuristic
def ml_fallback_predict(age, sex, trestbps, chol, max_hr, thalach):
    # sex: 0 female, 1 male
    score = 0.02 * (age - 50) + 0.015 * (trestbps - 120) + 0.01 * (chol - 200) + 0.02 * (max_hr - 100)
    # small sex bias and random smoothing
    score += 0.05 * sex
    prob = sigmoid(score)
    return float(np.clip(prob, 0, 0.999))

# Fallback NLP predictor: keyword matching
HEART_KEYWORDS = ["chest pain", "chest-pain", "chestpain", "shortness of breath", "shortness", "breath", "palpitations", "dizzy", "faint", "angina", "pressure"]

def nlp_fallback_predict(text):
    text_l = text.lower()
    hits = sum(1 for kw in HEART_KEYWORDS if kw in text_l)
    if hits >= 1:
        return {"label": "Symptoms related to heart disease", "score": min(0.99, 0.4 + 0.2 * hits)}
    else:
        return {"label": "Symptoms unlikely heart-related", "score": 0.15}

# Fallback image predictor: simple random placeholder or basic brightness heuristic
def image_fallback_predict(img: Image.Image):
    # small heuristic: very dark or very bright images -> mark as suspicious
    arr = np.array(img.convert('L').resize((128,128)))
    mean = arr.mean()
    if mean < 40 or mean > 220:
        return {"label": "Abnormal image (placeholder)", "score": 0.7}
    # otherwise random small chance
    return {"label": "Normal image (placeholder)", "score": 0.1}

# -------------------- attempt to load real models --------------------
ml_model = None
nlp_model = None
cnn_model = None

try:
    if os.path.exists(ML_MODEL_PATH):
        ml_model = joblib.load(ML_MODEL_PATH)
except Exception as e:
    print("Could not load ML model:", e)

try:
    if os.path.exists(NLP_MODEL_PATH):
        nlp_model = joblib.load(NLP_MODEL_PATH)
except Exception as e:
    print("Could not load NLP model:", e)

# Note: for CNN model loading we intentionally do not import heavy frameworks (tf/torch)
# unless the user provides a model and environment. If you upload a tensorflow .h5 model,
# you can uncomment the tensorflow import lines and load it. For now we keep fallback.
#
# Example to enable (if you include tensorflow in requirements):
# import tensorflow as tf
# try:
#     if os.path.exists(CNN_MODEL_PATH):
#         cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
# except Exception as e:
#     print("Could not load CNN model:", e)

# -------------------- prediction wrappers --------------------
def predict_all(age, sex, trestbps, chol, max_hr, thalach, symptoms, image):
    # ML prediction
    try:
        if ml_model is not None:
            X = np.array([[age, sex, trestbps, chol, max_hr, thalach]])
            ml_prob = float(ml_model.predict_proba(X)[0][1])
        else:
            ml_prob = ml_fallback_predict(age, sex, trestbps, chol, max_hr, thalach)
    except Exception as e:
        print("ML prediction error:", e)
        ml_prob = ml_fallback_predict(age, sex, trestbps, chol, max_hr, thalach)

    # NLP prediction
    try:
        if nlp_model is not None:
            # assuming nlp_model has a predict_proba or predict interface
            lab = nlp_model.predict([symptoms])[0]
            # try to get probabilities
            if hasattr(nlp_model, 'predict_proba'):
                score = float(nlp_model.predict_proba([symptoms])[0].max())
            else:
                score = 0.8
            nlp_res = {"label": str(lab), "score": score}
        else:
            nlp_res = nlp_fallback_predict(symptoms)
    except Exception as e:
        print("NLP prediction error:", e)
        nlp_res = nlp_fallback_predict(symptoms)

    # Image prediction
    try:
        if image is not None:
            if cnn_model is not None:
                # placeholder for real cnn preproc + predict
                # y = cnn_preprocess_and_predict(cnn_model, image)
                img_res = {"label": "Model result (placeholder)", "score": 0.6}
            else:
                img_res = image_fallback_predict(image)
        else:
            img_res = {"label": "No image provided", "score": 0.0}
    except Exception as e:
        print("Image prediction error:", e)
        img_res = {"label": "Image error", "score": 0.0}

    # Meta recommendation
    combined_score = 0.6 * ml_prob + 0.3 * nlp_res.get('score', 0) + 0.1 * img_res.get('score', 0)
    risk_percent = int(np.round(combined_score * 100))

    if combined_score > 0.6:
        recommendation = "High risk — recommend visiting cardiologist for further tests."
    elif combined_score > 0.3:
        recommendation = "Moderate risk — consider further evaluation and monitoring."
    else:
        recommendation = "Low risk — maintain healthy lifestyle and follow up if symptoms worsen."

    out = {
        "ml_probability": f"{ml_prob*100:.1f}%",
        "nlp_label": nlp_res.get('label'),
        "nlp_score": f"{nlp_res.get('score',0)*100:.1f}%",
        "image_label": img_res.get('label'),
        "image_score": f"{img_res.get('score',0)*100:.1f}%",
        "combined_risk": f"{risk_percent}%",
        "recommendation": recommendation
    }
    return out

# -------------------- Gradio interface --------------------
with gr.Blocks(title="Heart Disease Prediction & Detection") as demo:
    gr.Markdown("# Heart Disease Prediction & Detection (Demo)\nThis interface integrates tabular ML, symptom NLP, and image analysis.\n\n_No names will be displayed on the UI._")

    with gr.Row():
        with gr.Column(scale=2):
            age = gr.Slider(minimum=1, maximum=120, step=1, label="Age", value=50)
            sex = gr.Radio(choices=["Female", "Male"], value="Male", label="Sex")
            trestbps = gr.Number(value=120, label="Resting Blood Pressure (mm Hg)")
            chol = gr.Number(value=200, label="Cholesterol (mg/dl)")
            max_hr = gr.Number(value=100, label="Max Heart Rate achieved")
            thalach = gr.Number(value=100, label="Thalach (max heart rate)")
            symptoms = gr.Textbox(lines=3, placeholder="Type symptoms here (e.g. chest pain, shortness of breath)", label="Symptoms (text)")
            image_input = gr.Image(type="pil", label="ECG / X-ray image (optional)")
            submit = gr.Button("Run Prediction")

        with gr.Column(scale=1):
            ml_out = gr.Textbox(label="ML model probability (tabular)")
            nlp_out = gr.Textbox(label="NLP symptoms analysis")
            img_out = gr.Textbox(label="Image analysis result")
            combined = gr.Textbox(label="Combined Risk Score")
            rec = gr.Textbox(label="Recommendation")

    def run_and_format(age, sex_label, trestbps, chol, max_hr, thalach, symptoms, image):
        sex_val = 1 if sex_label.lower().startswith('m') else 0
        res = predict_all(age, sex_val, trestbps, chol, max_hr, thalach, symptoms or "", image)
        ml_text = f"Risk (from tabular ML): {res['ml_probability']}"
        nlp_text = f"{res['nlp_label']} (confidence: {res['nlp_score']})"
        img_text = f"{res['image_label']} (confidence: {res['image_score']})"
        return ml_text, nlp_text, img_text, res['combined_risk'], res['recommendation']

    submit.click(fn=run_and_format,
                 inputs=[age, sex, trestbps, chol, max_hr, thalach, symptoms, image_input],
                 outputs=[ml_out, nlp_out, img_out, combined, rec])

    gr.Markdown("---\n**How to replace placeholders with real models:**\n1. Put trained models at the paths shown in the comments at the top.\n2. Replace fallback functions with model loading and preprocessing.\n3. Ensure dependencies (tensorflow/torch) added to requirements.txt if using a CNN model.\n4. To deploy on Hugging Face Spaces, push this file and requirements.txt to a new Space repository.")

# -------------------- requirements.txt content (copy to a separate requirements.txt file) --------------------
# gradio
# numpy
# pandas
# scikit-learn
# pillow
# joblib
# transformers  # optional for advanced NLP
# tensorflow   # optional if you will use a .h5 CNN model

if __name__ == "__main__":
    demo.launch(share=True)