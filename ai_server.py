import os
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np
import librosa
import tensorflow as tf

# === Model Ayarları ===
MODEL_PATH = "resp_model.h5"
CLASS_NAMES = ["Asthma", "Bronchiectasis", "Bronchiolitis", "COPD", "Healthy"]
SAMPLE_RATE = 22050
N_MFCC = 13
DURATION_SEC = 5.0

# === Modeli Yükle ===
print("[INFO] Model yükleniyor...")
model = tf.keras.models.load_model(MODEL_PATH)
print("[OK] Model yüklendi:", MODEL_PATH)

# === Özellik Çıkarma (MFCC) ===
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    target_len = int(SAMPLE_RATE * DURATION_SEC)

    # Ses uzunluğu sabitle (5 saniye)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean

# === Flask Uygulaması ===
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)

@app.route("/health", methods=["GET"])
def health():
    """Test endpointi — API çalışıyor mu kontrolü"""
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    """Flutter'dan gelen ses dosyasını alır, tahmin yapar"""
    try:
        if "file" not in request.files:
            return jsonify({"error": "form-data içinde 'file' bulunamadı"}), 400

        file = request.files["file"]
        if not file.filename.endswith(".wav"):
            return jsonify({"error": "Yalnızca .wav dosyaları destekleniyor"}), 400

        # Dosyayı kaydet
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_DIR, filename)
        file.save(file_path)
        print(f"[INFO] Yeni dosya alındı: {file_path}")

        # Özellik çıkarımı
        features = extract_features(file_path)
        features = np.expand_dims(features, axis=(0, 2))  # (1, 13, 1)

        # Tahmin
        prediction = model.predict(features)
        pred_index = np.argmax(prediction)
        pred_label = CLASS_NAMES[pred_index]
        confidence = float(prediction[0][pred_index])

        print(f"[RESULT] Tahmin: {pred_label} ({confidence*100:.2f}%)")

        return jsonify({
            "prediction": pred_label,
            "confidence": round(confidence * 100, 2),
            "saved_as": filename
        }), 200

    except Exception as e:
        print("[ERROR]", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
