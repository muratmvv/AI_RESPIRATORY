# AI_RESPIRATORY

Yapay Zekâ Destekli Solunum Analizi Uygulaması
Proje Özeti

Bu proje, kullanıcının sesli solunum kaydını alarak Python tabanlı bir yapay zekâ modeli aracılığıyla analiz eden bir sistemdir.
Mobil uygulama Flutter ile, sunucu tarafı ise Python (Flask) ile geliştirilmiştir.
Flask API, ses dosyasını alır, MFCC özelliklerini çıkarır ve TensorFlow modeli kullanarak solunum hastalıklarını tahmin eder.

1. Proje Yapısı
D:\ai_sound
 ┣ ai_server.py             # Flask sunucusu
 ┣ resp_model.h5            # Eğitilmiş TensorFlow modeli
 ┣ requirements.txt         # Gerekli Python kütüphaneleri
 ┗ /venv                    # Sanal ortam klasörü

D:\ai_sound_app
 ┣ lib/
 ┃ ┣ screens/
 ┃ ┃ ┣ disease_analysis_screen.dart   # Ses kaydı ve analiz ekranı
 ┃ ┃ ┣ analysis_history_screen.dart   # Sonuç geçmişi ekranı
 ┃ ┗ main.dart
 ┗ pubspec.yaml

2. Gerekli Kurulumlar
Python tarafı
cd D:\ai_sound
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

Flutter tarafı
cd D:\ai_sound_app
flutter pub get

3. Flask Sunucusunu Başlatma
3.1. Flask dosyası (ai_server.py)
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import librosa, tempfile, os

app = Flask(__name__)
CORS(app)

MODEL_PATH = "resp_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)
CLASSES = ["asthma", "bronchitis", "pneumonia", "healthy"]

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Dosya bulunamadı"}), 400

    file = request.files["file"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        file.save(tmp.name)
        y, sr = librosa.load(tmp.name, sr=None)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
        x = mfcc.reshape(1, 13, 1)
        probs = model.predict(x)[0]
        idx = np.argmax(probs)
        prediction = CLASSES[idx]
        confidence = float(probs[idx] * 100)
    os.remove(tmp.name)
    return jsonify({"prediction": prediction, "confidence": confidence})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

3.2. Sunucuyu çalıştır
cd D:\ai_sound
venv\Scripts\activate
python ai_server.py


Eğer başarıyla başladıysa:

Running on http://127.0.0.1:5000
Running on http://192.168.1.34:5000

4. Flutter Tarafı Çalıştırma
4.1. Android izinleri

android/app/src/main/AndroidManifest.xml içine ekle:

<uses-permission android:name="android.permission.RECORD_AUDIO"/>
<uses-permission android:name="android.permission.READ_MEDIA_AUDIO"/>
<uses-permission android:name="android.permission.POST_NOTIFICATIONS"/>

4.2. HTTP bağlantısına izin ver

android/app/src/main/res/xml/network_security_config.xml:

<?xml version="1.0" encoding="utf-8"?>
<network-security-config>
  <base-config cleartextTrafficPermitted="true"/>
</network-security-config>


ve AndroidManifest.xml içinde:

<application
  android:networkSecurityConfig="@xml/network_security_config"
  ... >

4.3. Flask bağlantısı için IP seçimi

Emülatörde çalıştırıyorsan:

final uri = Uri.parse("http://10.0.2.2:5000/predict");


Gerçek cihazda (aynı Wi-Fi’da):

final uri = Uri.parse("http://192.168.1.34:5000/predict");

5. Çalıştırma Adımları

Flask sunucusunu başlat:

python ai_server.py


Flutter uygulamasını başlat:

flutter run


Uygulamada “Hastalık Analizi” sayfasına git.

“Kayda Başla” → nefes al/ver → “Kaydı Durdur”.

Sunucu sesi analiz eder, sonucu JSON olarak döndürür.

Uygulama sonucu ekranda gösterir ve geçmişe kaydeder.

6. Olası Sorunlar

Cihazdan API’ye bağlanamıyor:

Bilgisayar ve cihaz aynı Wi-Fi’da olmalı.

Firewall’da 5000 portuna izin ver.

HTTP hatası:

network_security_config.xml doğru eklendiğinden emin ol.

Model bulunamadı:

resp_model.h5 dosyası Flask dizininde olmalı.

7. Kısa Çalışma Özeti

Bu proje, staj sürecimde geliştirilmiş uçtan uca bir yapay zekâ çözümüdür.
Ses kaydı alır, modeli çalıştırır ve hastalık tahmini üretir.
Flutter ile mobil arayüz, Flask ile API bağlantısı kurulmuştur.
Veri işleme, model eğitimi ve entegrasyon süreçlerinin tamamı tarafımdan gerçekleştirilmiştir.
