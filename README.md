AI-Based Respiratory Sound Analysis for Disease Detection
Bu proje, akıllı telefonlar üzerinden alınan solunum seslerini analiz ederek hastalık teşhisi koyan uçtan uca bir sistemdir. Sistem; kullanıcı arayüzü için Flutter, sunucu tarafında veri işleme için Flask ve derin öğrenme tabanlı analiz için TensorFlow altyapısını kullanmaktadır.





1. Proje Mimarisi ve Bileşenler
Sistem, hesaplama yükünü sunucuya aktaran ve mobil cihazı bir sensör/arayüz olarak kullanan "Client-Server" mimarisine sahiptir.


Backend (Python/Flask) - D:\ai_sound

ai_server.py: Gelen ses dosyalarını karşılayan, MFCC özelliklerini çıkaran ve modeli çalıştıran ana API sunucusudur.




resp_model.h5: CNN ve BiLSTM katmanlarından oluşan, test setinde %92.15 doğruluk oranına sahip hibrit modeldir.



requirements.txt: Projenin bağımlılıklarını (tensorflow, librosa, flask, flask-cors) içerir.



Frontend (Flutter App) - D:\ai_sound_app

lib/screens: Ses kaydı alma, API'ye gönderme ve sonuçları görselleştirme işlemlerini yönetir.



AndroidManifest.xml: Uygulamanın çalışması için gerekli olan mikrofon ve ağ erişim izinlerini barındırır.


2. Sunucu Tarafı Operasyonel Akış (ai_server.py)
Sunucu tarafı, ham ses verisini tıbbi bir tanıya dönüştürmek için şu teknik adımları izler:

Veri Kabulü: Flask sunucusu /predict uç noktası üzerinden ses dosyalarını HTTP POST isteği ile kabul eder.


Ön İşleme: Ses sinyalleri 16 kHz örnekleme hızına getirilir ve genlik normalizasyonu uygulanır.

Özellik Çıkarımı (MFCC): Sinyalin frekans karakteristiğini yakalamak için 13 adet Mel-Frekans Cepstral Katsayısı hesaplanır.


Hibrit Analiz: CNN katmanları spektral özellikleri çıkarırken, BiLSTM katmanları sesteki zamansal değişimleri analiz eder.


Tanı Üretimi: Model; Astım, Bronşit, Zatürre veya Sağlıklı sınıflarından birini tahmin eder ve güven skorunu hesaplar.



3. Kurulum ve Çalıştırma Rehberi
Backend Kurulumu

Bash
cd D:\ai_sound
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python ai_server.py
Frontend Kurulumu

Bash
cd D:\ai_sound_app
flutter pub get
flutter run
4. Teknik Yapılandırma ve İzinler
Ağ Güvenliği: Yerel ağda test yapabilmek için Android tarafında temiz metin (HTTP) trafiğine izin verilmiştir.

API Bağlantısı: Emülatörler için 10.0.2.2, fiziksel cihazlar için bilgisayarın yerel IP adresi tanımlanmıştır.

Model Optimizasyonu: Sunucu, her istekte modeli yeniden yüklemek yerine başlangıçta belleğe alarak düşük gecikme süreli yanıt verir.

5. Başarım ve Akademik Katkı
Bu çalışma kapsamında geliştirilen sistem, "Asthma Detection Dataset v2" kullanılarak eğitilmiş ve test edilmiştir. Karmaşık ve gürültülü ses kayıtlarında dahi yüksek performans sergileyen bu model, tıbbi bilişim alanında mobil tabanlı erken teşhis çözümleri için bir örnek teşkil etmektedir.





Akademik Kabul: Bu projenin bilimsel bulguları, Nonlinear Science and Intelligent Applications (ISSN: 3105-7837) dergisinde yayınlanmak üzere kabul edilmiştir.
