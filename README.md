# 🤟 Türk İşaret Dili Tanıma Sistemi (AI Destekli)

Bu proje, **MediaPipe Holistic** ve **LSTM/GRU/CNN** gibi derin öğrenme modellerini kullanarak gerçek zamanlı Türk İşaret Dili (TİD) tanıma ve eğitim sistemi sunar. Kullanıcı dostu arayüzü ile kendi veri setinizi oluşturabilir, modelinizi eğitebilir ve canlı testler yapabilirsiniz.

![Uygulama Ekran Görüntüsü](preview.png) *<!-- Buraya ekran görüntüsü ekleyebilirsiniz -->*

---

## 🚀 Özellikler

*   **⚡ Gerçek Zamanlı Tanıma:** Kameradan alınan görüntüleri anlık olarak işler ve çevirir.
*   **🛠️ Kolay Veri Toplama:** Kendi işaretlerinizi kolayca kaydedin ve veri seti oluşturun.
*   **🧠 Esnek Model Eğitimi:** LSTM, GRU, CNN ve Bi-LSTM gibi farklı mimarilerle modelinizi eğitin.
*   **🧪 Test Modu:** Veri kaydetmeden sadece alıştırma yapmak için "Test Et" modu.
*   **🎨 Modern Arayüz:** CustomTkinter ile geliştirilmiş şık ve karanlık mod destekli arayüz.
*   **🇹🇷 Tamamen Türkçe:** Arayüz ve kod açıklamaları tamamen Türkçe'dir.

---

## 📦 Kurulum

1.  **Projeyi İndirin:**
    ```bash
    git clone https://github.com/kullaniciadi/proje-adi.git
    cd proje-adi
    ```

2.  **Gerekli Kütüphaneleri Yükleyin:**
    Python 3.10.11 önerilir.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Uygulamayı Başlatın:**
    ```bash
    python app.py
    ```

---

## 🎮 Kullanım

### 1. Veri Toplama (Data Collection)
*   **Yeni Kelime:** Öğretmek istediğiniz kelimeyi girin (örn. "Merkaba").
*   **Adet:** Kaç tane örnek video toplanacağını belirtin (varsayılan: 30).
*   **TOPLAMAYI BAŞLAT:** Kayıt işlemini başlatır. Hazırlanmanız için geri sayım yapar.
*   **TEST ET:** Veri kaydetmeden toplama sürecini simüle eder.

### 2. Eğitim (Training)
*   **Model Mimarisi:** İhtiyacınıza uygun modeli seçin (LSTM genelde iyi bir başlangıçtır).
*   **Epochs:** Eğitim süresi.
*   **Eğitimi Başlat:** Topladığınız verilerle modeli eğitin.
*   **Otomatik Optimizasyon:** En iyi parametreleri (Keras Tuner ile) otomatik bulmak için bu seçeneği kullanın.

### 3. Tahmin (Prediction)
*   **TAHMİNİ BAŞLAT:** Eğitilen modeli yükler ve kameradan gerçek zamanlı çeviri yapar.
*   Tahmin edilen kelime ve doğruluk oranı ekranda yeşil renkle gösterilir.

---

## 📂 Proje Yapısı

*   `app.py`: Ana uygulama ve kullanıcı arayüzü.
*   `model_trainer.py`: Model oluşturma ve eğitim işlemleri.
*   `data_collector.py`: Kamera ve Mediapipe işlemleri.
*   `predictor.py`: Canlı tahmin mantığı.
*   `utils.py`: MediaPipe çizim yardımcıları.
*   `config.py`: Ayarlar ve sabitler.

---

## 🤝 Katkıda Bulunma

Hata bildirimleri ve özellik istekleri için lütfen "Issues" kısmını kullanın. Pull request'ler memnuniyetle karşılanır!

---

**Geliştirici:** Yusuf
**Lisans:** MIT

