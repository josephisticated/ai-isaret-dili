# 🤟 Türk İşaret Dili Tanıma Sistemi (AI Destekli)

Bu proje, **MediaPipe Holistic** ve **LSTM/GRU/CNN** gibi derin öğrenme modellerini kullanarak gerçek zamanlı Türk İşaret Dili (TİD) tanıma ve eğitim sistemi sunar. Kullanıcı dostu arayüzü ile kendi veri setinizi oluşturabilir, modelinizi eğitebilir ve canlı testler yapabilirsiniz.

![Uygulama Ekran Görüntüsü](preview.png) *<!-- Buraya ekran görüntüsü ekleyebilirsiniz -->*

---

## 🚀 Özellikler

*   **⚡ Gerçek Zamanlı Tanıma:** Kameradan alınan görüntüleri anlık olarak işler ve çevirir.
*   **🛠️ Kolay Veri Toplama:** Kendi işaretlerinizi kolayca kaydedin. **"Test Et"** modu ile kayıt almadan pratik yapın.
*   **💾 Veri ve Model Yönetimi:** Veri setlerinizi ve eğitilmiş modellerinizi tek tıkla dışa aktarın (Zip) veya yükleyin.
*   **📊 Gelişmiş Grafikler:** Eğitim sonrası **F1-Score**, **Confusion Matrix** ve **Accuracy** grafiklerini inceleyin.
*   **🧠 Optimize Edilmiş Modeller:** Kararlı performans için sabitlenmiş LSTM mimarisi ve otomatik optimizasyon seçenekleri.
*   **🎨 Modern Arayüz:** CustomTkinter ile geliştirilmiş, tamamen Türkçe modern arayüz.
*   **🏗️ Modüler Mimari:** OOP prensipleri ile refaktör edilmiş, sürdürülebilir kod yapısı.

---

## 📦 Kurulum

1.  **Projeyi İndirin:**
    ```bash
    git clone https://github.com/josephisticated/ai-isaret-dili.git
    cd ai-isaret-dili
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
*   **Yeni Kelime:** Öğretmek istediğiniz kelimeyi girin (örn. "Merhaba").
*   **Adet:** Kaç tane örnek video toplanacağını belirtin.
*   **TOPLAMAYI BAŞLAT:** Geri sayım (3sn) sonrası kaydı başlatır. Ekranda **"BEKLE"** (Sarı) ve **"KAYIT"** (Kırmızı) komutlarını takip edin.
*   **TEST ET:** Veri kaydetmeden toplama sürecini simüle eder.
*   **İçe/Dışa Aktar:** Sol menüden veri klasörünüzü yedekleyebilir (Zip) veya yedeği geri yükleyebilirsiniz.

### 2. Eğitim (Training)
*   **Model Mimarisi:** LSTM, GRU, CNN veya Bi-LSTM seçeneklerinden birini seçin.
*   **Ayarlar:** Epoch (Döngü), Dropout ve Learning Rate gibi değerleri değiştirebilirsiniz.
*   **Eğitimi Başlat:** Topladığınız verilerle modeli eğitin. Sonuçlar (Loss/Accuracy) canlı olarak güncellenir.
*   **Grafikler:** Eğitim bitince doğruluk, kayıp ve **Confusion Matrix** grafiklerini sekmelerde inceleyin.

### 3. Tahmin (Prediction)
*   **TAHMİNİ BAŞLAT:** Eğitilen modeli yükler ve kameradan gerçek zamanlı çeviri yapar.
*   **Sonuç:** Tahmin edilen kelime ve doğruluk oranı (%) yeşil renkle videonun üzerine yazılır.
*   **Log:** Algılanan hareketler tarihçeli olarak alttaki kutuda listelenir.

---

## 📂 Proje Yapısı

*   `app.py`: Ana uygulama ve kullanıcı arayüzü (GUI).
*   `predictor.py`: Gerçek zamanlı tahmin mantığını içeren `SignLanguagePredictor` sınıfı.
*   `model_trainer.py`: Derin öğrenme modellerinin eğitimi ve değerlendirilmesi.
*   `data_collector.py`: Veri toplama ve veri artırma (augmentation) işlemleri.
*   `utils.py`: MediaPipe Holistic entegrasyonu ve yardımcı fonksiyonlar.
*   `config.py`: Proje genelindeki yollar ve parametre ayarları.
*   `main.py`: Komut satırı (CLI) üzerinden kullanım seçeneği.

---

## 🤝 Katkıda Bulunma

Hata bildirimleri ve özellik istekleri için lütfen "Issues" kısmını kullanın. Pull request'ler memnuniyetle karşılanır!

---

**Geliştirici:** josephisticated
**Lisans:** MIT
