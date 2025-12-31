# 🛠️ Geliştirme Raporu: Karşılaşılan Sorunlar ve Çözümler

Bu belge, "Türk İşaret Dili Tanıma Sistemi"nin geliştirilme sürecinde karşılaşılan teknik zorlukları, hataları ve bunlara üretilen mühendislik çözümlerini özetlemektedir. Proje raporunda "Karşılaşılan Zorluklar" veya "Geliştirme Süreci" başlığı altında kullanılabilir.

---

## 1. Kritik Hata: Kamera Görüntüsünün Gelmemesi (Thread Crash)
*   **Sorun:** Uygulama başlatıldığında veya "Veri Toplama" sekmesine geçildiğinde kamera görüntüsü ekrana gelmiyor, sistem tepkisiz kalıyordu.
*   **Sebep:** Kod sadeleştirme (refactoring) çalışmaları sırasında silinen "Easter Egg" (Sürpriz Yumurta) özelliğine ait eski bir fonksiyon çağrısının (`self.check_easter_egg`), ana video döngüsü (`_video_loop`) içinde unutulması. Bu durum, arka planda çalışan thread'in bir `AttributeError` fırlatarak sessizce çökmesine neden oluyordu.
*   **Çözüm:** Video işleme döngüsü satır satır incelendi (Debugging). Hatalı fonksiyon çağrısı tespit edilip temizlendi. Ayrıca, gelecekteki hataların thread'i çökertmesini engellemek için döngü kapsayıcı bir `try-except` bloğu ile güçlendirildi ve detaylı hata loglaması eklendi.

## 2. Syntax Hatası: Hatalı Hata Yakalama Bloğu
*   **Sorun:** Kamera düzeltmesi yapılırken uygulamanın hiç açılmaması.
*   **Sebep:** Video döngüsüne eklenen bir `try` bloğunun, `except` bloğu olmadan bırakılması (SyntaxError). Python'un girintileme (indentation) yapısındaki bir karışıklık sonucu ortaya çıktı.
*   **Çözüm:** Hatalı blok yapısı tespit edildi. `try-except` yapısı düzeltilerek kodun sözdizimi (syntax) geçerli hale getirildi.

## 3. Artık Veri Hatası: `is_paused` Değişkeni
*   **Sorun:** Kamera çalıştıktan kısa bir süre sonra uygulamanın kapanması.
*   **Sebep:** Yine kaldırılan özelliklerden kalan ve `__init__` metodundan silinen `self.is_paused` değişkeninin, video döngüsü içinde hala kontrol ediliyor olması.
*   **Çözüm:** Değişkenin kullanıldığı tüm satırlar (IDE'nin "Find References" özelliği ile) taranarak koddan tamamen temizlendi.

## 4. Thread Güvenliği (Thread Safety) ve UI Donmaları
*   **Sorun:** Video işleme sırasında arayüzün (butonlar, geçişler) donması veya titremesi.
*   **Sebep:** Python'un grafik arayüz kütüphanesi (Tkinter/CustomTkinter) "Thread-Safe" değildir. Arka plan thread'inden doğrudan arayüz elemanlarını güncellemeye çalışmak (örneğin `label.configure`), çakışmalara (race condition) yol açıyordu.
*   **Çözüm:** "Producer-Consumer" benzeri bir yapı kuruldu.
    1.  **Arka Plan Thread (`_video_loop`):** Sadece kameradan görüntü okur, MediaPipe ile işler ve sonucu bir değişkene (`self.current_frame_pil`) yazar. Arayüze dokunmaz.
    2.  **Ana Thread (`_update_video_ui`):** Belirli aralıklarla (30ms) değişkeni kontrol eder ve görüntüyü ekrana basar. `threading.Lock()` mekanizması ile veri bütünlüğü sağlandı.

## 5. Model Mimarisi ve Kullanıcı Deneyimi (UX) Basitleştirmesi
*   **Sorun:** Kullanıcıların "Units" (Nöron Sayısı) gibi derin öğrenme hiperparametrelerini yanlış ayarlaması sonucu eğitimin başarısız olması veya modelin hiçbir şey öğrenememesi.
*   **Çözüm:** LSTM modeli için en stabil performans veren mimari (64-128-64 katman yapısı) test edilerek belirlendi. Arayüzden bu karmaşık ayar kaldırılarak kod tarafında sabitlendi (Hardcoded). Böylece kullanıcı hatası minimize edildi ve model başarısı garanti altına alındı.

## 6. Veri Taşınabilirliği
*   **Sorun:** Toplanan verilerin (`.npy` dosyaları) klasör yapısını bozmadan başka bilgisayara taşınmasının zorluğu.
*   **Çözüm:** Python `zipfile` ve `shutil` kütüphaneleri kullanılarak tek tıkla **"Veri Dışa Aktar (.zip)"** ve **"Veri Klasörü Yükle"** özellikleri geliştirildi. Bu özellik, klasör hiyerarşisini (Eylem Adı -> Video ID -> Frame ID) koruyarak yedekleme yapmayı sağladı.

---
