import customtkinter as ctk
import cv2
import threading
import time
import numpy as np
from PIL import Image, ImageTk
import config
from utils import MediapipeHelper
from data_collector import DataCollector
from model_trainer import ModelTrainer
from predictor import SignLanguagePredictor
import os
import shutil
import zipfile
import sys
from tkinter import filedialog, messagebox
from sklearn.model_selection import train_test_split

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class CTkToolTip(ctk.CTkToplevel):
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave)
        self.widget.bind("<Unmap>", self.leave)
        self.id = None
        self.tooltip_window = None

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hide()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(200, self.show)

    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def show(self):
        # Widget artık görünür değilse göstermeyi engelle
        try:
            if not self.widget.winfo_exists() or not self.widget.winfo_viewable():
                return
        except:
            return

        try:
            # Herhangi bir widget için genel konumlandırma (Düğme, Çerçeve vb.)
            x = self.widget.winfo_rootx() + 20
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 10
            
            self.tooltip_window = ctk.CTkToplevel(self.widget)
            self.tooltip_window.wm_overrideredirect(True)
            self.tooltip_window.wm_geometry(f"+{x}+{y}")
            # En üstte kaldığından emin ol
            self.tooltip_window.attributes('-topmost', True)
            
            label = ctk.CTkLabel(self.tooltip_window, text=self.text, corner_radius=10, fg_color="gray20", text_color="white", padx=10, pady=5)
            label.pack()
        except Exception as e:
            print(f"Tooltip error: {e}") 

    def hide(self):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None

    @classmethod
    def hide_all(cls):
        # Gecikme mantığıyla, çoğunlukla 'ayrılma' etkinliklerine güveniyoruz, ancak bu hala
        # örnekleri sürekli izlersek yararlı olabilir. Basitlik ve kararlılık için etkinlik bağlamalarına güvenelim.
        # Ancak app.py'deki mevcut çağrıyı karşılamak için bunu bir geçiş olarak tutacağız veya gerekirse temel bir zorla temizleme uygulayacağız.
        pass

class IORedirector(object):
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.buffer = ""

    def write(self, str):
        self.text_widget.insert("end", str)
        self.text_widget.see("end")

    def flush(self):
        pass

class SignLanguageApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("İşaret Dili AI Eğitmeni & Çeviricisi")
        self.geometry("1300x850") # Daha iyi düzen için genişletildi

        # Mantık Bileşenleri
        self.cap = cv2.VideoCapture(0)
        self.mp_helper = MediapipeHelper()
        self.collector = DataCollector()
        self.trainer = ModelTrainer()
        self.predictor = None 
        self.actions = []
        
        # Durum Değişkenleri
        self.is_collecting = False
        self.is_testing = False  # Test modu durumu
        self.is_predicting = False
        self.draw_landmarks = True
        self.prediction_active = False
        self.collection_sequence_count = 0
        self.current_sequence_idx = 0
        self.collection_frame_count = 0
        self.stop_event = threading.Event()
        self.show_train_warning = True
        self.unsaved_changes = False
        
        
        # Ses Başlatma
        
        
        
        
        # İş Parçacığı Güvenliği
        self.current_frame_pil = None
        self.ui_lock = threading.Lock()
        
        # Geri sayım durumu
        self.countdown_active = False
        self.countdown_value = 0
        self.countdown_start_time = 0
        
        # Arayüzü Kur
        self._setup_ui()
        
        # Hareketleri Yükle
        self._update_actions_list()
        
        # Video Döngüsünü Başlat
        self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self.video_thread.start()
        
        # Arayüz Güncelleme Döngüsünü Başlat (Ana İş Parçacığı)
        self._update_video_ui()
        
        # Odağı sıfırlamak için tıklamayı bağla, ancak YALNIZCA bir girişe/düğmeye tıklanmıyorsa
        self.bind("<Button-1>", self._on_global_click)

    def _on_global_click(self, event):
        # Tıklanan widget'ın bir giriş widget'ı olup olmadığını kontrol et (Giriş, Metin vb.)
        try:
            widget_class = event.widget.winfo_class()
            # Giriş için standart Tkinter sınıfları
            if widget_class in ['Entry', 'Text', 'TEntry', 'TCombobox']:
                return
        except Exception:
            pass
            
        # Aksi takdirde giriş odağı temizlemek için köke odaklan
        self.focus()

    def _setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar ---
        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(8, weight=1) # Boşluk

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="SignAI Studio", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(40, 20))

        # Veri Yönetimi Bölümü
        ctk.CTkLabel(self.sidebar_frame, text="Veri Yönetimi", font=ctk.CTkFont(size=14, weight="bold")).grid(row=1, column=0, padx=20, sticky="w")
        
        self.btn_import_data = ctk.CTkButton(self.sidebar_frame, text="Veri Klasörü Yükle", command=self._import_data_folder)
        self.btn_import_data.grid(row=2, column=0, padx=20, pady=10)
        
        self.btn_export_data = ctk.CTkButton(self.sidebar_frame, text="Veri Dışa Aktar (Zip)", command=self._export_data_dialog)
        self.btn_export_data.grid(row=3, column=0, padx=20, pady=10)

        # Model Yönetimi Bölümü
        ctk.CTkLabel(self.sidebar_frame, text="Model Yönetimi", font=ctk.CTkFont(size=14, weight="bold")).grid(row=4, column=0, padx=20, pady=(20, 0), sticky="w")

        self.btn_import_model = ctk.CTkButton(self.sidebar_frame, text="Model Yükle", command=self._import_model)
        self.btn_import_model.grid(row=5, column=0, padx=20, pady=10)
        
        self.btn_export_model = ctk.CTkButton(self.sidebar_frame, text="Modeli Kaydet", command=self._export_model)
        self.btn_export_model.grid(row=6, column=0, padx=20, pady=10)

        # Kamera Seçimi
        ctk.CTkLabel(self.sidebar_frame, text="Kamera Seçimi", font=ctk.CTkFont(size=14, weight="bold")).grid(row=7, column=0, padx=20, pady=(20, 0), sticky="w")
        self.camera_var = ctk.StringVar(value="Kamera 0")
        self.camera_menu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Kamera 0", "Kamera 1", "Kamera 2"], command=self._change_camera, variable=self.camera_var)
        self.camera_menu.grid(row=8, column=0, padx=20, pady=10)


        # --- Ana Alan ---
        self.tabview = ctk.CTkTabview(self, width=250)
        self.tabview.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.tabview.add("Veri Toplama")
        self.tabview.add("Eğitim")
        self.tabview.add("Tahmin")
        
        # Sekme değişimi için komutu doğru şekilde ayarlama
        self.tabview.configure(command=self._on_tab_change_command)

        self._setup_collection_tab()
        self._setup_training_tab()
        self._setup_prediction_tab()
        
    def _on_tab_change_command(self):
        # Standart geri aramayı işlemek için sarmalayıcı
        self._on_tab_change(self.tabview.get())

    def _setup_collection_tab(self):
        tab = self.tabview.tab("Veri Toplama")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        # Kontrol Çerçevesi
        controls = ctk.CTkFrame(tab)
        controls.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        
        # Kelime Girişi
        self.entry_word = ctk.CTkEntry(controls, placeholder_text="Yeni Kelime Giriniz", width=200, height=35)
        self.entry_word.pack(side="left", padx=10, pady=10)
        CTkToolTip(self.entry_word, "Öğretmek istediğiniz kelimeyi buraya yazın.")

        vcmd = (self.register(self._validate_int_input), '%P')
        
        # Adet Girişi
        self.entry_count = ctk.CTkEntry(controls, placeholder_text="Adet", width=60, height=35,
                                        validate="key", validatecommand=vcmd)
        self.entry_count.insert(0, str(config.NO_SEQUENCES))
        self.entry_count.pack(side="left", padx=5, pady=10)
        self.entry_count.bind("<FocusOut>", lambda e: self._clamp_value(self.entry_count, 1, 30, int))
        CTkToolTip(self.entry_count, "Kaç adet video toplanacağı.")
        
        # Kaydırıcı
        slider_frame = ctk.CTkFrame(controls, fg_color="transparent")
        slider_frame.pack(side="left", padx=10)
        
        self.slider_delay = ctk.CTkSlider(slider_frame, from_=0, to=5, number_of_steps=5, command=self._update_delay_label)
        self.slider_delay.set(2)
        self.slider_delay.pack(pady=5)
        
        self.label_delay = ctk.CTkLabel(slider_frame, text="Delay: 2s")
        self.label_delay.pack(pady=0)
        
        # İskelet Göster/Gizle
        self.switch_landmarks = ctk.CTkSwitch(controls, text="İskeleti Göster", command=self._toggle_landmarks)
        self.switch_landmarks.select()
        self.switch_landmarks.pack(side="right", padx=10)
        
        # Veri Çoğaltma Düğmesi
        ctk.CTkButton(controls, text="Veri Çoğalt (Gürültü)", fg_color="purple", width=120, command=self._open_augmentation_dialog).pack(side="right", padx=10)

        # Durum Çubuğu
        status_frame = ctk.CTkFrame(tab, height=50)
        status_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        
        self.status_light = ctk.CTkLabel(status_frame, text="  ", fg_color="gray", width=30, height=30, corner_radius=15)
        self.status_light.pack(side="left", padx=20, pady=10)
        
        self.status_label = ctk.CTkLabel(status_frame, text="Durum: Hazır", font=("Arial", 16))
        self.status_label.pack(side="left", pady=10)
        
        self.count_label = ctk.CTkLabel(status_frame, text="0 / 30", font=("Arial", 16, "bold"))
        self.count_label.pack(side="right", padx=20)

        # Video Alanı
        self.video_label_col = ctk.CTkLabel(tab, text="", fg_color="black", corner_radius=10)
        self.video_label_col.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        
        self.video_label_col = ctk.CTkLabel(tab, text="", fg_color="black", corner_radius=10)
        self.video_label_col.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        
        # Eylem Düğmeleri Çerçevesi
        btn_frame = ctk.CTkFrame(tab, fg_color="transparent")
        btn_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=10)
        
        # Test Düğmesi
        self.btn_test = ctk.CTkButton(btn_frame, text="TEST ET", fg_color="orange", height=50, width=150, font=("Arial", 16, "bold"), command=self._toggle_test)
        self.btn_test.pack(side="left", padx=(0, 10), fill="x", expand=True)

        # Toplamayı Başlat Düğmesi
        self.btn_collect = ctk.CTkButton(btn_frame, text="TOPLAMAYI BAŞLAT", fg_color="green", height=50, font=("Arial", 16, "bold"), command=self._toggle_collection)
        self.btn_collect.pack(side="left", fill="x", expand=True)

    def _setup_training_tab(self):
        tab = self.tabview.tab("Eğitim")
        tab.grid_columnconfigure(0, weight=1)
        
        # --- Ayarlar ---
        settings_frame = ctk.CTkFrame(tab)
        settings_frame.pack(pady=10, padx=20, fill="x")
        
        # Model Seçimi
        ctk.CTkLabel(settings_frame, text="Model Mimarisi:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.model_var = ctk.StringVar(value="LSTM")
        models = ["LSTM", "GRU", "CNN", "Bi-LSTM"]
        for i, m in enumerate(models):
            rb = ctk.CTkRadioButton(settings_frame, text=m, variable=self.model_var, value=m)
            rb.grid(row=0, column=i+1, padx=10, pady=10)
            
        # Hiperparametreler
        self.hp_frame = ctk.CTkFrame(tab)
        self.hp_frame.pack(pady=10, padx=20, fill="x")
        
        # İpuçları ile Etiketler ve Girişler
        params = [
            ("Epochs:", config.EPOCHS, "Eğitim döngüsü sayısı. Ne kadar süreceğini belirler."),
            ("Dropout:", "0.2", "Unutma oranı (0.1-0.5). Ezberlemeyi önler."),
            ("LR:", "0.0001", "Öğrenme hızı. Küçük değerler daha hassas öğrenir.")
        ]
        
        self.hp_entries = {}
        for i, (label, default, tooltip) in enumerate(params):
            lbl = ctk.CTkLabel(self.hp_frame, text=label)
            lbl.grid(row=0, column=i*2, padx=5, pady=10)
            CTkToolTip(lbl, tooltip)
            
            entry = ctk.CTkEntry(self.hp_frame, width=80)
            entry.insert(0, str(default))
            entry.grid(row=0, column=i*2+1, padx=5, pady=10)
            self.hp_entries[label] = entry

        self.entry_epochs = self.hp_entries["Epochs:"]
        self.entry_dropout = self.hp_entries["Dropout:"]
        self.entry_lr = self.hp_entries["LR:"]
        
        # Eğitim Girişlerine Doğrulama Uygula
        vcmd = (self.register(self._validate_int_input), '%P')
        
        self.entry_epochs.configure(validate="key", validatecommand=vcmd)
        self.entry_epochs.bind("<FocusOut>", lambda e: self._clamp_value(self.entry_epochs, 1, 1000, int))
        
        self.entry_dropout.bind("<FocusOut>", lambda e: self._clamp_value(self.entry_dropout, 0.1, 0.9, float))
        self.entry_lr.bind("<FocusOut>", lambda e: self._clamp_value(self.entry_lr, 0.0001, 0.1, float))
        
        # Otomatik Geçiş
        self.auto_var = ctk.IntVar(value=0)
        self.chk_auto = ctk.CTkCheckBox(tab, text="Otomatik Optimizasyon (Keras Tuner)", variable=self.auto_var, command=self._toggle_auto_settings)
        self.chk_auto.pack(pady=5)
        CTkToolTip(self.chk_auto, "En iyi parametreleri otomatik olarak bulur.")

        # Erken Durdurma Geçişi
        self.early_stop_var = ctk.BooleanVar(value=True)
        self.chk_early = ctk.CTkCheckBox(tab, text="Erken Durdurma (Early Stopping)", variable=self.early_stop_var)
        self.chk_early.pack(pady=5)
        CTkToolTip(self.chk_early, "Gelişme durursa eğitimi otomatik bitirir.")

        # Eğitim Düğmesi & Metrikler Düğmesi
        action_frame = ctk.CTkFrame(tab, fg_color="transparent")
        action_frame.pack(pady=10, padx=20, fill="x")
        
        self.btn_train = ctk.CTkButton(action_frame, text="Eğitimi Başlat", fg_color="#1f538d", height=50, font=("Arial", 16, "bold"), command=self._start_training)
        self.btn_train.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        self.btn_metrics = ctk.CTkButton(action_frame, text="Metrikleri Göster", fg_color="purple", height=50, font=("Arial", 16, "bold"), command=self._show_current_metrics)
        self.btn_metrics.pack(side="right", fill="x", expand=True)
        
        # Günlükler
        self.log_box = ctk.CTkTextbox(tab, height=300, font=("Consolas", 12))
        self.log_box.pack(pady=10, padx=20, fill="both", expand=True)

    def _setup_prediction_tab(self):
        tab = self.tabview.tab("Tahmin")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        
        self.video_label_pred = ctk.CTkLabel(tab, text="", fg_color="black")
        self.video_label_pred.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        controls = ctk.CTkFrame(tab)
        controls.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        
        self.lbl_prediction = ctk.CTkLabel(controls, text="Tahmin: ...", font=("Arial", 24, "bold"), text_color="#00ff00")
        self.lbl_prediction.pack(side="left", padx=20)
        
        self.btn_toggle_pred = ctk.CTkButton(controls, text="TAHMİNİ BAŞLAT", fg_color="green", height=40, command=self._toggle_prediction)
        self.btn_toggle_pred.pack(side="right", padx=10, pady=10)

    # --- Arayüz Mantık Yöntemleri ---
    def _update_delay_label(self, value):
        self.label_delay.configure(text=f"Gecikme: {int(value)}sn")

    def _validate_int_input(self, P):
        if P == "": return True
        return P.isdigit()
        
    def _clamp_value(self, widget, min_val, max_val, type_func):
        try:
            val = type_func(widget.get())
            if val < min_val: val = min_val
            if val > max_val: val = max_val
        except ValueError:
            # Geçersizse, min_val değerine veya varsayılan mantığa geri dön
            val = min_val
            
        # Widget'ı güncelle
        widget.delete(0, "end")
        widget.insert(0, str(val))

    def _toggle_auto_settings(self):
        state = "disabled" if self.auto_var.get() == 1 else "normal"
        self.entry_epochs.configure(state=state)
        self.entry_dropout.configure(state=state)
        self.entry_lr.configure(state=state)
        # Otomatik ayarlama genellikle mimariyi arıyorsa Mimari seçimini devre dışı bırak
        # Ancak burada seçilen mimari İÇİN ayarlama yapıyoruz veya küresel.
        # Uygulama Planı "Seçilen modeli alır... ve hiper parametreleri optimize eder" dedi.
        # Bu yüzden model seçimini etkin tutuyoruz.

    # --- Veri ve Model Yönetimi ---
    
    def _import_data_folder(self):
        folder_selected = filedialog.askdirectory(title="İçe Aktarılacak Veri Klasörünü Seç")
        if not folder_selected:
            return
            
        def import_task():
            count = 0
            try:
                self.log_box.insert("end", "Veri içe aktarılıyor... Lütfen bekleyin.\n")
                self.btn_train.configure(state="disabled") # İçe aktarma sırasında eğitimi devre dışı bırak
                
                for item in os.listdir(folder_selected):
                    source_item = os.path.join(folder_selected, item)
                    if os.path.isdir(source_item):
                        action_name = item
                        target_action_dir = os.path.join(config.DATA_PATH, action_name)
                        if not os.path.exists(target_action_dir):
                            os.makedirs(target_action_dir)
                        
                        # Hedefteki son indeksi bul
                        # listdir üzerindeki izin hataları için try-except kullan
                        try:
                            numeric_files = [int(f) for f in os.listdir(target_action_dir) if f.isdigit()]
                        except:
                            numeric_files = []
                            
                        start_idx = max(numeric_files) + 1 if numeric_files else 0
                        
                        # Sekansları kopyala
                        for sub_item in os.listdir(source_item):
                            source_seq_dir = os.path.join(source_item, sub_item)
                            if os.path.isdir(source_seq_dir):
                                target_seq_dir = os.path.join(target_action_dir, str(start_idx))
                                try:
                                    shutil.copytree(source_seq_dir, target_seq_dir)
                                    start_idx += 1
                                    count += 1
                                except FileExistsError:
                                    # Çakışma varsa atla veya akıllıca hallet
                                    pass
                                except Exception as e:
                                    print(f"Copy error: {e}")

                self._update_actions_list()
                self.log_box.insert("end", f"İçe aktarma tamamlandı: {count} yeni sekans.\n")
                # Arayüz geri aramalarını zamanla
                self.after(0, lambda: messagebox.showinfo("Başarılı", f"{count} adet sekans başarıyla içe aktarıldı."))
                
            except Exception as e:
                self.log_box.insert("end", f"İçe aktarma hatası: {str(e)}\n")
                self.after(0, lambda: messagebox.showerror("Hata", f"İçe aktarma başarısız: {str(e)}"))
            finally:
                self.after(0, lambda: self.btn_train.configure(state="normal"))

        threading.Thread(target=import_task, daemon=True).start()

    def _open_augmentation_dialog(self):
        self._update_actions_list()
        if self.actions.size == 0:
            messagebox.showwarning("Uyarı", "Veri bulunamadı. Önce veri toplayın veya içe aktarın.")
            return

        dialog = ctk.CTkToplevel(self)
        dialog.title("Veri Çoğaltma (Augmentation)")
        dialog.geometry("350x500")
        
        ctk.CTkLabel(dialog, text="Hangi kelimeler çoğaltılsın? (Gürültü Ekleme)", font=("Arial", 14, "bold")).pack(pady=10)
        
        # Onay Kutuları
        scroll = ctk.CTkScrollableFrame(dialog, height=300)
        scroll.pack(fill="both", expand=True, padx=10)
        
        vars = []
        for action in self.actions:
            var = ctk.BooleanVar(value=True)
            chk = ctk.CTkCheckBox(scroll, text=action, variable=var)
            chk.pack(anchor="w", pady=2)
            vars.append((action, var))
            
        def perform_augmentation():
            selected = [a for a, v in vars if v.get()]
            if not selected:
                return
            
            dialog.destroy()
            
            def augment_task():
                self.btn_train.configure(state="disabled")
                self.log_box.insert("end", "Veri çoğaltma işlemi başlatıldı...\n")
                
                total_new = 0
                for action in selected:
                    count = self.collector.augment_action(action, num_copies=1)
                    total_new += count
                    self.log_box.insert("end", f"{action}: {count} kopya eklendi.\n")
                    self.log_box.see("end")
                    
                self.log_box.insert("end", f"Toplam {total_new} yeni veri oluşturuldu.\n")
                self.after(0, lambda: messagebox.showinfo("Başarılı", f"{total_new} adet yeni veri üretildi."))
                self.after(0, lambda: self.btn_train.configure(state="normal"))
                self._update_actions_list()
                
            threading.Thread(target=augment_task, daemon=True).start()

        ctk.CTkButton(dialog, text="Seçilenleri Çoğalt (x2)", command=perform_augmentation, fg_color="green").pack(pady=20)

    def _export_data_dialog(self):
        if not self.actions.size > 0:
            messagebox.showwarning("Uyarı", "Dışa aktarılacak veri yok.")
            return

        dialog = ctk.CTkToplevel(self)
        dialog.title("Veri Dışa Aktar")
        dialog.geometry("300x400")
        
        ctk.CTkLabel(dialog, text="Dışa Aktarılacak Kelimeleri Seç:").pack(pady=10)
        
        scroll = ctk.CTkScrollableFrame(dialog, height=250)
        scroll.pack(fill="both", expand=True, padx=10)
        
        vars = []
        for action in self.actions:
            var = ctk.BooleanVar(value=True) # Varsayılan olarak tümünü seç
            chk = ctk.CTkCheckBox(scroll, text=action, variable=var)
            chk.pack(anchor="w", pady=2)
            vars.append((action, var))
            
        def perform_export():
            selected_actions = [a for a, v in vars if v.get()]
            if not selected_actions:
                return
                
            file_path = filedialog.asksaveasfilename(defaultextension=".zip", filetypes=[("Zip files", "*.zip")])
            if not file_path:
                return
                
            try:
                with zipfile.ZipFile(file_path, 'w') as zipf:
                    for action in selected_actions:
                        action_path = os.path.join(config.DATA_PATH, action)
                        for root, dirs, files in os.walk(action_path):
                            for file in files:
                                full_path = os.path.join(root, file)
                                rel_path = os.path.relpath(full_path, config.DATA_PATH)
                                zipf.write(full_path, rel_path)
                messagebox.showinfo("Başarılı", "Veri başarıyla dışa aktarıldı.")
                dialog.destroy()
            except Exception as e:
                messagebox.showerror("Hata", f"Dışa aktarma başarısız: {str(e)}")

        ctk.CTkButton(dialog, text="Seçilenleri Dışa Aktar", command=perform_export).pack(pady=10)

    def _export_model(self):
        if not os.path.exists(config.MODEL_PATH):
            messagebox.showwarning("Uyarı", "Eğitilmiş model bulunamadı.")
            return
            
        file_path = filedialog.asksaveasfilename(defaultextension=".keras", filetypes=[("Keras Model", "*.keras")])
        if file_path:
            shutil.copy2(config.MODEL_PATH, file_path)
            self.unsaved_changes = False
            messagebox.showinfo("Başarılı", "Model dışa aktarıldı.")

    def _import_model(self):
        if self.unsaved_changes: # İdeal olarak eğitim bitiminde bunu True yap
             if not messagebox.askyesno("Kaydedilmemiş Değişiklikler", "Eğittiğiniz fakat kaydetmediğiniz bir model var. Devam ederseniz üzerine yazılacak. Devam edilsin mi?"):
                 return
                 
        file_path = filedialog.askopenfilename(filetypes=[("Keras Model", "*.keras")])
        if file_path:
             try:
                shutil.copy2(file_path, config.MODEL_PATH)
             except shutil.SameFileError:
                pass
             except Exception as e:
                messagebox.showerror("Hata", f"Model kopyalanamadı: {e}")
                return

             try:
                 # Predictor sınıfını başlat (Modeli otomatik yükler)
                 self.predictor = SignLanguagePredictor(self.actions)
                 self.unsaved_changes = False
                 messagebox.showinfo("Başarılı", "Model içe aktarıldı ve yüklendi.")
             except Exception as e:
                 messagebox.showerror("Hata", f"Model yüklenemedi: {e}")

    # --- Çekirdek Döngüler ve Mantık (Öncekiyle aynı, birleştirildi) ---
    def _toggle_landmarks(self):
        self.draw_landmarks = self.switch_landmarks.get() == 1

    def _on_tab_change(self, tab_name):
        # Takılı kalan ipuçlarını zorla kapat
        CTkToolTip.hide_all()
        
        if tab_name != "Tahmin" and self.prediction_active:
            self._toggle_prediction()
        # Sekme değişikliğinde eylemler yenilendi mi?
        if tab_name == "Eğitim" or tab_name == "Tahmin":
            self._update_actions_list()

    def _update_actions_list(self):
        if os.path.exists(config.DATA_PATH):
            sorted_actions = sorted([d for d in os.listdir(config.DATA_PATH) if os.path.isdir(os.path.join(config.DATA_PATH, d))])
            self.actions = np.array(sorted_actions)
        else:
            self.actions = np.array([])

    def _update_video_ui(self):
        with self.ui_lock:
            if self.current_frame_pil:
                # Ana iş parçacığında CTkImage oluştur
                ctk_img = ctk.CTkImage(light_image=self.current_frame_pil, dark_image=self.current_frame_pil, size=(640, 480))
                
                try:
                    current_tab = self.tabview.get()
                    if current_tab == "Veri Toplama":
                        self.video_label_col.configure(image=ctk_img)
                        self.video_label_col.image = ctk_img
                    elif current_tab == "Tahmin":
                        self.video_label_pred.configure(image=ctk_img)
                        self.video_label_pred.image = ctk_img
                except Exception:
                    pass
        
        # Bir sonraki güncellemeyi zamanla
        if not self.stop_event.is_set():
            self.after(30, self._update_video_ui)

    def _video_loop(self):
        print("BİLGİ: Video döngüsü başlatıldı")
        while not self.stop_event.is_set():


            ret, frame = self.cap.read()
            if not ret:
                print("BİLGİ: Kare alınamadı")
                time.sleep(0.01)
                continue
            
            # Ayna etkisi için çerçeveyi çevir
            frame = cv2.flip(frame, 1)
            
            image, results = self.mp_helper.detect_mediapipe(frame)
            
            # --- GERİ SAYIM MANTIĞI ---
            if self.countdown_active:
                elapsed = time.time() - self.countdown_start_time
                remaining = 3 - int(elapsed)
                
                if remaining > 0:
                    # Geri sayımı çiz
                    h, w, c = image.shape
                    cv2.putText(image, str(remaining), (w//2 - 50, h//2 + 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 165, 255), 10, cv2.LINE_AA)
                    self.status_label.configure(text=f"Başlıyor... {remaining}")
                else:
                    # Toplamayı başlat!
                    self.countdown_active = False
                    self.is_collecting = True
                    self.collection_frame_count = 0
                    self.next_sequence_time = time.time() # Hemen başlat
                    
                    self.btn_collect.configure(text="BAŞLADI!", fg_color="green")
                    self.status_light.configure(fg_color="green")
                    self.status_label.configure(text=f"Veri Toplanıyor: {self.current_sequence_idx}")
                    
                    h, w, c = image.shape # Geri sayımdan sonraki ilk kare olması durumunda boyutları yeniden al
                    cv2.putText(image, "BASLA!", (w//2 - 150, h//2 + 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5, cv2.LINE_AA)
            
            # İşaret noktalarını çiz (her zaman)
            if self.draw_landmarks:
                self.mp_helper.draw_styled_landmarks(image, results)
            
            # Geri sayım çalışıyorsa gerisini atla
            if self.countdown_active:
                 # Görüntüleme için dönüştür ve devam et
                 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                 img_pil = Image.fromarray(image)
                 
                 with self.ui_lock:
                     self.current_frame_pil = img_pil
                 
                 continue

            # --- TOPLAMA / TEST MANTIĞI ---
            if self.is_collecting or self.is_testing:
                current_time = time.time()
                if current_time < self.next_sequence_time:
                    # Bekleme süresi
                    wait_time = int(self.next_sequence_time - current_time) + 1
                    cv2.putText(image, f"BEKLE: {wait_time}sn", (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 4, cv2.LINE_AA)
                    self.status_light.configure(fg_color="orange")
                else:
                    # Kayıt süresi
                    self.status_light.configure(fg_color="red")
                    self.status_label.configure(text=f"Kaydediliyor {self.current_sequence_idx+1}" + (" (TEST)" if self.is_testing else ""))
                    
                    if not self.is_testing:
                        save_path = os.path.join(config.DATA_PATH, self.current_action, str(self.start_folder + self.current_sequence_idx))
                        os.makedirs(save_path, exist_ok=True)
                        
                        npy_path = os.path.join(save_path, str(self.collection_frame_count))
                        keypoints = self.mp_helper.extract_keypoints(results)
                        np.save(npy_path, keypoints)
                    
                    cv2.putText(image, f"KAYIT {self.collection_frame_count}/{config.SEQUENCE_LENGTH}" + (" [TEST]" if self.is_testing else ""), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                    
                    self.collection_frame_count += 1
                    
                    if self.collection_frame_count == config.SEQUENCE_LENGTH:
                        self.collection_frame_count = 0
                        self.current_sequence_idx += 1
                        
                        try:
                            # Hedef sayısını güvenli bir şekilde al, yapılandırmaya varsayılan
                            target_count = int(self.entry_count.get())
                        except:
                            target_count = config.NO_SEQUENCES
                            
                        self.count_label.configure(text=f"{self.current_sequence_idx} / {target_count}")
                        
                        self.next_sequence_time = time.time() + self.slider_delay.get()
                        
                        if self.current_sequence_idx >= target_count:
                            if self.is_testing:
                                self._stop_test() # Sadece dur, kaydetme mesajı yok
                            else:
                                self.is_collecting = False
                                self._finish_collection()

            # --- TAHMİN MANTIĞI ---
            elif self.prediction_active and self.predictor and len(self.actions) > 0:
                keypoints = self.mp_helper.extract_keypoints(results)
                
                try:
                    label, conf, _ = self.predictor.predict(keypoints)
                    
                    if label:
                        conf_text = f"Tahmin: {label} ({conf:.2f})"
                        self.lbl_prediction.configure(text=conf_text)
                        
                        # Kullanıcı görünürlüğü için algılamayı günlük kutusuna kaydet
                        if not hasattr(self, 'last_pred_log') or self.last_pred_log != label:
                            self.log_box.insert("end", f"Algılandı: {label} ({conf:.2f})\n")
                            self.log_box.see("end")
                            self.last_pred_log = label
                    else:
                        self.lbl_prediction.configure(text="Tahmin: ...")
            
                except Exception as e:
                    print(f"Prediction error: {e}")
                    self.log_box.insert("end", f"Prediction logic error: {e}\n")

            # İşaret noktalarını çiz
            if self.draw_landmarks:
                self.mp_helper.draw_styled_landmarks(image, results)
            
            # Basit Durum Bilgisi
            if self.prediction_active and len(self.actions) == 0:
                cv2.putText(image, "Hareket/Sinif Yok!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(image, "HATA: Model siniflari (Actions) yuklenemedi!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(image, "Lutfen 'Egitim' sekmesine gidip verileri kontrol edin.", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(image)
            
            with self.ui_lock:
                self.current_frame_pil = img_pil
            
            time.sleep(0.01)

    # --- Düğme Mantığı ---
    def _toggle_test(self):
        if self.is_testing:
            self._stop_test()
        else:
            self._start_test()

    def _start_test(self):
        if self.is_collecting: return # Toplama yapılıyorsa test etme
        
        self.is_testing = True
        self.btn_test.configure(text="TESTİ DURDUR", fg_color="red")
        # Simülasyon için sayaçları sıfırla
        self.current_sequence_idx = 0
        self.collection_frame_count = 0
        self.next_sequence_time = time.time()
        
        try:
             target = int(self.entry_count.get())
             self.count_label.configure(text=f"0 / {target}")
        except:
             self.count_label.configure(text=f"0 / {config.NO_SEQUENCES}")

    def _stop_test(self):
        self.is_testing = False
        self.btn_test.configure(text="TEST ET", fg_color="orange")
        self.btn_collect.configure(state="normal")
        self.status_light.configure(fg_color="gray")
        self.status_label.configure(text="Test Tamamlandı.")
        self.count_label.configure(text="0 / 30")


    def _toggle_collection(self):
        if self.is_testing: return # Test ediliyorsa veri toplama
        if self.is_collecting:
            self._stop_collection()
        else:
            self._start_collection()

    def _stop_collection(self):
        self.is_collecting = False
        self.btn_collect.configure(text="TOPLAMAYI BAŞLAT", fg_color="green")
        self.btn_test.configure(state="normal") # Test düğmesini etkinleştir
        self.status_light.configure(fg_color="gray")
        self.status_label.configure(text=f"Durduruldu. ({self.current_sequence_idx})")

    def _start_collection(self):
        action = self.entry_word.get().strip()
        if not action:
            self.status_label.configure(text="Hata: Önce bir kelime girin!")
            return
            
        # Geri sayımı başlat
        self.countdown_value = 3
        self.countdown_start_time = time.time()
        self.countdown_active = True
        
        # Geri sayım aşaması için arayüz güncellemeleri
        self.btn_collect.configure(text="HAZIRLAN...", fg_color="orange")
        self.btn_test.configure(state="disabled")
        self.status_light.configure(fg_color="orange")
        self.status_label.configure(text=f"Başlıyor... {self.countdown_value}")
        
        # Veri toplama için ön hesaplamalar
        try:
            target_seq = int(self.entry_count.get())
        except:
            target_seq = config.NO_SEQUENCES
            
        self.current_sequence_idx = self.collector.get_start_folder(action)
        self.count_label.configure(text=f"{self.current_sequence_idx} / {target_seq}")
        
        # is_collecting henüz True yapılmadı; video döngüsü geri sayımdan sonra yapacak
        
        self.current_action = action
        self.start_folder = self.collector.get_start_folder(action)
        self.current_sequence_idx = 0
        self.collection_frame_count = 0
        self.next_sequence_time = time.time()
        
        try:
            target = int(self.entry_count.get())
            self.count_label.configure(text=f"0 / {target}")
        except:
             self.count_label.configure(text=f"0 / {config.NO_SEQUENCES}")

        self.is_collecting = True
        self.btn_collect.configure(text="TOPLAMAYI DURDUR", fg_color="red")
        self.btn_test.configure(state="disabled") # Test düğmesini devre dışı bırak
    
    def _finish_collection(self):
        self.status_light.configure(fg_color="green")
        self.status_label.configure(text="Toplama Tamamlandı!")
        self.btn_collect.configure(text="TOPLAMAYI BAŞLAT", fg_color="green")
        self._update_actions_list()
        
        # Sadece güncel olmayabilecek yüklü bir modelimiz varsa uyarı göster
        if self.show_train_warning and self.model is not None:
            self.after(500, self._show_retrain_warning)

    def _show_retrain_warning(self):
        dialog = ctk.CTkToplevel(self)
        dialog.title("Dikkat")
        dialog.geometry("400x250")
        dialog.transient(self) 
        
        # Ortalama mantığı
        try:
            x = self.winfo_x() + (self.winfo_width() // 2) - 200
            y = self.winfo_y() + (self.winfo_height() // 2) - 125
            dialog.geometry(f"+{x}+{y}")
        except: pass

        lbl = ctk.CTkLabel(dialog, 
                          text="Yeni veri girişi yapıldı!\n\nModelin bu yeni verileri tanıması için\n'Training' sekmesinden modeli yeniden\neğitmeniz gerekmektedir.", 
                          font=("Arial", 14), 
                          wraplength=350)
        lbl.pack(pady=20, padx=20)
        
        dont_show_var = ctk.IntVar(value=0)
        chk = ctk.CTkCheckBox(dialog, text="Bir daha gösterme", variable=dont_show_var)
        chk.pack(pady=10)
        
        def close_dialog():
            if dont_show_var.get() == 1:
                self.show_train_warning = False
            dialog.destroy()
            
        btn = ctk.CTkButton(dialog, text="Anlaşıldı", command=close_dialog)
        btn.pack(pady=10)

    def _show_current_metrics(self):
        if not self.model or not self.trainer.model:
            # Model varsa ancak yüklenmemişse otomatik yüklemeye izin ver
            if self.trainer.load_trained_model():
                self.model = self.trainer.model
                self.log_box.insert("end", "Model diskten yüklendi.\n")
            else:
                self.log_box.insert("end", "Hata: Yüklü model bulunamadı.\n")
                return

        self._update_actions_list()
        if self.actions.size == 0:
            self.log_box.insert("end", "Hata: Veri bulunamadı. Metrikler için veri seti gereklidir.\n")
            return

        def eval_task():
            self.btn_metrics.configure(state="disabled")
            try:
                self.log_box.insert("end", "Metrikler hesaplanıyor...\n")
                # Using full dataset for on-demand check or we could do split. 
                # Let's do split to be consistent with 'Test Data' concept.
                X, y = self.trainer.load_data(self.actions)
                if len(X) == 0:
                     self.log_box.insert("end", "Hata: Veri yüklenemedi.\n")
                     self.btn_metrics.configure(state="normal")
                     return

                # Create a temporary split to simulate test condition
                _, X_test, _, y_test = train_test_split(X, y, test_size=0.05)
                
                metrics = self.trainer.evaluate_model(X_test, y_test)
                # Pass None for history as we don't have training history for saved model
                self.after(0, lambda: self._show_detailed_metrics(metrics, None))
                self.log_box.insert("end", "Metrikler hesaplandı.\n")
            except Exception as e:
                self.log_box.insert("end", f"Hata: {str(e)}\n")
            finally:
                self.btn_metrics.configure(state="normal")
        
        threading.Thread(target=eval_task, daemon=True).start()

    def _show_detailed_metrics(self, metrics, history):
        win = ctk.CTkToplevel(self)
        win.title("Eğitim ve Performans Sonuçları")
        win.geometry("1000x800")
        
        # Text Metrics Frame
        m_frame = ctk.CTkFrame(win)
        m_frame.pack(fill="x", padx=10, pady=10)
        
        m_text = f"Doğruluk: {metrics['accuracy']:.4f}  |  F1-Score: {metrics['f1_score']:.4f}  |  Gecikme: {metrics['latency_ms']:.2f} ms  |  FPS: {metrics['fps']:.2f}"
        ctk.CTkLabel(m_frame, text=m_text, font=("Arial", 16, "bold")).pack(pady=10)
        
        # Grafikler için Sekmeler
        tabs = ctk.CTkTabview(win)
        tabs.pack(fill="both", expand=True, padx=10, pady=10)
        tabs.add("Grafikler")
        tabs.add("Confusion Matrix")
        
        # 1. Loss/Acc Graph
        hist_img_pil = self.trainer.plot_history(history)
        if hist_img_pil:
            hist_ctk = ctk.CTkImage(light_image=hist_img_pil, dark_image=hist_img_pil, size=(900, 300))
            ctk.CTkLabel(tabs.tab("Grafikler"), text="", image=hist_ctk).pack(fill="both", expand=True)
        else:
            ctk.CTkLabel(tabs.tab("Grafikler"), text="Eğitim geçmişi bulunamadı.\n(Model sonradan yüklendiği için epoch bazlı grafikler gösterilemiyor.)", font=("Arial", 16)).pack(pady=50)
            
        # 2. Confusion Matrix
        cm_img_pil = self.trainer.plot_confusion_matrix(metrics['confusion_matrix'], self.actions)
        if cm_img_pil:
            cm_ctk = ctk.CTkImage(light_image=cm_img_pil, dark_image=cm_img_pil, size=(700, 500))
            ctk.CTkLabel(tabs.tab("Confusion Matrix"), text="", image=cm_ctk).pack(fill="both", expand=True)

    def _change_camera(self, choice):
        idx = int(choice.split(" ")[1])
        self.log_box.insert("end", f"Kamera değiştiriliyor: {idx}...\n")
        
        def switch_task():
            # Release old
            if self.cap is not None:
                self.cap.release()
            
            # Init new
            new_cap = cv2.VideoCapture(idx)
            if new_cap.isOpened():
                self.cap = new_cap
                self.after(0, lambda: self.log_box.insert("end", f"Kamera {idx} aktif.\n"))
            else:
                self.after(0, lambda: self.log_box.insert("end", f"Hata: Kamera {idx} açılamadı. 0'a dönülüyor.\n"))
                new_cap = cv2.VideoCapture(0)
                self.cap = new_cap
                self.after(0, lambda: self.camera_var.set("Kamera 0"))
                
        threading.Thread(target=switch_task, daemon=True).start()

    def _stop_training(self):
        if self.trainer:
            self.trainer.stop_training = True
            self.log_box.insert("end", "Durdurma isteği gönderildi... Model epoch sonunda duracak.\n")

    def _start_training(self):
        # Eğitimin devam edip etmediğini kontrol et (düğme metni kontrolü yeterli)
        if self.btn_train.cget("text") == "Eğitimi Durdur":
            self._stop_training()
            return

        # Parametreleri al
        try:
            epochs = int(self.entry_epochs.get())
            if not self.auto_var.get():
                units = 64 # Sabit mimari
                dropout = float(self.entry_dropout.get())
                lr = float(self.entry_lr.get())
                model_type = self.model_var.get()
                auto_tune = False
            else:
                units, dropout, lr, model_type = 0, 0, 0, ""
                auto_tune = True
                
            use_early_stopping = self.early_stop_var.get()
            
        except ValueError:
            self.log_box.insert("end", "Error: Invalid parameters.\n")
            return
        
        self.btn_train.configure(text="Eğitimi Durdur", fg_color="red")
        self.log_box.delete("0.0", "end")
        self.log_box.insert("0.0", "Eğitim başlatılıyor...\n")
        
        def train_task():
            self._update_actions_list()
            if len(self.actions) == 0:
                self.log_box.insert("end", "Hata: Veri bulunamadı.\n")
                self.btn_train.configure(state="normal", text="Eğitimi Başlat", fg_color="#1f538d")
                return

            def log_callback(msg):
                self.log_box.insert("end", msg)
                self.log_box.see("end")
            
            # Çıktıları yönlendir
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = IORedirector(self.log_box)
            sys.stderr = IORedirector(self.log_box)

            try:
                history, X_test, y_test = self.trainer.train(
                    self.actions, 
                    epochs=epochs, 
                    model_type=model_type,
                    units=units,
                    dropout=dropout,
                    lr=lr,
                    auto_tune=auto_tune,
                    use_early_stopping=use_early_stopping,
                    callback_fn=log_callback
                )
                if history:
                    self.log_box.insert("end", "Eğitim Tamamlandı!\n")
                    self.unsaved_changes = True # Kaydedilmemiş olarak işaretle
                    
                    # Değerlendirme
                    if X_test is not None:
                         metrics = self.trainer.evaluate_model(X_test, y_test)
                         self.after(0, lambda: self._show_detailed_metrics(metrics, history))
                else:
                    self.log_box.insert("end", "Eğitim durduruldu veya başarısız oldu.\n")
                     
            except Exception as e:
                self.log_box.insert("end", f"Hata: {str(e)}\n")
                import traceback
                traceback.print_exc()
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
            
            self.btn_train.configure(state="normal", text="Eğitimi Başlat", fg_color="#1f538d")
            self._load_model()
        
        threading.Thread(target=train_task, daemon=True).start()

    def _load_model(self):
        if self.trainer.load_trained_model():
            self.model = self.trainer.model
            self.log_box.insert("end", "Model tahmin için yüklendi.\n")

    def _toggle_prediction(self):
        if not self.prediction_active:
            if self.model is None:
                self._load_model()
                if self.model is None:
                    self.lbl_prediction.configure(text="Hata: Model Bulunamadı")
                    return
            
            self.prediction_active = True
            self.btn_toggle_pred.configure(text="TAHMİNİ DURDUR", fg_color="red")
        else:
            self.prediction_active = False
            self.btn_toggle_pred.configure(text="TAHMİNİ BAŞLAT", fg_color="green")
            self.lbl_prediction.configure(text="Tahmin: Durduruldu")

    def on_closing(self):
        self.stop_event.set()
        self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = SignLanguageApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
