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
        # Prevent showing if widget is no longer visible/mapped
        try:
            if not self.widget.winfo_exists() or not self.widget.winfo_viewable():
                return
        except:
            return

        try:
            # Generic positioning for any widget (Button, Frame, etc.)
            x = self.widget.winfo_rootx() + 20
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 10
            
            self.tooltip_window = ctk.CTkToplevel(self.widget)
            self.tooltip_window.wm_overrideredirect(True)
            self.tooltip_window.wm_geometry(f"+{x}+{y}")
            # Ensure it stays on top
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
        # With the delay logic, we rely on 'leave' events mostly, but this can still be useful 
        # if we tracked instances constantly. For simplicity and stability, let's rely on event bindings.
        # But to satisfy the existing call in app.py, we'll keep it as a pass or implement a basic forced cleanup if needed.
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
        self.geometry("1300x850") # Expanded for better layout

        # Logic Components
        self.cap = cv2.VideoCapture(0)
        self.mp_helper = MediapipeHelper()
        self.collector = DataCollector()
        self.trainer = ModelTrainer()
        self.model = None 
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
        
        # Easter Egg
        self.media_path = "media"
        if not os.path.exists(self.media_path):
            os.makedirs(self.media_path)
        self.is_paused = False
        
        # Setup UI
        self._setup_ui()
        
        # Load Actions
        self._update_actions_list()
        
        # Start Video Loop
        self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self.video_thread.start()
        
        # Bind click to reset focus, but ONLY if not clicking an entry/button
        self.bind("<Button-1>", self._on_global_click)

    def _on_global_click(self, event):
        # Check if the clicked widget is an input widget (Entry, Text, etc.)
        try:
            widget_class = event.widget.winfo_class()
            # Standard Tkinter classes for input
            if widget_class in ['Entry', 'Text', 'TEntry', 'TCombobox']:
                return
        except Exception:
            pass
            
        # Otherwise focus root to clear entry focus
        self.focus()

    def _setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar ---
        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(8, weight=1) # Spacer

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="SignAI Studio", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(40, 20))

        # Data Management Section
        ctk.CTkLabel(self.sidebar_frame, text="Veri Yönetimi", font=ctk.CTkFont(size=14, weight="bold")).grid(row=1, column=0, padx=20, sticky="w")
        
        self.btn_import_data = ctk.CTkButton(self.sidebar_frame, text="Veri Klasörü Yükle", command=self._import_data_folder)
        self.btn_import_data.grid(row=2, column=0, padx=20, pady=10)
        
        self.btn_export_data = ctk.CTkButton(self.sidebar_frame, text="Veri Dışa Aktar (Zip)", command=self._export_data_dialog)
        self.btn_export_data.grid(row=3, column=0, padx=20, pady=10)

        # Model Management Section
        # Model Management Section
        ctk.CTkLabel(self.sidebar_frame, text="Model Yönetimi", font=ctk.CTkFont(size=14, weight="bold")).grid(row=4, column=0, padx=20, pady=(20, 0), sticky="w")

        self.btn_import_model = ctk.CTkButton(self.sidebar_frame, text="Model Yükle", command=self._import_model)
        self.btn_import_model.grid(row=5, column=0, padx=20, pady=10)
        
        self.btn_export_model = ctk.CTkButton(self.sidebar_frame, text="Modeli Kaydet", command=self._export_model)
        self.btn_export_model.grid(row=6, column=0, padx=20, pady=10)

        # --- Main Area ---
        self.tabview = ctk.CTkTabview(self, width=250)
        self.tabview.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.tabview.add("Veri Toplama")
        self.tabview.add("Eğitim")
        self.tabview.add("Tahmin")
        
        # Correctly setting the command for tab change
        self.tabview.configure(command=self._on_tab_change_command)

        self._setup_collection_tab()
        self._setup_training_tab()
        self._setup_prediction_tab()
        
    def _on_tab_change_command(self):
        # Wrapper to handle standard callback
        self._on_tab_change(self.tabview.get())

    def _setup_collection_tab(self):
        tab = self.tabview.tab("Veri Toplama")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        # Controls Frame
        controls = ctk.CTkFrame(tab)
        controls.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        
        # Word Input
        self.entry_word = ctk.CTkEntry(controls, placeholder_text="Yeni Kelime Giriniz", width=200, height=35)
        self.entry_word.pack(side="left", padx=10, pady=10)
        CTkToolTip(self.entry_word, "Öğretmek istediğiniz kelimeyi buraya yazın.")

        vcmd = (self.register(self._validate_int_input), '%P')
        
        # Count Input
        self.entry_count = ctk.CTkEntry(controls, placeholder_text="Adet", width=60, height=35,
                                        validate="key", validatecommand=vcmd)
        self.entry_count.insert(0, str(config.NO_SEQUENCES))
        self.entry_count.pack(side="left", padx=5, pady=10)
        self.entry_count.bind("<FocusOut>", lambda e: self._clamp_value(self.entry_count, 1, 30, int))
        CTkToolTip(self.entry_count, "Kaç adet video toplanacağı.")
        
        # Slider
        slider_frame = ctk.CTkFrame(controls, fg_color="transparent")
        slider_frame.pack(side="left", padx=10)
        
        self.slider_delay = ctk.CTkSlider(slider_frame, from_=0, to=5, number_of_steps=5, command=self._update_delay_label)
        self.slider_delay.set(2)
        self.slider_delay.pack(pady=5)
        
        self.label_delay = ctk.CTkLabel(slider_frame, text="Delay: 2s")
        self.label_delay.pack(pady=0)
        
        # Landmarks Toggle
        # Landmarks Toggle
        self.switch_landmarks = ctk.CTkSwitch(controls, text="İskeleti Göster", command=self._toggle_landmarks)
        self.switch_landmarks.select()
        self.switch_landmarks.pack(side="right", padx=10)

        # Status Bar
        status_frame = ctk.CTkFrame(tab, height=50)
        status_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        
        self.status_light = ctk.CTkLabel(status_frame, text="  ", fg_color="gray", width=30, height=30, corner_radius=15)
        self.status_light.pack(side="left", padx=20, pady=10)
        
        self.status_label = ctk.CTkLabel(status_frame, text="Durum: Hazır", font=("Arial", 16))
        self.status_label.pack(side="left", pady=10)
        
        self.count_label = ctk.CTkLabel(status_frame, text="0 / 30", font=("Arial", 16, "bold"))
        self.count_label.pack(side="right", padx=20)

        # Video Area
        self.video_label_col = ctk.CTkLabel(tab, text="", fg_color="black", corner_radius=10)
        self.video_label_col.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        
        self.video_label_col = ctk.CTkLabel(tab, text="", fg_color="black", corner_radius=10)
        self.video_label_col.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        
        # Action Buttons Frame
        btn_frame = ctk.CTkFrame(tab, fg_color="transparent")
        btn_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=10)
        
        # Test Button
        self.btn_test = ctk.CTkButton(btn_frame, text="TEST ET", fg_color="orange", height=50, width=150, font=("Arial", 16, "bold"), command=self._toggle_test)
        self.btn_test.pack(side="left", padx=(0, 10), fill="x", expand=True)

        # Start Collection Button
        self.btn_collect = ctk.CTkButton(btn_frame, text="TOPLAMAYI BAŞLAT", fg_color="green", height=50, font=("Arial", 16, "bold"), command=self._toggle_collection)
        self.btn_collect.pack(side="left", fill="x", expand=True)

    def _setup_training_tab(self):
        tab = self.tabview.tab("Eğitim")
        tab.grid_columnconfigure(0, weight=1)
        
        # --- Settings ---
        settings_frame = ctk.CTkFrame(tab)
        settings_frame.pack(pady=10, padx=20, fill="x")
        
        # Model Selection
        ctk.CTkLabel(settings_frame, text="Model Mimarisi:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.model_var = ctk.StringVar(value="LSTM")
        models = ["LSTM", "GRU", "CNN", "Bi-LSTM"]
        for i, m in enumerate(models):
            rb = ctk.CTkRadioButton(settings_frame, text=m, variable=self.model_var, value=m)
            rb.grid(row=0, column=i+1, padx=10, pady=10)
            
        # Hyperparameters
        self.hp_frame = ctk.CTkFrame(tab)
        self.hp_frame.pack(pady=10, padx=20, fill="x")
        
        # Labels and Entries with Tooltips
        params = [
            ("Epochs:", config.EPOCHS, "Eğitim döngüsü sayısı. Ne kadar süreceğini belirler."),
            ("Dropout:", "0.2", "Unutma oranı (0.1-0.5). Ezberlemeyi önler."),
            ("Units:", "64", "Nöron sayısı. Karmaşıklığı artırır."),
            ("LR:", "0.001", "Öğrenme hızı. Küçük değerler daha hassas öğrenir.")
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
        self.entry_units = self.hp_entries["Units:"]
        self.entry_lr = self.hp_entries["LR:"]
        
        # Apply Validation to Training Inputs
        vcmd = (self.register(self._validate_int_input), '%P')
        
        self.entry_epochs.configure(validate="key", validatecommand=vcmd)
        self.entry_epochs.bind("<FocusOut>", lambda e: self._clamp_value(self.entry_epochs, 1, 1000, int))
        
        self.entry_units.configure(validate="key", validatecommand=vcmd)
        self.entry_units.bind("<FocusOut>", lambda e: self._clamp_value(self.entry_units, 16, 512, int))
        
        self.entry_dropout.bind("<FocusOut>", lambda e: self._clamp_value(self.entry_dropout, 0.1, 0.9, float))
        self.entry_lr.bind("<FocusOut>", lambda e: self._clamp_value(self.entry_lr, 0.0001, 0.1, float))
        
        # Auto Toggle
        self.auto_var = ctk.IntVar(value=0)
        self.chk_auto = ctk.CTkCheckBox(tab, text="Otomatik Optimizasyon (Keras Tuner)", variable=self.auto_var, command=self._toggle_auto_settings)
        self.chk_auto.pack(pady=10)
        CTkToolTip(self.chk_auto, "En iyi parametreleri otomatik olarak bulur.")

        # Train Button & Metrics Button
        action_frame = ctk.CTkFrame(tab, fg_color="transparent")
        action_frame.pack(pady=10, padx=20, fill="x")
        
        self.btn_train = ctk.CTkButton(action_frame, text="Eğitimi Başlat", fg_color="#1f538d", height=50, font=("Arial", 16, "bold"), command=self._start_training)
        self.btn_train.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        self.btn_metrics = ctk.CTkButton(action_frame, text="Metrikleri Göster", fg_color="purple", height=50, font=("Arial", 16, "bold"), command=self._show_current_metrics)
        self.btn_metrics.pack(side="right", fill="x", expand=True)
        
        # Logs
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

    # --- UI Logic Methods ---
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
            # If invalid, revert to min_val or default logic
            val = min_val
            
        # Update widget
        widget.delete(0, "end")
        widget.insert(0, str(val))

    def _toggle_auto_settings(self):
        state = "disabled" if self.auto_var.get() == 1 else "normal"
        self.entry_epochs.configure(state=state)
        self.entry_dropout.configure(state=state)
        self.entry_units.configure(state=state)
        self.entry_lr.configure(state=state)
        # Disable Architecture selection if auto-tuning typically searches architecture
        # But here we implement tuning FOR the selected architecture or global. 
        # Implementation Plan said "Takes selected model... and optimizes hyperparams".
        # So we keep model selection enabled.

    # --- Data & Model Management ---
    
    def _import_data_folder(self):
        folder_selected = filedialog.askdirectory(title="İçe Aktarılacak Veri Klasörünü Seç")
        if not folder_selected:
            return
            
        def import_task():
            count = 0
            try:
                self.log_box.insert("end", "Veri içe aktarılıyor... Lütfen bekleyin.\n")
                self.btn_train.configure(state="disabled") # Disable training during import
                
                for item in os.listdir(folder_selected):
                    source_item = os.path.join(folder_selected, item)
                    if os.path.isdir(source_item):
                        action_name = item
                        target_action_dir = os.path.join(config.DATA_PATH, action_name)
                        if not os.path.exists(target_action_dir):
                            os.makedirs(target_action_dir)
                        
                        # Find last index in target
                        # Use try-except for permission errors on listdir
                        try:
                            numeric_files = [int(f) for f in os.listdir(target_action_dir) if f.isdigit()]
                        except:
                            numeric_files = []
                            
                        start_idx = max(numeric_files) + 1 if numeric_files else 0
                        
                        # Copy sequences
                        for sub_item in os.listdir(source_item):
                            source_seq_dir = os.path.join(source_item, sub_item)
                            if os.path.isdir(source_seq_dir):
                                target_seq_dir = os.path.join(target_action_dir, str(start_idx))
                                try:
                                    shutil.copytree(source_seq_dir, target_seq_dir)
                                    start_idx += 1
                                    count += 1
                                except FileExistsError:
                                    # Skip if collision, or handle smart
                                    pass
                                except Exception as e:
                                    print(f"Copy error: {e}")

                self._update_actions_list()
                self.log_box.insert("end", f"İçe aktarma tamamlandı: {count} yeni sekans.\n")
                # Schedule GUI callbacks
                self.after(0, lambda: messagebox.showinfo("Başarılı", f"{count} adet sekans başarıyla içe aktarıldı."))
                
            except Exception as e:
                self.log_box.insert("end", f"İçe aktarma hatası: {str(e)}\n")
                self.after(0, lambda: messagebox.showerror("Hata", f"İçe aktarma başarısız: {str(e)}"))
            finally:
                self.after(0, lambda: self.btn_train.configure(state="normal"))

        threading.Thread(target=import_task, daemon=True).start()

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
            var = ctk.BooleanVar(value=True) # Default select all
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
        if self.unsaved_changes: # Ideally set this True on training finish
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

             if self.trainer.load_trained_model():
                 self.model = self.trainer.model
                 self.unsaved_changes = False
                 messagebox.showinfo("Başarılı", "Model içe aktarıldı ve yüklendi.")
             else:
                 messagebox.showerror("Hata", "Model yüklenemedi.")

    # --- Core Loops & Logic (Same as before, merged) ---
    def _toggle_landmarks(self):
        self.draw_landmarks = self.switch_landmarks.get() == 1

    def _on_tab_change(self, tab_name):
        # Force close any stuck tooltips
        CTkToolTip.hide_all()
        
        if tab_name != "Tahmin" and self.prediction_active:
            self._toggle_prediction()
        # Refreshed actions on tab change potentially?
        if tab_name == "Eğitim" or tab_name == "Tahmin":
            self._update_actions_list()

    def _update_actions_list(self):
        if os.path.exists(config.DATA_PATH):
            sorted_actions = sorted([d for d in os.listdir(config.DATA_PATH) if os.path.isdir(os.path.join(config.DATA_PATH, d))])
            self.actions = np.array(sorted_actions)
        else:
            self.actions = np.array([])

    def check_easter_egg(self, prediction):
        clean_pred = str(prediction).strip().lower() # Normalize to lowercase
        found_file = None
        
        # Case-insensitive search
        if os.path.exists(self.media_path):
            for f in os.listdir(self.media_path):
                if f.lower().startswith(clean_pred + "."):
                    # Check extension
                    if f.lower().endswith(('.mp4', '.png', '.jpg', '.jpeg')):
                        found_file = os.path.join(self.media_path, f)
                        break
        
        if found_file:
            self.log_box.insert("end", f"Easter Egg bulundu: {found_file}\n")
            self.is_paused = True
            
            # Create popup
            window = ctk.CTkToplevel(self)
            window.title(f"Sürpriz: {prediction}")
            window.geometry("600x400")
            
            # Center logic
            try:
                x = self.winfo_x() + (self.winfo_width() // 2) - 300
                y = self.winfo_y() + (self.winfo_height() // 2) - 200
                window.geometry(f"+{x}+{y}")
            except: pass
            
            video_label = ctk.CTkLabel(window, text="")
            video_label.pack(fill="both", expand=True)
            
            is_video = found_file.lower().endswith('.mp4')
            
            if is_video:
                cap = cv2.VideoCapture(found_file)
                
                def play_video():
                    if not window.winfo_exists():
                        cap.release()
                        self.is_paused = False
                        return
                        
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.resize(frame, (600, 400))
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(frame)
                        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(600, 400))
                        
                        video_label.configure(image=ctk_img)
                        video_label.image = ctk_img
                        window.after(33, play_video)
                    else:
                        cap.release()
                        window.destroy()
                        self.is_paused = False
                
                def on_close():
                    cap.release()
                    window.destroy()
                    self.is_paused = False
                    
                window.protocol("WM_DELETE_WINDOW", on_close)
                play_video()
                
            else: # Image
                pil_img = Image.open(found_file)
                # Resize keeping aspect ratio maybe? For now fixed size
                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(600, 400))
                video_label.configure(image=ctk_img)
                
                def close_after_delay():
                    if window.winfo_exists():
                        window.destroy()
                    self.is_paused = False
                    
                window.protocol("WM_DELETE_WINDOW", close_after_delay)
                # Show for 3 seconds
                window.after(3000, close_after_delay)

    def _video_loop(self):
        # ... (Previous Video Loop Code with Slider Delay Logic) ...
        # Copying exact logic from previous iteration to ensure no regression
        sequence = []
        sentence = []
        predictions = []
        threshold = 0.5
        res = None
        pred_label = "..."
        
        while not self.stop_event.is_set():
            if self.is_paused:
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            image, results = self.mp_helper.detect_mediapipe(frame)
            
            # --- COLLECTION / TEST LOGIC ---
            if self.is_collecting or self.is_testing:
                current_time = time.time()
                if current_time < self.next_sequence_time:
                    # Waiting period
                    wait_time = int(self.next_sequence_time - current_time) + 1
                    cv2.putText(image, f"BEKLE: {wait_time}sn", (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 4, cv2.LINE_AA)
                    self.status_light.configure(fg_color="orange")
                else:
                    # Recording period
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
                            # Safely get target count, defaulting to config
                            target_count = int(self.entry_count.get())
                        except:
                            target_count = config.NO_SEQUENCES
                            
                        self.count_label.configure(text=f"{self.current_sequence_idx} / {target_count}")
                        
                        self.next_sequence_time = time.time() + self.slider_delay.get()
                        
                        if self.current_sequence_idx >= target_count:
                            if self.is_testing:
                                self._stop_test() # Just stop, no save msg
                            else:
                                self.is_collecting = False
                                self._finish_collection()

            # --- PREDICTION LOGIC ---
            elif self.prediction_active and self.model and len(self.actions) > 0:
                keypoints = self.mp_helper.extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]
                
                if len(sequence) == 30:
                    try:
                        res = self.model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                        pred_idx = np.argmax(res)
                        
                        if pred_idx < len(self.actions):
                            pred_label = self.actions[pred_idx]
                        else:
                            pred_label = f"IndexError({pred_idx})"
                        
                        # Debug logic - print to log box occasionally?
                        # print(f"Pred: {pred_label} Conf: {res[pred_idx]:.2f}") 
                        
                        if res[pred_idx] > threshold:
                            conf_text = f"Tahmin: {pred_label} ({res[pred_idx]:.2f})"
                            self.lbl_prediction.configure(text=conf_text)
                            
                            # Log detection to log box for user visibility ("Type out what it detects")
                            # We throttle this to avoid spamming the log box (e.g. only if changed)
                            if not hasattr(self, 'last_pred_log') or self.last_pred_log != pred_label:
                                self.log_box.insert("end", f"Algılandı: {pred_label} ({res[pred_idx]:.2f})\n")
                                self.log_box.see("end")
                                self.last_pred_log = pred_label
                            
                            # Check for easter egg
                            self.after(0, lambda p=pred_label: self.check_easter_egg(p))
                        else:
                            self.lbl_prediction.configure(text="Tahmin: ...")
                        
                    except Exception as e:
                        print(f"Prediction error: {e}")
                        self.log_box.insert("end", f"Prediction logic error: {e}\n")

            if self.draw_landmarks:
                self.mp_helper.draw_styled_landmarks(image, results)
            
            # DEBUG OVERLAY
            if self.prediction_active:
                debug_color = (0, 255, 0) if len(self.actions) > 0 and len(sequence) == 30 else (0, 0, 255)
                debug_text = f"Durum: {len(sequence)}/30 | Siniflar: {len(self.actions)}"
                cv2.putText(image, debug_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, debug_color, 2)
                
                if res is not None and len(self.actions) > 0:
                     pred_max = np.max(res)
                     debug_conf_text = f"Top: {pred_label} ({pred_max:.2f})"
                     cv2.putText(image, debug_conf_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                if len(self.actions) == 0:
                     cv2.putText(image, "HATA: Model siniflari (Actions) yuklenemedi!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                     cv2.putText(image, "Lutfen 'Egitim' sekmesine gidip verileri kontrol edin.", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(image)
            
            # Use after() to schedule GUI updates safely from thread? 
            # Tkinter isn't thread safe but usually modifying image on label is accepted if careful.
            # Ideally use a queue, but direct config works for simple cases mostly.
            # Using try-except if widget destroyed
            try:
                current_tab = self.tabview.get()
                if current_tab == "Veri Toplama":
                    ctk_img = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(640, 480))
                    self.video_label_col.configure(image=ctk_img)
                    self.video_label_col.image = ctk_img
                elif current_tab == "Tahmin":
                    ctk_img = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(640, 480))
                    self.video_label_pred.configure(image=ctk_img)
                    self.video_label_pred.image = ctk_img
            except Exception:
                pass # App closed
            
            time.sleep(0.01)

    # --- Button Logic ---
    def _toggle_test(self):
        if self.is_testing:
            self._stop_test()
        else:
            self._start_test()

    def _start_test(self):
        if self.is_collecting: return # Don't test if collecting
        
        self.is_testing = True
        self.btn_test.configure(text="TESTİ DURDUR", fg_color="red")
        self.btn_collect.configure(state="disabled") # Disable collect button
        
        # Init counters for simulation
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
        if self.is_testing: return # Don't collect if testing
        if self.is_collecting:
            self._stop_collection()
        else:
            self._start_collection()

    def _stop_collection(self):
        self.is_collecting = False
        self.btn_collect.configure(text="TOPLAMAYI BAŞLAT", fg_color="green")
        self.btn_test.configure(state="normal") # Enable test button
        self.status_light.configure(fg_color="gray")
        self.status_label.configure(text=f"Durduruldu. ({self.current_sequence_idx})")

    def _start_collection(self):
        action = self.entry_word.get().strip()
        if not action:
            self.status_label.configure(text="Hata: Önce bir kelime girin!")
            return
        
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
        self.btn_test.configure(state="disabled") # Disable test button
    
    def _finish_collection(self):
        self.status_light.configure(fg_color="green")
        self.status_label.configure(text="Toplama Tamamlandı!")
        self.btn_collect.configure(text="TOPLAMAYI BAŞLAT", fg_color="green")
        self._update_actions_list()
        
        # Only show warning if we have a loaded model that might be outdated now
        if self.show_train_warning and self.model is not None:
            self.after(500, self._show_retrain_warning)

    def _show_retrain_warning(self):
        dialog = ctk.CTkToplevel(self)
        dialog.title("Dikkat")
        dialog.geometry("400x250")
        dialog.transient(self) 
        
        # Center logic
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
        # Allow auto-loading if model exists but not loaded
        if not self.model or not self.trainer.model:
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
        
        # Tabs for Graphs
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

    def _start_training(self):
        # ... Params fetching ...
        try:
            epochs = int(self.entry_epochs.get())
            if not self.auto_var.get():
                units = int(self.entry_units.get())
                dropout = float(self.entry_dropout.get())
                lr = float(self.entry_lr.get())
                model_type = self.model_var.get()
                auto_tune = False
            else:
                units, dropout, lr, model_type = 0, 0, 0, ""
                auto_tune = True
        except ValueError:
            self.log_box.insert("end", "Error: Invalid parameters.\n")
            return
        
        self.btn_train.configure(state="disabled", text="Eğitiliyor...")
        self.log_box.delete("0.0", "end")
        self.log_box.insert("0.0", "Eğitim başlatılıyor...\n")
        
        def train_task():
            self._update_actions_list()
            if len(self.actions) == 0:
                self.log_box.insert("end", "Hata: Veri bulunamadı.\n")
                self.btn_train.configure(state="normal", text="Eğitimi Başlat")
                return

            def log_callback(msg):
                self.log_box.insert("end", msg)
                self.log_box.see("end")
            
            # Redirect stdout/stderr
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
                    callback_fn=log_callback
                )
                self.log_box.insert("end", "Eğitim Tamamlandı!\n")
                self.unsaved_changes = True # Mark as unsaved
                
                # Evaluation
                if X_test is not None:
                     metrics = self.trainer.evaluate_model(X_test, y_test)
                     self.after(0, lambda: self._show_detailed_metrics(metrics, history))
                     
            except Exception as e:
                self.log_box.insert("end", f"Hata: {str(e)}\n")
                import traceback
                traceback.print_exc()
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
            
            self.btn_train.configure(state="normal", text="Eğitimi Başlat")
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
