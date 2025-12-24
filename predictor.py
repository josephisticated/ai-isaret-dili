import cv2
import numpy as np
import config
from utils import MediapipeHelper
from model_trainer import ModelTrainer

class SignLanguagePredictor:
    def __init__(self, actions, threshold=0.5):
        self.actions = np.array(actions)
        self.mp_helper = MediapipeHelper()
        self.trainer = ModelTrainer()
        self.threshold = threshold
        
        # Modeli yükle
        if not self.trainer.load_trained_model():
            raise Exception("Model yüklenemedi. Lütfen önce modeli eğitin.")
        
        self.model = self.trainer.model
        self.colors = [(245,117,16), (117,245,16), (16,117,245)] * 10
        
        # Tahmin durumu
        self.sequence = []
        self.sentence = []
        self.predictions = []

    def predict(self, keypoints):
        """
        Tek bir kare (keypoints) alır ve tahmin sonucunu döndürür.
        
        Döndürür:
            predicted_label (str veya None): Eğer bir tahmin yapıldıysa (eşik değeri aşıldıysa) etiket, aksi halde None.
            confidence (float): Tahmin güven oranı.
            current_sentence (list): Oluşturulan cümlenin kelime listesi.
        """
        self.sequence.append(keypoints)
        self.sequence = self.sequence[-30:] # Son 30 kareyi tut
        
        predicted_label = None
        confidence = 0.0
        
        if len(self.sequence) == 30:
            res = self.model.predict(np.expand_dims(self.sequence, axis=0), verbose=0)[0]
            best_idx = np.argmax(res)
            confidence = res[best_idx]
            
            self.predictions.append(best_idx)
            
            # Son 10 tahmin kararlı mı?
            if np.unique(self.predictions[-10:])[0] == best_idx:
                if confidence > self.threshold:
                    predicted_label = self.actions[best_idx]
                    
                    if len(self.sentence) > 0:
                        if predicted_label != self.sentence[-1]:
                            self.sentence.append(predicted_label)
                    else:
                        self.sentence.append(predicted_label)

            if len(self.sentence) > 5:
                self.sentence = self.sentence[-5:]
                
        return predicted_label, confidence, self.sentence

    def prob_viz(self, res, input_frame):
        # Bu fonksiyon şimdilik sadece görselleştirme için, model çıktısını (res) dışarıdan alması gerekebilir
        # ancak yeni yapıda 'res' predict içinde yerel.
        # Görselleştirme isteniyorsa predict metodunun 'res' (olasılık dağılımı) döndürmesi daha iyi olabilir.
        # Basitlik adına şimdilik run() içinde kullanacağız.
        pass

    def run(self):
        """
        Bağımsız çalıştırıldığında kamera açıp test etmeyi sağlar.
        """
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, results = self.mp_helper.detect_mediapipe(frame)
            self.mp_helper.draw_styled_landmarks(image, results)
            
            keypoints = self.mp_helper.extract_keypoints(results)
            
            # --- YENİ YAPI KULLANIMI ---
            label, conf, sentence = self.predict(keypoints)
            
            if label:
                 cv2.putText(image, f"{label} ({conf:.2f})", (3, 440), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow('OpenCV Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        self.mp_helper.close()
