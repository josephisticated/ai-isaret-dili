import cv2
import numpy as np
import config
from utils import MediapipeHelper
from model_trainer import ModelTrainer

class RealTimeTranslator:
    def __init__(self, actions):
        self.actions = np.array(actions)
        self.mp_helper = MediapipeHelper()
        self.trainer = ModelTrainer()
        
        # Modeli yükle
        if not self.trainer.load_trained_model():
            raise Exception("Model yüklenemedi. Lütfen önce modeli eğitin.")
        
        self.model = self.trainer.model
        self.colors = [(245,117,16), (117,245,16), (16,117,245)] * 10 # Repeat colors if many actions

    def prob_viz(self, res, input_frame):
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            # Ensure we don't go out of bounds if actions > colors
            color = self.colors[num % len(self.colors)]
            
            cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), color, -1)
            cv2.putText(output_frame, self.actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        return output_frame

    def run(self):
        sequence = []
        sentence = []
        predictions = []
        threshold = 0.5

        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, results = self.mp_helper.detect_mediapipe(frame)
            self.mp_helper.draw_styled_landmarks(image, results)
            
            # Tahmin mantığı
            keypoints = self.mp_helper.extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                res = self.model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))
                
                if np.unique(predictions[-10:])[0] == np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 
                        if len(sentence) > 0: 
                            if self.actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(self.actions[np.argmax(res)])
                        else:
                            sentence.append(self.actions[np.argmax(res)])

                if len(sentence) > 5: 
                    sentence = sentence[-5:]

                # Olasılıkları görselleştir
                image = self.prob_viz(res, image)
            
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('OpenCV Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        self.mp_helper.close()
