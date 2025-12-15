import cv2
import numpy as np
import mediapipe as mp
import config

class MediapipeHelper:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
            model_complexity=1,
            refine_face_landmarks=False
        )

    def detect_mediapipe(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def draw_styled_landmarks(self, image, results):
        # Yüz bağlantılarını çiz (Performans için kapatıldı/Sadeleştirildi)
        # if results.face_landmarks:
        #     self.mp_drawing.draw_landmarks(
        #         image, 
        #         results.face_landmarks, 
        #         self.mp_holistic.FACEMESH_TESSELATION, 
        #         self.mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
        #         self.mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        #     )
        
        # Sadece ana hatları çizmek isterseniz şunları açabilirsiniz (Daha hızlı):
        if results.face_landmarks:
             self.mp_drawing.draw_landmarks(
                 image, 
                 results.face_landmarks, 
                 self.mp_holistic.FACEMESH_CONTOURS, 
                 self.mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                 self.mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
             )
        
        # Vücut bağlantılarını çiz
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                self.mp_holistic.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                self.mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
            )
        
        # Sol el bağlantılarını çiz
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, 
                results.left_hand_landmarks, 
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                self.mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
            )
        
        # Sağ el bağlantılarını çiz
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, 
                results.right_hand_landmarks, 
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

    def extract_keypoints(self, results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])
    
    def close(self):
        self.holistic.close()
