import os
import cv2
import numpy as np
import time
import config
from utils import MediapipeHelper

class DataCollector:
    def __init__(self):
        self.mp_helper = MediapipeHelper()
        self.is_collecting = False
        self.should_stop = False

    def collect_step(self, frame, action_name, sequence_idx, frame_num, save_path):
        """
        Veri toplama için tek bir kareyi işler.
        Döndürür: İşlenmiş görüntü, anahtar noktalar
        """
        image, results = self.mp_helper.detect_mediapipe(frame)
        self.mp_helper.draw_styled_landmarks(image, results)
        
        # Anahtar noktaları dışa aktar
        keypoints = self.mp_helper.extract_keypoints(results)
        
        # Yol sağlanmışsa kaydet
        if save_path:
            npy_path = os.path.join(save_path, str(frame_num))
            np.save(npy_path, keypoints)
            
        return image

    def get_start_folder(self, action_name):
        action_path = os.path.join(config.DATA_PATH, action_name)
        if not os.path.exists(action_path):
            os.makedirs(action_path)
            return 0
        
        dir_list = os.listdir(action_path)
        numeric_dirs = [int(f) for f in dir_list if f.isdigit()]
        if numeric_dirs:
            return max(numeric_dirs) + 1
        return 0

    def close(self):
        self.mp_helper.close()
