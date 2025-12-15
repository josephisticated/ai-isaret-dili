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

    def apply_noise(self, keypoints, noise_level=0.05):
        noise = np.random.normal(0, noise_level, keypoints.shape)
        return keypoints + noise

    def augment_action(self, action_name, num_copies=1):
        action_path = os.path.join(config.DATA_PATH, action_name)
        if not os.path.exists(action_path):
            return 0

        # Get existing sequences
        sequences = [d for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d))]
        sequences = [int(s) for s in sequences if s.isdigit()]
        
        if not sequences:
            return 0

        start_idx = max(sequences) + 1
        count = 0

        for seq_idx in sequences:
            source_seq_path = os.path.join(action_path, str(seq_idx))
            
            # Read all frames in sequence
            frames = []
            valid_seq = True
            for frame_num in range(config.SEQUENCE_LENGTH):
                npy_path = os.path.join(source_seq_path, f"{frame_num}.npy")
                if os.path.exists(npy_path):
                    frames.append(np.load(npy_path))
                else:
                    valid_seq = False
                    break
            
            if not valid_seq:
                continue

            # Generate copies
            for _ in range(num_copies):
                target_seq_path = os.path.join(action_path, str(start_idx))
                os.makedirs(target_seq_path, exist_ok=True)
                
                for frame_num, frame_data in enumerate(frames):
                    augmented_data = self.apply_noise(frame_data)
                    np.save(os.path.join(target_seq_path, f"{frame_num}.npy"), augmented_data)
                
                start_idx += 1
                count += 1
                
        return count
