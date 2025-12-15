import os
import numpy as np

# Veri Yolları
DATA_PATH = os.path.join('MP_Data')
LOG_PATH = os.path.join('Logs')
MODEL_PATH = 'action.keras'

# Veri Toplama Parametreleri
NO_SEQUENCES = 30
SEQUENCE_LENGTH = 30
START_FOLDER = 0

# MediaPipe Parametreleri
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Model Parametreleri
EPOCHS = 100
