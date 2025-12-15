import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
# 2. Keypoints using MP Holistic
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections
def draw_styled_landmarks(image, results):
    # Yuz, pose ve elleri stilize sekilde ciz
    if results.face_landmarks:
        # HATA DUZELTILDİ: FACE_CONNECTIONS yerine FACEMESH_TESSELATION kullanıldı
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
cap = cv2.VideoCapture(0)
# Mediapipe model ayarlandi
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)
        cv2.imshow('OpenCV Feed', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])
# 4. Setup Folders for Collection
# Veri dizini ve eylemler
DATA_PATH = os.path.join('MP_Data')
actions = np.array(['hello','thanks','iloveyou'])
no_sequences = 30
sequence_length = 30
start_folder = 30

for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    if os.path.exists(action_path):
        file_list = os.listdir(action_path)
        numeric_files = [f for f in file_list if f.isdigit()]
        dirmax = np.max(np.array(numeric_files).astype(int)) if len(numeric_files)>0 else 0
    else:
        dirmax = 0
    for sequence in range(1, no_sequences+1):
        target_dir = os.path.join(DATA_PATH, action, str(dirmax+sequence))
        os.makedirs(target_dir, exist_ok=True)
# 5. Collect Keypoint Values for Training and Testing
import os
import cv2
import numpy as np

# --- DUZELTİLMESİ GEREKEN AYARLAR ---
DATA_PATH = os.path.join('MP_Data')
actions = np.array(['hello', 'thanks', 'iloveyou'])

# Modelin daha iyi öğrenmesi için veri sayısını artırıyoruz (30 video)
no_sequences = 30 

# KRİTİK: LSTM modelin 30 kare beklediği için bu değer 30 OLMALI (5 değil)
sequence_length = 30

start_folder = 0 # İlk kez çalıştırıyorsan 0, daha önce veri topladıysan son klasör numarasını yaz

cap = cv2.VideoCapture(0)

# MediaPipe Model Başlatma
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        for sequence in range(start_folder, start_folder + no_sequences):
            current_save_dir = os.path.join(DATA_PATH, action, str(sequence))
            os.makedirs(current_save_dir, exist_ok=True)
            
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret:
                    print("HATA: Kamera goruntusu alinmadi.")
                    break

                image, results = mediapipe_detection(frame, holistic)
                
                # Önceki adımda düzelttiğimiz fonksiyonu çağırdığından emin ol
                draw_styled_landmarks(image, results)
                
                # Ekrana Bilgi Yazdırma Mantığı
                if frame_num == 0:
                    cv2.putText(image, 'BASLIYOR...', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Video {action}: {sequence}', (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(500) # Hazırlanmak için 2 saniye bekle
                else:
                    cv2.putText(image, f'Video {action}: {sequence}', (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                
                # Keypointleri Çıkar ve Kaydet
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(current_save_dir, str(frame_num))
                np.save(npy_path, keypoints)
                
                # 'q' tuşu ile çıkış
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
    cap.release()
    cv2.destroyAllWindows()
# cap.release()
# cv2.destroyAllWindows()
# 6. Preprocess Data and Create Labels and Features
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
label_map = {label:num for num, label in enumerate(actions)}
label_map
sequences, labels = [], []

for action in actions:
    # Klasör listesini al ve integer'a çevir
    dir_list = np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int)
    
    for sequence in dir_list:
        # --- DÜZELTME 1: İSTENMEYEN KLASÖRÜ ATLA ---
        # Eğer klasör numarası 30 ise, bu turu pas geç
        if sequence == 30:
            continue
            
        window = []
        is_sequence_valid = True # Bu video sağlam mı kontrolü
        
        for frame_num in range(sequence_length):
            # --- DÜZELTME 2: GÜVENLİ YÜKLEME ---
            npy_path = os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num))
            
            # Dosya gerçekten var mı kontrol et (Hata almamak için)
            if os.path.exists(npy_path):
                res = np.load(npy_path)
                window.append(res)
            else:
                # Eğer tek bir kare bile eksikse bu videoyu komple iptal et
                print(f"UYARI: {action} klasöründeki {sequence} videosunun {frame_num}. karesi eksik. Video atlanıyor.")
                is_sequence_valid = False
                break 
        
        # Eğer video hatasız yüklendiyse listeye ekle
        if is_sequence_valid:
            sequences.append(window)
            labels.append(label_map[action])

print(f"Toplam yüklenen veri sayısı: {len(sequences)}")
def augment_data(X, y):
    """
    Mevcut veriye gürültü ekleyerek ve ölçekleyerek veriyi çoğaltır.
    """
    augmented_X = []
    augmented_y = []
    
    for i in range(len(X)):
        original_sample = X[i]
        original_label = y[i]
        
        # 1. ORİJİNAL VERİYİ EKLE
        augmented_X.append(original_sample)
        augmented_y.append(original_label)
        
        # 2. GÜRÜLTÜ EKLEME (JITTERING)
        # Koordinatlara ufak rastgele sayılar ekler (Titreme efekti)
        noise = np.random.normal(0, 0.05, original_sample.shape)
        noisy_sample = original_sample + noise
        augmented_X.append(noisy_sample)
        augmented_y.append(original_label)
        
        # 3. ÖLÇEKLEME (SCALING)
        # Veriyi %10 oranında büyütüp küçültür (Kameraya yaklaşma/uzaklaşma efekti)
        scale_factor = np.random.uniform(0.9, 1.1)
        scaled_sample = original_sample * scale_factor
        augmented_X.append(scaled_sample)
        augmented_y.append(original_label)

    return np.array(augmented_X), np.array(augmented_y)

# --- KULLANIM ---
# X ve y dizilerini oluşturduktan SONRA, train_test_split'ten ÖNCE bu kodu çalıştır:

print(f"Artırma öncesi veri sayısı: {len(X)}")

# Fonksiyonu çağır
X_augmented, y_augmented = augment_data(X, y)

# Yeni verileri ana değişkenlere ata
X = X_augmented
y = y_augmented

print(f"Artırma sonrası veri sayısı: {len(X)}")
print(f"Yeni X boyutu: {X.shape}")
X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
# 7. Build and Train LSTM Neural Network
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.callbacks import TensorBoard
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential()

# 1. Katman
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))

# 2. Katman
model.add(LSTM(128, return_sequences=True, activation='relu'))
# Dropout Eklemesi: Rastgele %20 nöronu kapatarak ezberlemeyi önler
model.add(Dropout(0.2)) 

# 3. Katman
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dropout(0.2))

# Dense Katmanlar
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

# Çıkış Katmanı (Action sayısı kadar nöron)
model.add(Dense(actions.shape[0], activation='softmax'))

# Modeli Derleme (Compile)
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Özet Görüntüleme
model.summary()
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=100, callbacks=[tb_callback])
model.summary()
# 8. Make Predictions
res = model.predict(X_test)
actions[np.argmax(res[0])]
actions[np.argmax(y_test[0])]
# 9. Save Weights
model.save('action.keras')
# del model
model.load_weights('action.keras')
# 10. Evaluation using Confusion Matrix and Accuracy
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
yhat = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()
multilabel_confusion_matrix(ytrue, yhat)
accuracy_score(ytrue, yhat)
# 11. Test in Real Time
from scipy import stats
colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame
plt.figure(figsize=(18,18))
plt.imshow(prob_viz(res, actions, image, colors))
# 1. New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.5

cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))
            
            
        #3. Viz logic
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5: 
                sentence = sentence[-5:]

            # Viz probabilities
            image = prob_viz(res, actions, image, colors)
            
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()