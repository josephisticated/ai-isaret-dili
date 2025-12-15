import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Conv1D, MaxPooling1D, Flatten, Bidirectional, Input
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import config
import keras_tuner as kt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, confusion_matrix, f1_score
import time
import io
from PIL import Image

class ModelTrainer:
    def __init__(self):
        self.model = None
        self.actions = np.array([])
        self.best_hps = None

    def load_data(self, actions):
        self.actions = np.array(actions)
        label_map = {label:num for num, label in enumerate(self.actions)}
        sequences, labels = [], []

        for action in self.actions:
            action_path = os.path.join(config.DATA_PATH, action)
            if not os.path.exists(action_path):
                continue
                
            dir_list = np.array([f for f in os.listdir(action_path) if f.isdigit()]).astype(int)
            
            for sequence in dir_list:
                window = []
                is_valid = True
                for frame_num in range(config.SEQUENCE_LENGTH):
                    res_path = os.path.join(action_path, str(sequence), "{}.npy".format(frame_num))
                    if os.path.exists(res_path):
                        res = np.load(res_path)
                        window.append(res)
                    else:
                        is_valid = False
                        break
                
                if is_valid:
                    sequences.append(window)
                    labels.append(label_map[action])

        return np.array(sequences), to_categorical(labels).astype(int)

    def build_custom_model(self, model_type, units, dropout, lr, input_shape, output_shape):
        model = Sequential()
        
        if model_type == 'LSTM':
            model.add(Input(shape=input_shape))
            model.add(LSTM(units, return_sequences=True, activation='relu'))
            model.add(LSTM(units*2, return_sequences=True, activation='relu'))
            model.add(Dropout(dropout))
            model.add(LSTM(units, return_sequences=False, activation='relu'))
            model.add(Dropout(dropout))
            
        elif model_type == 'GRU':
            model.add(Input(shape=input_shape))
            model.add(GRU(units, return_sequences=True, activation='relu'))
            model.add(GRU(units*2, return_sequences=True, activation='relu'))
            model.add(Dropout(dropout))
            model.add(GRU(units, return_sequences=False, activation='relu'))
            model.add(Dropout(dropout))
            
        elif model_type == 'CNN':
            model.add(Input(shape=input_shape))
            model.add(Conv1D(filters=units, kernel_size=3, activation='relu'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Conv1D(filters=units*2, kernel_size=3, activation='relu'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())
            model.add(Dropout(dropout))

        elif model_type == 'Bi-LSTM':
            model.add(Input(shape=input_shape))
            model.add(Bidirectional(LSTM(units, return_sequences=True, activation='relu')))
            model.add(Bidirectional(LSTM(units, return_sequences=False, activation='relu')))
            model.add(Dropout(dropout))

        model.add(Dense(units, activation='relu'))
        model.add(Dense(units//2, activation='relu'))
        model.add(Dense(output_shape, activation='softmax'))
        
        model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        return model

    def build_tuner_model(self, hp, input_shape, output_shape):
        model_type = hp.Choice('model_type', ['LSTM', 'GRU', 'CNN', 'Bi-LSTM'])
        units = hp.Int('units', min_value=32, max_value=128, step=32)
        dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
        lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        
        return self.build_custom_model(model_type, units, dropout, lr, input_shape, output_shape)

    def train(self, actions, model_type='LSTM', units=64, dropout=0.2, lr=0.001, epochs=None, auto_tune=False, callback_fn=None):
        if epochs is None:
            epochs = config.EPOCHS
            
        X, y = self.load_data(actions)
        if len(X) == 0:
            print("Eğitilecek veri bulunamadı.")
            return None, None, None

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
        input_shape = (config.SEQUENCE_LENGTH, 1662)
        output_shape = y.shape[1]

        log_dir = os.path.join(config.LOG_PATH)
        tb_callback = TensorBoard(log_dir=log_dir)
        early_stopping = EarlyStopping(monitor='val_categorical_accuracy', patience=10, restore_best_weights=True)
        checkpoint = ModelCheckpoint(filepath=config.MODEL_PATH, monitor='val_categorical_accuracy', save_best_only=True)
        
        callbacks = [tb_callback, early_stopping, checkpoint]
        if callback_fn:
            from tensorflow.keras.callbacks import Callback
            class CustomGUICallback(Callback):
                def on_epoch_end(self, epoch, logs=None):
                    msg = f"Epoch {epoch+1}/{epochs} - loss: {logs['loss']:.4f} - acc: {logs['categorical_accuracy']:.4f}\n"
                    callback_fn(msg)
            callbacks.append(CustomGUICallback())

        history = None
        if auto_tune:
            if callback_fn: callback_fn("Otomatik Optimizasyon (Keras Tuner) Başlatılıyor...\nBu işlem biraz zaman alabilir.\n")
            
            tuner = kt.Hyperband(
                lambda hp: self.build_tuner_model(hp, input_shape, output_shape),
                objective='val_categorical_accuracy',
                max_epochs=epochs,
                factor=3,
                directory='my_dir',
                project_name='sign_language_tuning',
                overwrite=True
            )
            
            tuner.search(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), callbacks=[early_stopping])
            self.best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            
            best_model = tuner.get_best_models(num_models=1)[0]
            msg = f"""
            Bulunan En İyi Hiperparametreler:
            Model: {self.best_hps.get('model_type')}
            Units (Nöron): {self.best_hps.get('units')}
            Dropout: {self.best_hps.get('dropout')}
            LR (Öğrenme Hızı): {self.best_hps.get('learning_rate')}
            """
            if callback_fn: callback_fn(msg)
            print(msg)
            
            self.model = best_model
            self.model.save(config.MODEL_PATH)
            
        else:
            if callback_fn: callback_fn(f"{model_type} modeli eğitiliyor...\n")
            self.model = self.build_custom_model(model_type, units, dropout, lr, input_shape, output_shape)
            history = self.model.fit(X_train, y_train, epochs=epochs, callbacks=callbacks, validation_data=(X_test, y_test))
            self.model.save(config.MODEL_PATH)
            
        print("Model eğitildi.")
        return history, X_test, y_test

    def evaluate_model(self, X_test, y_test):
        # 1. Predictions
        yhat = self.model.predict(X_test)
        ytrue = np.argmax(y_test, axis=1).tolist()
        yhat = np.argmax(yhat, axis=1).tolist()
        
        # 2. Metrics
        acc = accuracy_score(ytrue, yhat)
        f1 = f1_score(ytrue, yhat, average='weighted')
        
        # 3. Confusion Matrix
        cm = confusion_matrix(ytrue, yhat)
        
        # 4. Inference Time (Latency)
        start_time = time.time()
        _ = self.model.predict(np.expand_dims(X_test[0], axis=0), verbose=0)
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        fps = 1000 / latency_ms if latency_ms > 0 else 0
        
        return {
            "accuracy": acc,
            "f1_score": f1,
            "confusion_matrix": cm,
            "latency_ms": latency_ms,
            "fps": fps
        }

    def plot_confusion_matrix(self, cm, actions):
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=actions, yticklabels=actions)
        plt.xlabel('Tahmin Edilen')
        plt.ylabel('Gerçek')
        plt.title('Confusion Matrix')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        plt.close()
        return img

    def plot_history(self, history):
        if not history: return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        ax1.plot(history.history['loss'], label='Eğitim Loss')
        ax1.plot(history.history['val_loss'], label='Doğrulama Loss')
        ax1.set_title('Loss Grafiği')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Accuracy
        ax2.plot(history.history['categorical_accuracy'], label='Eğitim Acc')
        ax2.plot(history.history['val_categorical_accuracy'], label='Doğrulama Acc')
        ax2.set_title('Doğruluk (Accuracy) Grafiği')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        plt.close()
        return img

    def load_trained_model(self, model_path=config.MODEL_PATH):
        if os.path.exists(model_path):
            self.model = load_model(model_path)
            self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
            return True
        else:
            return False
