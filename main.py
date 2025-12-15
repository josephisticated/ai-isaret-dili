import sys
import os
import numpy as np
import config
from data_collector import DataCollector
from model_trainer import ModelTrainer
from predictor import RealTimeTranslator

def get_actions():
    # Helper to get current actions from data folder
    if not os.path.exists(config.DATA_PATH):
        return []
    return [d for d in os.listdir(config.DATA_PATH) if os.path.isdir(os.path.join(config.DATA_PATH, d))]

def main():
    while True:
        print("\n--- İşaret Dili Uygulaması ---")
        print("1. Veri Topla (Yeni kelime öğret)")
        print("2. Modeli Eğit")
        print("3. Çeviriciyi Çalıştır")
        print("4. Çıkış")
        
        choice = input("Seçiminizi girin (1-4): ")
        
        if choice == '1':
            action_name = input("Öğretmek istediğiniz kelimeyi girin: ").strip()
            if action_name:
                collector = DataCollector()
                print(f"'{action_name}' için veri toplamaya hazır olun...")
                collector.collect_data(action_name)
            else:
                print("Geçersiz isim.")
                
        elif choice == '2':
            actions = get_actions()
            if not actions:
                print("Veri bulunamadı. Lütfen önce veri toplayın.")
                continue
            
            print(f"Şu hareketler üzerinde eğitiliyor: {actions}")
            trainer = ModelTrainer()
            trainer.train(actions)
            
        elif choice == '3':
            actions = get_actions()
            if not actions:
                print("Veri bulunamadı. Lütfen önce veri toplayın.")
                continue
            
            try:
                translator = RealTimeTranslator(actions)
                print("Çevirici başlatılıyor... Çıkmak için 'q' basın.")
                translator.run()
            except Exception as e:
                print(f"Hata: {e}")
                
        elif choice == '4':
            print("Çıkılıyor...")
            break
        else:
            print("Geçersiz seçim.")

if __name__ == "__main__":
    main()
