# Rahim Ağzı Kanseri Tespiti (Cervical Cancer Detection)

Bu proje, rahim ağzı kanseri hücre örneklerini sınıflandırmak için geliştirilmiş bir yapay zekâ uygulamasıdır. Farklı derin öğrenme modelleri (ResNet, Inception, vb.) denenmiş, en yüksek doğruluk oranı **VGG16** modeli ile elde edilmiştir. Flask tabanlı bir API üzerinden modelin kullanımı sağlanmıştır ve basit bir frontend arayüzü ile test edilebilir hale getirilmiştir.

## Özellikler
- VGG16 tabanlı derin öğrenme modeli
- Flask REST API ile model servisi
- `/predict` endpoint’i üzerinden görsel sınıflandırma
- Kullanıcı dostu frontend arayüzü (HTML, CSS, JS)
- Model sınıf etiketleri `class_indices.json` dosyasında tutulur

## Kurulum ve Çalıştırma
1. Sanal ortam oluştur ve aktif et:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements.txt
   cd backend
   python app_2.py

http://127.0.0.1:5000/


## Model Bilgisi

Kullanılan mimari: VGG16

Giriş boyutu: 224x224 piksel

Normalizasyon: [0, 1] aralığı

Denenen modeller arasında en yüksek doğruluk oranı VGG16 ile elde edilmiştir.
